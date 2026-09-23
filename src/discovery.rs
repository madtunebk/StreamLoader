//! Resolves a user-supplied path (single file, index JSON, or directory)
//! into a concrete plan for which shard files to open, and enforces that
//! every shard path referenced from an index stays inside the checkpoint
//! root.

use std::collections::HashMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

use crate::error::{LoaderError, Result};

#[derive(Debug)]
pub enum Input {
    /// A single, self-contained `.safetensors` file.
    SingleFile(PathBuf),
    /// A `*.safetensors.index.json` file with a `weight_map`.
    Index(PathBuf),
    /// A directory containing multiple sibling `.safetensors` files and no
    /// index; treated as implicit shards of one model. Tensor names must
    /// be disjoint across them (checked while building descriptors) or the
    /// directory is rejected as ambiguous rather than silently merged.
    ImplicitShards(Vec<PathBuf>),
}

#[derive(Debug, Deserialize)]
pub struct IndexJson {
    pub weight_map: HashMap<String, String>,
}

pub fn resolve_input(path: &Path, allow_implicit_multi_shard: bool) -> Result<Input> {
    let meta = fs::metadata(path).map_err(|source| LoaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    if meta.is_file() {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if name.ends_with(".safetensors.index.json") {
            return Ok(Input::Index(path.to_path_buf()));
        }
        if name.ends_with(".safetensors") {
            return Ok(Input::SingleFile(path.to_path_buf()));
        }
        return Err(LoaderError::UnsupportedInput {
            path: path.to_path_buf(),
        });
    }

    if meta.is_dir() {
        let mut index_candidates = Vec::new();
        let mut shard_candidates = Vec::new();
        for entry in fs::read_dir(path).map_err(|source| LoaderError::Io {
            path: path.to_path_buf(),
            source,
        })? {
            let entry = entry.map_err(|source| LoaderError::Io {
                path: path.to_path_buf(),
                source,
            })?;
            let p = entry.path();
            if !p.is_file() {
                continue;
            }
            let name = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            if name.ends_with(".safetensors.index.json") {
                index_candidates.push(p);
            } else if name.ends_with(".safetensors") {
                shard_candidates.push(p);
            }
        }
        index_candidates.sort();
        shard_candidates.sort();

        if index_candidates.len() == 1 {
            return Ok(Input::Index(index_candidates.remove(0)));
        }
        if index_candidates.len() > 1 {
            return Err(LoaderError::AmbiguousDirectory {
                dir: path.to_path_buf(),
                reason: "multiple *.safetensors.index.json files found",
                candidates: index_candidates,
            });
        }
        if shard_candidates.is_empty() {
            return Err(LoaderError::NoShardsFound {
                dir: path.to_path_buf(),
            });
        }
        if shard_candidates.len() == 1 {
            return Ok(Input::SingleFile(shard_candidates.remove(0)));
        }
        if !allow_implicit_multi_shard {
            return Err(LoaderError::AmbiguousDirectory {
                dir: path.to_path_buf(),
                reason: "multiple .safetensors files with no index; disjoint tensor names alone \
                         don't prove they're shards of one model (they're also what several \
                         independent components dropped in the same folder would look like) — \
                         pass an explicit index/file path, or opt in via \
                         OpenOptions::allow_implicit_multi_shard if you already know these files \
                         belong to one checkpoint",
                candidates: shard_candidates,
            });
        }
        return Ok(Input::ImplicitShards(shard_candidates));
    }

    Err(LoaderError::UnsupportedInput {
        path: path.to_path_buf(),
    })
}

pub fn parse_index_json(path: &Path) -> Result<IndexJson> {
    let text = fs::read_to_string(path).map_err(|source| LoaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| LoaderError::InvalidIndexJson {
        path: path.to_path_buf(),
        source,
    })
}

/// Resolve a shard filename as it appears in an index's `weight_map`
/// against `root` (the index file's parent directory), and reject anything
/// that could escape it.
///
/// Resolution rule (documented per the mmap/path-safety requirement):
/// 1. The raw string must not be an absolute path.
/// 2. It must not contain any `..` component (rejected lexically, before
///    any filesystem access).
/// 3. The joined path is canonicalized (which resolves symlinks) and the
///    result must still start within the *trust boundary* — `trusted_root`
///    if the caller gave one, else `root` itself. This catches a shard
///    filename that looks like a plain relative path but is, or passes
///    through, a symlink pointing outside that boundary.
///
/// The shard file must already exist for step 3's canonicalization to
/// succeed; a dangling reference is reported as a plain I/O error rather
/// than a path-safety error.
///
/// # Why `trusted_root` exists
///
/// The default boundary (`root`, the index file's own directory) is the
/// strictest safe default: a downloaded/untrusted index.json paired with a
/// same-named local symlink cannot make this loader read anything outside
/// that one directory. But real Hugging Face hub caches deliberately
/// symlink shard files *out* of the snapshot directory into a
/// content-addressed `blobs/` directory shared across snapshots of the
/// same repo (for on-disk dedup) — that is a legitimate, standard layout,
/// not an attack, and the strict default would reject every such model.
/// A caller who already trusts a wider directory (e.g. their own local hub
/// cache entry for one specific repo) can pass it explicitly as
/// `trusted_root`; this is never inferred or auto-detected from the
/// directory layout, only ever set by explicit caller opt-in.
pub fn resolve_shard_path(
    root: &Path,
    trusted_root: Option<&Path>,
    index_path: &Path,
    raw: &str,
) -> Result<PathBuf> {
    let candidate = Path::new(raw);
    if candidate.is_absolute() {
        return Err(LoaderError::UnsafeShardPath {
            index_path: index_path.to_path_buf(),
            raw: raw.to_string(),
            reason: "absolute shard paths are not allowed",
        });
    }
    for component in candidate.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => {
                return Err(LoaderError::UnsafeShardPath {
                    index_path: index_path.to_path_buf(),
                    raw: raw.to_string(),
                    reason: "'..' path traversal is not allowed",
                });
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(LoaderError::UnsafeShardPath {
                    index_path: index_path.to_path_buf(),
                    raw: raw.to_string(),
                    reason: "absolute shard paths are not allowed",
                });
            }
        }
    }

    let joined = root.join(candidate);
    let boundary = trusted_root.unwrap_or(root);
    let canonical_boundary = boundary.canonicalize().map_err(|source| LoaderError::Io {
        path: boundary.to_path_buf(),
        source,
    })?;
    let canonical_joined = joined.canonicalize().map_err(|source| LoaderError::Io {
        path: joined.clone(),
        source,
    })?;
    if !canonical_joined.starts_with(&canonical_boundary) {
        return Err(LoaderError::UnsafeShardPath {
            index_path: index_path.to_path_buf(),
            raw: raw.to_string(),
            reason: "resolves outside the trusted checkpoint root (possible symlink escape)",
        });
    }

    Ok(canonical_joined)
}
