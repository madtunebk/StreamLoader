use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::block::{self, Block, BlockConfig};
use crate::descriptor::{ShardId, TensorDescriptor};
use crate::discovery::{self, Input};
use crate::error::{LoaderError, Result};
use crate::shard::Shard;

/// Process-wide counter handing out a unique id to each `Model` as it's
/// constructed, so a `Block` can record which `Model` it came from and
/// later calls can reject a `Block` from a *different* `Model` instead of
/// silently reading whatever happens to be at that shard/offset in the
/// wrong checkpoint. Starts at 1 so 0 can mean "not yet stamped" (the
/// placeholder used by the free functions in `block.rs`, which don't know
/// about `Model` at all).
static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

/// Tunables for opening a model. `max_header_bytes` bounds how large a
/// declared SafeTensors JSON header we accept per shard, checked before
/// that header is parsed — independent of, and typically much smaller
/// than, the `safetensors` crate's own fixed 100MB ceiling.
#[derive(Debug, Clone)]
pub struct OpenOptions {
    pub max_header_bytes: u64,
    /// Explicit, caller-opted-in widening of the symlink-escape trust
    /// boundary for index-referenced shard paths. `None` (the default)
    /// means "no shard may resolve outside the index file's own
    /// directory" — the strictest safe default. See
    /// [`crate::discovery::resolve_shard_path`] for why a real Hugging
    /// Face hub cache (which symlinks shards into a shared `blobs/`
    /// sibling directory) needs this widened explicitly to be usable.
    pub trusted_root: Option<PathBuf>,
    /// Whether a directory with multiple sibling `.safetensors` files and
    /// no index may be treated as implicit shards of *one* model. `false`
    /// (the default) rejects such a directory as ambiguous instead:
    /// disjoint tensor names alone don't prove several files belong to
    /// the same model — they're also exactly what you'd see if the
    /// directory actually holds several independent components (e.g. a
    /// text encoder and a VAE dropped in the same folder). Set this to
    /// `true` only when you already know every `.safetensors` file in the
    /// directory is a genuine shard of a single checkpoint.
    pub allow_implicit_multi_shard: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            max_header_bytes: 64 * 1024 * 1024,
            trusted_root: None,
            allow_implicit_multi_shard: false,
        }
    }
}

/// An opened, indexed checkpoint: one or more memory-mapped shards plus an
/// owned table of tensor descriptors. Holds no tensor payload in owned
/// heap buffers — only byte-range metadata. Byte views handed out by
/// [`Model::tensor_bytes`] borrow directly from the shards' mmaps.
#[derive(Debug)]
pub struct Model {
    id: u64,
    shards: Vec<Shard>,
    tensors: BTreeMap<String, TensorDescriptor>,
    root_input: PathBuf,
}

impl Model {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, &OpenOptions::default())
    }

    pub fn open_with(path: impl AsRef<Path>, opts: &OpenOptions) -> Result<Self> {
        let path = path.as_ref();
        let input = discovery::resolve_input(path, opts.allow_implicit_multi_shard)?;
        let mut shards: Vec<Shard> = Vec::new();
        let mut tensors: BTreeMap<String, TensorDescriptor> = BTreeMap::new();

        match input {
            Input::SingleFile(shard_path) => {
                let (shard, metadata) = Shard::open(&shard_path, opts.max_header_bytes)?;
                let shard_id = shards.len();
                shards.push(shard);
                let mut names: Vec<String> = metadata.tensors().into_keys().collect();
                names.sort();
                for name in names {
                    let info = metadata
                        .info(&name)
                        .expect("name came from metadata.tensors()");
                    insert_descriptor(&mut tensors, &shards, shard_id, &name, info)?;
                }
            }
            Input::Index(index_path) => {
                let index = discovery::parse_index_json(&index_path)?;
                let root = index_path.parent().unwrap_or_else(|| Path::new("."));

                let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::new();
                for (tensor_name, shard_raw) in &index.weight_map {
                    by_shard
                        .entry(shard_raw.clone())
                        .or_default()
                        .push(tensor_name.clone());
                }

                for (shard_raw, mut tensor_names) in by_shard {
                    tensor_names.sort();
                    let resolved = discovery::resolve_shard_path(
                        root,
                        opts.trusted_root.as_deref(),
                        &index_path,
                        &shard_raw,
                    )?;
                    let (shard, metadata) = Shard::open(&resolved, opts.max_header_bytes)?;
                    let shard_id = shards.len();
                    shards.push(shard);

                    let mut seen: std::collections::HashSet<String> =
                        std::collections::HashSet::new();
                    for tensor_name in &tensor_names {
                        let info = metadata.info(tensor_name).ok_or_else(|| {
                            LoaderError::TensorMissingFromShard {
                                index_path: index_path.clone(),
                                shard_path: shards[shard_id].path.clone(),
                                tensor: tensor_name.clone(),
                            }
                        })?;
                        insert_descriptor(&mut tensors, &shards, shard_id, tensor_name, info)?;
                        seen.insert(tensor_name.clone());
                    }

                    for (name, _info) in metadata.tensors() {
                        if !seen.contains(&name) {
                            return Err(LoaderError::TensorMissingFromIndex {
                                index_path: index_path.clone(),
                                shard_path: shards[shard_id].path.clone(),
                                tensor: name,
                            });
                        }
                    }
                }
            }
            Input::ImplicitShards(paths) => {
                for shard_path in paths {
                    let (shard, metadata) = Shard::open(&shard_path, opts.max_header_bytes)?;
                    let shard_id = shards.len();
                    shards.push(shard);
                    let mut names: Vec<String> = metadata.tensors().into_keys().collect();
                    names.sort();
                    for name in names {
                        let info = metadata
                            .info(&name)
                            .expect("name came from metadata.tensors()");
                        insert_descriptor(&mut tensors, &shards, shard_id, &name, info)?;
                    }
                }
            }
        }

        Ok(Model {
            id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            shards,
            tensors,
            root_input: path.to_path_buf(),
        })
    }

    /// Opaque identity of this opened model. Two `Model`s opened from the
    /// identical path are still distinct instances with distinct ids
    /// (e.g. `tests/sharded.rs`'s two `Model::open` calls for the same
    /// directory) — this identifies *this in-memory instance*, not the
    /// checkpoint's location on disk.
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn root_input(&self) -> &Path {
        &self.root_input
    }

    pub fn shard_paths(&self) -> impl Iterator<Item = &Path> {
        self.shards.iter().map(|s| s.path.as_path())
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn descriptors(&self) -> impl Iterator<Item = &TensorDescriptor> {
        self.tensors.values()
    }

    pub fn descriptor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(name)
    }

    /// Zero-copy byte view for one tensor, borrowed from the owning
    /// shard's mmap. Never reads/copies bytes beyond this tensor's range.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let d = self
            .tensors
            .get(name)
            .ok_or_else(|| LoaderError::TensorNotFound(name.to_string()))?;
        let shard = &self.shards[d.shard_id];
        shard.byte_range(name, d.file_offset, d.file_offset + d.byte_len)
    }

    /// Group all tensors into logical blocks per `config`. Metadata-only —
    /// never touches tensor payload bytes.
    ///
    /// Callers doing this repeatedly in a hot loop (e.g. a prefetch
    /// scheduler walking blocks in order) should call this once and reuse
    /// the returned `Vec<Block>` rather than calling it again per lookup —
    /// this method itself does the full grouping pass every time it's
    /// called, it does not memoize across calls.
    pub fn blocks(&self, config: &BlockConfig) -> Vec<Block> {
        let all: Vec<TensorDescriptor> = self.tensors.values().cloned().collect();
        block::group_blocks(&all, config)
            .into_iter()
            .map(|mut b| {
                b.model_id = self.id;
                b
            })
            .collect()
    }

    /// Descriptors for exactly one block id. Errors if nothing matched —
    /// `transformer_blocks.1` never silently matches `transformer_blocks.10`.
    /// Unlike `blocks()`, this does not build every other block first.
    pub fn block(&self, config: &BlockConfig, block_id: &str) -> Result<Block> {
        let all: Vec<TensorDescriptor> = self.tensors.values().cloned().collect();
        block::find_block(&all, config, block_id)
            .map(|mut b| {
                b.model_id = self.id;
                b
            })
            .ok_or_else(|| LoaderError::BlockNotFound(block_id.to_string()))
    }

    /// Every `Model::block`/`Model::blocks` result-consuming method starts
    /// with this: reject a `Block` that didn't come from `self`. Without
    /// this, `model_a.copy_block(&block_from_model_b)` would silently
    /// index into `model_a`'s shards using `model_b`'s byte offsets —
    /// wrong data at best, an out-of-bounds error at worst, neither of
    /// which says "you mixed up two models."
    fn check_provenance(&self, blk: &Block) -> Result<()> {
        if blk.model_id != self.id {
            return Err(LoaderError::BlockFromDifferentModel {
                block_id: blk.id.clone(),
            });
        }
        Ok(())
    }

    /// Zero-copy byte views for every tensor in `block`, in the block's
    /// tensor order.
    pub fn block_views<'a>(&'a self, blk: &'a Block) -> Result<Vec<(&'a str, &'a [u8])>> {
        self.check_provenance(blk)?;
        blk.tensors
            .iter()
            .map(|d| {
                self.tensor_bytes(&d.name)
                    .map(|bytes| (d.name.as_str(), bytes))
            })
            .collect()
    }

    /// Explicitly copy every tensor in `block` into one caller-owned
    /// buffer, distinct from the zero-copy view API. Returns the buffer,
    /// the offset layout within it, and total bytes copied.
    pub fn copy_block(&self, blk: &Block) -> Result<CopiedBlock> {
        self.check_provenance(blk)?;
        let total: u64 = blk.tensors.iter().map(|d| d.byte_len).sum();
        let mut buffer = vec![0u8; total as usize];
        let layout = self.copy_block_into(blk, &mut buffer)?;
        Ok(CopiedBlock {
            buffer,
            layout,
            bytes_copied: total,
        })
    }

    /// Same as `copy_block`, but writes into a caller-provided
    /// destination instead of allocating a fresh `Vec` — for a caller
    /// that already owns a buffer (e.g. a pinned host buffer) and wants
    /// this crate to never allocate on their behalf. Errors if `dest` is
    /// smaller than the block's total byte length; never partially
    /// writes past a validated size mismatch.
    pub fn copy_block_into(&self, blk: &Block, dest: &mut [u8]) -> Result<Vec<TensorLayout>> {
        self.check_provenance(blk)?;
        let total: u64 = blk.tensors.iter().map(|d| d.byte_len).sum();
        if (dest.len() as u64) < total {
            return Err(LoaderError::DestinationTooSmall {
                needed: total,
                got: dest.len() as u64,
            });
        }
        let mut layout = Vec::with_capacity(blk.tensors.len());
        let mut offset: u64 = 0;
        for d in &blk.tensors {
            let bytes = self.tensor_bytes(&d.name)?;
            let start = offset as usize;
            let end = start + bytes.len();
            dest[start..end].copy_from_slice(bytes);
            layout.push(TensorLayout {
                name: d.name.clone(),
                offset,
                len: d.byte_len,
            });
            offset += d.byte_len;
        }
        Ok(layout)
    }

    /// Deterministic per-tensor checksum (BLAKE3 of the exact raw tensor
    /// bytes, dtype-agnostic) for one tensor. This reads the tensor's
    /// payload bytes and may cause disk I/O; it is not a metadata-only
    /// operation.
    pub fn checksum_tensor(&self, name: &str) -> Result<TensorChecksum> {
        let bytes = self.tensor_bytes(name)?;
        let hash = blake3::hash(bytes);
        Ok(TensorChecksum {
            name: name.to_string(),
            algorithm: "blake3",
            hex: hash.to_hex().to_string(),
            bytes: bytes.len() as u64,
        })
    }

    pub fn checksum_block(&self, blk: &Block) -> Result<Vec<TensorChecksum>> {
        self.check_provenance(blk)?;
        blk.tensors
            .iter()
            .map(|d| self.checksum_tensor(&d.name))
            .collect()
    }
}

/// Layout entry for one tensor within a [`CopiedBlock`]'s buffer.
#[derive(Debug, Clone)]
pub struct TensorLayout {
    pub name: String,
    pub offset: u64,
    pub len: u64,
}

/// Result of [`Model::copy_block`]: an explicit, caller-owned copy —
/// distinct from and never required by the zero-copy view API.
#[derive(Debug, Clone)]
pub struct CopiedBlock {
    pub buffer: Vec<u8>,
    pub layout: Vec<TensorLayout>,
    pub bytes_copied: u64,
}

#[derive(Debug, Clone)]
pub struct TensorChecksum {
    pub name: String,
    pub algorithm: &'static str,
    pub hex: String,
    pub bytes: u64,
}

/// Build one descriptor and insert it, or fail with contextual
/// `DuplicateTensorName`/overflow errors. `shards` must already contain
/// the shard identified by `shard_id` (pushed before this is called), so
/// duplicate errors can report both the first and second source path.
fn insert_descriptor(
    tensors: &mut BTreeMap<String, TensorDescriptor>,
    shards: &[Shard],
    shard_id: ShardId,
    name: &str,
    info: &safetensors::tensor::TensorInfo,
) -> Result<()> {
    let current_shard = &shards[shard_id];
    if let Some(existing) = tensors.get(name) {
        return Err(LoaderError::DuplicateTensorName {
            tensor: name.to_string(),
            first_path: shards[existing.shard_id].path.clone(),
            second_path: current_shard.path.clone(),
        });
    }
    let (rel_start, rel_end) = (info.data_offsets.0 as u64, info.data_offsets.1 as u64);
    let (start, end) = current_shard.absolute_range(name, rel_start, rel_end)?;
    let byte_len = end
        .checked_sub(start)
        .ok_or_else(|| LoaderError::OffsetOverflow {
            path: current_shard.path.clone(),
            tensor: name.to_string(),
        })?;
    tensors.insert(
        name.to_string(),
        TensorDescriptor {
            name: name.to_string(),
            dtype: info.dtype,
            shape: info.shape.clone(),
            shard_id,
            file_offset: start,
            byte_len,
        },
    );
    Ok(())
}
