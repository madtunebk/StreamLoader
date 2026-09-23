use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::tensor::{Metadata, SafeTensors};

use crate::error::{LoaderError, Result};

/// Length in bytes of the little-endian u64 header-length prefix every
/// SafeTensors file starts with.
const HEADER_LEN_PREFIX: u64 = 8;

/// One opened, memory-mapped checkpoint shard.
///
/// # Safety contract (mmap)
///
/// This holds a read-only `memmap2::Mmap` over `path`. Per the `memmap2`
/// contract, the backing file must not be modified or truncated for as
/// long as this mapping is alive: doing so is undefined behavior (it can
/// produce a SIGBUS on access, or silently expose different bytes than
/// what was validated at open time), because the OS maps pages of the
/// file lazily and on demand rather than copying them up front. Opening
/// the file read-only prevents *this process* from writing through the
/// mapping, but it does not stop another process, or the same file being
/// replaced/truncated on disk, from invalidating pages after they were
/// validated here. Callers that need that guarantee must arrange their
/// own external invariant (e.g. don't point this at files a concurrent
/// writer can touch).
#[derive(Debug)]
pub struct Shard {
    pub path: PathBuf,
    mmap: Mmap,
    /// Absolute file offset where tensor payload bytes begin, i.e.
    /// `8 + header_json_len`.
    data_start: u64,
}

impl Shard {
    /// Open and mmap `path`, then parse and validate its SafeTensors
    /// header using the `safetensors` crate. `max_header_bytes` bounds how
    /// large a declared header we're willing to let the JSON parser touch,
    /// checked *before* handing the buffer to `safetensors`, independent of
    /// that crate's own fixed 100MB ceiling.
    ///
    /// Returns the opened shard plus the fully validated, owned tensor
    /// metadata (name -> dtype/shape/relative offsets) so the caller can
    /// build [`crate::descriptor::TensorDescriptor`]s. `Metadata` here is
    /// fully owned (no borrow of the mmap), so no self-referential struct
    /// is needed to keep it around after this call returns.
    pub fn open(path: &Path, max_header_bytes: u64) -> Result<(Self, Metadata)> {
        let file = File::open(path).map_err(|source| LoaderError::Io {
            path: path.to_path_buf(),
            source,
        })?;

        // SAFETY: `file` is opened read-only and kept open only for the
        // duration of this call (mmap does not need the fd afterwards).
        // The caller of `Shard::open` (the discovery/model layer) is
        // responsible for the broader "don't mutate files we've mapped"
        // contract described on `Shard` itself; nothing below assumes
        // exclusive access to the file, only that its bytes are readable.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| LoaderError::Mmap {
            path: path.to_path_buf(),
            source,
        })?;

        if (mmap.len() as u64) < HEADER_LEN_PREFIX {
            return Err(LoaderError::HeaderTooSmall {
                path: path.to_path_buf(),
            });
        }

        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&mmap[0..8]);
        let declared_header_len = u64::from_le_bytes(len_bytes);
        if declared_header_len > max_header_bytes {
            return Err(LoaderError::HeaderTooLarge {
                path: path.to_path_buf(),
                declared: declared_header_len,
                limit: max_header_bytes,
            });
        }

        let (header_len, metadata) =
            SafeTensors::read_metadata(&mmap).map_err(|source| LoaderError::InvalidHeader {
                path: path.to_path_buf(),
                source,
            })?;

        let data_start = HEADER_LEN_PREFIX
            .checked_add(header_len as u64)
            .ok_or_else(|| LoaderError::OffsetOverflow {
                path: path.to_path_buf(),
                tensor: String::new(),
            })?;

        Ok((
            Shard {
                path: path.to_path_buf(),
                mmap,
                data_start,
            },
            metadata,
        ))
    }

    /// Absolute file offset for a byte range that is relative to the
    /// SafeTensors data section (i.e. as stored in `TensorInfo::data_offsets`).
    pub fn absolute_range(&self, tensor: &str, rel_start: u64, rel_end: u64) -> Result<(u64, u64)> {
        let start =
            self.data_start
                .checked_add(rel_start)
                .ok_or_else(|| LoaderError::OffsetOverflow {
                    path: self.path.clone(),
                    tensor: tensor.to_string(),
                })?;
        let end =
            self.data_start
                .checked_add(rel_end)
                .ok_or_else(|| LoaderError::OffsetOverflow {
                    path: self.path.clone(),
                    tensor: tensor.to_string(),
                })?;
        Ok((start, end))
    }

    /// Borrow a byte range `[start, end)` (absolute file offsets) as a
    /// zero-copy slice into the mmap. The returned slice's lifetime is
    /// tied to `&self`, which is tied to the `Shard` (and transitively the
    /// `Model` that owns it) — never detached or transmuted.
    pub fn byte_range(&self, tensor: &str, start: u64, end: u64) -> Result<&[u8]> {
        let file_len = self.mmap.len() as u64;
        if start > end || end > file_len {
            return Err(LoaderError::OutOfBounds {
                path: self.path.clone(),
                tensor: tensor.to_string(),
                start,
                end,
                file_len,
            });
        }
        Ok(&self.mmap[start as usize..end as usize])
    }

    pub fn file_len(&self) -> u64 {
        self.mmap.len() as u64
    }
}
