use std::path::PathBuf;

use safetensors::tensor::SafeTensorError;

/// All errors this crate can return. Every variant carries enough context
/// (path, tensor/block name, offsets) to act on without re-deriving it.
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("failed to open {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to mmap {path}: {source}")]
    Mmap {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "header of {path} declares {declared} bytes, which exceeds the configured limit of {limit} bytes"
    )]
    HeaderTooLarge {
        path: PathBuf,
        declared: u64,
        limit: u64,
    },

    #[error("{path} is smaller than the 8-byte SafeTensors header length prefix")]
    HeaderTooSmall { path: PathBuf },

    #[error("failed to parse/validate SafeTensors header of {path}: {source}")]
    InvalidHeader {
        path: PathBuf,
        #[source]
        source: SafeTensorError,
    },

    #[error("arithmetic overflow computing byte range for tensor {tensor} in {path}")]
    OffsetOverflow { path: PathBuf, tensor: String },

    #[error(
        "tensor {tensor} in {path} has byte range [{start}, {end}) which is out of bounds for a file of {file_len} bytes"
    )]
    OutOfBounds {
        path: PathBuf,
        tensor: String,
        start: u64,
        end: u64,
        file_len: u64,
    },

    #[error("shard path {raw:?} referenced from {index_path} is rejected: {reason}")]
    UnsafeShardPath {
        index_path: PathBuf,
        raw: String,
        reason: &'static str,
    },

    #[error("could not parse index JSON {path}: {source}")]
    InvalidIndexJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error(
        "tensor {tensor} is listed in index {index_path} for shard {shard_path} but that shard does not contain a tensor with that name"
    )]
    TensorMissingFromShard {
        index_path: PathBuf,
        shard_path: PathBuf,
        tensor: String,
    },

    #[error(
        "shard {shard_path} contains tensor {tensor}, but index {index_path} does not list it in weight_map"
    )]
    TensorMissingFromIndex {
        index_path: PathBuf,
        shard_path: PathBuf,
        tensor: String,
    },

    #[error(
        "duplicate tensor name {tensor:?}: first seen in {first_path}, seen again in {second_path}"
    )]
    DuplicateTensorName {
        tensor: String,
        first_path: PathBuf,
        second_path: PathBuf,
    },

    #[error("no .safetensors files found under {dir}")]
    NoShardsFound { dir: PathBuf },

    #[error(
        "ambiguous directory {dir}: {reason} ({candidates:?}); pass an explicit index or file path"
    )]
    AmbiguousDirectory {
        dir: PathBuf,
        reason: &'static str,
        candidates: Vec<PathBuf>,
    },

    #[error(
        "unsupported input path {path}: expected a .safetensors file, a *.safetensors.index.json file, or a directory"
    )]
    UnsupportedInput { path: PathBuf },

    #[error("tensor {0:?} not found")]
    TensorNotFound(String),

    #[error("block {0:?} not found (no tensor matched this block id)")]
    BlockNotFound(String),

    #[error(
        "block {block_id:?} was produced by a different Model instance than the one it was passed to \
         (mismatched provenance) — descriptors from one model must not be read through another model's shards"
    )]
    BlockFromDifferentModel { block_id: String },

    #[error("destination buffer too small: block needs {needed} bytes, got {got}")]
    DestinationTooSmall { needed: u64, got: u64 },

    #[error("block family {0:?} is not configured")]
    UnknownFamily(String),

    #[error("invalid usage: {0}")]
    InvalidUsage(String),
}

pub type Result<T> = std::result::Result<T, LoaderError>;
