//! Rust model loader v1: mmap-backed SafeTensors indexing and logical
//! transformer-block grouping.
//!
//! This is a weight loader, not an inference engine. It opens original
//! SafeTensors checkpoints (single file, sharded with an index, or a
//! directory of implicit shards) read-only via `mmap`, builds owned
//! byte-range descriptors for every tensor without materializing payloads,
//! and groups tensors into logical blocks by configurable name prefixes.
//!
//! `mmap` is lazy virtual-memory mapping, not a guarantee of residency,
//! pinning, or GPU placement — see [`Model`] and `shard` for the exact
//! safety/consistency contract this relies on.

pub mod block;
pub mod descriptor;
pub mod discovery;
pub mod error;
pub mod model;
pub mod shard;

pub use block::{Block, BlockConfig, BlockFamily, UNASSIGNED};
pub use descriptor::{ShardId, TensorDescriptor};
pub use error::{LoaderError, Result};
pub use model::{CopiedBlock, Model, OpenOptions, TensorChecksum, TensorLayout};
pub use safetensors::Dtype;
