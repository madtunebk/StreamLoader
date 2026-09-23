use safetensors::Dtype;

/// Which on-disk shard a tensor's bytes live in, as an index into
/// [`crate::model::Model::shards`].
pub type ShardId = usize;

/// An owned, lightweight description of one tensor: everything needed to
/// locate and interpret its bytes, without holding any of the bytes
/// themselves. Cheap to clone and to collect into `Vec`s for JSON output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    /// Exact original tensor name, unmodified.
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub shard_id: ShardId,
    /// Absolute byte offset from the start of the shard file (i.e. already
    /// includes the 8-byte length prefix and the JSON header — this is a
    /// file offset, not a SafeTensors "data section" offset).
    pub file_offset: u64,
    pub byte_len: u64,
}

impl TensorDescriptor {
    pub fn file_range(&self) -> std::ops::Range<u64> {
        self.file_offset..self.file_offset + self.byte_len
    }
}
