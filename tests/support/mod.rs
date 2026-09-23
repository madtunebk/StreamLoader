//! Shared fixture-generation helpers for integration tests. Builds small,
//! deterministic SafeTensors files/directories entirely in Rust using the
//! real `safetensors` crate serializer, so fixtures are guaranteed to be
//! valid per the format (no hand-rolled byte layout to get subtly wrong).

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::Dtype;
use safetensors::tensor::TensorView;

pub struct FakeTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

/// An F32 tensor filled with a deterministic, seed-dependent ramp so two
/// fixtures built with different seeds are guaranteed to differ, and the
/// same seed always reproduces identical bytes.
pub fn f32_tensor(name: &str, shape: Vec<usize>, seed: u32) -> FakeTensor {
    let n: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n * 4);
    for i in 0..n {
        let v = seed as f32 + (i as f32) * 0.5;
        data.extend_from_slice(&v.to_le_bytes());
    }
    FakeTensor {
        name: name.to_string(),
        dtype: Dtype::F32,
        shape,
        data,
    }
}

pub fn scalar_f32(name: &str, value: f32) -> FakeTensor {
    FakeTensor {
        name: name.to_string(),
        dtype: Dtype::F32,
        shape: vec![],
        data: value.to_le_bytes().to_vec(),
    }
}

pub fn empty_tensor(name: &str, shape: Vec<usize>) -> FakeTensor {
    // A tensor whose declared shape has a zero dimension has zero elements
    // and therefore zero payload bytes, regardless of the other dims.
    FakeTensor {
        name: name.to_string(),
        dtype: Dtype::F32,
        shape,
        data: Vec::new(),
    }
}

/// Serialize `tensors` into one valid `.safetensors` file at `path` using
/// `safetensors::serialize` (the real crate's own writer).
pub fn write_safetensors_file(path: &Path, tensors: &[FakeTensor]) -> Vec<u8> {
    let mut sorted: Vec<&FakeTensor> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));

    let views: Vec<(String, TensorView)> = sorted
        .iter()
        .map(|t| {
            let view = TensorView::new(t.dtype, t.shape.clone(), &t.data)
                .expect("fixture tensor shape/dtype/data length must agree");
            (t.name.clone(), view)
        })
        .collect();

    let bytes = safetensors::serialize(views, None).expect("serialize fixture");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, &bytes).expect("write fixture file");
    bytes
}

pub struct ShardSpec {
    pub filename: &'static str,
    pub tensors: Vec<FakeTensor>,
}

/// Write a sharded model: one `.safetensors` file per `ShardSpec`, plus a
/// `model.safetensors.index.json` with a `weight_map` covering every
/// tensor. Returns the index file path.
pub fn write_sharded_model(dir: &Path, shards: &[ShardSpec]) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
    for shard in shards {
        let path = dir.join(shard.filename);
        write_safetensors_file(&path, &shard.tensors);
        for t in &shard.tensors {
            weight_map.insert(t.name.clone(), shard.filename.to_string());
        }
    }
    let index = serde_json::json!({
        "metadata": {"format": "pt"},
        "weight_map": weight_map,
    });
    let index_path = dir.join("model.safetensors.index.json");
    fs::write(&index_path, serde_json::to_vec_pretty(&index).unwrap()).unwrap();
    index_path
}

/// A small FLUX-shaped set of block tensors: two `transformer_blocks`
/// (indices 2 and 10, to exercise numeric-vs-lexicographic ordering), one
/// `single_transformer_blocks`, and one clearly unrelated shared tensor —
/// spread across two shards so at least one block spans shards.
pub fn flux_like_shards() -> Vec<ShardSpec> {
    vec![
        ShardSpec {
            filename: "shard-00001-of-00002.safetensors",
            tensors: vec![
                f32_tensor("transformer_blocks.2.attn.qkv.weight", vec![4, 4], 2),
                f32_tensor("transformer_blocks.2.attn.qkv.bias", vec![4], 20),
                f32_tensor("transformer_blocks.10.attn.qkv.weight", vec![4, 4], 10),
                f32_tensor("single_transformer_blocks.0.proj.weight", vec![4, 4], 100),
                scalar_f32("shared.time_embed.scale", 1.5),
            ],
        },
        ShardSpec {
            filename: "shard-00002-of-00002.safetensors",
            tensors: vec![
                // Same logical block (transformer_blocks.2) continues in
                // the second shard: a block spanning shards.
                f32_tensor("transformer_blocks.2.mlp.fc1.weight", vec![4, 8], 21),
                f32_tensor("transformer_blocks.10.attn.qkv.bias", vec![4], 101),
                f32_tensor("single_transformer_blocks.0.proj.bias", vec![4], 101),
            ],
        },
    ]
}
