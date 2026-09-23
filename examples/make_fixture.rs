//! Generates a small, deterministic, FLUX-shaped dummy checkpoint directory
//! for manually poking at with the CLI (or a script driving the CLI).
//! This is not a performance stand-in for a real 28GB checkpoint -- it is
//! just enough structure (multiple shards, an index, numeric block indices
//! including a 2-vs-10 case, and one shared/unassigned tensor) to exercise
//! every CLI command end to end.
//!
//! Usage: cargo run --release --example make_fixture -- <out_dir>

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

use safetensors::Dtype;
use safetensors::tensor::TensorView;

struct FakeTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn f32_tensor(shape: &[usize], seed: u32) -> FakeTensor {
    let n: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n * 4);
    for i in 0..n {
        let v = seed as f32 + (i as f32) * 0.5;
        data.extend_from_slice(&v.to_le_bytes());
    }
    FakeTensor {
        dtype: Dtype::F32,
        shape: shape.to_vec(),
        data,
    }
}

fn write_shard(path: &Path, tensors: Vec<(&str, FakeTensor)>) -> Vec<String> {
    let mut sorted = tensors;
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut names = Vec::new();
    let views: Vec<(String, TensorView)> = sorted
        .iter()
        .map(|(name, t)| {
            names.push(name.to_string());
            (
                name.to_string(),
                TensorView::new(t.dtype, t.shape.clone(), &t.data).expect("valid fixture tensor"),
            )
        })
        .collect();
    let bytes = safetensors::serialize(views, None).expect("serialize fixture shard");
    fs::write(path, bytes).expect("write shard file");
    names
}

fn main() {
    let out_dir = env::args()
        .nth(1)
        .unwrap_or_else(|| "./fixture_checkpoint".to_string());
    let out_dir = Path::new(&out_dir);
    fs::create_dir_all(out_dir).expect("create output directory");

    let mut weight_map: BTreeMap<String, String> = BTreeMap::new();

    // Shard 1: most of transformer_blocks.0/2/10 plus the start of
    // single_transformer_blocks.0.
    let shard1 = "diffusion_pytorch_model-00001-of-00002.safetensors";
    let names1 = write_shard(
        &out_dir.join(shard1),
        vec![
            (
                "transformer_blocks.0.attn.qkv.weight",
                f32_tensor(&[8, 8], 0),
            ),
            ("transformer_blocks.0.attn.qkv.bias", f32_tensor(&[8], 1)),
            (
                "transformer_blocks.2.attn.qkv.weight",
                f32_tensor(&[8, 8], 2),
            ),
            ("transformer_blocks.2.attn.qkv.bias", f32_tensor(&[8], 3)),
            (
                "transformer_blocks.10.attn.qkv.weight",
                f32_tensor(&[8, 8], 10),
            ),
            (
                "single_transformer_blocks.0.proj.weight",
                f32_tensor(&[8, 16], 20),
            ),
        ],
    );

    // Shard 2: the rest of transformer_blocks.10, the rest of
    // single_transformer_blocks.0 (so that block spans both shards), and
    // one clearly shared/unassigned tensor.
    let shard2 = "diffusion_pytorch_model-00002-of-00002.safetensors";
    let names2 = write_shard(
        &out_dir.join(shard2),
        vec![
            ("transformer_blocks.10.attn.qkv.bias", f32_tensor(&[8], 11)),
            (
                "single_transformer_blocks.0.proj.bias",
                f32_tensor(&[8], 21),
            ),
            ("shared.time_embed.weight", f32_tensor(&[4, 4], 99)),
        ],
    );

    for n in names1 {
        weight_map.insert(n, shard1.to_string());
    }
    for n in names2 {
        weight_map.insert(n, shard2.to_string());
    }

    let index = serde_json::json!({
        "metadata": {"format": "pt", "source": "streamloader make_fixture example"},
        "weight_map": weight_map,
    });
    let index_path = out_dir.join("diffusion_pytorch_model.safetensors.index.json");
    fs::write(&index_path, serde_json::to_vec_pretty(&index).unwrap()).expect("write index");

    eprintln!("wrote fixture checkpoint to {}", out_dir.display());
    eprintln!("try:");
    eprintln!(
        "  cargo run --release -- inspect {} --json",
        out_dir.display()
    );
    eprintln!(
        "  cargo run --release -- blocks {} --preset flux --json",
        out_dir.display()
    );
    eprintln!(
        "  cargo run --release -- block {} --id transformer_blocks.10 --preset flux --json",
        out_dir.display()
    );
    eprintln!(
        "  cargo run --release -- verify {} --block transformer_blocks.10 --preset flux --json",
        out_dir.display()
    );
}
