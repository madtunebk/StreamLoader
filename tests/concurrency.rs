mod support;

use std::sync::Arc;
use std::thread;

use streamloader::Model;
use support::{ShardSpec, f32_tensor, write_sharded_model};

/// Safe ownership/lifetimes: a `Model` can be shared (read-only) across
/// threads and produce concurrent zero-copy views without any unsafe code
/// at the call site, because `tensor_bytes` borrows from `&self` rather
/// than handing out anything self-referential or lifetime-erased.
#[test]
fn concurrent_reads_from_multiple_threads() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = write_sharded_model(
        dir.path(),
        &[
            ShardSpec {
                filename: "a.safetensors",
                tensors: vec![f32_tensor("layers.0.weight", vec![64], 1)],
            },
            ShardSpec {
                filename: "b.safetensors",
                tensors: vec![f32_tensor("layers.1.weight", vec![64], 2)],
            },
        ],
    );
    let model = Arc::new(Model::open(&index_path).unwrap());

    let mut handles = Vec::new();
    for i in 0..8 {
        let model = Arc::clone(&model);
        handles.push(thread::spawn(move || {
            let name = if i % 2 == 0 {
                "layers.0.weight"
            } else {
                "layers.1.weight"
            };
            for _ in 0..50 {
                let bytes = model.tensor_bytes(name).unwrap();
                assert_eq!(bytes.len(), 64 * 4);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}
