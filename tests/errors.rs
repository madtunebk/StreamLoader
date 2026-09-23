mod support;

use std::fs;
use std::io::Write;

use streamloader::{LoaderError, Model};
use support::{ShardSpec, f32_tensor, write_safetensors_file, write_sharded_model};

#[test]
fn missing_shard_file_referenced_by_index() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = dir.path().join("model.safetensors.index.json");
    let index = serde_json::json!({
        "weight_map": { "layers.0.weight": "does-not-exist.safetensors" }
    });
    fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

    let err = Model::open(&index_path).unwrap_err();
    // A dangling reference is a plain I/O error surfaced during path
    // resolution/canonicalization, not a path-safety violation.
    assert!(matches!(err, LoaderError::Io { .. }));
}

#[test]
fn missing_tensor_declared_in_index_but_absent_from_shard() {
    let dir = tempfile::tempdir().unwrap();
    write_safetensors_file(
        &dir.path().join("shard.safetensors"),
        &[f32_tensor("layers.0.weight", vec![2], 1)],
    );
    let index_path = dir.path().join("model.safetensors.index.json");
    let index = serde_json::json!({
        "weight_map": {
            "layers.0.weight": "shard.safetensors",
            "layers.999.weight": "shard.safetensors",
        }
    });
    fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

    let err = Model::open(&index_path).unwrap_err();
    assert!(matches!(err, LoaderError::TensorMissingFromShard { .. }));
}

#[test]
fn inconsistent_index_missing_a_tensor_that_the_shard_actually_has() {
    let dir = tempfile::tempdir().unwrap();
    write_safetensors_file(
        &dir.path().join("shard.safetensors"),
        &[
            f32_tensor("layers.0.weight", vec![2], 1),
            f32_tensor("layers.1.weight", vec![2], 2),
        ],
    );
    let index_path = dir.path().join("model.safetensors.index.json");
    // weight_map only declares one of the two tensors actually in the shard.
    let index = serde_json::json!({
        "weight_map": { "layers.0.weight": "shard.safetensors" }
    });
    fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

    let err = Model::open(&index_path).unwrap_err();
    assert!(matches!(err, LoaderError::TensorMissingFromIndex { .. }));
}

#[test]
fn truncated_file_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let bytes = write_safetensors_file(&path, &[f32_tensor("layers.0.weight", vec![4, 4], 1)]);
    // Chop off the last few payload bytes -- the header still declares the
    // full length, so the buffer is now shorter than the header promises.
    let truncated = &bytes[..bytes.len() - 4];
    fs::write(&path, truncated).unwrap();

    let err = Model::open(&path).unwrap_err();
    assert!(matches!(err, LoaderError::InvalidHeader { .. }));
}

#[test]
fn malformed_header_bytes_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    // A length prefix claiming a huge header, followed by garbage that is
    // neither valid UTF-8 JSON nor long enough to satisfy that length.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(1000u64).to_le_bytes());
    bytes.extend_from_slice(b"not json");
    fs::write(&path, &bytes).unwrap();

    let err = Model::open(&path).unwrap_err();
    assert!(matches!(err, LoaderError::InvalidHeader { .. }));
}

#[test]
fn header_smaller_than_length_prefix_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    fs::write(&path, [1, 2, 3]).unwrap();
    let err = Model::open(&path).unwrap_err();
    assert!(matches!(err, LoaderError::HeaderTooSmall { .. }));
}

#[test]
fn declared_header_larger_than_configured_limit_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let mut f = fs::File::create(&path).unwrap();
    // Declare an enormous header without actually writing that much data;
    // the configurable limit must reject this before any large allocation
    // or JSON parse is attempted.
    f.write_all(&(10_000_000_000u64).to_le_bytes()).unwrap();
    f.write_all(b"{}").unwrap();
    drop(f);

    let opts = streamloader::OpenOptions {
        max_header_bytes: 1024,
        ..Default::default()
    };
    let err = Model::open_with(&path, &opts).unwrap_err();
    assert!(matches!(err, LoaderError::HeaderTooLarge { .. }));
}

#[test]
fn duplicate_tensor_names_across_implicit_shards_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    write_safetensors_file(
        &dir.path().join("a.safetensors"),
        &[f32_tensor("layers.0.weight", vec![2], 1)],
    );
    write_safetensors_file(
        &dir.path().join("b.safetensors"),
        &[f32_tensor("layers.0.weight", vec![2], 2)],
    );
    let opts = streamloader::OpenOptions {
        allow_implicit_multi_shard: true,
        ..Default::default()
    };
    let err = Model::open_with(dir.path(), &opts).unwrap_err();
    assert!(matches!(err, LoaderError::DuplicateTensorName { .. }));
}

#[test]
fn implicit_multi_shard_without_opt_in_is_ambiguous_not_merged() {
    let dir = tempfile::tempdir().unwrap();
    write_safetensors_file(
        &dir.path().join("a.safetensors"),
        &[f32_tensor("layers.0.weight", vec![2], 1)],
    );
    write_safetensors_file(
        &dir.path().join("b.safetensors"),
        &[f32_tensor("layers.1.weight", vec![2], 2)], // distinct name, still not merged by default
    );
    let err = Model::open(dir.path()).unwrap_err();
    assert!(matches!(err, LoaderError::AmbiguousDirectory { .. }));
}

#[test]
fn absolute_shard_path_in_index_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    write_safetensors_file(
        &dir.path().join("shard.safetensors"),
        &[f32_tensor("layers.0.weight", vec![2], 1)],
    );
    let index_path = dir.path().join("model.safetensors.index.json");
    let index = serde_json::json!({
        "weight_map": { "layers.0.weight": "/etc/passwd" }
    });
    fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

    let err = Model::open(&index_path).unwrap_err();
    assert!(matches!(err, LoaderError::UnsafeShardPath { .. }));
}

#[test]
fn parent_dir_traversal_in_index_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    // A real file that does live outside the checkpoint root, to prove
    // this is rejected lexically/by containment and not just because the
    // target happens to be missing.
    let outside = dir.path().join("outside.safetensors");
    write_safetensors_file(&outside, &[f32_tensor("secret", vec![1], 1)]);

    let root = dir.path().join("checkpoint");
    fs::create_dir_all(&root).unwrap();
    let index_path = root.join("model.safetensors.index.json");
    let index = serde_json::json!({
        "weight_map": { "layers.0.weight": "../outside.safetensors" }
    });
    fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

    let err = Model::open(&index_path).unwrap_err();
    assert!(matches!(err, LoaderError::UnsafeShardPath { .. }));
}

#[cfg(unix)]
#[test]
fn trusted_root_explicitly_widens_the_symlink_boundary() {
    // Mirrors a real Hugging Face hub cache: shard files in the snapshot
    // directory are symlinks into a shared `blobs/` sibling directory
    // outside the snapshot. That must be rejected by default, and allowed
    // only once the caller explicitly opts in via `trusted_root`.
    use std::os::unix::fs::symlink;

    let hub_entry = tempfile::tempdir().unwrap();
    let blobs = hub_entry.path().join("blobs");
    fs::create_dir_all(&blobs).unwrap();
    let real_shard = blobs.join("deadbeef");
    write_safetensors_file(&real_shard, &[f32_tensor("layers.0.weight", vec![2], 1)]);

    let snapshot = hub_entry.path().join("snapshots/abc123/transformer");
    fs::create_dir_all(&snapshot).unwrap();
    symlink(&real_shard, snapshot.join("model.safetensors")).unwrap();

    let index_path = snapshot.join("model.safetensors.index.json");
    let index = serde_json::json!({
        "weight_map": { "layers.0.weight": "model.safetensors" }
    });
    fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

    // Default: rejected, since the blob lives outside the snapshot dir.
    let err = Model::open(&index_path).unwrap_err();
    assert!(matches!(err, LoaderError::UnsafeShardPath { .. }));

    // Explicitly trusting the whole hub entry (snapshot + blobs) allows it.
    let opts = streamloader::OpenOptions {
        trusted_root: Some(hub_entry.path().to_path_buf()),
        ..Default::default()
    };
    let model = Model::open_with(&index_path, &opts).unwrap();
    assert_eq!(model.tensor_count(), 1);
}

#[cfg(unix)]
#[test]
fn symlink_escape_in_index_is_rejected() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    let outside = dir.path().join("outside.safetensors");
    write_safetensors_file(&outside, &[f32_tensor("secret", vec![1], 1)]);

    let root = dir.path().join("checkpoint");
    fs::create_dir_all(&root).unwrap();
    let link = root.join("shard.safetensors");
    symlink(&outside, &link).unwrap();

    let index_path = root.join("model.safetensors.index.json");
    let index = serde_json::json!({
        "weight_map": { "layers.0.weight": "shard.safetensors" }
    });
    fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

    let err = Model::open(&index_path).unwrap_err();
    assert!(matches!(err, LoaderError::UnsafeShardPath { .. }));
}

#[test]
fn ambiguous_directory_with_multiple_index_files_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    write_sharded_model(
        dir.path(),
        &[ShardSpec {
            filename: "a.safetensors",
            tensors: vec![f32_tensor("layers.0.weight", vec![2], 1)],
        }],
    );
    // A second, differently-named index file makes the directory ambiguous.
    fs::write(
        dir.path()
            .join("diffusion_pytorch_model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"weight_map": {}})).unwrap(),
    )
    .unwrap();

    let err = Model::open(dir.path()).unwrap_err();
    assert!(matches!(err, LoaderError::AmbiguousDirectory { .. }));
}

#[test]
fn empty_directory_reports_no_shards_found() {
    let dir = tempfile::tempdir().unwrap();
    let err = Model::open(dir.path()).unwrap_err();
    assert!(matches!(err, LoaderError::NoShardsFound { .. }));
}
