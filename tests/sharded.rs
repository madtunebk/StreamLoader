mod support;

use streamloader::Model;
use support::{ShardSpec, f32_tensor, flux_like_shards, write_sharded_model};

#[test]
fn multi_shard_valid_weight_map() {
    let dir = tempfile::tempdir().unwrap();
    let shards = vec![
        ShardSpec {
            filename: "shard-a.safetensors",
            tensors: vec![f32_tensor("layers.0.weight", vec![2], 1)],
        },
        ShardSpec {
            filename: "shard-b.safetensors",
            tensors: vec![f32_tensor("layers.1.weight", vec![2], 2)],
        },
    ];
    let index_path = write_sharded_model(dir.path(), &shards);

    let model = Model::open(&index_path).unwrap();
    assert_eq!(model.tensor_count(), 2);
    assert!(model.tensor_bytes("layers.0.weight").is_ok());
    assert!(model.tensor_bytes("layers.1.weight").is_ok());

    // opening the directory itself (index auto-discovered) must agree
    let model2 = Model::open(dir.path()).unwrap();
    assert_eq!(model2.tensor_count(), 2);
}

#[test]
fn block_spans_shards_and_nonadjacent_ranges() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = write_sharded_model(dir.path(), &flux_like_shards());
    let model = Model::open(&index_path).unwrap();

    let config = streamloader::BlockConfig::flux();
    let block = model.block(&config, "transformer_blocks.2").unwrap();

    let names: Vec<&str> = block.tensors.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"transformer_blocks.2.attn.qkv.weight"));
    assert!(names.contains(&"transformer_blocks.2.attn.qkv.bias"));
    assert!(names.contains(&"transformer_blocks.2.mlp.fc1.weight"));

    // the three tensors must come from at least two distinct shards
    let shard_ids: std::collections::HashSet<_> =
        block.tensors.iter().map(|d| d.shard_id).collect();
    assert!(
        shard_ids.len() >= 2,
        "block should span more than one shard"
    );

    // listing/getting the block must not require reading payload bytes;
    // only after we ask for views do bytes get touched.
    let views = model.block_views(&block).unwrap();
    assert_eq!(views.len(), block.tensors.len());
    for (name, bytes) in views {
        let d = block.tensors.iter().find(|d| d.name == name).unwrap();
        assert_eq!(bytes.len() as u64, d.byte_len);
    }
}

#[test]
fn implicit_multi_shard_directory_rejected_by_default() {
    // Multiple sibling .safetensors files with no index and no explicit
    // opt-in must NOT be silently merged into one model -- disjoint names
    // alone don't prove they belong together.
    let dir = tempfile::tempdir().unwrap();
    support::write_safetensors_file(
        &dir.path().join("part1.safetensors"),
        &[f32_tensor("layers.0.weight", vec![2], 1)],
    );
    support::write_safetensors_file(
        &dir.path().join("part2.safetensors"),
        &[f32_tensor("layers.1.weight", vec![2], 2)],
    );

    let err = Model::open(dir.path()).unwrap_err();
    assert!(matches!(
        err,
        streamloader::LoaderError::AmbiguousDirectory { .. }
    ));
}

#[test]
fn implicit_shards_directory_without_index_with_explicit_opt_in() {
    let dir = tempfile::tempdir().unwrap();
    support::write_safetensors_file(
        &dir.path().join("part1.safetensors"),
        &[f32_tensor("layers.0.weight", vec![2], 1)],
    );
    support::write_safetensors_file(
        &dir.path().join("part2.safetensors"),
        &[f32_tensor("layers.1.weight", vec![2], 2)],
    );

    let opts = streamloader::OpenOptions {
        allow_implicit_multi_shard: true,
        ..Default::default()
    };
    let model = Model::open_with(dir.path(), &opts).unwrap();
    assert_eq!(model.tensor_count(), 2);
}

#[test]
fn implicit_shards_reject_duplicate_names_even_with_opt_in() {
    let dir = tempfile::tempdir().unwrap();
    support::write_safetensors_file(
        &dir.path().join("part1.safetensors"),
        &[f32_tensor("layers.0.weight", vec![2], 1)],
    );
    support::write_safetensors_file(
        &dir.path().join("part2.safetensors"),
        &[f32_tensor("layers.0.weight", vec![2], 999)], // same name, different data
    );

    let opts = streamloader::OpenOptions {
        allow_implicit_multi_shard: true,
        ..Default::default()
    };
    let err = Model::open_with(dir.path(), &opts).unwrap_err();
    assert!(matches!(
        err,
        streamloader::LoaderError::DuplicateTensorName { .. }
    ));
}
