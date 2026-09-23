mod support;

use streamloader::{BlockConfig, BlockFamily, Model, UNASSIGNED};
use support::{flux_like_shards, write_sharded_model};

fn open_flux_fixture() -> (tempfile::TempDir, Model) {
    let dir = tempfile::tempdir().unwrap();
    let index_path = write_sharded_model(dir.path(), &flux_like_shards());
    let model = Model::open(&index_path).unwrap();
    (dir, model)
}

#[test]
fn numeric_ordering_and_flux_family_order() {
    let (_dir, model) = open_flux_fixture();
    let config = BlockConfig::flux();
    let blocks = model.blocks(&config);

    let ids: Vec<&str> = blocks.iter().map(|b| b.id.as_str()).collect();
    // transformer_blocks (2 before 10, numerically) must all precede
    // single_transformer_blocks, regardless of string/lexicographic order.
    assert_eq!(
        ids,
        vec![
            "transformer_blocks.2",
            "transformer_blocks.10",
            "single_transformer_blocks.0",
            UNASSIGNED,
        ]
    );
}

#[test]
fn shared_unassigned_and_explicit_prefix_selection() {
    let (_dir, model) = open_flux_fixture();
    let config = BlockConfig::flux();
    let blocks = model.blocks(&config);
    let unassigned = blocks.iter().find(|b| b.id == UNASSIGNED).unwrap();
    let names: Vec<&str> = unassigned.tensors.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names, vec!["shared.time_embed.scale"]);

    // explicit prefix selection: fetch the unassigned bucket directly by id.
    let fetched = model.block(&config, UNASSIGNED).unwrap();
    assert_eq!(fetched.tensors.len(), 1);
}

#[test]
fn exact_block_matching_1_does_not_include_10() {
    let dir = tempfile::tempdir().unwrap();
    let shards = vec![support::ShardSpec {
        filename: "shard.safetensors",
        tensors: vec![
            support::f32_tensor("transformer_blocks.1.weight", vec![2], 1),
            support::f32_tensor("transformer_blocks.10.weight", vec![2], 10),
        ],
    }];
    let index_path = write_sharded_model(dir.path(), &shards);
    let model = Model::open(&index_path).unwrap();

    let config = BlockConfig::flux();
    let block1 = model.block(&config, "transformer_blocks.1").unwrap();
    assert_eq!(block1.tensors.len(), 1);
    assert_eq!(block1.tensors[0].name, "transformer_blocks.1.weight");

    let block10 = model.block(&config, "transformer_blocks.10").unwrap();
    assert_eq!(block10.tensors.len(), 1);
    assert_eq!(block10.tensors[0].name, "transformer_blocks.10.weight");
}

#[test]
fn nonexistent_block_is_an_error() {
    let (_dir, model) = open_flux_fixture();
    let config = BlockConfig::flux();
    let err = model.block(&config, "transformer_blocks.999").unwrap_err();
    assert!(matches!(err, streamloader::LoaderError::BlockNotFound(_)));
}

#[test]
fn generic_preset_covers_blocks_and_layers_conventions() {
    let dir = tempfile::tempdir().unwrap();
    let shards = vec![support::ShardSpec {
        filename: "shard.safetensors",
        tensors: vec![
            support::f32_tensor("blocks.3.weight", vec![2], 1),
            support::f32_tensor("layers.5.weight", vec![2], 2),
        ],
    }];
    let index_path = write_sharded_model(dir.path(), &shards);
    let model = Model::open(&index_path).unwrap();

    let config = BlockConfig::generic();
    let blocks = model.blocks(&config);
    let ids: Vec<&str> = blocks.iter().map(|b| b.id.as_str()).collect();
    assert!(ids.contains(&"blocks.3"));
    assert!(ids.contains(&"layers.5"));
}

#[test]
fn explicit_model_prefix_does_not_strip_unrelated_names() {
    let dir = tempfile::tempdir().unwrap();
    let shards = vec![support::ShardSpec {
        filename: "shard.safetensors",
        tensors: vec![
            support::f32_tensor("model.layers.0.weight", vec![2], 1),
            // Not under the "model" prefix at all -- must not be silently
            // treated as if it were.
            support::f32_tensor("other_model.layers.0.weight", vec![2], 2),
        ],
    }];
    let index_path = write_sharded_model(dir.path(), &shards);
    let model = Model::open(&index_path).unwrap();

    let config = BlockConfig::new(vec![BlockFamily::new("layers")]).with_model_prefix("model");
    let block = model.block(&config, "layers.0").unwrap();
    let names: Vec<&str> = block.tensors.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names, vec!["model.layers.0.weight"]);

    let blocks = model.blocks(&config);
    let unassigned = blocks.iter().find(|b| b.id == UNASSIGNED).unwrap();
    assert_eq!(
        unassigned
            .tensors
            .iter()
            .map(|d| d.name.as_str())
            .collect::<Vec<_>>(),
        vec!["other_model.layers.0.weight"]
    );
}
