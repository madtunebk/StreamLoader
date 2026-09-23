mod support;

use streamloader::{BlockConfig, LoaderError, Model};
use support::{ShardSpec, f32_tensor, write_sharded_model};

fn one_block_model(seed: u32) -> (tempfile::TempDir, Model) {
    let dir = tempfile::tempdir().unwrap();
    let shards = vec![ShardSpec {
        filename: "shard.safetensors",
        tensors: vec![
            f32_tensor("transformer_blocks.0.weight", vec![4], seed),
            f32_tensor("transformer_blocks.0.bias", vec![2], seed + 1),
        ],
    }];
    let index_path = write_sharded_model(dir.path(), &shards);
    let model = Model::open(&index_path).unwrap();
    (dir, model)
}

#[test]
fn block_views_rejects_a_block_from_a_different_model() {
    let (_dir_a, model_a) = one_block_model(1);
    let (_dir_b, model_b) = one_block_model(2);

    let config = BlockConfig::flux();
    let block_from_b = model_b.block(&config, "transformer_blocks.0").unwrap();

    let err = model_a.block_views(&block_from_b).unwrap_err();
    assert!(matches!(err, LoaderError::BlockFromDifferentModel { .. }));
}

#[test]
fn copy_block_rejects_a_block_from_a_different_model() {
    let (_dir_a, model_a) = one_block_model(1);
    let (_dir_b, model_b) = one_block_model(2);

    let config = BlockConfig::flux();
    let block_from_b = model_b.block(&config, "transformer_blocks.0").unwrap();

    let err = model_a.copy_block(&block_from_b).unwrap_err();
    assert!(matches!(err, LoaderError::BlockFromDifferentModel { .. }));
}

#[test]
fn copy_block_into_rejects_a_block_from_a_different_model() {
    let (_dir_a, model_a) = one_block_model(1);
    let (_dir_b, model_b) = one_block_model(2);

    let config = BlockConfig::flux();
    let block_from_b = model_b.block(&config, "transformer_blocks.0").unwrap();
    let mut buf = vec![0u8; 1024];

    let err = model_a
        .copy_block_into(&block_from_b, &mut buf)
        .unwrap_err();
    assert!(matches!(err, LoaderError::BlockFromDifferentModel { .. }));
}

#[test]
fn checksum_block_rejects_a_block_from_a_different_model() {
    let (_dir_a, model_a) = one_block_model(1);
    let (_dir_b, model_b) = one_block_model(2);

    let config = BlockConfig::flux();
    let block_from_b = model_b.block(&config, "transformer_blocks.0").unwrap();

    let err = model_a.checksum_block(&block_from_b).unwrap_err();
    assert!(matches!(err, LoaderError::BlockFromDifferentModel { .. }));
}

#[test]
fn same_model_own_block_works_normally() {
    let (_dir, model) = one_block_model(1);
    let config = BlockConfig::flux();
    let block = model.block(&config, "transformer_blocks.0").unwrap();
    assert!(model.block_views(&block).is_ok());
    assert!(model.copy_block(&block).is_ok());
    assert!(model.checksum_block(&block).is_ok());
}

#[test]
fn two_models_of_the_same_checkpoint_have_distinct_ids() {
    // Opening the identical path twice still yields two distinct in-memory
    // instances -- a Block from one must not be usable on the other, even
    // though they describe the same bytes on disk.
    let dir = tempfile::tempdir().unwrap();
    let shards = vec![ShardSpec {
        filename: "shard.safetensors",
        tensors: vec![f32_tensor("layers.0.weight", vec![2], 1)],
    }];
    let index_path = write_sharded_model(dir.path(), &shards);
    let model1 = Model::open(&index_path).unwrap();
    let model2 = Model::open(&index_path).unwrap();
    assert_ne!(model1.id(), model2.id());

    let config = BlockConfig::generic();
    let block_from_1 = model1.block(&config, "layers.0").unwrap();
    let err = model2.block_views(&block_from_1).unwrap_err();
    assert!(matches!(err, LoaderError::BlockFromDifferentModel { .. }));
}

#[test]
fn copy_block_into_matches_copy_block() {
    let (_dir, model) = one_block_model(7);
    let config = BlockConfig::flux();
    let block = model.block(&config, "transformer_blocks.0").unwrap();

    let via_copy_block = model.copy_block(&block).unwrap();

    let mut buf = vec![0xAAu8; via_copy_block.buffer.len()];
    let layout = model.copy_block_into(&block, &mut buf).unwrap();

    assert_eq!(buf, via_copy_block.buffer);
    assert_eq!(layout.len(), via_copy_block.layout.len());
    for (a, b) in layout.iter().zip(via_copy_block.layout.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.offset, b.offset);
        assert_eq!(a.len, b.len);
    }
}

#[test]
fn copy_block_into_rejects_too_small_destination() {
    let (_dir, model) = one_block_model(1);
    let config = BlockConfig::flux();
    let block = model.block(&config, "transformer_blocks.0").unwrap();

    let mut tiny = vec![0u8; 1];
    let err = model.copy_block_into(&block, &mut tiny).unwrap_err();
    assert!(matches!(err, LoaderError::DestinationTooSmall { .. }));
}

#[test]
fn find_block_does_not_build_unrelated_blocks() {
    // block() on one id must not require every other block's tensors to
    // be cloned along the way; this is a behavioral/perf regression
    // guard, checked indirectly via the returned block containing only
    // the tensors for the requested id (never leaking tensors from other
    // blocks into the result).
    let dir = tempfile::tempdir().unwrap();
    let shards = vec![ShardSpec {
        filename: "shard.safetensors",
        tensors: vec![
            f32_tensor("transformer_blocks.1.weight", vec![2], 1),
            f32_tensor("transformer_blocks.2.weight", vec![2], 2),
            f32_tensor("single_transformer_blocks.0.weight", vec![2], 3),
        ],
    }];
    let index_path = write_sharded_model(dir.path(), &shards);
    let model = Model::open(&index_path).unwrap();
    let config = BlockConfig::flux();

    let block = model.block(&config, "transformer_blocks.1").unwrap();
    assert_eq!(block.tensors.len(), 1);
    assert_eq!(block.tensors[0].name, "transformer_blocks.1.weight");
}
