mod support;

use std::process::Command;

use serde_json::Value;
use support::{flux_like_shards, write_sharded_model};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_streamloader")
}

#[test]
fn inspect_json_lists_every_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = write_sharded_model(dir.path(), &flux_like_shards());

    let output = Command::new(bin())
        .args([
            "inspect",
            index_path.parent().unwrap().to_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["tensor_count"], 8);
    assert_eq!(json["shard_count"], 2);
}

#[test]
fn blocks_json_orders_flux_numerically_then_by_family() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = write_sharded_model(dir.path(), &flux_like_shards());

    let output = Command::new(bin())
        .args([
            "blocks",
            index_path.parent().unwrap().to_str().unwrap(),
            "--preset",
            "flux",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());

    let json: Value = serde_json::from_slice(&output.stdout).unwrap();
    let ids: Vec<&str> = json
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        vec![
            "transformer_blocks.2",
            "transformer_blocks.10",
            "single_transformer_blocks.0",
            "shared/unassigned",
        ]
    );
}

#[test]
fn verify_checksum_matches_independently_computed_hash() {
    let dir = tempfile::tempdir().unwrap();
    let shards = vec![support::ShardSpec {
        filename: "shard.safetensors",
        tensors: vec![support::f32_tensor("layers.0.weight", vec![4], 42)],
    }];
    let index_path = write_sharded_model(dir.path(), &shards);
    let expected_bytes = support::f32_tensor("layers.0.weight", vec![4], 42).data;
    let expected_hex = blake3::hash(&expected_bytes).to_hex().to_string();

    let output = Command::new(bin())
        .args([
            "verify",
            index_path.parent().unwrap().to_str().unwrap(),
            "--tensor",
            "layers.0.weight",
            "--preset",
            "generic",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["algorithm"], "blake3");
    assert_eq!(json["tensors"][0]["hex"], expected_hex);
    assert_eq!(json["tensors"][0]["bytes"], 16);
}

#[test]
fn nonexistent_block_exits_nonzero() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = write_sharded_model(dir.path(), &flux_like_shards());

    let output = Command::new(bin())
        .args([
            "block",
            index_path.parent().unwrap().to_str().unwrap(),
            "--id",
            "transformer_blocks.9999",
            "--preset",
            "flux",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!output.stderr.is_empty(), "diagnostics should go to stderr");
}

#[test]
fn nonexistent_tensor_exits_nonzero() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = write_sharded_model(dir.path(), &flux_like_shards());

    let output = Command::new(bin())
        .args([
            "verify",
            index_path.parent().unwrap().to_str().unwrap(),
            "--tensor",
            "does.not.exist",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[test]
fn copy_flag_reports_bytes_copied_in_json() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = write_sharded_model(dir.path(), &flux_like_shards());

    let output = Command::new(bin())
        .args([
            "block",
            index_path.parent().unwrap().to_str().unwrap(),
            "--id",
            "transformer_blocks.2",
            "--preset",
            "flux",
            "--copy",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let json: Value = serde_json::from_slice(&output.stdout).unwrap();
    let bytes_copied = json["copy"]["bytes_copied"].as_u64().unwrap();
    assert!(bytes_copied > 0);
}
