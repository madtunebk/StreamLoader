mod support;

use safetensors::Dtype;
use streamloader::Model;
use support::{empty_tensor, f32_tensor, scalar_f32, write_safetensors_file};

#[test]
fn known_tensor_bytes_shapes_dtypes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let t = f32_tensor("layers.0.weight", vec![2, 3], 7);
    let expected_bytes = t.data.clone();
    write_safetensors_file(&path, &[t]);

    let model = Model::open(&path).unwrap();
    assert_eq!(model.tensor_count(), 1);

    let d = model.descriptor("layers.0.weight").unwrap();
    assert_eq!(d.dtype, Dtype::F32);
    assert_eq!(d.shape, vec![2, 3]);
    assert_eq!(d.byte_len, 24); // 2*3*4 bytes

    let bytes = model.tensor_bytes("layers.0.weight").unwrap();
    assert_eq!(bytes, expected_bytes.as_slice());
}

#[test]
fn empty_and_scalar_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    write_safetensors_file(
        &path,
        &[
            scalar_f32("layers.0.scale", 3.5),
            empty_tensor("layers.0.empty", vec![0, 4]),
        ],
    );

    let model = Model::open(&path).unwrap();

    let scalar = model.descriptor("layers.0.scale").unwrap();
    assert!(scalar.shape.is_empty());
    assert_eq!(scalar.byte_len, 4);
    assert_eq!(model.tensor_bytes("layers.0.scale").unwrap().len(), 4);

    let empty = model.descriptor("layers.0.empty").unwrap();
    assert_eq!(empty.shape, vec![0, 4]);
    assert_eq!(empty.byte_len, 0);
    assert_eq!(model.tensor_bytes("layers.0.empty").unwrap().len(), 0);
}

#[test]
fn multiple_instances_same_names_different_data() {
    let dir = tempfile::tempdir().unwrap();
    let path_a = dir.path().join("a.safetensors");
    let path_b = dir.path().join("b.safetensors");
    write_safetensors_file(&path_a, &[f32_tensor("layers.0.weight", vec![2], 1)]);
    write_safetensors_file(&path_b, &[f32_tensor("layers.0.weight", vec![2], 99)]);

    let model_a = Model::open(&path_a).unwrap();
    let model_b = Model::open(&path_b).unwrap();

    let bytes_a = model_a.tensor_bytes("layers.0.weight").unwrap();
    let bytes_b = model_b.tensor_bytes("layers.0.weight").unwrap();
    assert_ne!(bytes_a, bytes_b);
}

#[test]
fn tensor_not_found_is_a_clean_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    write_safetensors_file(&path, &[f32_tensor("layers.0.weight", vec![2], 1)]);
    let model = Model::open(&path).unwrap();
    let err = model.tensor_bytes("does.not.exist").unwrap_err();
    assert!(matches!(err, streamloader::LoaderError::TensorNotFound(_)));
}
