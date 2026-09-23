import torch
import streamloader_engine as se

torch.cuda.init()
stream_ptr = torch.cuda.current_stream().cuda_stream

failures = 0
for trial, fill in enumerate([1.5, -3.25, 42.0, 0.001, 999.5, 7.0]):
    n = 1 << 16  # 65536 elements
    cap = se.debug_dlpack_roundtrip(fill, n, stream_ptr)
    t = torch.utils.dlpack.from_dlpack(cap)

    # Deliberately NOT calling torch.cuda.synchronize() here -- the whole
    # point is proving the cross-stream CUDA event wait alone is
    # sufficient for PyTorch's own stream to see the completed transfer.
    assert t.is_cuda, "expected a CUDA tensor"
    assert t.dtype == torch.float32, t.dtype
    assert tuple(t.shape) == (n,), t.shape

    total = t.sum().item()
    expected = fill * n
    ok = abs(total - expected) < 1e-1
    status = "OK" if ok else "MISMATCH"
    print(f"trial {trial}: fill={fill} sum={total} expected={expected} {status}")
    if not ok:
        failures += 1

if failures:
    print(f"\n{failures} trial(s) FAILED -- cross-stream sync is broken")
    raise SystemExit(1)
print("\nAll trials passed with no torch.cuda.synchronize() call -- cross-stream event handoff is correct.")
