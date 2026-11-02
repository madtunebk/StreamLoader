#!/usr/bin/env python3
"""
Offline WEIGHT-ONLY int8 quantization of a DiT transformer, for real
transfer-halving streaming through the Rust engine with a CORRECT bf16
compute path.

Why this exists (vs. quantize_transformer.py, the bitsandbytes version):
bitsandbytes' Linear8bitLt force-casts *activations* to fp16 before its
int8 matmul. FLUX.2's attention activations have outliers that are finite
in bf16 but overflow fp16's 65504 ceiling to inf -> the hidden state
blows up from block 0 and the image is pure noise. That is intrinsic to
bnb's kernel (it has no bf16 compute path) and unfixable by wrapping or
threshold tuning -- see docs/ENGINE_VALIDATION.md and the project memory.

Weight-only int8 sidesteps that entirely: only the WEIGHTS are int8 (so
the bytes streamed host->GPU are halved, the actual goal), and the
forward pass dequantizes each weight back to bf16 and runs an ordinary
bf16 matmul. Activations never touch fp16, so no overflow.

Format per 2D Linear weight `W` of shape [out, in]:
  - `<name>`        int8   [out, in]   -- round(W / scale[:,None])
  - `<name>.scale`  bf16   [out]       -- per-output-channel absmax / 127
Symmetric, per-output-row (per-output-channel) scaling. Dequant is
exactly `W_bf16 = CB.to(bf16) * scale[:, None]`. 1D tensors (norms,
biases) are copied through unquantized as bf16.

Writes to a NEW directory; never touches the source. Uses
safetensors.torch directly (one-time offline preprocessing, not a hot
path).
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file


def quantize_weight_wo(t: torch.Tensor):
    """Per-output-channel symmetric int8. Returns (cb int8 [out,in],
    scale bf16 [out]). Round-to-nearest, clamped to [-127, 127] (we
    reserve -128 so dequant is exactly symmetric)."""
    w = t.to(torch.float32)
    absmax = w.abs().amax(dim=1)  # [out]
    # Guard a fully-zero row: scale 0 would give 0/0; use 1.0, the row
    # dequantizes back to all-zeros either way.
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    cb = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return cb, scale.to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="source transformer dir (safetensors + index.json)")
    parser.add_argument("--dst", required=True, help="output dir for the quantized checkpoint")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    index = json.load(open(src / "diffusion_pytorch_model.safetensors.index.json"))
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    new_weight_map = {}
    total_original_bytes = 0
    total_quantized_bytes = 0
    quantized_count = 0
    copied_count = 0

    for shard_idx, shard in enumerate(shards):
        out_tensors = {}
        with safe_open(src / shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                total_original_bytes += t.numel() * t.element_size()

                if t.dim() == 2:
                    cb, scale = quantize_weight_wo(t)
                    out_tensors[name] = cb
                    out_tensors[f"{name}.scale"] = scale
                    total_quantized_bytes += cb.numel() * cb.element_size()
                    total_quantized_bytes += scale.numel() * scale.element_size()
                    quantized_count += 1
                else:
                    # Keep norms/biases in bf16 (the model's compute dtype).
                    tb = t.to(torch.bfloat16)
                    out_tensors[name] = tb
                    total_quantized_bytes += tb.numel() * tb.element_size()
                    copied_count += 1

        out_name = f"diffusion_pytorch_model-{shard_idx+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(out_tensors, dst / out_name, metadata={"format": "pt", "quantization": "int8_weight_only_rowwise"})
        for name in out_tensors:
            new_weight_map[name] = out_name
        print(f"shard {shard}: {len(out_tensors)} tensors -> {out_name}")

    new_index = {
        "metadata": {"format": "pt", "quantization": "int8_weight_only_rowwise", "source": str(src)},
        "weight_map": new_weight_map,
    }
    with open(dst / "diffusion_pytorch_model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    shutil.copy(src / "config.json", dst / "config.json")

    print(f"\nquantized {quantized_count} tensors, copied {copied_count} unquantized (as bf16)")
    print(f"original:  {total_original_bytes/1e9:.2f} GB")
    print(f"quantized: {total_quantized_bytes/1e9:.2f} GB "
          f"({total_quantized_bytes/total_original_bytes*100:.1f}% of original)")
    print(f"written to {dst}")


if __name__ == "__main__":
    main()
