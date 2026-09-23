#!/usr/bin/env python3
"""
Offline INT8 quantization of FLUX.2-klein-9B's transformer, for real
(not simulated) int8 inference via bitsandbytes' Linear8bitLt.

Every 2D weight tensor (every Linear layer's weight -- confirmed via
Flux2Transformer2DModel's parameter shapes: 100% of this model's
parameters are 2D) is quantized row-wise via
bitsandbytes.functional.int8_vectorwise_quant, matching exactly what
Linear8bitLt(has_fp16_weights=False) does internally on `.to(device)` --
this produces the same CB (int8 weight, row-major) / SCB (per-output-row
fp16 scale) pair bitsandbytes expects, so the result is a real, usable
quantized checkpoint, not an ad hoc format.

Non-2D tensors (this model has none among its transformer weights, but
the script handles them anyway for robustness) are copied through
unquantized.

Writes to a NEW directory -- never touches the original Hugging Face
cache. Uses safetensors.torch directly (not the Rust loader) since this
is a one-time offline preprocessing step, not something on any hot path.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import bitsandbytes as bnb
from safetensors.torch import safe_open, save_file

DEFAULT_SRC = (
    "/home/nobus/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9B"
    "/snapshots/92196c8e11f7b6cf2b7493e037d8c5345c559216/transformer"
)
DEFAULT_DST = "/home/nobus/Raid0/RustStream/inference/quantized_model/transformer"


def quantize_tensor(t: torch.Tensor):
    """Row-wise (per-output-channel) int8 quantization, matching
    bitsandbytes' own Linear8bitLt(has_fp16_weights=False) internals
    exactly (Int8Params._quantize calls this same function on fp16 data)."""
    t_fp16 = t.to(torch.float16)
    cb, scb, _ = bnb.functional.int8_vectorwise_quant(t_fp16)
    return cb, scb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--dst", default=DEFAULT_DST)
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
                    cb, scb = quantize_tensor(t)
                    out_tensors[name] = cb  # int8, same shape as original weight
                    out_tensors[f"{name}.SCB"] = scb  # fp16, shape [out_features]
                    total_quantized_bytes += cb.numel() * cb.element_size() + scb.numel() * scb.element_size()
                    quantized_count += 1
                else:
                    out_tensors[name] = t
                    total_quantized_bytes += t.numel() * t.element_size()
                    copied_count += 1

        out_name = f"diffusion_pytorch_model-{shard_idx+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(out_tensors, dst / out_name, metadata={"format": "pt", "quantization": "bnb_int8_row_wise"})
        for name in out_tensors:
            new_weight_map[name] = out_name
        print(f"shard {shard}: {len(out_tensors)} tensors written to {out_name}")

    new_index = {
        "metadata": {"format": "pt", "quantization": "bnb_int8_row_wise", "source": str(src)},
        "weight_map": new_weight_map,
    }
    with open(dst / "diffusion_pytorch_model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # config.json needed for reconstructing the module structure later
    shutil.copy(src / "config.json", dst / "config.json")

    print(f"\nquantized {quantized_count} tensors, copied {copied_count} unquantized")
    print(f"original: {total_original_bytes/1e9:.2f} GB")
    print(f"quantized: {total_quantized_bytes/1e9:.2f} GB ({total_quantized_bytes/total_original_bytes*100:.1f}% of original)")
    print(f"written to {dst}")


if __name__ == "__main__":
    main()
