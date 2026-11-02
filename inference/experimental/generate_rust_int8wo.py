#!/usr/bin/env python3
"""
FLUX.2-klein-9B inference with WEIGHT-ONLY int8 weights streamed through
the Rust engine. Same streaming architecture as inference/generate_flux.py
(pinned host cache + reusable VRAM ring + DLPack, per-block prefetch
hooks); the only change is that TRANSFORMER_DIR points at the weight-only
int8 checkpoint (quantize_transformer_wo.py's output, ~9.3GB vs 18.16GB),
so every streamed Linear weight arrives as an (int8 CB, bf16 scale) pair.

Unlike the bitsandbytes path (inference/experimental/generate_rust_int8.py,
a confirmed dead-end -- it force-casts activations to fp16 and FLUX's
attention outliers overflow fp16 -> noise), this dequantizes each weight
to bf16 inside the forward and runs an ordinary bf16 matmul. Activations
never touch fp16, so there is no overflow. The int8 weights still halve
the host->GPU transfer, which is the whole point.

Run with:
  HF_HOME=/path/to/cache UV_NO_SYNC=1 \
    .venv/bin/python inference/experimental/generate_rust_int8wo.py \
      --transformer-dir /path/to/quantized_model/transformer --prompt "..."
"""

import argparse
import json
import os
import sys
import time
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import cpu_offload, init_empty_weights
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoTokenizer, Qwen3ForCausalLM

import streamloader_engine as se

MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
PINNED_BUDGET = 12 * 1024 * 1024 * 1024  # weight-only int8 transformer is ~9.3GB
VRAM_SLOTS = 2
PREFETCH_AHEAD = 1


class Int8WeightOnlyLinear(nn.Module):
    """A Linear whose weight is stored int8 (per-output-channel scale)
    and dequantized to bf16 at forward time, then run as a plain bf16
    matmul. Weight + scale are assigned per block by the streaming hook,
    so they start as meta/None and are swapped in before each use.

    `weight` holds the int8 CB tensor (named `weight` so the streamed
    `<...>.weight` key lands here via load_state_dict(assign=True)).
    `weight_scale` holds the bf16 per-row scale (streamed as
    `<...>.weight.scale`; see apply_tensors)."""

    def __init__(self, in_features, out_features, bias=False, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # int8 weight placeholder; real bytes come from the engine.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8, device=device),
            requires_grad=False,
        )
        # bf16 per-output-channel scale; assigned alongside the weight.
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device=device), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        # Dequantize to bf16: W[o,i] = CB[o,i] * scale[o]. Matmul stays
        # bf16 throughout -- no fp16 cast anywhere, so no overflow on
        # FLUX's outlier activations.
        w = self.weight.to(torch.bfloat16) * self.weight_scale[:, None]
        return F.linear(x.to(torch.bfloat16), w, self.bias)


def build_meta_transformer_int8wo(transformer_dir):
    cfg = json.load(open(f"{transformer_dir}/config.json"))
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    with init_empty_weights():
        model = Flux2Transformer2DModel(**cfg)

    replaced = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if type(child) is nn.Linear:
                new_mod = Int8WeightOnlyLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device="meta",
                )
                setattr(module, child_name, new_mod)
                replaced += 1
    print(f"replaced {replaced} nn.Linear modules with Int8WeightOnlyLinear")
    model.eval()
    return model


def apply_tensors(transformer, tensors):
    """Route a streamed name->tensor dict into the right parameters. A
    quantized Linear arrives as two keys: `<mod>.weight` (int8) and
    `<mod>.weight.scale` (bf16). The scale must be assigned to the
    module's `weight_scale` parameter, not loaded as a state-dict key
    (there is no `<mod>.weight.scale` in the module's state dict).
    Everything else (norms, biases) loads normally."""
    plain = {}
    for name, t in tensors.items():
        if name.endswith(".weight.scale"):
            mod_path = name[: -len(".weight.scale")]
            mod = transformer.get_submodule(mod_path)
            assert isinstance(mod, Int8WeightOnlyLinear), f"{mod_path} has a .weight.scale but is {type(mod).__name__}"
            mod.weight_scale = nn.Parameter(t, requires_grad=False)
        else:
            plain[name] = t
    if plain:
        transformer.load_state_dict(plain, strict=False, assign=True)


def attach_engine_wo(transformer, engine, compute_stream_ptr, prefetch_ahead=1):
    if engine.vram_slots() <= prefetch_ahead:
        raise ValueError(f"vram_slots ({engine.vram_slots()}) must be > prefetch_ahead ({prefetch_ahead})")

    block_ids = engine.block_ids()
    pos_of = {bid: i for i, bid in enumerate(block_ids)}

    def block_module(block_id):
        family, idx = block_id.rsplit(".", 1)
        return getattr(transformer, family)[int(idx)]

    def make_pre_hook(block_id):
        def hook(module, args, kwargs):
            tensors = engine.get_block(block_id, compute_stream_ptr)
            apply_tensors(transformer, tensors)
            pos = pos_of[block_id] + prefetch_ahead
            if pos < len(block_ids):
                engine.prefetch(block_ids[pos])
            return args, kwargs
        return hook

    def make_post_hook(block_id):
        def hook(module, args, output):
            engine.mark_block_done(block_id, compute_stream_ptr)
            return output
        return hook

    handles = []
    for bid in block_ids:
        mod = block_module(bid)
        handles.append(mod.register_forward_pre_hook(make_pre_hook(bid), with_kwargs=True))
        handles.append(mod.register_forward_hook(make_post_hook(bid)))

    apply_tensors(transformer, engine.get_shared())
    engine.prefetch(block_ids[0])
    return handles


def build_pipeline(device, snapshot, transformer_dir):
    print("loading scheduler/tokenizer/text_encoder/vae (real weights, separate offload policy)...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(snapshot, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, subfolder="tokenizer")
    text_encoder = Qwen3ForCausalLM.from_pretrained(snapshot, subfolder="text_encoder", dtype=torch.bfloat16)
    cpu_offload(text_encoder, execution_device=device)
    vae = AutoencoderKLFlux2.from_pretrained(snapshot, subfolder="vae", dtype=torch.bfloat16)
    cpu_offload(vae, execution_device=device)

    print("building meta transformer with Int8WeightOnlyLinear placeholders...")
    transformer = build_meta_transformer_int8wo(transformer_dir)
    pipe = Flux2KleinPipeline(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, transformer=transformer, is_distilled=True,
    )
    return pipe, transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-dir", required=True, help="weight-only int8 checkpoint dir (quantize_transformer_wo.py output)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default=f"inference/output/{uuid.uuid4()}.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    device = torch.device("cuda:0")
    torch.cuda.init()
    stream_ptr = torch.cuda.current_stream(device).cuda_stream

    # Text encoder / vae / scheduler still come from the ORIGINAL snapshot
    # (only the transformer was quantized). Resolve it from the HF cache.
    snapshot = snapshot_download(MODEL_ID, allow_patterns=["scheduler/*", "tokenizer/*", "text_encoder/*", "vae/*", "*.json", "*.txt"])
    hub = os.path.dirname(os.path.dirname(args.transformer_dir.rstrip("/")))

    t0 = time.time()
    pipe, transformer = build_pipeline(device, snapshot, args.transformer_dir)

    print(f"initializing Rust engine from {args.transformer_dir} (budget={PINNED_BUDGET/1e9:.1f}GB)...")
    engine = se.Engine(args.transformer_dir, PINNED_BUDGET, VRAM_SLOTS, 0, hub, None, 0)
    print(f"engine stats after init: {engine.stats()}")
    attach_engine_wo(transformer, engine, stream_ptr, prefetch_ahead=PREFETCH_AHEAD)
    load_time = time.time() - t0
    print(f"\ntotal load time: {load_time:.2f}s")

    print(f"\ngenerating {args.width}x{args.height}: {args.prompt!r} (seed={args.seed}, steps={args.steps})")
    t0 = time.time()
    with torch.no_grad():
        image = pipe(prompt=args.prompt, width=args.width, height=args.height,
                     num_inference_steps=args.steps,
                     generator=torch.Generator(device="cpu").manual_seed(args.seed)).images[0]
    gen_time = time.time() - t0
    print(f"generation time: {gen_time:.2f}s ({gen_time/args.steps:.3f}s/it)")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    image.save(args.out)
    print(f"saved to {args.out}")

    stats = engine.stats()
    print("\n=== engine stats (whole run) ===")
    print(f"  bytes transferred H2D:   {stats['bytes_h2d']/1e9:.3f} GB")
    print(f"  transfer count:          {stats['transfer_count']}")
    print(f"  pinned host bytes:       {stats['pinned_bytes']/1e9:.3f} GB")
    print(f"  VRAM bytes (engine):     {stats['vram_bytes']/1e9:.3f} GB")


if __name__ == "__main__":
    sys.exit(main())
