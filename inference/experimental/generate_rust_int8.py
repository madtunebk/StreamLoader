#!/usr/bin/env python3
"""
Same streaming architecture as generate_rust.py (Rust engine: pinned
host cache + reusable VRAM ring + DLPack, per-block prefetch hooks) --
the ONLY thing that changes is that TRANSFORMER_DIR now points at the
int8-quantized checkpoint (quantize_transformer.py's output, ~9.09GB vs
the original 18.16GB). This deliberately does NOT try to fit the whole
model resident in VRAM and bypass streaming -- the point of this script
is to isolate one variable (transfer volume) against the existing
streaming design and measure whether it moves generation time, not to
build a different architecture.

Each streamed Linear weight now arrives as a (CB int8, SCB fp32) pair
(bitsandbytes' row-wise int8 format) instead of a single bf16 tensor.
Every block-hook re-fetch constructs a fresh bnb.nn.Int8Params from
that pair and assigns it to the corresponding Linear8bitLt module's
`.weight` -- bitsandbytes' own Linear8bitLt.forward() detects a fresh
non-None `.weight.CB` and re-derives `.state.CB`/`.state.SCB` from it on
its own (see Linear8bitLt.forward's `init_8bit_state()` call), so this
composes correctly with data that changes on every single forward call,
which is exactly what per-block streaming does.
"""

import argparse
import json
import sys
import time

import bitsandbytes as bnb
import torch
import torch.nn as nn
from accelerate import cpu_offload, init_empty_weights
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, Qwen3ForCausalLM

import streamloader_engine as se

HUB = "/home/nobus/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9B"
SNAPSHOT = f"{HUB}/snapshots/92196c8e11f7b6cf2b7493e037d8c5345c559216"
TRANSFORMER_DIR = "/home/nobus/Raid0/RustStream/inference/quantized_model/transformer"
PINNED_BUDGET = 12 * 1024 * 1024 * 1024  # quantized transformer is ~9.09GB
VRAM_SLOTS = 2
PREFETCH_AHEAD = 1


def build_meta_transformer_int8():
    cfg = json.load(open(f"{TRANSFORMER_DIR}/config.json"))
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    with init_empty_weights():
        model = Flux2Transformer2DModel(**cfg)

    # Replace every nn.Linear with a bnb Linear8bitLt placeholder (meta
    # device -- never read before our hooks overwrite it with real
    # streamed data, so no point materializing real throwaway weights).
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and not isinstance(child, bnb.nn.Linear8bitLt):
                new_mod = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,
                    # LLM.int8()'s outlier threshold: activation columns
                    # with any value exceeding this run in fp16 instead
                    # of int8. threshold=0.0 (disabled) produced pure
                    # noise output on the real model -- diffusion
                    # transformer activations have real outliers this
                    # model's original bf16 dynamic range accommodates
                    # but naive int8 (and bnb's internal bf16->fp16 cast
                    # before quantizing) does not. 6.0 is the standard
                    # LLM.int8() paper default.
                    threshold=6.0,
                    device="meta",
                )
                setattr(module, child_name, new_mod)
                replaced += 1
    print(f"replaced {replaced} nn.Linear modules with Linear8bitLt")

    model.eval()
    return model


def apply_tensors(transformer, tensors, stream_ptr):
    """Split a name->tensor dict into (Linear8bitLt weight, its .SCB
    companion) pairs vs. everything else (norms, biases -- ordinary
    tensors), and apply each the right way."""
    scb_names = {n for n in tensors if n.endswith(".SCB")}
    weight_names_with_scb = {n[: -len(".SCB")] for n in scb_names}

    plain = {}
    for name, t in tensors.items():
        if name in scb_names:
            continue
        if name in weight_names_with_scb:
            mod_path, attr = name.rsplit(".", 1)
            lin_mod = transformer.get_submodule(mod_path)
            assert isinstance(lin_mod, bnb.nn.Linear8bitLt), f"{mod_path} is not Linear8bitLt but has a .SCB tensor"
            scb = tensors[name + ".SCB"]
            lin_mod.weight = bnb.nn.Int8Params(t, requires_grad=False, has_fp16_weights=False, CB=t, SCB=scb)
        else:
            plain[name] = t

    if plain:
        transformer.load_state_dict(plain, strict=False, assign=True)


def attach_engine(transformer, engine, stream_ptr):
    block_ids = engine.block_ids()
    pos_of = {bid: i for i, bid in enumerate(block_ids)}

    def block_module(block_id):
        family, idx = block_id.rsplit(".", 1)
        return getattr(transformer, family)[int(idx)]

    def make_pre_hook(block_id):
        def hook(module, args, kwargs):
            tensors = engine.get_block(block_id, stream_ptr)
            apply_tensors(transformer, tensors, stream_ptr)
            pos = pos_of[block_id] + PREFETCH_AHEAD
            if pos < len(block_ids):
                engine.prefetch(block_ids[pos])
            return args, kwargs

        return hook

    def make_post_hook(block_id):
        def hook(module, args, output):
            engine.mark_block_done(block_id, stream_ptr)
            return output

        return hook

    handles = []
    for bid in block_ids:
        mod = block_module(bid)
        handles.append(mod.register_forward_pre_hook(make_pre_hook(bid), with_kwargs=True))
        handles.append(mod.register_forward_hook(make_post_hook(bid)))

    shared = engine.get_shared()
    apply_tensors(transformer, shared, stream_ptr)
    print(f"shared weights loaded: {len(shared)} tensors")

    engine.prefetch(block_ids[0])
    return handles


def build_pipeline(device):
    print("loading scheduler/tokenizer/text_encoder/vae (real weights, separate offload policy)...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SNAPSHOT, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, subfolder="tokenizer")

    text_encoder = Qwen3ForCausalLM.from_pretrained(SNAPSHOT, subfolder="text_encoder", dtype=torch.bfloat16)
    cpu_offload(text_encoder, execution_device=device)

    vae = AutoencoderKLFlux2.from_pretrained(SNAPSHOT, subfolder="vae", dtype=torch.bfloat16)
    cpu_offload(vae, execution_device=device)

    print("building meta transformer with Linear8bitLt placeholders...")
    transformer = build_meta_transformer_int8()

    pipe = Flux2KleinPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        is_distilled=True,
    )
    return pipe, transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default="/home/nobus/Raid0/RustStream/inference/output_int8.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    torch.cuda.init()
    device = torch.device("cuda:0")
    stream_ptr = torch.cuda.current_stream().cuda_stream

    t0 = time.time()
    pipe, transformer = build_pipeline(device)

    print(f"initializing Rust engine from {TRANSFORMER_DIR} (budget={PINNED_BUDGET/1e9:.1f}GB, {VRAM_SLOTS} VRAM slots)...")
    engine = se.Engine(TRANSFORMER_DIR, PINNED_BUDGET, VRAM_SLOTS, 0, None)
    print(f"engine stats after init: {engine.stats()}")
    attach_engine(transformer, engine, stream_ptr)

    load_time = time.time() - t0
    print(f"\ntotal load time (pipeline + engine + hooks): {load_time:.2f}s")

    print(f"\ngenerating {args.width}x{args.height}: {args.prompt!r} (seed={args.seed}, steps={args.steps})")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t0 = time.time()
    with torch.no_grad():
        image = pipe(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]
    gen_time = time.time() - t0
    print(f"generation time (steady-state, post-warmup): {gen_time:.2f}s")

    image.save(args.out)
    print(f"saved to {args.out}")

    stats = engine.stats()
    print("\n=== engine stats (whole run) ===")
    print(f"  bytes transferred H2D:   {stats['bytes_h2d']/1e9:.3f} GB")
    print(f"  transfer count:          {stats['transfer_count']}")
    print(f"  cache hits:              {stats['cache_hits']}")
    print(f"  pinned host bytes:       {stats['pinned_bytes']/1e9:.3f} GB")
    print(f"  VRAM bytes (engine):     {stats['vram_bytes']/1e9:.3f} GB")
    print(f"\nload_time={load_time:.2f}s generation_time={gen_time:.2f}s")


if __name__ == "__main__":
    sys.exit(main())
