#!/usr/bin/env python3
"""
Real diffusers inference for Qwen-Image-2.1 where the *transformer's*
weights are served by the Rust engine (streamloader_engine) instead of
diffusers' own loader/offload machinery -- same mechanism as
generate_rust.py (FLUX.2), proving the engine generalizes to a different
DiT architecture with zero changes to engine/src/*.rs:

- transformer is constructed with meta (0-byte) weights via
  accelerate.init_empty_weights() -- diffusers' loader never materializes
  a CPU copy of it at all.
- Rust's Engine owns one pinned host buffer (built directly from the
  mmap'd checkpoint) and a small ring of reusable VRAM buffers, plus an
  optional static ResidentPool (see docs/RESIDENT_POOL_TODO.md).
- Per-block forward hooks (se.attach_engine) pull each block's real CUDA
  tensors from the engine (DLPack, zero payload copy) and
  load_state_dict(..., assign=True) them into the meta module right
  before it runs, then tell the engine the block is done right after.
- text_encoder and vae are NOT managed by the engine -- ordinary real
  weights + accelerate.cpu_offload (each runs exactly once per
  generation, not once per denoising step, so offloading them is
  essentially free).
- Qwen-Image-2.1's transformer has only ONE block family
  (`transformer_blocks`, no double/single-stream split like FLUX) and a
  flat, deterministic forward loop, verified against the installed
  diffusers source before this was written -- including that its default
  KV-cache mode (checkpoint config: causal_condition=true) still calls
  the transformer exactly once per denoising step, so it doesn't violate
  the engine's block-order-determinism assumption.
- If the engine fails to initialize, this script does not catch it and
  fall back to some other loading path -- it crashes loudly.
"""

import argparse
import json
import os
import sys
import time
import uuid

import torch
from accelerate import cpu_offload, init_empty_weights
from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline, QwenImage21Transformer2DModel
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

import streamloader_engine as se

# No hardcoded local path: snapshot_download resolves (and downloads if
# needed) using the caller's own HF cache -- portable across machines,
# no username/home-directory baked into the script.
MODEL_ID = "Qwen/Qwen-Image-2.1"
SNAPSHOT = snapshot_download(MODEL_ID, allow_patterns=["transformer/*", "*.json", "*.txt"])
HUB = os.path.dirname(os.path.dirname(SNAPSHOT))  # models--org--name/ -- needed as the engine's trust_root
TRANSFORMER_DIR = f"{SNAPSHOT}/transformer"
PINNED_BUDGET = 20 * 1024 * 1024 * 1024  # transformer is ~14.23GB
VRAM_SLOTS = 2
PREFETCH_AHEAD = 1

# Aspect-ratio presets, tested today: "full" sizes need vae.enable_tiling()
# to avoid a real VAE-decode OOM at 2048x2048 (see docs/RESIDENT_POOL_TODO.md);
# "-lo" presets are lighter/faster, for a quick test or a weaker card.
ASPECT_RATIOS = {
    "1:1":  (2048, 2048),
    "4:3":  (2400, 1792),
    "3:4":  (1792, 2400),
    "3:2":  (2528, 1696),
    "2:3":  (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
    "1:1-lo":  (1024, 1024),
    "4:3-lo":  (1152, 896),
    "3:4-lo":  (896, 1152),
    "16:9-lo": (1280, 720),
    "9:16-lo": (720, 1280),
}


def build_meta_transformer():
    cfg = json.load(open(f"{TRANSFORMER_DIR}/config.json"))
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    with init_empty_weights():
        model = QwenImage21Transformer2DModel(**cfg)
    model.eval()
    return model


def build_pipeline(device):
    print("loading scheduler/processor/text_encoder/vae (real weights, separate offload policy)...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    processor = Qwen3VLProcessor.from_pretrained(MODEL_ID, subfolder="processor")

    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, subfolder="text_encoder", dtype=torch.bfloat16)
    cpu_offload(text_encoder, execution_device=device)

    vae = AutoencoderKLQwenImage21.from_pretrained(MODEL_ID, subfolder="vae", dtype=torch.bfloat16)
    vae.enable_slicing()  # bounds peak VRAM for batch>1 (decodes one image at a time)
    vae.enable_tiling()   # bounds peak VRAM for large single-image resolutions (e.g. 2048x2048)
    cpu_offload(vae, execution_device=device)

    print("building meta transformer (0 bytes materialized by diffusers)...")
    transformer = build_meta_transformer()

    pipe = QwenImage21Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        processor=processor,
        transformer=transformer,
    )
    return pipe, transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default=f"inference/output/{str(uuid.uuid4())}.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aspect", choices=sorted(ASPECT_RATIOS), default=None,
                         help="pick a preset instead of --width/--height")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    # Not step-distilled (unlike FLUX.2-klein): pipeline default is 40.
    # ~30 is a functional minimum, 45-50 for full-quality small text --
    # see docs/RESIDENT_POOL_TODO.md's step-count sweep.
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--resident-gb", type=float, default=0.0)
    parser.add_argument("--batch", type=int, default=1)
    # Reference-image conditioning (QwenImage21Pipeline's `image=` param).
    parser.add_argument("--image", default=None, help="path to a reference image, enables image-conditioning mode")
    args = parser.parse_args()

    if args.aspect:
        args.width, args.height = ASPECT_RATIOS[args.aspect]

    torch.cuda.init()
    device = torch.device("cuda:0")
    stream_ptr = torch.cuda.current_stream().cuda_stream

    t0 = time.time()
    pipe, transformer = build_pipeline(device)

    resident_budget = int(args.resident_gb * 1024**3)
    print(f"initializing Rust engine from {TRANSFORMER_DIR} (budget={PINNED_BUDGET/1e9:.1f}GB, {VRAM_SLOTS} VRAM slots, resident={args.resident_gb:.2f}GB)...")
    engine = se.Engine(TRANSFORMER_DIR, PINNED_BUDGET, VRAM_SLOTS, 0, HUB, ["transformer_blocks"], resident_budget)
    print(f"resident blocks chosen: {engine.resident_block_ids()}")
    print(f"engine stats after init: {engine.stats()}")
    se.attach_engine(transformer, engine, stream_ptr, prefetch_ahead=PREFETCH_AHEAD)

    load_time = time.time() - t0
    print(f"\ntotal load time (pipeline + engine + hooks): {load_time:.2f}s")

    payload = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "num_images_per_prompt": args.batch,
        "generator": torch.Generator(device="cpu").manual_seed(args.seed),
    }
    if args.image:
        payload["image"] = Image.open(args.image)

    print(f"\ngenerating {args.width}x{args.height} x{args.batch}: {args.prompt!r} (seed={args.seed})")
    t0 = time.time()
    with torch.no_grad():
        images = pipe(**payload).images
    gen_time = time.time() - t0
    print(f"generation time (steady-state, post-warmup): {gen_time:.2f}s ({gen_time/args.batch:.2f}s/image)")

    if args.batch == 1:
        images[0].save(args.out)
        print(f"saved to {args.out}")
    else:
        base, ext = args.out.rsplit(".", 1)
        for i, image in enumerate(images):
            out_path = f"{base}_{i}.{ext}"
            image.save(out_path)
        print(f"saved {args.batch} images to {base}_0.{ext} .. {base}_{args.batch-1}.{ext}")

    stats = engine.stats()
    print("\n=== engine stats (whole run) ===")
    print(f"  bytes transferred H2D:   {stats['bytes_h2d']/1e9:.3f} GB")
    print(f"  transfer count:          {stats['transfer_count']}")
    print(f"  cache hits:              {stats['cache_hits']}")
    print(f"  pinned host bytes:       {stats['pinned_bytes']/1e9:.3f} GB")
    print(f"  VRAM bytes (engine):     {stats['vram_bytes']/1e9:.3f} GB")
    print(f"  resident bytes:          {stats['resident_bytes']/1e9:.3f} GB")
    print(f"  resident hits:           {stats['resident_hits']}")
    print(f"\nload_time={load_time:.2f}s generation_time={gen_time:.2f}s resident_gb={args.resident_gb:.2f} batch={args.batch}")


if __name__ == "__main__":
    sys.exit(main())
