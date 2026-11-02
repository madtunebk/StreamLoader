#!/usr/bin/env python3
"""
Real diffusers inference for FLUX.2-klein-9B where the *transformer's*
weights are served by the Rust engine (streamloader_engine) instead of
diffusers' own loader/offload machinery:

- transformer is constructed with meta (0-byte) weights via
  accelerate.init_empty_weights() -- diffusers' loader never materializes
  a CPU copy of it at all.
- Rust's Engine owns one pinned host buffer (built directly from the
  mmap'd checkpoint) and a small ring of reusable VRAM buffers.
- Per-block forward hooks pull that block's real CUDA tensors from the
  engine (DLPack, zero payload copy) and load_state_dict(..., assign=True)
  them into the meta module right before it runs, then tell the engine
  the block is done right after.
- text_encoder and vae are NOT managed by the engine -- they get their
  own, separate policy: ordinary real weights + accelerate.cpu_offload
  (each runs exactly once per generation, unlike the transformer's 50
  denoising steps, so offloading them is essentially free).
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
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoTokenizer, Qwen3ForCausalLM

import streamloader_engine as se

# No hardcoded local path: snapshot_download resolves (and downloads if
# needed) using the caller's own HF cache -- portable across machines,
# no username/home-directory baked into the script.
MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
SNAPSHOT = snapshot_download(MODEL_ID, allow_patterns=["transformer/*", "*.json", "*.txt"])
HUB = os.path.dirname(os.path.dirname(SNAPSHOT))  # models--org--name/ -- needed as the engine's trust_root
TRANSFORMER_DIR = f"{SNAPSHOT}/transformer"
PINNED_BUDGET = 20 * 1024 * 1024 * 1024  # transformer is ~18.16GB
VRAM_SLOTS = 2
PREFETCH_AHEAD = 1


def build_meta_transformer():
    cfg = json.load(open(f"{TRANSFORMER_DIR}/config.json"))
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    with init_empty_weights():
        model = Flux2Transformer2DModel(**cfg)
    model.eval()
    return model


def build_pipeline(device):
    print("loading scheduler/tokenizer/text_encoder/vae (real weights, separate offload policy)...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")

    text_encoder = Qwen3ForCausalLM.from_pretrained(MODEL_ID, subfolder="text_encoder", dtype=torch.bfloat16)
    cpu_offload(text_encoder, execution_device=device)

    vae = AutoencoderKLFlux2.from_pretrained(MODEL_ID, subfolder="vae", dtype=torch.bfloat16)
    # Decode one image of the batch at a time instead of all at once --
    # off by default. This is what actually reduces VAE decode's peak
    # VRAM for batch>1 (not enable_tiling(), which splits large per-image
    # spatial dims, not large batches -- irrelevant to the batch=6 OOM
    # this was added for, see RESIDENT_POOL_TODO.md).
    vae.enable_slicing()
    cpu_offload(vae, execution_device=device)

    print("building meta transformer (0 bytes materialized by diffusers)...")
    transformer = build_meta_transformer()

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
    parser.add_argument("--out",  default=f"inference/output/{str(uuid.uuid4())}.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    # This is a step-distilled model (config.json: is_distilled=true,
    # guidance_embeds=false) -- per user guidance, ~18-20 steps is the
    # useful range; the pipeline's own default (50) is more than this
    # architecture needs. Kept overridable, not hardcoded.
    parser.add_argument("--steps", type=int, default=20)
    # Stage 3 (ResidentPool, static/init-time-only): how many GB of the
    # transformer to keep permanently resident in VRAM. Resident blocks
    # are never re-transferred, so every GB resident removes that GB from
    # the H2D traffic of EVERY denoising step -- the main lever for cutting
    # CPU->GPU transfer. 0 restores the pure-streaming baseline (whole
    # model re-transferred every step). Default 4.0 is a safe fit on a 12GB
    # card alongside the ring+shared slots (~2.4GB) and forward-pass
    # activations (~2-3GB); raise it if you have more free VRAM, lower it
    # if you OOM at large resolutions or batch>1. See RESIDENT_POOL_TODO.md
    # for the full budget sweep (gains flatten past ~6-8GB).
    parser.add_argument("--resident-gb", type=float, default=4.0)
    # Batched generation: the Rust engine only ever serves WEIGHT tensors
    # (no batch dimension), so it's fully agnostic to batch size -- only
    # the activations/latents PyTorch itself allocates scale with this,
    # which is workspace VRAM the engine has no visibility into (see the
    # review's §7). Default 1 keeps existing single-image behavior and
    # output naming identical.
    parser.add_argument("--batch", type=int, default=1)
    # Flux2KleinPipeline.__call__ takes `image` as its first param --
    # img2img/reference-conditioning, same mechanism as QwenImage21Pipeline.
    # Verified from source (pipeline_flux2_klein.py) before wiring this up.
    parser.add_argument("--image", default=None, help="path to a reference image, enables image-conditioning mode")
    # Block-level torch.compile. FLUX on a 3060 is COMPUTE-bound (proven:
    # halving transfer bytes via weight-only int8 did not speed it up),
    # so the only real s/it lever is cutting compute -- which is what
    # torch.compile's kernel fusion does. Compiled per block (not whole
    # model) so the per-step load_state_dict(assign=True) weight swap
    # changes tensor data, not graph structure: each block compiles once
    # on its first step and is reused, rather than recompiling every step.
    # Same pattern as generate_qwen_compiled.py. Off by default (first run
    # pays a one-time compile cost); measure --compile vs eager at the
    # same seed/steps.
    parser.add_argument("--compile", action="store_true", help="block-level torch.compile the transformer (compute speedup)")
    args = parser.parse_args()

    torch.cuda.init()
    device = torch.device("cuda:0")
    stream_ptr = torch.cuda.current_stream().cuda_stream

    t0 = time.time()
    pipe, transformer = build_pipeline(device)

    resident_budget = int(args.resident_gb * 1024**3)
    print(f"initializing Rust engine from {TRANSFORMER_DIR} (budget={PINNED_BUDGET/1e9:.1f}GB, {VRAM_SLOTS} VRAM slots, resident={args.resident_gb:.2f}GB)...")
    engine = se.Engine(TRANSFORMER_DIR, PINNED_BUDGET, VRAM_SLOTS, 0, HUB, None, resident_budget)
    print(f"resident blocks chosen: {engine.resident_block_ids()}")
    print(f"engine stats after init: {engine.stats()}")
    se.attach_engine(transformer, engine, stream_ptr, prefetch_ahead=PREFETCH_AHEAD)

    if args.compile:
        # Persistent inductor cache so the compile cost is paid once across
        # runs, not every process start.
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor"))
        # Compile each DiT block's forward individually. FLUX has two
        # families: double-stream (transformer_blocks) and single-stream
        # (single_transformer_blocks). fullgraph=False tolerates the
        # forward-hook boundary (the engine's pre/post hooks are not part
        # of the compiled region, so a graph break there is expected and
        # fine). The streamed weights arrive via assign=True as new tensor
        # objects each step; compiling the block (stable structure) rather
        # than the whole model keeps that a data change, not a recompile.
        n = 0
        for family in ("transformer_blocks", "single_transformer_blocks"):
            for block in getattr(transformer, family):
                block.forward = torch.compile(block.forward, mode="default", fullgraph=False)
                n += 1
        print(f"torch.compile enabled on {n} blocks (first step will pay a one-time compile cost)")

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

    # inference/output/ is gitignored (generated content) -- git tracks
    # no empty directories, so a fresh clone has no such folder at all
    # until something creates it. Found by an actual clean-clone test,
    # not assumed: this crashed at the save step otherwise.
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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
