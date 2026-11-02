#!/usr/bin/env python3
"""Qwen-Image-2.1 inference with Rust-managed transformer weights."""

import argparse
import gc
import json
import os
import sys
import time
import uuid
from pathlib import Path

import torch
from accelerate import cpu_offload, init_empty_weights
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

import streamloader_engine as se


MODEL_ID = "Qwen/Qwen-Image-2.1"
SNAPSHOT = Path(snapshot_download(
    MODEL_ID,
    allow_patterns=["transformer/*", "*.json", "*.txt"],
))
HUB = SNAPSHOT.parent.parent
TRANSFORMER_DIR = SNAPSHOT / "transformer"

PINNED_BUDGET = 20 * 1024**3
VRAM_SLOTS = 2
PREFETCH_AHEAD = 1

ASPECT_RATIOS = {
    "1:1": (2048, 2048), "4:3": (2400, 1792),
    "3:4": (1792, 2400), "3:2": (2528, 1696),
    "2:3": (1696, 2528), "16:9": (2752, 1536),
    "9:16": (1536, 2752), "1:1-lo": (1024, 1024),
    "4:3-lo": (1152, 896), "3:4-lo": (896, 1152),
    "16:9-lo": (1280, 720), "9:16-lo": (720, 1280),
}


def build_meta_transformer():
    with (TRANSFORMER_DIR / "config.json").open() as f:
        cfg = json.load(f)

    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)

    with init_empty_weights():
        model = QwenImage21Transformer2DModel(**cfg)

    model.eval()

    if not all(p.is_meta for p in model.parameters()):
        raise RuntimeError("Transformer parameters were not all initialized on meta.")

    return model


def build_pipeline(device):
    print("Loading scheduler, processor, text encoder and VAE...")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_ID, subfolder="scheduler"
    )
    processor = Qwen3VLProcessor.from_pretrained(
        MODEL_ID, subfolder="processor"
    )

    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    cpu_offload(text_encoder, execution_device=device)

    vae = AutoencoderKLQwenImage21.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16
    )
    vae.enable_slicing()
    vae.enable_tiling()
    cpu_offload(vae, execution_device=device)

    print("Building meta transformer...")
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
    parser.add_argument("--out", default=f"inference/output/{uuid.uuid4()}.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aspect", choices=sorted(ASPECT_RATIOS))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40)
    # GB of the transformer kept permanently resident in VRAM. Resident
    # blocks are never re-transferred, so each GB removes that GB from
    # every denoising step's H2D traffic. 0 = pure-streaming baseline.
    # Default 4.0 fits safely on a 12GB card; raise/lower per free VRAM.
    parser.add_argument("--resident-gb", type=float, default=4.0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--images", nargs="+", default=None)
    args = parser.parse_args()

    if args.aspect:
        args.width, args.height = ASPECT_RATIOS[args.aspect]

    if not args.prompt.strip():
        parser.error("--prompt cannot be empty")
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        parser.error("--width and --height must be positive multiples of 16")
    if args.steps < 1 or args.batch < 1 or args.resident_gb < 0:
        parser.error("--steps and --batch must be positive; --resident-gb cannot be negative")
    if not Path(args.out).suffix:
        parser.error("--out must include a file extension, e.g. output.png")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check PyTorch and your NVIDIA driver.")

    device = torch.device("cuda:0")
    torch.cuda.init()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    start = time.perf_counter()

    pipe, transformer = build_pipeline(device)

    resident_budget = int(args.resident_gb * 1024**3)
    print(f"Initializing Rust engine from {TRANSFORMER_DIR}...")

    engine = se.Engine(
        str(TRANSFORMER_DIR),
        PINNED_BUDGET,
        VRAM_SLOTS,
        0,
        str(HUB),
        ["transformer_blocks"],
        resident_budget,
    )

    print("Resident blocks:", engine.resident_block_ids())
    print("Engine stats:", engine.stats())

    se.attach_engine(
        transformer,
        engine,
        torch.cuda.current_stream(device).cuda_stream,
        prefetch_ahead=PREFETCH_AHEAD,
    )
    # Enable persistent PyTorch compilation cache.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.cache/torchinductor")

    # Compile the transformer.
    transformer = torch.compile(transformer)
 
    load_time = time.perf_counter() - start
    print(f"Load time: {load_time:.2f}s")

    payload = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "num_images_per_prompt": args.batch,
        "generator": torch.Generator(device="cpu").manual_seed(args.seed),
    }

    reference_images = []
    try:
        if args.images:
            for path in args.images:
                with Image.open(path) as im:
                    reference_images.append(im.convert("RGB"))
            payload["image"] = reference_images

        print(f"Generating {args.width}x{args.height}, batch={args.batch}...")
        start = time.perf_counter()

        with torch.inference_mode():
            images = pipe(**payload).images

        torch.cuda.synchronize(device)
        gen_time = time.perf_counter() - start

        if not images:
            raise RuntimeError("Pipeline returned no images.")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)

        if len(images) == 1:
            images[0].save(out)
            print(f"Saved: {out}")
        else:
            for i, image in enumerate(images):
                filename = out.with_name(f"{out.stem}_{i}{out.suffix}")
                image.save(filename)
                print(f"Saved: {filename}")

        stats = engine.stats()
        print("\n=== Engine stats ===")
        for key in (
            "bytes_h2d", "transfer_count", "cache_hits",
            "pinned_bytes", "vram_bytes", "resident_bytes", "resident_hits",
        ):
            value = stats.get(key)
            if value is not None:
                if key.endswith("_bytes") or key == "bytes_h2d":
                    print(f"{key}: {value / 1e9:.3f} GB")
                else:
                    print(f"{key}: {value}")

        print(f"\nload_time={load_time:.2f}s generation_time={gen_time:.2f}s")

    finally:
        for image in reference_images:
            image.close()
        gc.collect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)