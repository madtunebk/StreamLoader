#!/usr/bin/env python3
"""
Real diffusers inference for FLUX.2-klein-9B, loaded from the local
Hugging Face cache (local_files_only=True -- never re-downloads).
Uses model CPU offloading so the ~18GB transformer + Qwen3 text encoder +
VAE, which don't all fit in one 12GB RTX 3060, stream through the GPU one
component at a time instead of requiring everything resident at once.
"""

import sys
import time

import torch
from diffusers import Flux2KleinPipeline

MODEL_DIR = (
    "/home/nobus/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9B"
    "/snapshots/92196c8e11f7b6cf2b7493e037d8c5345c559216"
)

PROMPT = "a bottle on a table with the milky way galaxy swirling inside it"
WIDTH = 1024
HEIGHT = 1024
OUT_PATH = "/home/nobus/Raid0/RustStream/inference/output.png"


def main():
    print(f"loading pipeline from local cache: {MODEL_DIR}", flush=True)
    t0 = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    # Both the ~18GB transformer and the ~15.4GB Qwen3-8B text encoder
    # individually exceed one 12GB RTX 3060, so whole-component offload
    # (enable_model_cpu_offload) OOMs on the first component it moves.
    # Sequential offload streams one layer at a time instead -- slower,
    # but only ever needs a thin slice of any component resident on GPU.
    pipe.enable_sequential_cpu_offload()

    print(f"generating {WIDTH}x{HEIGHT}: {PROMPT!r}", flush=True)
    t0 = time.time()
    generator = torch.Generator(device="cpu").manual_seed(0)
    # num_inference_steps/guidance_scale left at the pipeline's own defaults
    # (50 steps, guidance_scale=4.0) rather than guessed values.
    image = pipe(
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        generator=generator,
    ).images[0]
    print(f"generated in {time.time() - t0:.1f}s", flush=True)

    image.save(OUT_PATH)
    print(f"saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
