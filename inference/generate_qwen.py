#!/usr/bin/env python3
"""
Minimal Qwen-Image-2.1 generation with the Rust streaming engine -- same
mechanism as generate_simple.py (FLUX.2), proving the engine generalizes
to a different DiT architecture without any Rust changes.

Qwen-Image-2.1's transformer has only ONE block family (`transformer_blocks`,
no double/single-stream split like FLUX) and a flat, deterministic
forward loop (verified against the installed diffusers source before
this was written) -- so block_families=["transformer_blocks"] is all
the engine needs to know.
"""

import json

import torch
from accelerate import cpu_offload, init_empty_weights
from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline, QwenImage21Transformer2DModel
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

import streamloader_engine as se

HUB = "/home/nobus/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1"
SNAPSHOT = f"{HUB}/snapshots/790c92633540aa0cb11d9abf19eb46d861714758"
TRANSFORMER_DIR = f"{SNAPSHOT}/transformer"

PROMPT = "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
SEED = 42
STEPS = 20
RESIDENT_GB = 2.0
OUT = "inference/output/qwen_out.png"

device = torch.device("cuda:0")
torch.cuda.init()
stream_ptr = torch.cuda.current_stream().cuda_stream

# --- ordinary diffusers loading for everything except the transformer ---
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SNAPSHOT, subfolder="scheduler")
processor = Qwen3VLProcessor.from_pretrained(SNAPSHOT, subfolder="processor")
text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(SNAPSHOT, subfolder="text_encoder", dtype=torch.bfloat16)
cpu_offload(text_encoder, execution_device=device)
vae = AutoencoderKLQwenImage21.from_pretrained(SNAPSHOT, subfolder="vae", dtype=torch.bfloat16)
vae.enable_slicing()
cpu_offload(vae, execution_device=device)

# --- transformer: meta weights, real weights served by the Rust engine ---
cfg = json.load(open(f"{TRANSFORMER_DIR}/config.json"))
cfg.pop("_class_name", None)
cfg.pop("_diffusers_version", None)
with init_empty_weights():
    transformer = QwenImage21Transformer2DModel(**cfg)
transformer.eval()

engine = se.Engine(
    TRANSFORMER_DIR, 20 * 1024**3, 2, 0, HUB,
    ["transformer_blocks"],  # only block family Qwen-Image-2.1 has -- see module docstring
    int(RESIDENT_GB * 1024**3),
)
se.attach_engine(transformer, engine, stream_ptr)

# --- ordinary diffusers pipeline call, same as any other model ---
pipe = QwenImage21Pipeline(
    scheduler=scheduler, vae=vae, text_encoder=text_encoder, processor=processor,
    transformer=transformer,
)

with torch.no_grad():
    image = pipe(
        prompt=PROMPT,
        width=1024,
        height=1024,
        num_inference_steps=STEPS,
        generator=torch.Generator(device="cpu").manual_seed(SEED),
    ).images[0]

image.save(OUT)
print(f"saved to {OUT}")
