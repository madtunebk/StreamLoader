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
from PIL import Image

import streamloader_engine as se

HUB = "/home/nobus/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1"
SNAPSHOT = f"{HUB}/snapshots/790c92633540aa0cb11d9abf19eb46d861714758"
TRANSFORMER_DIR = f"{SNAPSHOT}/transformer"

PROMPT = "This is an RGBA image with transparency. A World pixel art , represent earh. The image has alpha channel and the background is transparent."


image_edit  = False
input_image = Image.open("/home/nobus/Pictures/Wallpapers/1296611123151.jpg")

ASPECT_RATIOS = {
    # full-size (tested today up to 2048x2048 -- held around 7/12GB VRAM)
    "1:1":  (2048, 2048),
    "4:3":  (2400, 1792),
    "3:4":  (1792, 2400),
    "3:2":  (2528, 1696),
    "2:3":  (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
    # budget presets -- lighter/faster, for weaker cards or a quick test
    "1:1-lo":  (1024, 1024),
    "4:3-lo":  (1152, 896),
    "3:4-lo":  (896, 1152),
    "16:9-lo": (1280, 720),
    "9:16-lo": (720, 1280),
}
ASPECT = "1:1-lo"  # pick a key from ASPECT_RATIOS above

SEED = 20260923
STEPS = 40
RESIDENT_GB = 4.0
WIDTH, HEIGHT = ASPECT_RATIOS[ASPECT]

GB = 1024**3
PINNED_BUDGET_BYTES = 20 * GB  # transformer is ~14.23GB, this just needs headroom above that
VRAM_SLOTS = 2
DEVICE_ORDINAL = 0

OUT = "inference/output/qwen_dragon_sticker.png"

device = torch.device(f"cuda:{DEVICE_ORDINAL}")
torch.cuda.init()
stream_ptr = torch.cuda.current_stream().cuda_stream

# --- ordinary diffusers loading for everything except the transformer ---
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SNAPSHOT, subfolder="scheduler")
processor = Qwen3VLProcessor.from_pretrained(SNAPSHOT, subfolder="processor")
text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(SNAPSHOT, subfolder="text_encoder", dtype=torch.bfloat16)
cpu_offload(text_encoder, execution_device=device)
vae = AutoencoderKLQwenImage21.from_pretrained(SNAPSHOT, subfolder="vae", dtype=torch.bfloat16)
vae.enable_slicing()  # bounds peak VRAM for batch>1 (decodes one image at a time)
vae.enable_tiling()   # bounds peak VRAM for large single-image resolutions (e.g. 2048x2048)
cpu_offload(vae, execution_device=device)

# --- transformer: meta weights, real weights served by the Rust engine ---
cfg = json.load(open(f"{TRANSFORMER_DIR}/config.json"))
cfg.pop("_class_name", None)
cfg.pop("_diffusers_version", None)
with init_empty_weights():
    transformer = QwenImage21Transformer2DModel(**cfg)
transformer.eval()

engine = se.Engine(
    TRANSFORMER_DIR, PINNED_BUDGET_BYTES, VRAM_SLOTS, DEVICE_ORDINAL, HUB,
    ["transformer_blocks"],  # only block family Qwen-Image-2.1 has -- see module docstring
    int(RESIDENT_GB * GB),
)
se.attach_engine(transformer, engine, stream_ptr)

# --- ordinary diffusers pipeline call, same as any other model ---
pipe = QwenImage21Pipeline(
    scheduler=scheduler, vae=vae, text_encoder=text_encoder, processor=processor,
    transformer=transformer,
)

payload = {
    "prompt": PROMPT,
    "width":  WIDTH,
    "height": HEIGHT,
    "num_inference_steps": STEPS,
    "generator": torch.Generator(device="cpu").manual_seed(SEED)
}

if image_edit:
    payload['image'] = input_image

with torch.no_grad():
    image = pipe(**payload).images[0]

image.save(OUT)
print(f"saved to {OUT}")
