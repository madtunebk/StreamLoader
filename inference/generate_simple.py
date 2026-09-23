#!/usr/bin/env python3
"""
Minimal FLUX.2-klein generation with the Rust streaming engine -- same
mechanism as generate_rust.py, stripped of CLI/argparse/stats printing.
Edit the constants below directly, like a plain diffusers example script.
"""

import torch
from accelerate import cpu_offload, init_empty_weights
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, Qwen3ForCausalLM

import streamloader_engine as se

HUB = "/home/nobus/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9B"
SNAPSHOT = f"{HUB}/snapshots/92196c8e11f7b6cf2b7493e037d8c5345c559216"
TRANSFORMER_DIR = f"{SNAPSHOT}/transformer"

PROMPT = "a fancy speech bubble, cosmic background"
SEED = 0
STEPS = 20
RESIDENT_GB = 2.0  # 0 = pure streaming, no resident blocks
OUT = "inference/output/simple_out.png"

device = torch.device("cuda:0")
torch.cuda.init()
stream_ptr = torch.cuda.current_stream().cuda_stream

# --- ordinary diffusers loading for everything except the transformer ---
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SNAPSHOT, subfolder="scheduler")
tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, subfolder="tokenizer")
text_encoder = Qwen3ForCausalLM.from_pretrained(SNAPSHOT, subfolder="text_encoder", dtype=torch.bfloat16)
cpu_offload(text_encoder, execution_device=device)
vae = AutoencoderKLFlux2.from_pretrained(SNAPSHOT, subfolder="vae", dtype=torch.bfloat16)
vae.enable_slicing()
cpu_offload(vae, execution_device=device)

# --- transformer: meta weights, real weights served by the Rust engine ---
import json
cfg = json.load(open(f"{TRANSFORMER_DIR}/config.json"))
cfg.pop("_class_name", None)
cfg.pop("_diffusers_version", None)
with init_empty_weights():
    transformer = Flux2Transformer2DModel(**cfg)
transformer.eval()

engine = se.Engine(TRANSFORMER_DIR, 20 * 1024**3, 2, 0, HUB, None, int(RESIDENT_GB * 1024**3))

block_ids = engine.block_ids()
pos_of = {bid: i for i, bid in enumerate(block_ids)}


def block_module(block_id):
    family, idx = block_id.rsplit(".", 1)
    return getattr(transformer, family)[int(idx)]


for bid in block_ids:
    mod = block_module(bid)

    def pre_hook(module, args, kwargs, bid=bid):
        transformer.load_state_dict(engine.get_block(bid, stream_ptr), strict=False, assign=True)
        pos = pos_of[bid] + 1
        if pos < len(block_ids):
            engine.prefetch(block_ids[pos])
        return args, kwargs

    def post_hook(module, args, output, bid=bid):
        engine.mark_block_done(bid, stream_ptr)
        return output

    mod.register_forward_pre_hook(pre_hook, with_kwargs=True)
    mod.register_forward_hook(post_hook)

transformer.load_state_dict(engine.get_shared(), strict=False, assign=True)
engine.prefetch(block_ids[0])

# --- ordinary diffusers pipeline call, same as any other model ---
pipe = Flux2KleinPipeline(
    scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
    transformer=transformer, is_distilled=True,
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
