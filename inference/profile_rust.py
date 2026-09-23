#!/usr/bin/env python3
"""
Quick diagnostic: split per-block hook wall-clock time into its parts
(engine.get_block call, load_state_dict, engine.prefetch call,
mark_block_done) to see whether the Rust-engine path is bound by PCIe
transfer or by Python/PyTorch-side bookkeeping overhead, before deciding
what to optimize. Runs only a handful of steps -- the point is the
per-step ratio, not a full image.
"""

import json
import time

import torch
from accelerate import cpu_offload, init_empty_weights
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, Qwen3ForCausalLM

import streamloader_engine as se

HUB = "/home/nobus/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9B"
SNAPSHOT = f"{HUB}/snapshots/92196c8e11f7b6cf2b7493e037d8c5345c559216"
TRANSFORMER_DIR = f"{SNAPSHOT}/transformer"
PINNED_BUDGET = 20 * 1024 * 1024 * 1024
VRAM_SLOTS = 2
PREFETCH_AHEAD = 1
STEPS = 4

timings = {"get_block": 0.0, "load_state_dict": 0.0, "prefetch": 0.0, "mark_done": 0.0, "shared_load": 0.0}
counts = {"get_block": 0}


def build_meta_transformer():
    cfg = json.load(open(f"{TRANSFORMER_DIR}/config.json"))
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    with init_empty_weights():
        model = Flux2Transformer2DModel(**cfg)
    model.eval()
    return model


def attach_engine(transformer, engine, stream_ptr):
    block_ids = engine.block_ids()
    pos_of = {bid: i for i, bid in enumerate(block_ids)}

    def block_module(block_id):
        family, idx = block_id.rsplit(".", 1)
        return getattr(transformer, family)[int(idx)]

    def make_pre_hook(block_id):
        def hook(module, args, kwargs):
            t0 = time.perf_counter()
            tensors = engine.get_block(block_id, stream_ptr)
            t1 = time.perf_counter()
            transformer.load_state_dict(tensors, strict=False, assign=True)
            t2 = time.perf_counter()
            pos = pos_of[block_id] + PREFETCH_AHEAD
            if pos < len(block_ids):
                engine.prefetch(block_ids[pos])
            t3 = time.perf_counter()
            timings["get_block"] += t1 - t0
            timings["load_state_dict"] += t2 - t1
            timings["prefetch"] += t3 - t2
            counts["get_block"] += 1
            return args, kwargs

        return hook

    def make_post_hook(block_id):
        def hook(module, args, output):
            t0 = time.perf_counter()
            engine.mark_block_done(block_id, stream_ptr)
            timings["mark_done"] += time.perf_counter() - t0
            return output

        return hook

    handles = []
    for bid in block_ids:
        mod = block_module(bid)
        handles.append(mod.register_forward_pre_hook(make_pre_hook(bid), with_kwargs=True))
        handles.append(mod.register_forward_hook(make_post_hook(bid)))

    t0 = time.perf_counter()
    shared = engine.get_shared()
    transformer.load_state_dict(shared, strict=False, assign=True)
    timings["shared_load"] = time.perf_counter() - t0

    engine.prefetch(block_ids[0])
    return handles


def main():
    torch.cuda.init()
    device = torch.device("cuda:0")
    stream_ptr = torch.cuda.current_stream().cuda_stream

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SNAPSHOT, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, subfolder="tokenizer")
    text_encoder = Qwen3ForCausalLM.from_pretrained(SNAPSHOT, subfolder="text_encoder", dtype=torch.bfloat16)
    cpu_offload(text_encoder, execution_device=device)
    vae = AutoencoderKLFlux2.from_pretrained(SNAPSHOT, subfolder="vae", dtype=torch.bfloat16)
    cpu_offload(vae, execution_device=device)
    transformer = build_meta_transformer()
    pipe = Flux2KleinPipeline(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer, is_distilled=True
    )

    engine = se.Engine(TRANSFORMER_DIR, PINNED_BUDGET, VRAM_SLOTS, 0, HUB)
    attach_engine(transformer, engine, stream_ptr)

    generator = torch.Generator(device="cpu").manual_seed(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        pipe(
            prompt="a bottle on a table with the milky way galaxy swirling inside it",
            width=1024,
            height=1024,
            num_inference_steps=STEPS,
            generator=generator,
        )
    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    hook_total = sum(timings.values())
    print(f"\n=== profile over {STEPS} steps ({counts['get_block']} block calls) ===")
    print(f"total wall time (incl. VAE decode, cuda-synced):  {total:.3f}s")
    print(f"shared_load (one-time):                           {timings['shared_load']*1000:.1f}ms")
    print(f"sum(get_block call overhead):                      {timings['get_block']:.3f}s  ({timings['get_block']/total*100:.1f}% of total)")
    print(f"sum(load_state_dict per block):                    {timings['load_state_dict']:.3f}s  ({timings['load_state_dict']/total*100:.1f}% of total)")
    print(f"sum(prefetch call overhead):                       {timings['prefetch']:.3f}s  ({timings['prefetch']/total*100:.1f}% of total)")
    print(f"sum(mark_done call overhead):                      {timings['mark_done']:.3f}s  ({timings['mark_done']/total*100:.1f}% of total)")
    print(f"sum(all hook overhead, CPU-side Python/PyO3):      {hook_total:.3f}s  ({hook_total/total*100:.1f}% of total)")
    print(f"remainder (actual GPU compute + unaccounted):      {total-hook_total:.3f}s  ({(total-hook_total)/total*100:.1f}% of total)")
    print(f"avg per get_block call:                            {timings['get_block']/counts['get_block']*1000:.3f}ms")
    print(f"avg per load_state_dict call:                      {timings['load_state_dict']/counts['get_block']*1000:.3f}ms")


if __name__ == "__main__":
    main()
