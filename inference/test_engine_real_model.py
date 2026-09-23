"""
Real correctness test: pull a real block from FLUX.2-klein-9B through the
whole engine (mmap -> pinned host cache -> VRAM slot -> DLPack -> real
torch.Tensor), copy it back to CPU, and compare it BYTE FOR BYTE against
an independent, from-scratch Python parse of the same SafeTensors shard
(same technique as the Phase-A interact.py cross-check, not reusing any
Rust code). This is the strongest test available: not "looks right", not
"hash matches", but the exact bytes are identical.
"""

import json
import struct
import sys
import time

import torch
import streamloader_engine as se

HUB = "/home/nobus/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9B"
TRANSFORMER_DIR = f"{HUB}/snapshots/92196c8e11f7b6cf2b7493e037d8c5345c559216/transformer"
BUDGET = 20 * 1024 * 1024 * 1024  # 20GB pinned budget (transformer is ~18.16GB)


def independent_tensor_bytes(shard_path, name):
    with open(shard_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        start, end = header[name]["data_offsets"]
        f.seek(8 + header_len + start)
        return f.read(end - start)


def main():
    torch.cuda.init()
    stream_ptr = torch.cuda.current_stream().cuda_stream

    print(f"initializing engine from {TRANSFORMER_DIR} (budget={BUDGET/1e9:.1f}GB)...")
    t0 = time.time()
    engine = se.Engine(TRANSFORMER_DIR, BUDGET, 2, 0, HUB)
    print(f"engine init: {time.time()-t0:.2f}s")
    print(f"stats after init: {engine.stats()}")

    block_ids = engine.block_ids()
    print(f"{len(block_ids)} blocks: {block_ids[:3]} ... {block_ids[-3:]}")
    assert "transformer_blocks.7" in block_ids

    print("\n=== fetching shared block ===")
    shared = engine.get_shared()
    print(f"{len(shared)} shared tensors: {list(shared.keys())[:3]}...")

    print("\n=== fetching transformer_blocks.7 ===")
    engine.prefetch("transformer_blocks.7")
    tensors = engine.get_block("transformer_blocks.7", stream_ptr)
    print(f"{len(tensors)} tensors in block")
    print("stats after one block fetch:", engine.stats())

    # Cross-check every tensor byte-for-byte against an independent parse
    # of the real shard file.
    inspect_shards = [
        f"{TRANSFORMER_DIR}/diffusion_pytorch_model-00001-of-00002.safetensors",
        f"{TRANSFORMER_DIR}/diffusion_pytorch_model-00002-of-00002.safetensors",
    ]

    mismatches = []
    for name, t in tensors.items():
        assert t.is_cuda, f"{name} is not a CUDA tensor"
        assert t.dtype == torch.bfloat16, f"{name} dtype={t.dtype}"

        gpu_bytes = t.contiguous().cpu().view(torch.int16).numpy().tobytes()

        found = None
        for shard in inspect_shards:
            try:
                found = independent_tensor_bytes(shard, name)
                break
            except KeyError:
                continue
        if found is None:
            mismatches.append((name, "not found in any shard by independent parser"))
            continue

        ok = found == gpu_bytes
        status = "OK" if ok else "MISMATCH"
        print(f"  {status} {name}: {len(gpu_bytes)} bytes")
        if not ok:
            mismatches.append((name, f"byte mismatch: {len(found)} vs {len(gpu_bytes)} bytes"))

    # Capture reference bytes BEFORE anything might evict this block's slot.
    reference_bytes = {name: t.contiguous().cpu().view(torch.int16).numpy().tobytes() for name, t in tensors.items()}
    engine.mark_block_done("transformer_blocks.7", stream_ptr)

    print("\n=== forcing eviction: with only 2 VRAM slots, fetch two more distinct blocks ===")
    # slot layout so far: slot0=transformer_blocks.7, slot1=<empty>
    engine.prefetch("single_transformer_blocks.0")
    t_single = engine.get_block("single_transformer_blocks.0", stream_ptr)  # -> slot1
    engine.mark_block_done("single_transformer_blocks.0", stream_ptr)
    print("stats after single_transformer_blocks.0 (slot1, fresh transfer expected):", engine.stats())

    engine.prefetch("transformer_blocks.6")
    t_six = engine.get_block("transformer_blocks.6", stream_ptr)  # -> slot0, EVICTS transformer_blocks.7
    engine.mark_block_done("transformer_blocks.6", stream_ptr)
    print("stats after transformer_blocks.6 (slot0, evicts block 7):", engine.stats())

    print("\n=== re-fetching transformer_blocks.7 (must be a genuine cache miss + fresh transfer now) ===")
    stats_before = engine.stats()
    tensors7b = engine.get_block("transformer_blocks.7", stream_ptr)
    stats_after = engine.stats()
    assert stats_after["transfer_count"] == stats_before["transfer_count"] + 1, (
        "expected a real transfer (block 7's slot was evicted), but transfer_count didn't increase -- "
        "this would mean the eviction test above wasn't actually testing eviction"
    )
    for name, expected_bytes in reference_bytes.items():
        reread_bytes = tensors7b[name].contiguous().cpu().view(torch.int16).numpy().tobytes()
        if expected_bytes != reread_bytes:
            mismatches.append((name, "re-fetch after genuine slot eviction returned different bytes than the first fetch"))
    print(f"stats after re-fetch: {stats_after}")

    if mismatches:
        print(f"\n{len(mismatches)} MISMATCH(ES):")
        for name, reason in mismatches:
            print(f"  {name}: {reason}")
        sys.exit(1)

    print("\nAll tensors byte-identical to an independent parse of the real checkpoint.")
    print("Buffer reuse (same VRAM slot, different block) did not corrupt data.")


if __name__ == "__main__":
    main()
