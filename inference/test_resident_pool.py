"""
Stage 4 correctness test for ResidentPool (see RESIDENT_POOL_TODO.md).

Same rigor as test_engine_real_model.py -- byte-exact against an
independent, from-scratch parse of the real checkpoint, not "looks right".
Two things that test doesn't cover, which this one does:

1. Every tensor of EVERY resident block, at more than one budget (so both
   an all-double-stream resident set and a mixed double+single set get
   checked), not just one block at one budget.
2. A full multi-cycle replay of the real per-block hook sequence
   (prefetch -> get_block -> mark_block_done, in block_order, wrapping
   like generate_rust.py's hooks do) across 2 full "diffusion steps",
   asserting resident blocks are NEVER re-transferred while non-resident
   blocks behave exactly as they did before ResidentPool existed
   (re-transferred once per cycle, same as the pre-stage-3 engine).
"""

import json
import struct
import sys

import torch
import streamloader_engine as se

HUB = "/home/nobus/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9B"
TRANSFORMER_DIR = f"{HUB}/snapshots/92196c8e11f7b6cf2b7493e037d8c5345c559216/transformer"
BUDGET = 20 * 1024 * 1024 * 1024
GB = 1024**3

SHARDS = [
    f"{TRANSFORMER_DIR}/diffusion_pytorch_model-00001-of-00002.safetensors",
    f"{TRANSFORMER_DIR}/diffusion_pytorch_model-00002-of-00002.safetensors",
]


def independent_tensor_bytes(shard_path, name):
    with open(shard_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        start, end = header[name]["data_offsets"]
        f.seek(8 + header_len + start)
        return f.read(end - start)


def find_independent_bytes(name):
    for shard in SHARDS:
        try:
            return independent_tensor_bytes(shard, name)
        except KeyError:
            continue
    return None


def check_block_byte_exact(engine, block_id, stream_ptr, mismatches):
    engine.prefetch(block_id)
    tensors = engine.get_block(block_id, stream_ptr)
    for name, t in tensors.items():
        gpu_bytes = t.contiguous().cpu().view(torch.int16).numpy().tobytes()
        expected = find_independent_bytes(name)
        if expected is None:
            mismatches.append((block_id, name, "not found in any shard by independent parser"))
            continue
        if expected != gpu_bytes:
            mismatches.append((block_id, name, f"byte mismatch: {len(expected)} vs {len(gpu_bytes)} bytes"))
    engine.mark_block_done(block_id, stream_ptr)
    return len(tensors)


def run_budget_point(resident_gb, stream_ptr, mismatches):
    resident_budget = int(resident_gb * GB)
    print(f"\n{'='*70}\nbudget = {resident_gb}GB\n{'='*70}")
    engine = se.Engine(TRANSFORMER_DIR, BUDGET, 2, 0, HUB, None, resident_budget)
    block_ids = engine.block_ids()
    resident_ids = set(engine.resident_block_ids())
    non_resident_ids = [b for b in block_ids if b not in resident_ids]
    print(f"resident: {sorted(resident_ids)}")
    print(f"non-resident (streaming): {len(non_resident_ids)} blocks")

    # 1. Byte-exact check for EVERY resident block, not just one.
    total_tensors = 0
    for bid in sorted(resident_ids):
        n = check_block_byte_exact(engine, bid, stream_ptr, mismatches)
        total_tensors += n
    print(f"checked {total_tensors} tensors across {len(resident_ids)} resident block(s)")

    stats_after_resident_checks = engine.stats()
    assert stats_after_resident_checks["transfer_count"] == 0, (
        f"resident-block fetches must never cause a real H2D transfer, "
        f"got transfer_count={stats_after_resident_checks['transfer_count']}"
    )
    print(f"PASS: zero transfers after touching every resident block (stats: {stats_after_resident_checks})")

    # 2. Full multi-cycle replay of the real hook sequence, 2 complete
    # cycles over ALL blocks (resident + streaming), matching
    # generate_rust.py's attach_engine pattern: prefetch one ahead,
    # get_block this one, mark_block_done after.
    pos_of = {bid: i for i, bid in enumerate(block_ids)}
    n_blocks = len(block_ids)
    engine.prefetch(block_ids[0])
    for cycle in range(2):
        for bid in block_ids:
            tensors = engine.get_block(bid, stream_ptr)
            pos = pos_of[bid] + 1
            if pos < n_blocks:
                engine.prefetch(block_ids[pos])
            engine.mark_block_done(bid, stream_ptr)
        stats = engine.stats()
        expected_transfers = (cycle + 1) * len(non_resident_ids)
        assert stats["transfer_count"] == expected_transfers, (
            f"after cycle {cycle+1}: expected transfer_count={expected_transfers} "
            f"(one per non-resident block per cycle so far), got {stats['transfer_count']}"
        )
        print(f"cycle {cycle+1}: transfer_count={stats['transfer_count']} (expected {expected_transfers}) -- OK")

    print(f"PASS: resident blocks never re-transferred across 2 full cycles; "
          f"non-resident blocks re-transferred exactly once per cycle, same as pre-ResidentPool behavior")


def main():
    torch.cuda.init()
    stream_ptr = torch.cuda.current_stream().cuda_stream
    mismatches = []

    # 1GB: homogeneous resident set (all double-stream blocks that fit).
    # 4GB: mixed resident set (double + single-stream), the shape that
    # showed the (still-unconfirmed-as-real) s/it dip in the sweep --
    # correctness must hold regardless of whether that dip is real.
    for resident_gb in (1.0, 4.0):
        run_budget_point(resident_gb, stream_ptr, mismatches)

    if mismatches:
        print(f"\n{len(mismatches)} MISMATCH(ES):")
        for block_id, name, reason in mismatches:
            print(f"  {block_id} / {name}: {reason}")
        sys.exit(1)

    print("\nAll resident-block tensors byte-identical to an independent parse, "
          "at every budget tested. Resident blocks never re-transferred across "
          "repeated cycles. Non-resident/streaming blocks behave exactly as before "
          "ResidentPool existed.")


if __name__ == "__main__":
    main()
