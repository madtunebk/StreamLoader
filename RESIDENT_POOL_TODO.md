# ResidentPool — implementation TODO

Tracks progress against the staged plan in the review doc (§12):
https://claude.ai/artifact/15cYLvrAqdjpPawGdR1xCR#58e61048-dca1

Source brief: `prompt.md`. Baseline numbers: `hardware.md`.

Rule for every stage: build (`cargo build --release` in `engine/`), install
(`maturin develop --release`, see `BUILD.md`), then re-run
`inference/test_engine_real_model.py` against the real model before moving on.
Stages 1-2 are additive/zero-behavior-change by design — if that test's
`transfer_count`/`bytes_h2d`/`cache_hits` shape ever changes at those stages,
stop and treat it as a regression, not progress.

## Done

- [x] **Stage 1 — per-block instrumentation** (2026-09-22)
  - `BlockTiming` struct + `block_timings: Vec<BlockTiming>` in `RustEngine`
    (`engine/src/engine.rs`); timing events recorded in `issue_transfer`,
    read back lazily via `block_timings_ms()` — never synced on the hot path.
  - Exposed as `Engine.block_timings()` in `engine/src/lib.rs`.
  - Verified: `test_engine_real_model.py` still byte-identical;
    `transfer_count`/`bytes_h2d`/`cache_hits` unchanged in shape.
  - Bonus finding confirmed live: double-stream blocks measure 872.4MB,
    single-stream 436.2MB (clean 2:1, matches the review's §3 hypothesis);
    H2D bandwidth ~13.35GB/s, matching `hardware.md`'s ~13.54GiB/s PCIe burst.
  - Full 16-step real generation re-run afterward
    (`inference/generate_rust.py`) reproduced `hardware.md` almost exactly:
    279.173GB H2D, 512 transfers, 497 cache hits — identical to baseline.

- [x] **Stage 2 — static next-use table** (2026-09-22)
  - `RustEngine::next_use_distances(from_block_id)` in `engine.rs`: pure
    query over the already-fixed `block_order`, no VRAM/transfer side
    effects. Exposed as `Engine.next_use_distances()`.
  - Verified: cyclic wrap-around correct (last block → first block =
    distance 1, not 31); farthest-next-use from a mid-cycle block correctly
    lands on the block right behind it. Confirms §8's cyclic-access
    assumption in code, not just from reading `transformer_flux2.py`.
  - `test_engine_real_model.py` still byte-identical.

## Not done

- [ ] **Stage 3 — static, init-time-only ResidentPool**
  - At `RustEngine::new()`, given a VRAM budget, greedily select the block
    subset to keep resident (knapsack framing, §8) using
    `next_use_distances`/block sizes from stage 1-2's instrumentation.
  - Give selected blocks separate, permanent CUDA allocations — never a
    repurposed ring slot (§5 hazard).
  - New branch in `issue_transfer`, checked *before* the ring-slot scan:
    resident block → return its fixed slot, no transfer, no round-robin.
  - This is the first stage that actually changes behavior (fewer H2D
    transfers) — needs the benchmark table in §10 run at a few budget points
    (start small: 0.5-1GB) before calling it a win.

- [ ] **Stage 4 — correctness test for stage 3**
  - Modeled on `test_engine_real_model.py`: at 2-3 budget points, byte-compare
    every resident block against an independent parse; force a full 2-step
    cycle; confirm resident blocks are *never* re-transferred while
    non-resident blocks behave identically to today.

- [ ] **Stage 5 — revisit dynamic resize (only if 3+4 show a real win)**
  - Shrinking ResidentPool for VAE/workspace pressure (§11's cross-phase OOM
    risk), mid-run promotion/eviction (§5-§6 hazards: `mark_block_done`'s
    stale-event risk, D2D promotion cost).
  - Do not start this speculatively — the brief itself allows "H2D falls,
    s/it flat" as an acceptable outcome from stage 3 alone.

## Open questions before stage 3

- What resident budget to test first? (leaning: 1GB — enough for ~1
  double-stream or ~2 single-stream blocks, small enough to keep VRAM
  headroom risk low on the 12GB 3060 per `hardware.md`'s ~7.1GB free)
- Confirm the greedy block-selection set (knapsack, §8) against a naive
  "always keep the N smallest blocks" baseline, per §10's methodology note.
