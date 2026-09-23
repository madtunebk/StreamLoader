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

## Done (continued)

- [x] **Stage 3 — static, init-time-only ResidentPool** (2026-09-23)
  - New `resident_budget_bytes` param on `RustEngine::new()`/`Engine()`
    (default `0` — fully backward compatible, verified against
    `test_engine_real_model.py` with no behavior change at 0).
  - Selection: first-fit-decreasing knapsack by block size (`engine.rs`,
    in `new()`, before ring allocation) — exact for this model's two
    distinct block sizes. Selected blocks get their own permanent CUDA
    allocation, filled synchronously once at init (same pattern as
    `shared_slot`), never a repurposed ring slot.
  - `issue_transfer` checks the resident map *before* the ring scan;
    `mark_block_done` no-ops for resident blocks (never evicted, nothing
    to protect). New `Stats::resident_bytes`/`resident_hits`, kept
    separate from `cache_hits` per §9's semantics warning.
  - Bonus: ring's `max_block_bytes` now computed only over the
    *non*-resident subset, so promoting a large block out of the ring
    also shrinks the ring's own footprint (directly addresses the §3
    padding-waste finding, not just adds a new allocation on top of it).
  - New `resident_block_ids()` accessor; `generate_rust.py` gained
    `--resident-gb` (default 0.0, same backward-compat contract).
  - Verified: byte-exact vs. independent parse across 3 repeated fetch
    cycles on a resident block, `transfer_count` stays 0 for it throughout
    (only `resident_hits` increments), non-resident/streaming blocks
    behave identically to pre-stage-3. Real generation run at 1GB budget
    in progress for the §10 benchmark table.

- [x] **Stage 4 — correctness test for stage 3** (2026-09-23)
  - New `inference/test_resident_pool.py`, same rigor as
    `test_engine_real_model.py`: byte-exact vs. independent parse for
    EVERY resident block (not just one), at two budgets (1GB — homogeneous
    all-double-stream set; 4GB — mixed double+single set), plus 2 full
    replayed diffusion-step cycles over all 32 blocks confirming resident
    blocks are NEVER re-transferred while non-resident blocks are
    re-transferred exactly once per cycle (unchanged from pre-stage-3).
  - **Found and fixed a real, pre-existing bug in the process** (not
    ResidentPool-specific): `Drop for RustEngine` freed device memory
    without first synchronizing `transfer_stream`. `alloc_device` zeroes
    every new buffer via an async memset that `new()` never waited on;
    creating a SECOND `RustEngine` in the same process after dropping the
    first (stage 4's test is the first thing in this project to ever do
    that — every prior test/script created exactly one engine per OS
    process) hit `CUDA_ERROR_ILLEGAL_ADDRESS` reproducibly, with zero
    block fetches involved. Confirmed via a minimal repro with
    `resident_budget=0` (i.e. ResidentPool fully disabled) that this bug
    predates stage 3 entirely. Fixed with one line
    (`self.transfer_stream.synchronize()` at the top of `drop()`).
    Verified: 3 engines created sequentially with no crash, full
    `test_engine_real_model.py` still byte-identical after the fix, and
    the new stage-4 test itself now passes cleanly end to end.

## Not done
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

## Resolved (were "open questions before stage 3")

- First test budget: 1GB, as planned. Actual result: the knapsack selects
  exactly ONE double-stream block (872.4MB) — 1GB isn't enough headroom
  left over (201MB) to also fit a 436.2MB single-stream block. Correct
  behavior, not a bug; worth re-testing at 1.5GB where a second block
  should fit.
- Knapsack-vs-naive-baseline comparison (§10 methodology note): not done
  yet — needs a second selection strategy implemented to compare against,
  or can be reasoned about analytically since this model only has two
  distinct block sizes (see the doc's §8 update once the benchmark table
  is filled in).

## Benchmark results so far (§10 sweep, 16 steps, same prompt/seed as `hardware.md`)

| Resident | s/it (steady) | generation | H2D | transfers | resident hits |
| --- | --- | --- | --- | --- | --- |
| 0 GB (baseline) | ~3.75-3.77s | 63.78s | 279.173 GB | 512 | 0 |
| 1 GB (1 double-stream block) | ~3.4-3.45s | 58.34s | 265.215 GB | 496 | 17 |
| 2 GB (2 double-stream blocks) | ~3.4s | 57.85s | 251.256 GB | 480 | 49 |
| 4 GB (4 double + 1 single, `transformer_blocks.0-3` + `single_transformer_blocks.0`) | ~3.45-3.47s | 58.52s (worse than 2GB) | 216.359 GB | 432 | 145 |
| 6 GB (7 double-stream blocks, `transformer_blocks.0-6`) | ~3.4s | 57.84s (back to the 2GB level, NOT worse) | 181.463 GB | 400 | 209 |

**Real total VRAM at 6GB (user's live nvtop reading, not just the engine's own `vram_bytes` stat)**: avg **10.45-11.20GB / 12GB** — only ~0.8-1.55GB headroom left. The engine's own `vram_bytes=8.561GB` undercounts real pressure by ~2.4-2.6GB (PyTorch's own workspace/VAE/text-encoder overhead on top), consistent with the ~2.4-2.5GB gap already visible at 0GB residency in `hardware.md` (2.45GB engine vs. ~4.9GB total observed). Confirms §11's cross-phase VRAM risk is real and close at this budget, even though this particular run didn't OOM.

| 8 GB (8 double + 3 single, all 8 `transformer_blocks.*` + 3 `single_transformer_blocks.*`) | ~3.4s | 58.56s | **146.566 GB** | 336 | 337 |

**8GB is where it actually breaks — just not with a hard crash.** User's live nvtop reading during the run: **11.98/12GB (99.8%)**, essentially saturated. The run's own stderr shows PyTorch's `CUDACachingAllocator` hitting real OOM **dozens of times** during generation (e.g. `memory allocation failed with OOM on device 0 while trying to allocate 538968064 bytes (free: 383778816, total: 12481003520)` — 366MB free, trying to allocate 514MB — repeated with different sizes: 514MB, 770MB, 1.03GB, over and over). The process did NOT crash (exit 0, image saved) because PyTorch's allocator automatically empties its own cache and retries on OOM before actually raising — but this is real allocator thrashing happening silently, exactly the failure mode named in §10/§11, just manifesting as repeated internal retries rather than a visible crash. This is arguably a MORE important finding than a clean crash would have been: it shows a third outcome category between "works cleanly" and "crashes outright" — "works, but is quietly fighting the allocator the whole time," which could tip into a real crash under a slightly different prompt/resolution/batch size. Engine `vram_bytes=9.869GB` (8.288GB resident + shrunk ring, now sized to the remaining single-stream max since all doubles are resident) undercounts the real pressure just as badly as at 6GB.

1GB result matches prediction almost exactly: H2D dropped by exactly one
double-stream block's bytes × 16 steps (872,416,256 × 16 = 13.959GB;
279.173 − 13.959 = 265.214GB, actual 265.215GB); transfer_count dropped by
exactly 16 (512→496, one fewer per step). Generation time dropped ~8.5%,
not just H2D bytes — real s/it improvement, satisfying the review's
primary objective, not just the secondary "minimize H2D" one.

2GB result: knapsack picked `transformer_blocks.0` and `.1` (2×872.4MB =
1.745GB; a third double-stream block or any single-stream block no longer
fits in the ~402MB left over). H2D/transfer_count again match prediction
exactly (2 blocks × 16 steps worth of bytes/transfers removed). But
generation time barely moved (58.34s → 57.85s, only −0.49s) despite H2D
dropping another 14GB — **diminishing returns already visible at just 2
resident blocks**, exactly the "don't assume larger is better" warning in
§10. Working hypothesis (not yet confirmed): the first resident block
removed the binding bottleneck in the prefetch/compute overlap; the
second block's saved transfer was already mostly hidden by overlap even
before it was resident, so avoiding it saves bytes but not much wall
time. Needs 3GB/4GB points to see whether this flattens further or
reverses.

`resident_hits=17`/`49` (not 32/64) is correct, not a bug: `generate_rust.py`'s
prefetch-ahead only fires when `pos_of[block]+1 < len(block_ids)`
(`inference/generate_rust.py`'s `make_pre_hook`) — the last block
(`single_transformer_blocks.23`) never prefetches block 0 for the next
step's wraparound. So block 0 gets one `prefetch()` ever (the initial one
in `attach_engine`) plus one `get_block()` per step = 1 + 16 = 17; block 1
gets one `prefetch()` per step (from block 0's own pre-hook, every step)
plus one `get_block()` per step = 16 + 16 = 32; 17 + 32 = 49, matching the
2GB run exactly.

## Key finding: non-monotonic s/it, likely single-run noise — NOT confirmed as real

Generation time: 63.78s (0GB) → 58.34s (1GB) → 57.85s (2GB) → 58.52s
(4GB) → 57.84s (6GB). 6GB — 7 resident double-stream blocks, 8.561GB
engine VRAM, right up against the 12GB ceiling — did NOT OOM (exit 0,
full pipeline including VAE decode completed) and came back down to
match 2GB's result almost exactly, rather than continuing the apparent
4GB regression.

**Honest caveat, stated plainly per `prompt.md`'s own rigor requirement**:
every budget point above is a SINGLE run, no repeated trials. The
0.5-0.7s spread between the 2GB/4GB/6GB results is small enough that it
could be ordinary run-to-run noise (GPU clock/thermal variance) rather
than a real effect tied to which specific blocks got selected resident.
Do not treat the "4GB regression" as confirmed until at least one of
these points is re-run to check its own variance. This is exactly the
kind of claim `prompt.md` says not to make until benchmarks demonstrate
it repeatably, not just once.

## Open questions before stage 5 (not before stage 3 anymore)

- Re-run 2GB and/or 4GB at least once more each to establish whether the
  spread is noise or real before drawing any conclusion about an
  inflection point (4GB's dip may actually be an early, milder version of
  8GB's allocator-thrashing pattern — worth checking with the same
  attention to stderr OOM warnings this time, not just the timing).
- The true hard-crash edge (a real `RuntimeError`, not just retried
  warnings) is still not found — 8GB thrashes but survives; would need
  higher still (or a less generous PyTorch allocator retry budget) to
  force an actual crash. Given 8GB already shows real degradation, this
  may not be worth chasing further for its own sake — the practical
  answer for stage 5 is "stay well below 8GB on this card."
- `block_timings()` (stage 1) could give per-block GPU wait time to help
  explain the 4GB dip mechanistically, and to directly measure retry-
  induced stalls at 8GB — not yet done.
- Knapsack-vs-naive-baseline comparison: still not done.
