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

## Evidence the 4GB "regression" was noise, not real (user's own re-run)

User ran 4GB resident again independently (different prompt, seed=55555,
20 steps not 16). Full numbers, verified exactly against our own
formulas:

- Same knapsack selection as before (`transformer_blocks.0-3` +
  `single_transformer_blocks.0`, 3.926GB) — confirms the selection is
  deterministic/reproducible, not prompt-dependent (expected, since it's
  purely a function of block sizes, not model input).
- `transfer_count=540` = 27 non-resident blocks × 20 steps, exact.
- `bytes_h2d=270.449GB` = (17.45 − 3.926)GB/step × 20 steps, exact.
- `resident_hits=181` = 21 (block 0: one initial prefetch + 20
  `get_block` calls) + 40×4 (blocks 1/2/3 and `single.0`: 20 `prefetch` +
  20 `get_block` each) = 181, exact — confirms the "21 + 40×(n-1
  contiguous/chained resident blocks)" hit-counting formula generalizes
  beyond the one case it was first derived from.
- Steady-state **3.30s/it** (tqdm) — *better* than the original 16-step
  sweep's own 4GB measurement (~3.45-3.47s/it) at the exact same resident
  block set, and roughly on par with the sweep's best point (2GB,
  ~3.4s/it). Different prompt/seed/step-count so not strictly
  apples-to-apples, but this is real evidence *against* treating the
  original 4GB dip as a confirmed regression tied to that specific block
  set. Current best read: 2-6GB is one broad "good" zone, and the
  ordering between points within it in any single run is mostly noise.

## Batching (`--batch`, added to generate_rust.py, orthogonal to ResidentPool)

Added `num_images_per_prompt`/`--batch` support since the Rust engine only
serves weight tensors (no batch dimension) — confirmed engine stats
(H2D/transfer_count) are byte-identical between batch=1 and batch=2 at the
same resident budget (251.256GB/480 transfers, 2GB resident, both cases).

Tested batch=2 at 2GB resident, 16 steps: 117.90s total = 58.95s/image,
vs. 57.85s for a single batch=1 run at the same settings — **no speedup
from batching**, just linear scaling (steady-state ~7.28s/it at batch=2
vs. ~3.4s/it at batch=1, ratio ≈2.1x). Confirms the compute-bound
conclusion from earlier: batching only pays off when the GPU has spare
compute capacity to fill; this card is already saturated at batch=1, so
a bigger batch just serializes more work rather than overlapping it.
**Practical takeaway**: for generating many images (10-20), a large
batch is not the right lever on this hardware — no throughput gain, and
real OOM risk from activations/attention scaling with batch size, likely
before even a modest batch size given how tight VRAM already is at
higher resident budgets. Two parallel processes (one per physical GPU,
`device_ordinal=0`/`1` — `ENGINE.md` currently scopes dual-GPU out, but
the parameter already exists) would give genuine 2x throughput instead.
Not implemented/tested yet.

**batch=6 confirmed §11's cross-phase VRAM risk for real** (user's own
run, 2GB resident, 20 steps, seed=55555): the transformer denoising loop
survived (2 OOM warnings early on, then stabilized — unlike 8GB's dozens
of repeated retries), but the run ultimately crashed with an actual OOM
at **VAE decode** — the exact failure mode §11 named on day one
("VAE decode happens *after* the transformer's per-step loop and after
ResidentPool has already claimed its budget... a real cross-phase
hazard"). It took batch=6's much larger per-phase activation footprint
to actually cross from "thrashes but survives" (8GB, batch=1) into "hard
crash", not resident budget alone — same underlying risk, different
lever. No output files were produced (process exited before any
`image.save()` call). This is the first real, reproduced crash in the
whole ResidentPool investigation, and it happened exactly where the
original review predicted it would.

**Root cause identified, not just diagnosed**: `AutoencoderKLFlux2` has
`enable_slicing()` (`vae.py:913`, `autoencoder_kl_flux2.py:146,248`),
which decodes a batch one image at a time (`if self.use_slicing and
z.shape[0] > 1`) instead of the whole batch's latents at once — exactly
the mechanism for reducing VAE decode's peak VRAM with batch>1. It's
`False` by default and `generate_rust.py` never calls it. Note this is
**not** the same as `enable_tiling()` (also unused, also off by default)
— tiling splits large spatial dimensions per image, slicing splits large
batches; batch=6's OOM was a batch-size problem, so slicing is the
relevant fix, not tiling. One-line, pure-diffusers fix, never implemented
yet: `vae.enable_slicing()` after building the VAE in `build_pipeline()`.

**Follow-up successful run, no crash**: 1.5GB resident, batch=2, 20
steps, seed=55555 (two distinct, fully-rendered speech-bubble images,
different styles). Knapsack picked `transformer_blocks.0` +
`single_transformer_blocks.0` (1.309GB, matches predicted knapsack fill
for 1.5GB exactly: one double fits, remaining ~738MB isn't enough for a
second double but is for one single). `transfer_count=600` (30
non-resident × 20 steps), `bytes_h2d=322.794GB` (matches (17.45−1.309)
× 20 predicted). `resident_hits=61` — validates the hit-counting formula
in a NEW shape: this time the two resident blocks are non-adjacent in
`block_order` (not a contiguous chain like the 4GB case), so neither
gets the other's prefetch: `transformer_blocks.0` = 21 (1 initial
prefetch + 20 get_block, same as always), `single_transformer_blocks.0`
= 40 (20 prefetch from its own non-resident predecessor + 20 get_block,
since it's an island) = 61 total. Formula generalizes correctly to
non-contiguous resident sets, not just contiguous chains.

**Fix verified: `vae.enable_slicing()` added to `build_pipeline()` in
`generate_rust.py`, batch=6 re-run with the EXACT crashing settings
(2GB resident, 20 steps, seed=55555) now succeeds — all 6 images saved.**
The same 2 early OOM warnings still appear (those are the transformer's
own activation pressure, unaffected by VAE slicing), but the run no
longer crashes at VAE decode. Cost: 425.30s total = 70.88s/image, worse
per-image than batch=2 (58.95s/image) — steady-state ~20.74s/it, almost
exactly the ~20.4s/it a naive 6x-linear-scaling prediction from the
single-image rate would give. Confirms once more: batch never wins on
this hardware, it only ever costs more per image as it grows, but at
least now it doesn't crash.

**Cleanest confirmation yet of batch-agnosticism**: batch=1 at the exact
same config (1.5GB resident, 20 steps, seed=55555) as the earlier
batch=2 run produced IDENTICAL engine stats down to the last digit
(`bytes_h2d=322.794GB`, `transfer_count=600`, `resident_hits=61` in
both). Generation time 72.63s/image (batch=1) vs. 71.23s/image (batch=2,
same config) — statistically indistinguishable, well within the noise
already established elsewhere in this sweep. Small batches (1-2) are
genuinely neutral on this hardware; only large batches (6+) start
costing measurably more per image.

## `attach_engine` moved into the package (DRY, not a new feature)

Was duplicated identically in `generate_rust.py` and `generate_simple.py`.
Moved into `engine/streamloader_engine/__init__.py` (hand-written, not
maturin's auto-generated stub) as `streamloader_engine.attach_engine(transformer,
engine, compute_stream_ptr, prefetch_ahead=1)` — pure Python, calls only
the already-exposed Rust methods (`block_ids`/`get_block`/`prefetch`/
`mark_block_done`/`get_shared`), no Rust changes needed.

Maturin note: this required converting to its "mixed Rust/Python" layout.
The working recipe (after one false start): put `__init__.py` at
`engine/streamloader_engine/__init__.py` directly (NOT `engine/python/streamloader_engine/`
— that's a different, also-valid maturin convention but not the one this
maturin version auto-detects without extra config). No `Cargo.toml`
changes needed once the file is in the right place; `maturin develop`
then installs it as an editable `.pth` pointing straight at `engine/`,
so further pure-Python edits to `__init__.py` take effect without
rebuilding (only Rust changes still need `maturin develop --release`).

Both call sites updated (`se.attach_engine(...)` instead of a local
`attach_engine` function); `generate_rust.py` lost one diagnostic print
(`shared weights loaded: N tensors...`) as a result, since that detail
lived inside the old local function — acceptable, `stats()` still covers
everything else. Re-verified: `test_engine_real_model.py` and
`test_resident_pool.py` both still pass byte-exact, and both
`generate_rust.py` and `generate_simple.py` still generate correctly
(same resident_hits/transfer_count formulas hold).

## Cross-architecture validation: Qwen-Image-2.1 (not just FLUX.2)

Confirmed empirically, not just theoretically, that the engine
generalizes to a genuinely different DiT architecture, downloaded fresh
today (`Qwen/Qwen-Image-2.1`, 31GB, released 2026-09-19 per its own HF
commit history):

- **Required diffusers 0.41.0.dev0 (git main)** — the model is too new
  for 0.40.0 (latest PyPI release at the time). Installed via
  `uv pip install git+https://github.com/huggingface/diffusers.git`.
  Re-verified `generate_simple.py` (FLUX.2) still works identically after
  the upgrade (same ~3.44-3.46s/it) before touching anything Qwen-related
  — no regression from the diffusers bump.
- **Architecture differences from FLUX.2, verified from source before
  running anything**: single block family (`transformer_blocks`, no
  double/single-stream split), and a KV-cache mode
  (`kv_cache_mode="extract"` on step 0, `"cached"` after, gated by the
  checkpoint's own `causal_condition: true` config) enabled BY DEFAULT
  (`use_kv_cache=True`) for plain text-to-image generation — this is
  exactly the exception case flagged in the original review's §8.
  Read the actual pipeline source (`pipeline_qwenimage21.py`) to confirm:
  `self.transformer(...)` is still called exactly ONCE per denoising
  step regardless of kv_cache_mode; the cache only changes internal
  attention math (skip recomputing K/V for the fixed text+condition
  tokens), never block invocation count or order. Confirms the engine's
  block-order-determinism assumption holds even here.
- **New script**: `inference/generate_qwen.py`, same minimal style as
  `generate_simple.py`, using `se.attach_engine` unchanged (zero Rust
  code touched, zero engine.rs changes) with
  `block_families=["transformer_blocks"]`.
- **One new dependency needed**: `torchvision` (for
  `Qwen3VLProcessor`'s internal video sub-processor, even though we
  never touch video) — installed matching the existing torch/cu130
  build, no torch version change.
- **First real run: SUCCESS.** 20 steps, 1024x1024, resident=2GB:
  **58s total, ~3.0-3.02s/it steady** — actually faster per-step than
  FLUX.2 (~3.4-3.46s/it at similar settings). Output: perfectly legible
  neon sign text ("QWEN IMAGE 2.1"), correct rainy-night scene, matching
  the prompt exactly. No correctness issues, no VRAM issues, first try
  after fixing the two environment gaps above (diffusers version,
  torchvision).

This is the strongest evidence yet for the "compatible with most DiT
transformers" design goal from earlier today: a brand-new architecture,
never seen before this session, worked through the SAME Rust engine
with zero Rust changes, just new Python glue + one config list.

**Step-count finding (user's rule of thumb, confirmed)**: unlike
FLUX.2-klein (step-distilled, ~18-20 steps is enough), Qwen-Image-2.1 is
NOT distilled (`num_inference_steps` pipeline default is 40) and needs
more steps for clean small-text rendering. Same 1280x720 "Chihuahua
Drift" poster prompt/seed at three step counts, 2GB resident:

| Steps | time | s/it steady | side-caption text |
| --- | --- | --- | --- |
| 20 | 54s | ~2.73-2.84s | one side caption garbled/overlapping |
| 30 | 83s | ~2.81-2.85s | side caption readable but rough |
| 48 | 132s | ~2.76-2.82s | **every text element perfectly legible** |

s/it is flat across step counts (as expected, steps don't change
per-step engine load), only total time scales. Practical rule for this
model: **30 steps = functional minimum, 45-50 = full quality**,
matching the user's own observation exactly.

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
