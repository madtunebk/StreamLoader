# ENGINE: pinned-cache + VRAM-buffer weight server for PyTorch

`engine/` is a second crate (`streamloader-engine`), built on top of the
`streamloader` library, that serves transformer weights to a real
PyTorch/diffusers forward pass — the counterpart to `streamloader` (which
is deliberately just an indexer/loader, no compute). This document covers
only the engine; see `DESIGN.md`/`VALIDATION.md` for the loader itself.

## Why this exists

The loader alone proves a checkpoint can be indexed and read correctly
without pre-splitting or full materialization. It says nothing about
actually running inference. Running FLUX.2-klein-9B (an 18.16GB
transformer + a ~15GB Qwen3 text encoder) on a 12GB GPU requires some
weight-streaming strategy no matter what language you write it in — this
engine is one specific strategy (pinned host cache + a small reusable
VRAM buffer ring + real CUDA-event-based prefetch), implemented in Rust
and handed to PyTorch via DLPack, so PyTorch does 100% of the actual
compute and Rust does 100% of the weight residency/transfer bookkeeping.

## Scope boundary (single GPU, one component)

- Only the **transformer** is Rust-managed. The text encoder and VAE keep
  their own, separate, ordinary policy: real weights loaded normally,
  then `accelerate.cpu_offload()` applied individually to each — each
  runs exactly once per generation (prompt encoding, final VAE decode),
  so offloading them costs nothing meaningful, unlike the transformer
  which runs once per denoising step (50 times in the benchmark below).
- Single GPU only (`device_ordinal`, fixed one context, one transfer
  stream). No dual-GPU orchestration, no ring attention — deliberately
  out of scope until this single-GPU path was correct and measured (it
  now is; see below).
- No RAM-budget policy, LRU eviction, or cross-process sharing beyond
  what's described here. The pinned cache holds the *whole* transformer
  or refuses to start (`EngineError::BudgetExceeded`) — no partial-cache
  silent degradation.

## Architecture

```
streamloader::Model (mmap, CPU-only, from the earlier deliverable)
        │  Model::blocks()/copy_block_into() -- called ONCE at init
        ▼
RustEngine::new()
        │  one cudaHostAlloc'd pinned buffer, sized to the whole
        │  transformer, filled directly from the mmap (no intermediate
        │  Vec, no second CPU-side cache of any kind)
        ▼
 ┌──────────────────────────────────────────────────────────┐
 │  pinned host buffer (18.16GB for this model)              │
 └──────────────────────────────────────────────────────────┘
        │  async H2D memcpy on a dedicated transfer stream
        ▼
 ┌─────────────┐   ┌─────────────┐   ┌───────────────────────┐
 │ shared slot │   │ VRAM slot 0 │   │ VRAM slot 1            │
 │ (permanent, │   │ (reused     │   │ (reused across every   │
 │  ~0.7GB)    │   │  across     │   │  non-shared block)     │
 │             │   │  blocks)    │   │                        │
 └─────────────┘   └─────────────┘   └───────────────────────┘
        │                  │                    │
        └──── DLPack (zero payload copy) ───────┘
                           ▼
              real torch.Tensor, on the SAME
              CUDA device, no extra copy
```

Per-block sequence (`get_block`/`prefetch`/`mark_block_done`, called from
PyTorch forward hooks — see `inference/generate_rust.py`):

1. `prefetch(next_block_id)` — issue an async H2D copy for the *next*
   block into whichever VRAM slot is next in the ring, on the transfer
   stream. If that slot's previous occupant hasn't been marked done yet,
   the transfer stream first waits (via a real CUDA event, not a CPU
   sync) for that.
2. `get_block(this_block_id, compute_stream_ptr)` — usually a cache hit
   (the prior `prefetch` call already issued the transfer); makes
   PyTorch's actual current stream wait on the transfer's completion
   event, builds DLPack-backed `torch.Tensor`s for every tensor in the
   block, and returns them as a name→tensor dict.
3. The caller `load_state_dict(tensors, strict=False, assign=True)`s
   that dict into the (meta-device) transformer, right before that
   block's submodule runs.
4. After the block's forward returns, `mark_block_done(this_block_id,
   compute_stream_ptr)` records a CUDA event on PyTorch's *compute*
   stream — this is what the *next* transfer into this slot waits on
   before overwriting it, so a buffer is never reused while a kernel
   might still be reading it.

### The interop pieces, and how each was verified before being relied on

- **cudarc** (`0.19.9`) for the CUDA driver API: context, streams,
  events, pinned host allocation, async memcpy. Verified it actually
  initializes this machine's RTX 3060 and does a real pinned-memory
  H2D/D2H roundtrip before writing anything else (`/tmp/cudarc_check`,
  not part of this deliverable, ad hoc smoke test).
- **`CudaContext::new()` uses `cuDevicePrimaryCtxRetain`** (confirmed by
  reading cudarc's source, `src/driver/safe/core.rs`), i.e. the *same*
  primary CUDA context the CUDA runtime API (and therefore PyTorch) uses
  for that device — this is what makes it valid to record/wait on CUDA
  events across a cudarc stream and PyTorch's own stream in the same
  process. This was the single biggest correctness risk in the whole
  design and was checked against cudarc's actual source before relying
  on it, not assumed.
- **Cross-stream event handoff without any `torch.cuda.synchronize()`**:
  proven with a standalone test (`inference/test_dlpack.py`) *before*
  building the real engine — an async H2D copy on our own stream, a CUDA
  event recorded on it, that event awaited by PyTorch's raw stream
  pointer via the raw driver API, then a `torch.Tensor` read back with no
  synchronous wait anywhere on the Python side. Run across 6 different
  fill values; every one read back correct. If the cross-stream wait were
  wrong, this is exactly the kind of test that can pass by luck under one
  timing and silently return stale data under another — it was designed
  to have no synchronization *except* the mechanism under test.
- **DLPack**: hand-implemented against the DLPack C ABI
  (`engine/src/dlpack.rs`) using the raw `pyo3_ffi::PyCapsule_*`
  functions specifically because the ownership-handoff convention (a
  capsule named `"dltensor"`, renamed to `"used_dltensor"` by a
  successful `from_dlpack` import, with the *capsule's own* destructor
  needing to check which state it's in before deciding whether to free
  anything) is not something a higher-level capsule wrapper exposes, and
  getting it wrong is a double-free or a leak, not a lint warning.
- **Pinned buffer lifetime**: an earlier version of `RustEngine::new()`
  captured the pinned buffer's raw pointer but let the owning
  `PinnedHostSlice` (a local variable) drop at the end of the function —
  freeing the allocation the instant `new()` returned. This was caught,
  not avoided by luck: `test_engine_real_model.py` segfaulted on the
  first real block fetch after init (the shared-block transfer, which
  happens *during* `new()`, worked fine — only later accesses through
  the now-dangling pointer crashed). Fixed by storing the
  `PinnedHostSlice` in the struct for the engine's whole lifetime. See
  the `_pinned` field's doc comment in `engine/src/engine.rs`.

## Loader fixes made specifically to support this (see VALIDATION.md for detail)

- `Model::copy_block_into(&Block, dest: &mut [u8])` — the primitive the
  pinned-cache fill loop uses; added specifically so filling the pinned
  buffer never needs an intermediate `Vec`.
- `Block` provenance checking — irrelevant to normal engine operation
  (the engine only ever uses `Block`s from its own single `Model`), but
  closes a real correctness gap that would have been easy to hit while
  iterating on this engine (e.g. accidentally reusing a `Block` handle
  across two `Model::open` calls during development).

## Build and run

```bash
cd engine
cargo build --release                    # sanity check, no Python needed for this step

# from a Python venv with torch + diffusers + transformers + accelerate + maturin installed:
VIRTUAL_ENV=/path/to/venv maturin develop --release
```

`inference/pyproject.toml` describes the venv's contents for `uv run`,
but `streamloader-engine` is a locally maturin-built extension (not a
PyPI package `uv sync` knows how to reproduce — `engine/` has no
`pyproject.toml`, only `Cargo.toml`), so always pass `--no-sync` (or set
`UV_NO_SYNC=1`) to make `uv run` use the venv as-is instead of trying to
reconcile it against the dependency list:

```bash
cd inference

# standalone cross-stream correctness proof (no real model needed):
uv run --no-sync python test_dlpack.py

# real-model correctness proof (byte-exact vs. an independent parse, plus
# forced eviction/reuse correctness):
uv run --no-sync python test_engine_real_model.py

# full pipeline, transformer served by the Rust engine:
uv run --no-sync python generate_rust.py --prompt "..." --seed 0 --steps 20 \
    --out output.png
```

(Equivalently, without `uv`: `.venv/bin/python generate_rust.py ...`.)

`generate_rust.py` hardcodes this session's local model path; point
`TRANSFORMER_DIR`/`HUB` at your own checkpoint to reuse it elsewhere.
`--steps` defaults to 20 (this is a step-distilled model per its own
`config.json`: `is_distilled: true`, `guidance_embeds: false` — the
pipeline's own unmodified default of 50 is more than the architecture
needs).

If the engine can't start (bad path, budget too small, no CUDA device),
`se.Engine(...)` raises `RuntimeError` and the script crashes — there is
no fallback to any other loading path anywhere in `generate_rust.py`.

## Honest performance results

See `ENGINE_VALIDATION.md` for the full measured numbers, methodology,
and — per the explicit requirement this was built against — an honest
accounting of *why* the result is what it is, not just what it is.
