# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Despite the directory name `RustStream`, the project is **StreamLoader**. It
has three layers, each building on the one below:

1. **`streamloader`** (`src/`, `Cargo.toml`) — a CPU-only Rust library + CLI
   that `mmap`-indexes SafeTensors checkpoints and groups tensors into
   logical transformer blocks by name prefix. No model compute, no GPU, no
   Python. A pure loader/indexer.
2. **`streamloader-engine`** (`engine/`, `engine/Cargo.toml`) — a second Rust
   crate (PyO3 + cudarc) that consumes the loader's `Model`/`Block` API and
   serves a DiT transformer's weights to a live PyTorch/diffusers forward
   pass via DLPack, streaming blocks in just ahead of execution. Lets a
   checkpoint larger than VRAM run inference. Built as a CPython extension,
   **not** a PyPI package.
3. **`inference/`** — Python diffusers scripts that drive the engine
   (`generate_flux.py` → FLUX.2-klein-9B, `generate_qwen.py` →
   Qwen-Image-2.1).

The core design point: PyTorch does 100% of the compute; Rust does 100% of
the weight residency/transfer bookkeeping. The engine is model-agnostic
across DiT transformers — supporting a new one is Python glue + a block-family
config string, with no change to `engine/src/*.rs`.

## Rust: build, test, run

```bash
cargo build --release                 # both crates
cargo test                            # loader: 48 tests (lib + integration + CLI)
cargo fmt --check
cargo clippy --all-targets
cargo test --test blocks              # run a single integration test file (tests/blocks.rs)

# Try the CLI against a generated fixture (no real checkpoint needed):
cargo run --release --example make_fixture -- /tmp/fixture_checkpoint
cargo run --release -- blocks /tmp/fixture_checkpoint --preset flux --json
```

Toolchain: rustc/cargo 1.97+, 2024 edition. No network beyond `cargo fetch`.

## Python engine: build + run (read `docs/BUILD.md` fully before touching this)

Two rules that cause real, recurring breakage if ignored:

- **Never run a bare `uv run ...` in this repo. Always `uv run --no-sync`**
  (or `UV_NO_SYNC=1`). A bare `uv run` triggers `uv sync`, which reconciles
  `.venv` strictly against the root `pyproject.toml` — and will *remove*
  `streamloader_engine` (locally built, not a PyPI package) and silently
  downgrade `diffusers` off its deliberate git-main pin (which breaks
  Qwen-Image-2.1 imports). Both have actually happened.
- **`streamloader_engine` is built by `maturin`, never by pip/uv.** Rebuild
  it after *every* change to `engine/src/*.rs`:

  ```bash
  cd engine
  VIRTUAL_ENV=/abs/path/to/repo/.venv \
    /abs/path/to/repo/.venv/bin/maturin develop --release
  ```

  `VIRTUAL_ENV` must be set explicitly — `engine/` intentionally has no
  `pyproject.toml`, so without it maturin walks up to the root `pyproject.toml`
  (which has no `[build-system]`) and fails.

```bash
# Run inference (requires NVIDIA GPU w/ CUDA 13.0, Python 3.10 exactly):
uv run --no-sync inference/generate_flux.py --prompt "..."

# Python correctness tests (byte-exact, against the real downloaded model):
uv run --no-sync inference/tests/test_dlpack.py            # cross-stream DLPack handoff
uv run --no-sync inference/tests/test_engine_real_model.py # the canonical regression test
```

Python **3.10 specifically** — the extension is built for CPython 3.10's ABI
(`cp310`). After editing `pyproject.toml` and running a deliberate `uv sync`,
always re-run the `maturin develop` step before assuming anything imports.

`test_engine_real_model.py` is the regression oracle: `transfer_count` /
`bytes_h2d` / `cache_hits` and byte-identical output are the invariants.
Stages 1–2 of ResidentPool are additive by design — if those numbers change
there, treat it as a regression.

## Engine architecture (the part that needs multiple files to grasp)

`RustEngine::new()` (`engine/src/engine.rs`) does one `cudaHostAlloc` sized to
the *whole* transformer and fills it directly from the loader's mmap (no
intermediate `Vec`, no second CPU cache). It either holds the whole
transformer pinned or refuses to start (`EngineError::BudgetExceeded`) — there
is no partial/LRU pinned cache. VRAM is a small ring of reusable slots (plus
one permanent shared slot). Per denoising block, driven from PyTorch forward
hooks in `inference/generate_flux.py`:

1. `prefetch(next_id)` — async H2D copy of the *next* block into the next ring
   slot, on a dedicated transfer stream (waits on a CUDA event if that slot's
   prior occupant isn't marked done).
2. `get_block(this_id, compute_stream)` — usually a cache hit; makes PyTorch's
   stream wait on the transfer event, returns DLPack-backed `torch.Tensor`s
   (zero payload copy).
3. Caller does `load_state_dict(tensors, strict=False, assign=True)` into the
   meta-device transformer right before that submodule runs.
4. `mark_block_done(this_id, compute_stream)` records an event on PyTorch's
   compute stream; the next transfer into that slot waits on it, so a buffer
   is never overwritten while a kernel is still reading it.

Correctness rests on cudarc's `CudaContext::new()` using
`cuDevicePrimaryCtxRetain` — the *same* primary context PyTorch uses — which
makes cross-stream CUDA events between cudarc and PyTorch valid. DLPack
ownership handoff is hand-implemented against the C ABI (`engine/src/dlpack.rs`)
because getting the capsule-rename convention wrong is a double-free, not a
lint. **Do not weaken these without reading `docs/ENGINE.md`.**

Only the transformer is Rust-managed; the text encoder and VAE use ordinary
real weights + `accelerate.cpu_offload` (they run once per generation, so
offload is nearly free). Single-GPU only by design.

`ResidentPool` (`docs/RESIDENT_POOL_TODO.md`) is an optional static VRAM cache
for the hottest blocks, layered on top of streaming.

## Loader design invariants (the part that is easy to break)

- **Zero-copy is the default.** `tensor_bytes`/`block_views` return borrowed
  `&[u8]` into the mmap, allocate nothing, retain nothing. `tests/ram_stability.rs`
  proves zero net heap growth over 40 passes via a byte-counting allocator —
  keep it that way. `copy_block`/`copy_block_into` are the *only* real heap
  copies and are opt-in (they report `bytes_copied`).
- **Block provenance.** `block_views`/`copy_block*`/`checksum_block` reject a
  `Block` not produced by the same `Model` instance
  (`LoaderError::BlockFromDifferentModel`) — never index one `Model`'s block
  into another's shards.
- **Shard-path security.** Index-referenced shard paths are confined to the
  index file's own directory (absolute paths and `..` rejected lexically; real
  path must resolve inside after symlink resolution). `--trust-root` is the
  *only* way to widen this (needed for real HF hub caches that symlink to a
  `blobs/` sibling); it is never auto-detected. See `src/discovery.rs`.
- **Block families / `--preset`.** `generic` tries `transformer_blocks`,
  `single_transformer_blocks`, `blocks`, `layers`; `flux` tries only the first
  two. Numeric ordering (`2` before `10`), never lexicographic. Unmatched
  tensors land in `shared/unassigned` — a grouping label, not a GPU-residency
  claim.

## Docs map

- `docs/BUILD.md` — the `uv`/`maturin` workflow and the pitfalls above.
- `docs/ENGINE.md` — engine architecture + the correctness evidence for every
  interop piece (cudarc context, cross-stream events, DLPack, pinned lifetime).
- `docs/RESIDENT_POOL_TODO.md` — ResidentPool staged plan, benchmark sweeps.
- `docs/DESIGN.md` — loader internal design.
- `docs/VALIDATION.md` / `docs/ENGINE_VALIDATION.md` — measured correctness &
  performance evidence (incl. the int8 quantization experiment, which is **not
  a working feature** — read before touching `inference/experimental/`).

## Working-tree note

The git status at session start shows `inference/generate_flux3.py`,
`generate_qwen_compiled.py`, `inference/flux3/`, and a modified
`generate_qwen.py` as untracked/uncommitted exploratory work. The committed,
documented entry points are `generate_flux.py` and `generate_qwen.py`; prefer
those as references unless the user points you at the newer files.
