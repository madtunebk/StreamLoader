# StreamLoader

A Rust checkpoint loader (`streamloader`) plus a GPU weight-streaming
engine built on top of it (`engine/`) — together, they let a DiT
transformer checkpoint larger than a GPU's VRAM still run real inference,
streaming weight blocks in just ahead of when each one executes rather
than requiring the whole model resident at once.

## `streamloader`: the loader crate

A Rust library and CLI that opens original SafeTensors checkpoints (single
file, sharded with a `*.index.json`, or a directory of implicit shards),
indexes tensor byte ranges via `mmap`, and groups tensors into logical
transformer blocks by configurable name prefixes — without pre-splitting
the checkpoint into per-block files and without copying the checkpoint
into a Rust heap buffer.

**This is a weight loader, not an inference engine.** It does not run any
model computation (no attention, no sampling, no VAE). CPU-only, no
Python/PyTorch/CUDA toolkit required to build or run it.

For the GPU weight-streaming engine built on top of this loader (`engine/`,
Rust + PyO3, serves real DiT transformer weights to PyTorch via DLPack) and
the inference scripts that drive it (`inference/`), see `docs/ENGINE.md`.

## Documentation

- [`inference/README.md`](inference/README.md) — how to run the
  inference scripts, including setup **without** `uv` (plain `pip`/`venv`)
- [`docs/BUILD.md`](docs/BUILD.md) — building/installing the `engine/`
  PyO3 extension and running inference
- [`docs/ENGINE.md`](docs/ENGINE.md) — the GPU weight-streaming engine
  (pinned-cache + reusable VRAM buffers + CUDA-event prefetch)
- [`docs/RESIDENT_POOL_TODO.md`](docs/RESIDENT_POOL_TODO.md) — ResidentPool
  (static VRAM residency) design, benchmarks, and staged implementation log
- [`docs/DESIGN.md`](docs/DESIGN.md) — this loader crate's internal design
- [`docs/VALIDATION.md`](docs/VALIDATION.md) /
  [`docs/ENGINE_VALIDATION.md`](docs/ENGINE_VALIDATION.md) — measured
  correctness/performance evidence for the loader and the engine
  respectively
- [`docs/hardware.md`](docs/hardware.md) — baseline hardware numbers this
  work was measured against
- [`docs/prompt.md`](docs/prompt.md) — the original ResidentPool task brief

## The GPU engine, in short

`engine/` is a second crate (`streamloader-engine`) built on top of this
loader: a pinned-host-cache + reusable-VRAM-buffer weight-streaming engine
that serves a real DiT transformer's weights to a live PyTorch/diffusers
forward pass via DLPack, with CUDA-event-based prefetch — so a checkpoint
larger than the GPU's VRAM can still run inference, streaming blocks in
just ahead of when each one executes. `ResidentPool` (docs/RESIDENT_POOL_TODO.md)
extends this with an optional static VRAM cache for the blocks used most,
cutting H2D traffic and steady-state generation time on top of streaming
alone.

Real, measured results on a single RTX 3060 12GB:
- **FLUX.2-klein-9B** (18.16GB transformer): streaming alone already beats
  the plain-diffusers baseline; adding a 2GB ResidentPool budget cuts
  generation time further (~9% faster than the 0GB-resident baseline at
  the sweep's best point). Past ~6-8GB resident, gains flatten or reverse
  as VRAM pressure and allocator overhead start to dominate — see
  `docs/RESIDENT_POOL_TODO.md` for the full budget sweep, including a real
  bug this work found and fixed (a CUDA stream-synchronization gap in
  engine teardown, unrelated to ResidentPool itself, only surfaced by
  creating a second engine instance in one process).
- **Qwen-Image-2.1** (a second, unrelated DiT architecture, released days
  before this was tested): the same engine, unmodified, streams it
  correctly with only new Python glue and a one-line block-family config
  change — no changes to `engine/src/*.rs` at all. Confirms the engine's
  design generalizes across DiT transformers, not just one model family.

See `docs/ENGINE.md` for the architecture and correctness evidence, and
`docs/RESIDENT_POOL_TODO.md` for the full benchmark log.

## Build

```bash
cargo build --release
```

Requires a Rust toolchain (built and tested with rustc/cargo 1.97.1,
2024 edition). No network access is required beyond the initial
`cargo fetch`/`cargo build` dependency download.

## CLI usage

```bash
# Generate a small deterministic dummy checkpoint to try commands against
# (2 shards, an index, a block spanning both shards, numeric 2-vs-10
# ordering, one shared/unassigned tensor).
cargo run --release --example make_fixture -- /tmp/fixture_checkpoint

# List every tensor's name/dtype/shape/shard/byte-range.
cargo run --release -- inspect /tmp/fixture_checkpoint --json

# List logical blocks (transformer_blocks before single_transformer_blocks,
# numeric order, shared/unassigned last).
cargo run --release -- blocks /tmp/fixture_checkpoint --preset flux --json

# Every tensor belonging to exactly one block id.
cargo run --release -- block /tmp/fixture_checkpoint --id transformer_blocks.10 --preset flux --json

# Same, plus an explicit owned copy of the block (distinct from the
# zero-copy view path) with the resulting buffer layout reported.
cargo run --release -- block /tmp/fixture_checkpoint --id transformer_blocks.10 --preset flux --copy --json

# Read the real payload bytes and emit deterministic per-tensor BLAKE3
# checksums for a block, or for one exact tensor (the selection path for
# shared/unassigned tensors).
cargo run --release -- verify /tmp/fixture_checkpoint --block transformer_blocks.10 --preset flux --json
cargo run --release -- verify /tmp/fixture_checkpoint --tensor shared.time_embed.scale --json
```

Every command accepts `--json` (machine-readable output on stdout,
diagnostics on stderr) or plain text without it. A nonexistent block or
tensor name exits with a nonzero status.

### `--preset` and `--model-prefix`

`--preset generic` (the default) tries, in order: `transformer_blocks`,
`single_transformer_blocks`, `blocks`, `layers`. `--preset flux` tries only
`transformer_blocks` then `single_transformer_blocks` (in that order —
FLUX's double-stream blocks before its single-stream blocks), each sorted
numerically (`2` before `10`, never lexicographically). Tensors matching no
family land in the `shared/unassigned` bucket — a grouping label only, not
a claim about GPU residency or execution order.

`--model-prefix <name>` matches tensors like `model.layers.0.*` by
stripping exactly that leading segment before family matching; a tensor
that doesn't start with the configured prefix is matched (and can still
be assigned) using its full original name — the prefix is never silently
stripped from unrelated tensors.

### `--trust-root` (real Hugging Face caches)

Index-referenced shard paths are constrained to the index file's own
directory by default: absolute paths and `..` are rejected lexically, and
after symlink resolution the real path must still resolve inside that
directory — otherwise a malicious/untrusted `index.json` paired with a
same-named local symlink could make the loader read arbitrary files.

Real Hugging Face hub caches symlink shard files *out* of the snapshot
directory into a shared `blobs/` sibling (for on-disk dedup across
snapshots of the same repo) — a legitimate layout that the strict default
rejects. Pass `--trust-root <dir>` to explicitly widen the boundary to a
directory you trust (e.g. the whole
`~/.cache/huggingface/hub/models--org--repo/` entry):

```bash
cargo run --release -- inspect \
  ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9B/snapshots/<hash>/transformer \
  --trust-root ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9B \
  --json
```

This is never inferred or auto-detected from the directory layout, only
set by explicit opt-in. See `src/discovery.rs` for the full resolution
rule.

## Supported inputs

1. A single `.safetensors` file.
2. A `*.safetensors.index.json` file (any suffix matching that pattern,
   e.g. `diffusion_pytorch_model.safetensors.index.json`, not only
   `model.safetensors.index.json`) with a `weight_map`.
3. A directory: an unambiguous single `*.safetensors.index.json` is used
   if present. With none, and exactly one `.safetensors` file present,
   that file is used unambiguously. With none and *multiple* sibling
   `.safetensors` files, the directory is rejected as ambiguous by
   default (`OpenOptions.allow_implicit_multi_shard`, default `false`) —
   disjoint tensor names alone don't prove several files are shards of
   one checkpoint (that's also what independent components dropped in the
   same folder would look like); pass an explicit index/file path, or set
   `allow_implicit_multi_shard: true` only once you already know every
   `.safetensors` file there belongs to one model, in which case tensor
   names are still required to be disjoint across them (rejected as a
   duplicate rather than silently merged if not). Multiple index files,
   or an empty directory, are reported as actionable errors.

## Public library API (`src/lib.rs`)

- `Model::open` / `Model::open_with(path, &OpenOptions)` — index a
  checkpoint. `Model::id()` returns a process-unique id for this opened
  instance (two `Model::open` calls on the same path are still distinct).
- `Model::descriptors()` / `Model::descriptor(name)` — tensor metadata.
- `Model::tensor_bytes(name)` — zero-copy `&[u8]` view borrowed from the
  owning shard's mmap.
- `Model::blocks(&BlockConfig)` / `Model::block(&BlockConfig, id)` —
  logical block grouping/lookup.
- `Model::block_views(&Block)` — zero-copy views for every tensor in a
  block.
- `Model::copy_block(&Block)` — explicit owned copy into a caller buffer,
  with byte-offset layout and bytes-copied count; distinct from and never
  required by the zero-copy path.
- `Model::copy_block_into(&Block, dest: &mut [u8])` — same, but writes
  into a caller-provided destination instead of allocating a `Vec`; the
  primitive `copy_block` is now built on top of.
- `Model::checksum_tensor` / `Model::checksum_block` — BLAKE3 over the
  real payload bytes.
- `block_views`/`copy_block`/`copy_block_into`/`checksum_block` all
  reject a `Block` that wasn't produced by `self` (a different `Model`
  instance, even for the identical checkpoint path) with
  `LoaderError::BlockFromDifferentModel`, rather than silently indexing
  into the wrong shards with someone else's byte offsets.

## Memory semantics (read this before assuming anything about RAM)

- `mmap` is lazy virtual-memory mapping. Opening/indexing a checkpoint
  does **not** mean the whole file is resident, pinned, or on GPU, and it
  is not equivalent to allocating an anonymous heap buffer of that size.
  File-backed pages become resident (and count toward RSS) only once
  actually read, and the kernel may evict and re-read them.
- Indexing never materializes tensor payloads into owned buffers — only
  small per-tensor descriptors (name/dtype/shape/offset/length) are
  allocated.
- `tensor_bytes`/`block_views` return borrowed slices into the mmap; they
  allocate nothing and retain nothing between calls. Verified directly:
  `tests/ram_stability.rs` uses a byte-counting global allocator to prove
  40 repeated passes over the same block cause zero net heap growth,
  whether read via the zero-copy path or the explicit `copy_block` path.
- `copy_block` performs a real, visible, caller-requested heap copy
  (reports `bytes_copied`) — it is opt-in, never implicit.
- `verify`/`checksum_*` read real payload bytes and may cause real disk
  I/O; this is explicitly not a metadata-only operation, and checksumming
  time is not a proxy for raw disk bandwidth or any GPU transfer speed.
- No RAM-budget policy, shared cache across GPU workers, or pinned-memory
  staging exists in this crate — those are explicitly deferred to a later
  milestone (see `docs/DESIGN.md`). This loader cannot reproduce the older
  Python reference implementation's RAM-doubling bug (a pageable cache
  plus a separate pinned cache of the same block, neither ever evicted)
  because it has no persistent cache of any kind to double.

## Limitations / what this crate is not

This section describes the `streamloader` loader crate (`src/`) alone —
see [The GPU engine, in short](#the-gpu-engine-in-short) above for what
`engine/` adds on top of it.

- Not an inference engine: no attention, sampling, VAE, or CUDA compute
  of any kind.
- No GPU transfer, pinned-memory staging, or CUDA-event-based buffer
  reuse — CPU-only. (`engine/` does exactly this, as a separate crate
  consuming this one's `Model`/`Block` API — see above.)
- No Python bindings (a later milestone for this crate; `engine/` already
  has PyO3 bindings for its own, separate API).
- No automatic model download; point it at checkpoints already on disk.
  (The Python inference scripts in `inference/` do resolve/download
  models via `huggingface_hub.snapshot_download` — that's their own
  behavior, not this crate's.)
- `verify`'s BLAKE3 checksums are a correctness/reproducibility tool, not
  a benchmark of storage or transfer bandwidth.

## Benchmark / example

```bash
cargo run --release --example make_fixture -- /tmp/fixture_checkpoint
cargo run --release -- inspect /tmp/fixture_checkpoint --json    # open + header-index timing is what you're timing here
cargo run --release -- verify  /tmp/fixture_checkpoint --block transformer_blocks.10 --preset flux --json
```

Run any command twice in a row to compare a first-observed pass against a
subsequent one; do not assume the first pass reflects a cold OS page
cache unless you have actually controlled for that (e.g. dropped caches
yourself — this tool does not, and never requires, privileged cache
flushing).

See `docs/VALIDATION.md` for actual measured numbers (including peak RSS)
against both the generated fixture and a real 18.16GB, 233-tensor
diffusers checkpoint.

## Tests

```bash
cargo test            # 48 tests: library + integration + CLI
cargo fmt --check
cargo clippy --all-targets
```

See `docs/VALIDATION.md` for the full list of what each test covers and the
exact recorded output of the last run.
