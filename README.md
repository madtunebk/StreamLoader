# streamloader

A Rust library and CLI that opens original SafeTensors checkpoints (single
file, sharded with a `*.index.json`, or a directory of implicit shards),
indexes tensor byte ranges via `mmap`, and groups tensors into logical
transformer blocks by configurable name prefixes — without pre-splitting
the checkpoint into per-block files and without copying the checkpoint
into a Rust heap buffer.

**This is a weight loader, not an inference engine.** It does not run any
model computation (no attention, no sampling, no VAE). CPU-only, no
Python/PyTorch/CUDA toolkit required to build or run it.

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
  milestone (see DESIGN.md). This loader cannot reproduce the older
  Python reference implementation's RAM-doubling bug (a pageable cache
  plus a separate pinned cache of the same block, neither ever evicted)
  because it has no persistent cache of any kind to double.

## Limitations / what this is not

- Not an inference engine: no attention, sampling, VAE, or CUDA compute
  of any kind.
- No GPU transfer, pinned-memory staging, or CUDA-event-based buffer
  reuse — CPU-only.
- No Python bindings (a later milestone).
- No automatic model download; point it at checkpoints already on disk.
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

See `VALIDATION.md` for actual measured numbers (including peak RSS)
against both the generated fixture and a real 18.16GB, 233-tensor
diffusers checkpoint.

## Tests

```bash
cargo test            # 48 tests: library + integration + CLI
cargo fmt --check
cargo clippy --all-targets
```

See `VALIDATION.md` for the full list of what each test covers and the
exact recorded output of the last run.
