# VALIDATION

## Environment

- OS: Linux 6.8.0-137-generic x86_64 (Ubuntu-based), local development machine
- `rustc 1.97.1 (8bab26f4f 2026-07-14)`, `cargo 1.97.1 (c980f4866 2026-06-30)`,
  edition 2024
- `clippy 0.1.97 (8bab26f4f6 2026-07-14)`, `rustfmt 1.9.0-stable`
- Key dependency versions (see `Cargo.lock` for the full resolved set):
  `safetensors 0.8.0`, `memmap2 0.9.11`, `clap 4.6.6`, `blake3 1.8.7`,
  `serde_json 1.0.151`, `thiserror 2.0.20`, `tempfile 3.27.0` (dev-only)
- Target hardware context: 64GB system RAM, 2x NVIDIA RTX 3060 12GB
  (irrelevant to this crate's own operation — CPU-only — but relevant to
  why the real-model validation below uses a checkpoint this large)

No network access was required to run any of the commands below beyond
the one-time `cargo build`/`cargo test` dependency fetch already recorded
in `Cargo.lock`. No model weights were downloaded by this work — the real
checkpoint used below was already present in the user's local Hugging
Face cache.

## Revision note

This is the second version of this document. The first version was
reviewed and two of its claims were flagged as overstated; both are
corrected below rather than silently rewritten:

1. It said `verify`'s peak-RSS measurement "**confirms** `verify` does
   not implicitly touch anything outside the requested block." RSS
   tracking real bytes-read only shows the number is *consistent with*
   that — it does not rule out the OS reading in a few extra
   page-aligned bytes at a tensor's boundary (ordinary mmap page-in
   behavior, not a bug), and RSS is a coarse enough signal that "confirms"
   overstated what was actually measured. Reworded below.
2. The review also identified real correctness gaps that hadn't been
   tested at all (block-provenance checking, and a directory-discovery
   default that could silently merge unrelated model components) — see
   "Fixes made in response to review" below. Those weren't overclaims in
   the first VALIDATION.md so much as coverage gaps it didn't mention;
   listed here for completeness.

## Fixes made in response to review

- **`copy_block`/`block_views`/`checksum_block` now validate block
  provenance.** Previously, calling e.g. `model_a.copy_block(&block)`
  where `block` was produced by a *different* `Model` (even one opened
  from a different checkpoint) would silently index into `model_a`'s
  shards using `block`'s byte offsets — wrong data, or an out-of-bounds
  error with no indication two models had been mixed up. Every `Model`
  now gets a process-unique id at construction; `Block` records which
  model produced it; the three methods above reject a mismatch with the
  new `LoaderError::BlockFromDifferentModel`. Covered by
  `tests/provenance.rs` (9 tests, including that two `Model::open` calls
  on the *same* path still get distinct ids and reject each other's
  blocks).
- **`copy_block_into(&self, blk, dest: &mut [u8])` added** — copies into
  a caller-provided buffer with no intermediate `Vec` allocation.
  `copy_block` is now a thin wrapper over it (allocates the `Vec` the
  caller asked for, then delegates). Rejects an undersized destination
  via `LoaderError::DestinationTooSmall` rather than panicking or
  truncating. Covered by `copy_block_into_matches_copy_block` (byte-for-
  byte identical output to the old path) and
  `copy_block_into_rejects_too_small_destination`.
- **`Model::block()` no longer builds every other block to find one.**
  It previously called the same full-grouping routine `blocks()` uses and
  threw away everything but the one requested; `block::find_block` now
  does a single filtering pass over the tensor list for just the
  requested id. `blocks()` is unchanged (it genuinely needs to group
  everything) but its doc comment now says explicitly that a caller doing
  this in a loop should call it once and reuse the result, not call it
  again per lookup.
- **Directory discovery no longer silently merges multiple
  `.safetensors` files with no index by default.** Disjoint tensor names
  don't prove several files are shards of one checkpoint — that's also
  exactly what independent components (e.g. a text encoder and a VAE)
  dropped in the same directory would look like. `OpenOptions.allow_implicit_multi_shard`
  (default `false`) now gates this; without it, such a directory returns
  `LoaderError::AmbiguousDirectory` instead of guessing. Covered by
  `implicit_multi_shard_directory_rejected_by_default`,
  `implicit_shards_directory_without_index_with_explicit_opt_in`, and
  `implicit_multi_shard_without_opt_in_is_ambiguous_not_merged`. This is
  a breaking default-behavior change from the first delivery, made
  deliberately.

Real-model numbers in this document were re-measured after these fixes
(see "Real-model validation" below) — none of them touch descriptor/offset
computation, so the numbers are unchanged from the first pass, but they
were re-run rather than assumed unchanged.

## Static checks

```
$ cargo fmt --check
(no output, exit 0)

$ cargo clippy --all-targets
    Finished `dev` profile [unoptimized + debuginfo] target(s)
(zero warnings)
```

## Automated tests: `cargo test` — 48/48 passed

```
$ cargo test --release
running 6 tests   (tests/blocks.rs)        ... ok. 6 passed
running 6 tests   (tests/cli.rs)           ... ok. 6 passed
running 1 test    (tests/concurrency.rs)   ... ok. 1 passed
running 15 tests  (tests/errors.rs)        ... ok. 15 passed
running 9 tests   (tests/provenance.rs)    ... ok. 9 passed
running 2 tests   (tests/ram_stability.rs) ... ok. 2 passed
running 5 tests   (tests/sharded.rs)       ... ok. 5 passed
running 4 tests   (tests/single_file.rs)   ... ok. 4 passed
```

All fixtures are generated in-process via `safetensors::serialize` /
`TensorView::new` (`tests/support/mod.rs`) — no Python fixture generator,
no fixtures checked into the repo as binary blobs.

Coverage against the spec's required test list (file:test):

1. **Known tensor bytes/shapes/dtypes** — `single_file.rs::known_tensor_bytes_shapes_dtypes`
2. **Multiple shards, valid weight_map** — `sharded.rs::multi_shard_valid_weight_map`
3. **Block spanning shards, nonadjacent ranges** — `sharded.rs::block_spans_shards_and_nonadjacent_ranges`
4. **Numeric ordering (2 vs 10) + FLUX family order** — `blocks.rs::numeric_ordering_and_flux_family_order`
5. **Shared/unassigned + explicit prefix selection** — `blocks.rs::shared_unassigned_and_explicit_prefix_selection`, `blocks.rs::explicit_model_prefix_does_not_strip_unrelated_names`
6. **Exact block match, 1 ≠ 10** — `blocks.rs::exact_block_matching_1_does_not_include_10`, `provenance.rs::find_block_does_not_build_unrelated_blocks`
7. **Missing shard/tensor, inconsistent index** — `errors.rs::missing_shard_file_referenced_by_index`, `errors.rs::missing_tensor_declared_in_index_but_absent_from_shard`, `errors.rs::inconsistent_index_missing_a_tensor_that_the_shard_actually_has`
8. **Truncation, malformed metadata/offsets, duplicate names, overflow, path escape** — `errors.rs::truncated_file_is_rejected`, `errors.rs::malformed_header_bytes_are_rejected`, `errors.rs::header_smaller_than_length_prefix_is_rejected`, `errors.rs::declared_header_larger_than_configured_limit_is_rejected`, `errors.rs::duplicate_tensor_names_across_implicit_shards_are_rejected`, `errors.rs::absolute_shard_path_in_index_is_rejected`, `errors.rs::parent_dir_traversal_in_index_is_rejected`, `errors.rs::symlink_escape_in_index_is_rejected`, `errors.rs::trusted_root_explicitly_widens_the_symlink_boundary`, `errors.rs::implicit_multi_shard_without_opt_in_is_ambiguous_not_merged`
9. **Empty/scalar tensors** — `single_file.rs::empty_and_scalar_tensors`
10. **Multiple model instances, identical names, different data** — `single_file.rs::multiple_instances_same_names_different_data`, `provenance.rs::two_models_of_the_same_checkpoint_have_distinct_ids`
11. **Safe ownership/lifetimes + concurrent reads** — `concurrency.rs::concurrent_reads_from_multiple_threads` (8 threads x 50 reads each, no `unsafe` at any call site)
12. **CLI JSON, verification checksum against known bytes, nonzero errors** — `cli.rs::inspect_json_lists_every_tensor`, `cli.rs::verify_checksum_matches_independently_computed_hash`, `cli.rs::nonexistent_block_exits_nonzero`, `cli.rs::nonexistent_tensor_exits_nonzero`, `cli.rs::copy_flag_reports_bytes_copied_in_json`

Plus, beyond the required list:
- `ram_stability.rs` proves (via a byte-counting global allocator, not RSS
  sampling) that 40 repeated passes over the same block through either
  the zero-copy view API or the explicit `copy_block` API cause zero net
  heap growth — the specific concern that motivated this crate's design
  (see DESIGN.md).
- `provenance.rs` (9 tests, see "Fixes made in response to review" above)
  covers block-provenance rejection across all four methods that accept
  a `Block`, plus `copy_block_into`'s buffer-reuse and undersized-
  destination behavior.

## Real-model validation

Fixture tests prove correctness of the format handling; they do not prove
this works on a real, large, real-world checkpoint shape. It was
additionally run against a real model already present in the local
Hugging Face cache, both in the first validation pass and again after the
fixes above.

**Model**: `black-forest-labs/FLUX.2-klein-9B`, `transformer/` component
(diffusers layout: `diffusion_pytorch_model.safetensors.index.json` + 2
shards), 233 tensors, BF16, **18.16GB** on disk (9.80GB + 8.36GB across
the two shards). Not downloaded by this work — already present locally.

```
$ ./target/release/streamloader inspect .../transformer --json
error: shard path "diffusion_pytorch_model-00001-of-00002.safetensors"
  referenced from .../diffusion_pytorch_model.safetensors.index.json
  is rejected: resolves outside the trusted checkpoint root
  (possible symlink escape)
exit=1
```

This is the strict default correctly firing: this model's cache directory
symlinks its shard files into a `blobs/` sibling outside the snapshot
directory (standard Hugging Face hub cache layout for on-disk dedup
across snapshots). Confirmed intentional, not a bug, by inspecting the
actual symlinks (`ls -la transformer/`, `realpath`) before adding the
`--trust-root` opt-in described in `README.md`/`DESIGN.md`.

```
$ ./target/release/streamloader inspect .../transformer \
    --trust-root ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9B --json
exit=0   time=1.0ms    peak child RSS=10.88MB
tensor_count=233  shard_count=2
```

Peak RSS while indexing an 18.16GB checkpoint: **~11MB** (10.88MB on the
post-fix re-run, 11.25MB on the first pass — both effectively "metadata-
sized," the small difference being ordinary run-to-run noise, not a
regression or improvement worth reading into) — i.e. indexing cost is
metadata-sized, not file-sized, matching README.md's memory-semantics
claim.

```
$ ./target/release/streamloader blocks .../transformer --trust-root ... --preset flux --json
exit=0   time=1.0ms
block_count=33
order: transformer_blocks.0, .1, .2 ... single_transformer_blocks.22, .23, shared/unassigned
```

33 blocks: `transformer_blocks.0`–`.7` (8), `single_transformer_blocks.0`–`.23`
(24), `shared/unassigned` last (1 bucket: `context_embedder.weight`,
`double_stream_modulation_img.linear.weight`,
`double_stream_modulation_txt.linear.weight`, `norm_out.linear.weight`,
`proj_out.weight`, `single_stream_modulation.linear.weight`,
`time_guidance_embed.timestep_embedder.linear_{1,2}.weight`,
`x_embedder.weight`) — confirms the FLUX preset's naming assumptions hold
against a real, independently-named checkpoint, not just the fixture that
was built to match them.

```
$ ./target/release/streamloader verify .../transformer --trust-root ... \
    --block transformer_blocks.7 --preset flux --json
exit=0   time=122.3ms (warm page cache; 397ms measured on an earlier, cold-er pass
                        in the same session — not a controlled cold-cache measurement,
                        reported per README's first-observed-vs-subsequent guidance)
peak child RSS=835.12MB
total_bytes=872416256  (872MB)  num_tensors=16
```

Peak RSS while checksumming an 872MB block: **835MB**, i.e. proportional
to bytes actually read rather than to the 18.16GB checkpoint total. This
is *consistent with* `verify` not touching payload outside the requested
block — it is not proof of that at the byte level (RSS also reflects
ordinary mmap page-alignment/readahead around the tensor boundaries, and
is a process-wide counter, not a per-call one). The stronger, byte-level
guarantee — that `tensor_bytes`/`block_views` only ever return the exact
`[file_offset, file_offset+byte_len)` range and nothing else — comes from
reading `src/shard.rs::byte_range`, not from this RSS measurement; the
RSS number here is corroborating field evidence on a real large
checkpoint, not the proof itself.

```
$ ./target/release/streamloader block .../transformer --trust-root ... \
    --id transformer_blocks.9999 --preset flux --json
error: block "transformer_blocks.9999" not found (no tensor matched this block id)
exit=1
```

Cross-checked independently: a from-scratch, stdlib-only Python script
(written during this session, not committed to this repo) re-parsed the
SafeTensors header of every shard itself (`struct.unpack` + `json.loads`,
no shared code with this crate or with `safetensors`) and recomputed
every one of the 9 fixture tensors' absolute offset/length independently
— all 9 matched the CLI's `inspect --json` output exactly. It also
independently confirmed the `blocks --preset flux` ordering and the
nonzero exit on a bad block id. Not re-run against the full 233-tensor
real model in that form, but the same header format and offset arithmetic
is what both the fixture and real-model runs above exercise.

## What is fixture-tested vs. real-model-tested vs. not tested

**Fixture-tested** (via `cargo test`, deterministic, in-process): every
correctness property in the required test list above — dtype/shape/byte
preservation, multi-shard + index parsing, shard-spanning blocks, numeric
+ FLUX ordering, exact block matching, shared/unassigned + prefix
handling, every error path (missing/inconsistent/duplicate/truncated/
malformed/overflow/path-escape/ambiguous-implicit-merge), empty/scalar
tensors, multiple independent instances, block provenance across all
four block-consuming methods, `copy_block_into` correctness and
undersized-destination rejection, concurrent reads, CLI JSON + checksum +
exit codes, and RAM-flatness across repeated passes.

**Real-model-tested** (against FLUX.2-klein-9B's `transformer/`
component, 18.16GB, 233 tensors, re-run after the provenance/copy/
discovery fixes above, commands and output shown above): directory/index
discovery, the symlink-escape default rejecting a real Hugging Face cache
layout and `--trust-root` correctly permitting it, inspect/blocks/verify
correctness and timing/RSS against a real multi-hundred-MB-per-block,
multi-GB-per-shard checkpoint, and a nonexistent-block error path.
Block-provenance rejection and `copy_block_into` were *not* re-run
against this real model specifically — they're covered by fixture tests
(`provenance.rs`) using the same code paths, not by a second real-model
pass, since nothing about those two features is checkpoint-size-dependent.

**Not tested**:
- The `vae/` (single-file) and `text_encoder/` (4-shard, different
  naming/architecture) components of the same downloaded model were not
  run through the loader — only the `transformer/` DiT component was.
- No checkpoint anywhere near the ~28GB target size mentioned in the
  original task context was used (18.16GB was the largest real checkpoint
  available locally); no claim is made about behavior at that larger
  size beyond what mmap's documented semantics imply.
- No true cold-page-cache measurement was taken (would require dropping
  caches, which this validation did not do — see the `verify` timing note
  above, reported honestly as warm-cache rather than omitted).
- Nothing about GPU transfer, pinned memory, or multi-process/multi-GPU
  sharing was tested in *this crate* — see the separate pinned-memory/GPU
  engine work (if delivered alongside this) for that; it has its own,
  separately reported benchmarks against a measured baseline, because
  "the loader is correct" and "the GPU engine is fast" are different
  claims requiring different evidence.
- No fuzzing; malformed-input tests are targeted, hand-constructed cases,
  not a fuzz corpus.
