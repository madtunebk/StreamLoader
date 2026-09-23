# DESIGN

## Goal

Index arbitrary original SafeTensors checkpoints (single file, sharded
with an index, or a directory of implicit shards) and expose byte-range
access to individual tensors and logical blocks, without pre-splitting the
checkpoint on disk and without copying the checkpoint into a Rust heap
buffer. CPU-only; no inference.

## Descriptor ownership and lifetimes

The central ownership problem: a `Model` needs to (a) own its memory
mappings for as long as it's alive, and (b) hand out byte slices that
borrow from those mappings, without becoming a self-referential struct
(the classic trap: a struct holding both an owned buffer and a borrow
into that same buffer, which is not expressible safely in ordinary Rust
without crates like `ouroboros`/`yoke`, or `unsafe` lifetime transmutes).

This is avoided structurally, not by care alone:

```
Shard::open(path) -> (Shard, safetensors::tensor::Metadata)
```

`safetensors::SafeTensors::read_metadata()` parses and fully validates a
shard's header and returns `(header_len, Metadata)` where `Metadata` is
**fully owned** — `Vec<TensorInfo>` plus a name→index map, no borrow of
the input buffer at all (only `TensorInfo::data_offsets: (usize, usize)`,
not byte slices). So `Shard::open` can parse the header, extract that
owned metadata, and return it *decoupled* from the `Shard` — no lifetime
ties the two together.

`Model::open_with` then builds `TensorDescriptor { name, dtype, shape,
shard_id, file_offset, byte_len }` — a small, fully owned, `Clone`-able
struct — for every tensor, using each shard's `absolute_range()` to
convert the format's data-section-relative offsets into absolute file
offsets (checked arithmetic throughout: `checked_add`/`checked_sub`,
never silently wrapping). `Model` then owns:

- `shards: Vec<Shard>` (each `Shard` owns one `memmap2::Mmap`)
- `tensors: BTreeMap<String, TensorDescriptor>` (metadata only, no bytes)

`Model::tensor_bytes(name)` is an ordinary borrowing method:

```rust
pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
    let d = self.tensors.get(name)...;
    let shard = &self.shards[d.shard_id];
    shard.byte_range(name, d.file_offset, d.file_offset + d.byte_len)
}
```

The returned `&[u8]`'s lifetime is tied to `&self` by ordinary Rust borrow
checking — it can never outlive the `Model`, never needs `unsafe`, and is
never constructed from a raw pointer or transmuted lifetime. This is the
"equivalent owned handles retaining the backing" contract from the spec,
achieved via the simplest possible mechanism: don't keep the borrowing
type (`SafeTensors<'data>`) around at all past the moment metadata is
extracted from it; keep only owned descriptors plus the thing they borrow
from (`Mmap`, itself owned).

The one `unsafe` block in the whole crate is `Mmap::map(&file)` in
`src/shard.rs`, annotated with its actual precondition (the backing file
must not be mutated/truncated for the mapping's lifetime; a read-only
mapping does not stop *another process* from doing so) rather than a
generic "trust me."

## Block grouping

`BlockConfig` is an ordered list of `BlockFamily { label }` plus an
optional `model_prefix`. Grouping (`src/block.rs::group_blocks`) is pure
string/integer parsing over already-built `TensorDescriptor`s — it never
touches tensor payload bytes, so listing or looking up a block costs
nothing beyond the metadata already in memory:

1. If `model_prefix` is set, strip exactly `"{prefix}."` from the front of
   a name if present; if the name doesn't start with it, match against
   the *original* name instead (never silently discards the tensor or
   mis-parses an unrelated name that happens to share a suffix).
2. Split the (possibly-stripped) name on `.` and try each family in
   *configured* order (not alphabetical): family matches if its label's
   dot-separated parts are a prefix of the tensor's parts and the next
   part is all-ASCII-digits, parsed as `u64`.
3. Group tensors by `(family, index)` using a `BTreeMap<u64, _>` keyed by
   the parsed integer — sorting falls out of the data structure, so `2`
   sorts before `10` for free; there's no string-sort trap to avoid.
4. Anything matching no family goes to a single `shared/unassigned`
   bucket, listed last.
5. Block ids are always `"{family}.{index}"`, deliberately excluding
   `model_prefix` — ids are stable regardless of that optional wrapper,
   and `transformer_blocks.1` can never match `transformer_blocks.10`
   because the match requires either end-of-name or another `.` right
   after the digits (enforced by the split-on-`.` matching, not a
   prefix/`starts_with` check).

This intentionally has no plugin system: a caller (library user, or a
future preset added to the CLI) builds a `BlockConfig::new(vec![...])`
with whatever family list/order their architecture needs. FLUX and the
generic four-convention default are just two `BlockConfig` values, not
special-cased branches in the grouping algorithm itself.

## Copy vs. view: two distinct, opt-in paths

`Model::block_views` returns borrowed `&[u8]` slices (no allocation).
`Model::copy_block` performs one real `Vec<u8>` allocation, copies every
tensor's bytes into it, and returns the buffer plus a byte-offset
`TensorLayout` per tensor and a `bytes_copied` count. Nothing in `Model`
retains a reference to that buffer afterward — the caller owns it
entirely and decides whether to reuse it across calls. This is verified,
not just asserted: `tests/ram_stability.rs` wraps a byte-counting global
allocator around 40 repeated `copy_block` calls (each buffer dropped
immediately) and asserts net heap allocation doesn't grow, i.e. nothing
inside `Model` is accumulating copies pass over pass.

## Path-safety boundary (`src/discovery.rs`)

Shard filenames from a `weight_map` are resolved against the index file's
own directory by default, rejected lexically first (absolute path, any
`..` component) before any filesystem access, then canonicalized and
required to still resolve inside that same directory — catching a
same-named local symlink pointing elsewhere. `OpenOptions.trusted_root`
lets a caller explicitly widen that boundary (documented at length in
`discovery.rs` and `README.md`); this exists because real Hugging Face
hub caches legitimately symlink shards into a shared `blobs/` sibling
directory outside the snapshot, which the strict default would otherwise
reject as if it were an attack. The widening is never inferred from the
directory layout — only ever set by explicit caller opt-in — so the
default posture stays deny-by-default.

## Future integration boundaries (explicitly not built here)

This crate is deliberately shaped so a later milestone can add, without
restructuring what exists:

- **RAM budget / shared cache across GPU workers**: would sit *above*
  `Model` (e.g. a cache keyed by `(model_identity, block_id)`, not just
  `block_id`, per the spec's caching-identity requirement) and call
  `Model::block_views`/`copy_block` as its only interface into this
  crate. `Model` itself has, and should keep having, no cache to conflict
  with such a layer's eviction policy.
- **Pinned-memory staging / direct pinned population**: `copy_block`'s
  plain `Vec<u8>` is the natural handoff point — a pinned-memory version
  would be a different, explicitly-named method/type (e.g. returning
  pinned host memory instead of a `Vec`), never a silent change to what
  `copy_block` returns, and never implemented as "mmap plus a pinned flag"
  as if the two were interchangeable.
- **CUDA transfer/compute events, resident VRAM buffers, prefetch
  scheduling**: entirely out of this crate; would consume
  `TensorDescriptor`/byte views as input and own its own
  completion-event bookkeeping. Rust ownership of a Rust-side buffer says
  nothing about whether an async CUDA copy from it has actually
  completed — that tracking has to be explicit wherever GPU code is
  eventually added, not assumed away.
- **Python bindings**: would wrap `Model`'s public API (likely via
  PyO3), most naturally exposing `copy_block`'s owned buffer as the
  hand-off point to a `torch.Tensor`/numpy array, since a zero-copy view
  tied to a Rust-owned mmap has no safe direct Python-object equivalent
  without careful buffer-protocol lifetime management — a problem for
  that later milestone, not papered over here.
