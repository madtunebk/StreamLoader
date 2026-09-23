//! The pinned-cache + reusable-VRAM-buffer engine.
//!
//! Layering, matching the explicit ask this was built against:
//! - `streamloader::Model` (mmap, CPU-only) is the only thing that ever
//!   touches the checkpoint file. This module never reads the file
//!   itself -- it only calls `Model::copy_block_into`.
//! - Exactly ONE pinned host allocation is made, sized to the whole
//!   transformer, filled directly from the mmap via
//!   `copy_block_into` (no intermediate `Vec`, no second CPU-side
//!   cache of any kind).
//! - A small fixed ring of reusable VRAM buffers (`VramSlot`), sized to
//!   the largest block, is transferred into round-robin as blocks are
//!   requested/prefetched. Buffers are never reallocated per block.
//! - Every tensor handed to Python is a DLPack view into one of those
//!   VRAM buffers -- zero payload copy at that boundary.
//! - Cross-stream correctness (our transfer stream vs. PyTorch's compute
//!   stream) is enforced with real CUDA events recorded/waited via the
//!   raw driver API against PyTorch's own raw stream pointer -- verified
//!   working (see `test_dlpack.py`) before this module was written.
//! - Single GPU only. No cross-process/cross-GPU sharing.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::sys as cu_sys;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream, result as cu_result};
use pyo3::prelude::*;
use streamloader::Dtype;
use streamloader::{Block, BlockConfig, BlockFamily, Model, OpenOptions, UNASSIGNED};

use crate::dlpack::{self, DlDtype};
use crate::error::{EngineError, Result};

fn dl_dtype(name: &str, dtype: Dtype) -> Result<DlDtype> {
    Ok(match dtype {
        Dtype::F32 => DlDtype::F32,
        Dtype::F16 => DlDtype::F16,
        Dtype::BF16 => DlDtype::Bf16,
        Dtype::F64 => DlDtype::F64,
        Dtype::I64 => DlDtype::I64,
        Dtype::I32 => DlDtype::I32,
        Dtype::I16 => DlDtype::I16,
        Dtype::I8 => DlDtype::I8,
        Dtype::U8 => DlDtype::U8,
        Dtype::BOOL => DlDtype::Bool,
        other => {
            return Err(EngineError::UnsupportedDtype {
                name: name.to_string(),
                dtype: other,
            });
        }
    })
}

/// Where one tensor lives within the whole pinned buffer, plus what a
/// GPU view of it should look like.
#[derive(Clone)]
struct TensorMeta {
    name: String,
    pinned_offset: u64,
    len: u64,
    dtype: Dtype,
    shape: Vec<usize>,
}

struct BlockMeta {
    id: String,
    tensors: Vec<TensorMeta>,
    total_bytes: u64,
}

/// One reusable device buffer. `device_ptr`/`capacity` are fixed for the
/// engine's whole lifetime; `occupant` names whichever block's bytes
/// currently live there.
struct VramSlot {
    device_ptr: u64,
    capacity: u64,
    occupant: Option<String>,
    /// Recorded on the transfer stream right after the H2D copy for
    /// `occupant` is issued. The compute stream waits on this before
    /// any kernel reads the buffer.
    ready_event: CudaEvent,
    /// Recorded on the *compute* stream (raw, from Python) by
    /// `mark_block_done` after the consumer is finished reading. The
    /// transfer stream waits on this before overwriting the buffer for
    /// the next occupant -- this is what makes buffer reuse safe.
    free_event: Option<CudaEvent>,
}

#[derive(Default, Clone, Copy)]
pub struct Stats {
    pub bytes_h2d: u64,
    pub transfer_count: u64,
    pub cache_hits: u64,
    pub pinned_bytes: u64,
    pub vram_bytes: u64,
    /// Total bytes of the blocks chosen resident at init (§8's static
    /// knapsack selection). Fixed for the engine's whole lifetime in
    /// stage 3 -- there is no eviction yet, so this never changes after
    /// `new()` returns.
    pub resident_bytes: u64,
    /// Incremented every time `issue_transfer` resolves a block from the
    /// resident pool instead of the streaming ring. Deliberately separate
    /// from `cache_hits` (a ring cache hit still required a prior
    /// transfer this run; a resident hit never transfers at all) -- see
    /// the review's §9 caution about not conflating the two.
    pub resident_hits: u64,
}

/// One real H2D transfer's identity + GPU-side timing handles. Populated
/// only on an actual transfer (never on a cache hit -- those are already
/// counted by `Stats::cache_hits`). `start`/`end` are recorded on the
/// transfer stream at issue time but deliberately never waited on there;
/// `elapsed_ms` is computed lazily, only when a caller asks for timings
/// (see `RustEngine::block_timings_ms`), so instrumentation adds no
/// synchronization to the hot per-block path -- consistent with the
/// engine's existing "no CPU sync on the hot path" discipline.
struct BlockTiming {
    id: String,
    bytes: u64,
    /// Always `false`: only the streaming path (a real H2D transfer)
    /// pushes a `BlockTiming` entry at all -- a resident-pool hit
    /// (`issue_transfer`'s other branch) never transfers, so it never
    /// reaches this struct. Kept as an explicit field anyway so the
    /// Python-facing tuple shape (`block_timings()`) doesn't need to
    /// change if timing ever needs to distinguish transfer *kinds*.
    resident: bool,
    start: CudaEvent,
    end: CudaEvent,
}

pub struct RustEngine {
    _model: Model, // kept alive: the mmap must outlive nothing here (copying already happened), but Model also owns shard file handles indirectly via Mmap; dropping is fine post-init. Retained mainly for introspection/debugging.
    ctx: Arc<CudaContext>,
    transfer_stream: Arc<CudaStream>,
    /// Owns the pinned host allocation for the engine's whole lifetime.
    /// MUST be kept here, not just its raw pointer -- `PinnedHostSlice`
    /// frees the allocation on `Drop`, and every block fetched after
    /// init reads through `pinned_base` (a raw pointer derived from this
    /// once, for hot-path reuse without paying `PinnedHostSlice`'s
    /// sync-gated accessor cost on every call). Dropping `_pinned` before
    /// `RustEngine` itself would turn `pinned_base` into a dangling
    /// pointer into freed page-locked memory -- exactly the bug an
    /// earlier version of this file had (caught by
    /// `test_engine_real_model.py` segfaulting on the first real block
    /// fetch after init, not on the shared-block fetch that happens
    /// during `new()` itself before the would-be free).
    _pinned: cudarc::driver::PinnedHostSlice<u8>,
    pinned_base: *mut u8,
    #[allow(dead_code)]
    pinned_total: u64,
    tensors: HashMap<String, TensorMeta>,
    blocks: HashMap<String, BlockMeta>,
    block_order: Vec<String>,
    shared_slot: Option<VramSlot>,
    /// Blocks chosen resident at init (§8's static knapsack, stage 3).
    /// Disjoint from `slots` by construction: a resident block is never
    /// placed in the streaming ring, and `issue_transfer` checks this map
    /// *before* the ring scan so a resident block id can never appear in
    /// both places. Never mutated after `new()` returns in stage 3 -- no
    /// eviction, no promotion, matching the "safe, synchronous, init-time
    /// only" pattern `shared_slot` already uses.
    resident: HashMap<String, VramSlot>,
    slots: Vec<VramSlot>,
    next_slot: usize,
    stats: Stats,
    /// Append-only, one entry per real H2D transfer for the engine's
    /// whole lifetime. Not cleared between generations -- same
    /// whole-run-accumulation discipline as `stats`.
    block_timings: Vec<BlockTiming>,
}

// SAFETY: `pinned_base` points at a `cudaHostAlloc`'d buffer that is
// never freed until `RustEngine` (and the `PinnedHostSlice` that
// produced it, which we intentionally leak -- see `new()`) is dropped;
// nothing here mutates through the pointer after init, only reads for
// building memcpy source slices, and those reads are only ever issued
// through `&mut self`/`&self` methods that PyO3 itself serializes via
// the GIL + its per-object borrow checking (a `#[pyclass]` never allows
// two Rust-side borrows of the same instance to run concurrently) --
// there is no path by which two threads actually dereference
// `pinned_base` at once. `CudaContext`/`CudaStream` are already
// `Send + Sync` per cudarc.
unsafe impl Send for RustEngine {}
unsafe impl Sync for RustEngine {}

impl RustEngine {
    pub fn new(
        path: &str,
        budget_bytes: u64,
        vram_slots: usize,
        device_ordinal: usize,
        trust_root: Option<&str>,
        block_families: Option<Vec<String>>,
        resident_budget_bytes: u64,
    ) -> Result<Self> {
        let model = Model::open_with(
            path,
            &OpenOptions {
                trusted_root: trust_root.map(std::path::PathBuf::from),
                ..Default::default()
            },
        )?;
        // Block-family prefixes are per-architecture, not hardcoded to
        // FLUX -- any DiT with a different naming convention (e.g.
        // PixelDiT's `model.patch_blocks.N.*` / `model.pixel_blocks.N.*`)
        // just needs its own family list. Defaults to FLUX's families
        // only for backwards compatibility with existing callers.
        let config = match block_families {
            Some(families) => BlockConfig::new(families.into_iter().map(BlockFamily::new).collect()),
            None => BlockConfig::flux(),
        };
        let mut ordered_blocks: Vec<Block> = model.blocks(&config);
        // `shared/unassigned` (embedders, norm_out, etc.) must be resident
        // before any real block runs; pull it to the front if present.
        if let Some(pos) = ordered_blocks.iter().position(|b| b.id == UNASSIGNED) {
            let shared = ordered_blocks.remove(pos);
            ordered_blocks.insert(0, shared);
        }

        let total_bytes: u64 = ordered_blocks
            .iter()
            .map(|b| b.tensors.iter().map(|d| d.byte_len).sum::<u64>())
            .sum();
        if total_bytes > budget_bytes {
            return Err(EngineError::BudgetExceeded {
                needed: total_bytes,
                budget: budget_bytes,
            });
        }

        let ctx = CudaContext::new(device_ordinal)?;
        let transfer_stream = ctx.new_stream()?;

        let mut pinned = unsafe { ctx.alloc_pinned::<u8>(total_bytes as usize) }?;
        let pinned_base = pinned.as_mut_ptr()?;

        let mut tensors: HashMap<String, TensorMeta> = HashMap::new();
        let mut blocks: HashMap<String, BlockMeta> = HashMap::new();
        let mut block_order: Vec<String> = Vec::new();
        let mut offset: u64 = 0;
        let mut shared_bytes: u64 = 0;

        for block in &ordered_blocks {
            let block_bytes: u64 = block.tensors.iter().map(|d| d.byte_len).sum();
            // SAFETY: `pinned_base..pinned_base+total_bytes` is one
            // allocation we own exclusively at this point (init, before
            // any GPU transfer or Python handoff); `offset+block_bytes
            // <= total_bytes` by construction (running sum over the same
            // descriptors used to compute `total_bytes` above).
            let dest = unsafe {
                std::slice::from_raw_parts_mut(
                    pinned_base.add(offset as usize),
                    block_bytes as usize,
                )
            };
            let layout = model.copy_block_into(block, dest)?;

            let mut meta_tensors = Vec::with_capacity(block.tensors.len());
            for (d, l) in block.tensors.iter().zip(layout.iter()) {
                dl_dtype(&d.name, d.dtype)?; // validate now, fail loudly at init not at first use
                let meta = TensorMeta {
                    name: d.name.clone(),
                    pinned_offset: offset + l.offset,
                    len: l.len,
                    dtype: d.dtype,
                    shape: d.shape.clone(),
                };
                tensors.insert(d.name.clone(), meta.clone());
                meta_tensors.push(meta);
            }

            if block.id == UNASSIGNED {
                shared_bytes = block_bytes;
            } else {
                block_order.push(block.id.clone());
            }
            blocks.insert(
                block.id.clone(),
                BlockMeta {
                    id: block.id.clone(),
                    tensors: meta_tensors,
                    total_bytes: block_bytes,
                },
            );
            offset += block_bytes;
        }

        let mut vram_bytes: u64 = 0;

        let shared_slot = if shared_bytes > 0 {
            let shared_meta = &blocks[UNASSIGNED];
            let dev_ptr = alloc_device(&transfer_stream, shared_bytes)?;
            vram_bytes += shared_bytes;
            let src = unsafe {
                std::slice::from_raw_parts(
                    pinned_base.add(block_pinned_start_offset(shared_meta)),
                    shared_bytes as usize,
                )
            };
            unsafe { cu_result::memcpy_htod_async(dev_ptr, src, transfer_stream.cu_stream()) }?;
            let ev = ctx.new_event(None)?;
            ev.record(&transfer_stream)?;
            ev.synchronize()?; // one-time init cost: shared weights must be ready before anything else runs
            Some(VramSlot {
                device_ptr: dev_ptr,
                capacity: shared_bytes,
                occupant: Some(UNASSIGNED.to_string()),
                ready_event: ev, // already recorded+synchronized above, unlike the ring slots' placeholder
                free_event: None,
            })
        } else {
            None
        };

        // Stage 3: static, init-time-only ResidentPool. Selection is a
        // first-fit-decreasing knapsack by block size -- every non-shared
        // block executes exactly once per diffusion step (verified against
        // the real FLUX.2 forward pass, see the review's §8), so keeping
        // any block resident saves exactly that block's own bytes every
        // step, forever; there is no per-step eviction decision to make in
        // this stage, only a one-time "which bytes fit the budget" choice.
        // Sorting by size descending and taking whatever still fits, in
        // order, is a standard bin-packing heuristic and (with only two
        // distinct block sizes in this model) lands at or very near the
        // exact optimum.
        let mut by_size_desc: Vec<(&String, u64)> = block_order
            .iter()
            .map(|id| (id, blocks[id].total_bytes))
            .collect();
        by_size_desc.sort_by_key(|b| std::cmp::Reverse(b.1));

        let mut resident_ids: Vec<String> = Vec::new();
        let mut resident_bytes_total: u64 = 0;
        for (id, bytes) in &by_size_desc {
            if resident_bytes_total + bytes <= resident_budget_bytes {
                resident_ids.push((*id).clone());
                resident_bytes_total += bytes;
            }
        }

        let mut resident: HashMap<String, VramSlot> = HashMap::new();
        for id in &resident_ids {
            let meta = &blocks[id];
            let bytes = meta.total_bytes;
            let dev_ptr = alloc_device(&transfer_stream, bytes)?;
            vram_bytes += bytes;
            let src = unsafe {
                std::slice::from_raw_parts(
                    pinned_base.add(block_pinned_start_offset(meta)),
                    bytes as usize,
                )
            };
            unsafe { cu_result::memcpy_htod_async(dev_ptr, src, transfer_stream.cu_stream()) }?;
            let ev = ctx.new_event(None)?;
            ev.record(&transfer_stream)?;
            // One-time init cost, same precedent as `shared_slot` above --
            // valid here specifically BECAUSE this only ever runs once,
            // synchronously, before any streaming/prefetch protocol
            // starts (see the review's §5: this pattern is unsafe to
            // reuse for a mid-run promotion, but correct at init).
            ev.synchronize()?;
            resident.insert(
                id.clone(),
                VramSlot {
                    device_ptr: dev_ptr,
                    capacity: bytes,
                    occupant: Some(id.clone()),
                    ready_event: ev,
                    free_event: None,
                },
            );
        }

        // Ring sizing only needs to cover blocks that actually still
        // stream -- a block promoted to the resident pool above no longer
        // needs ring capacity budgeted for it, which is exactly the
        // "ring already wastes VRAM padding smaller blocks to the max
        // size" finding from the review's §3: shrinking the max over the
        // *remaining* streaming set (not all blocks) avoids compounding
        // that waste on top of what ResidentPool is already saving.
        let max_block_bytes: u64 = block_order
            .iter()
            .filter(|id| !resident.contains_key(*id))
            .map(|id| blocks[id].total_bytes)
            .max()
            .unwrap_or(0);

        let slot_count = vram_slots.max(1);
        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            let dev_ptr = alloc_device(&transfer_stream, max_block_bytes.max(1))?;
            vram_bytes += max_block_bytes.max(1);
            slots.push(VramSlot {
                device_ptr: dev_ptr,
                capacity: max_block_bytes,
                occupant: None,
                // Never-recorded placeholder: `occupant: None` guarantees
                // `issue_transfer`'s cache-hit check can't match this slot
                // until its first real transfer replaces this event with
                // one that's actually been recorded, so it's never waited
                // on in this state.
                ready_event: ctx.new_event(None)?,
                free_event: None,
            });
        }

        Ok(RustEngine {
            _model: model,
            ctx,
            transfer_stream,
            _pinned: pinned,
            pinned_base,
            pinned_total: total_bytes,
            tensors,
            blocks,
            block_order,
            shared_slot,
            resident,
            slots,
            next_slot: 0,
            stats: Stats {
                pinned_bytes: total_bytes,
                vram_bytes,
                resident_bytes: resident_bytes_total,
                ..Default::default()
            },
            block_timings: Vec::new(),
        })
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Per-transfer (id, bytes, resident, elapsed_ms) for every real H2D
    /// transfer issued so far. Computing `elapsed_ms` calls
    /// `cudaEventElapsedTime` under the hood, which requires both events
    /// to have completed -- this blocks until the GPU work they bracket
    /// is done, exactly like `bench_h2d`'s existing use of the same
    /// mechanism (`lib.rs`). Call this after a run (or a phase of one),
    /// never inside the per-block hot path.
    pub fn block_timings_ms(&self) -> Result<Vec<(String, u64, bool, f64)>> {
        self.block_timings
            .iter()
            .map(|t| Ok((t.id.clone(), t.bytes, t.resident, t.start.elapsed_ms(&t.end)? as f64)))
            .collect()
    }

    pub fn block_ids(&self) -> Vec<String> {
        self.block_order.clone()
    }

    /// Block ids chosen resident at init (stage 3's static knapsack
    /// selection). Empty when `resident_budget_bytes` was 0 -- the
    /// default, fully-backward-compatible case.
    pub fn resident_block_ids(&self) -> Vec<String> {
        self.resident.keys().cloned().collect()
    }

    /// Cyclic next-use distance from `from_block_id` to every non-shared
    /// block, in the same units as `block_order`'s positions (0 = the
    /// block currently executing).
    ///
    /// This is the "farthest-next-use" table a future ResidentPool
    /// eviction decision would consult -- computed purely from
    /// `block_order`, which is fixed for the engine's whole lifetime
    /// (built once in `new()` from `model.blocks(&config)` and never
    /// rebuilt). It assumes execution wraps from the last block straight
    /// back to the first every diffusion step, which is what the real
    /// FLUX forward pass does (verified independently against
    /// `transformer_flux2.py`, not just this crate's own docs) -- this
    /// method does not re-verify that assumption at runtime, it only
    /// requires it to already be true.
    pub fn next_use_distances(&self, from_block_id: &str) -> Result<Vec<(String, u64)>> {
        let n = self.block_order.len() as u64;
        let from_idx = self
            .block_order
            .iter()
            .position(|b| b == from_block_id)
            .ok_or_else(|| EngineError::UnknownBlock(from_block_id.to_string()))? as u64;
        Ok(self
            .block_order
            .iter()
            .enumerate()
            .map(|(idx, id)| {
                let dist = (idx as u64 + n - from_idx) % n;
                (id.clone(), dist)
            })
            .collect())
    }

    /// Issue (or no-op if already resident/in-flight) the H2D transfer
    /// for `block_id` into whichever slot is next in the ring, without
    /// waiting for it or building any tensors. Meant to be called for
    /// the *next* block while the *current* one is still computing.
    pub fn prefetch(&mut self, block_id: &str) -> Result<()> {
        self.issue_transfer(block_id).map(|_| ())
    }

    /// Returns DLPack-backed CUDA tensors for every tensor in `block_id`,
    /// guaranteed (via a real CUDA event, not a CPU wait) to be visible
    /// to `compute_stream_ptr` before any kernel enqueued on it after
    /// this call runs.
    pub fn get_block(
        &mut self,
        py: Python<'_>,
        block_id: &str,
        compute_stream_ptr: u64,
    ) -> Result<HashMap<String, Py<PyAny>>> {
        let (device_ptr, ready_event) = self.issue_transfer(block_id)?;

        let compute_stream = compute_stream_ptr as cu_sys::CUstream;
        unsafe {
            cu_result::stream::wait_event(
                compute_stream,
                ready_event,
                cu_sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
        }?;

        self.build_tensors(py, block_id, device_ptr)
    }

    pub fn get_shared(&self, py: Python<'_>) -> Result<HashMap<String, Py<PyAny>>> {
        let Some(slot) = &self.shared_slot else {
            return Ok(HashMap::new());
        };
        self.build_tensors(py, UNASSIGNED, slot.device_ptr)
    }

    /// Record that `compute_stream_ptr` is done reading `block_id`'s
    /// tensors (call right after the block's forward pass returns). The
    /// event recorded here is what the *next* transfer into this slot
    /// waits on before overwriting it.
    pub fn mark_block_done(&mut self, block_id: &str, compute_stream_ptr: u64) -> Result<()> {
        // Resident blocks are never evicted/overwritten in stage 3, so
        // there is no slot to protect and nothing to record -- a
        // `free_event` only exists to gate the *next* transfer into a
        // slot, and a resident block's slot never has a next occupant.
        if self.resident.contains_key(block_id) {
            return Ok(());
        }
        let slot_idx = self.slot_holding(block_id)?;
        let ev = self.ctx.new_event(None)?;
        let compute_stream = compute_stream_ptr as cu_sys::CUstream;
        unsafe { cu_result::event::record(ev.cu_event(), compute_stream) }?;
        self.slots[slot_idx].free_event = Some(ev);
        Ok(())
    }

    fn slot_holding(&self, block_id: &str) -> Result<usize> {
        self.slots
            .iter()
            .position(|s| s.occupant.as_deref() == Some(block_id))
            .ok_or_else(|| EngineError::BlockNotResident(block_id.to_string()))
    }

    /// Ensures `block_id`'s bytes are resident somewhere on the GPU
    /// (issuing a streaming transfer if needed) and returns
    /// `(device_ptr, ready_event)` for it. `ready_event` is the raw CUDA
    /// event handle the caller's compute stream must wait on before
    /// reading -- returned by value (not a borrow of `self`) specifically
    /// so this method can answer for either the resident pool or the
    /// streaming ring without the caller needing to know which.
    fn issue_transfer(&mut self, block_id: &str) -> Result<(u64, cu_sys::CUevent)> {
        // Resident pool checked FIRST, before the ring scan: a resident
        // block is never placed in `self.slots`, never re-transferred,
        // never counted as a ring cache hit.
        if let Some(slot) = self.resident.get(block_id) {
            self.stats.resident_hits += 1;
            return Ok((slot.device_ptr, slot.ready_event.cu_event()));
        }

        if let Some(idx) = self
            .slots
            .iter()
            .position(|s| s.occupant.as_deref() == Some(block_id))
        {
            self.stats.cache_hits += 1;
            let slot = &self.slots[idx];
            return Ok((slot.device_ptr, slot.ready_event.cu_event()));
        }

        let meta = self
            .blocks
            .get(block_id)
            .ok_or_else(|| EngineError::UnknownBlock(block_id.to_string()))?;
        let block_bytes = meta.total_bytes;
        let pinned_offset = meta.tensors.first().map(|t| t.pinned_offset).unwrap_or(0);

        let idx = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slots.len();

        // Don't overwrite a slot the compute stream might still be
        // reading: make the TRANSFER stream wait for that slot's
        // free_event (set by mark_block_done for whatever was there
        // before) before issuing the new copy.
        if let Some(free_event) = self.slots[idx].free_event.take() {
            unsafe {
                cu_result::stream::wait_event(
                    self.transfer_stream.cu_stream(),
                    free_event.cu_event(),
                    cu_sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                )
            }?;
        }

        // Timing-only event, recorded immediately before the copy is
        // issued. Never waited on here -- only ever read back later via
        // `block_timings_ms`, so this adds no synchronization to the hot
        // path (same instrumentation discipline as `bench_h2d`).
        let timing_start = self
            .ctx
            .new_event(Some(cu_sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        timing_start.record(&self.transfer_stream)?;

        let src = unsafe {
            std::slice::from_raw_parts(
                self.pinned_base.add(pinned_offset as usize),
                block_bytes as usize,
            )
        };
        unsafe {
            cu_result::memcpy_htod_async(
                self.slots[idx].device_ptr,
                src,
                self.transfer_stream.cu_stream(),
            )
        }?;
        let ev = self.ctx.new_event(None)?;
        ev.record(&self.transfer_stream)?;
        // Separate from `ev`/`ready_event` deliberately: `ready_event` is
        // owned by the slot and consumed by the cross-stream wait in
        // `get_block`; this one is owned by the timing record and must
        // never be touched by that correctness-critical path.
        let timing_end = self
            .ctx
            .new_event(Some(cu_sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        timing_end.record(&self.transfer_stream)?;
        self.slots[idx].ready_event = ev;
        self.slots[idx].occupant = Some(block_id.to_string());

        self.block_timings.push(BlockTiming {
            id: block_id.to_string(),
            bytes: block_bytes,
            resident: false,
            start: timing_start,
            end: timing_end,
        });

        self.stats.bytes_h2d += block_bytes;
        self.stats.transfer_count += 1;
        Ok((self.slots[idx].device_ptr, self.slots[idx].ready_event.cu_event()))
    }

    fn build_tensors(
        &self,
        py: Python<'_>,
        block_id: &str,
        slot_base: u64,
    ) -> Result<HashMap<String, Py<PyAny>>> {
        let meta = self
            .blocks
            .get(block_id)
            .ok_or_else(|| EngineError::UnknownBlock(block_id.to_string()))?;
        // Resolved once per call, not per tensor -- `from_dlpack` is a
        // plain Python function, cheap to hold a reference to for the
        // duration of one block's tensors.
        let from_dlpack = py
            .import("torch")?
            .getattr("utils")?
            .getattr("dlpack")?
            .getattr("from_dlpack")?;

        let mut out = HashMap::with_capacity(meta.tensors.len());
        let block_pinned_start = meta
            .tensors
            .iter()
            .map(|t| t.pinned_offset)
            .min()
            .unwrap_or(0);
        for t in &meta.tensors {
            let within_block = t.pinned_offset - block_pinned_start;
            let device_ptr = slot_base + within_block;
            let dtype = dl_dtype(&t.name, t.dtype)?;
            let capsule = dlpack::make_cuda_capsule(
                py,
                device_ptr,
                self.ctx.ordinal() as i32,
                &t.shape,
                dtype,
            )?;
            let tensor = from_dlpack.call1((capsule,))?;
            out.insert(t.name.clone(), tensor.unbind());
        }
        Ok(out)
    }
}

fn block_pinned_start_offset(meta: &BlockMeta) -> usize {
    meta.tensors
        .iter()
        .map(|t| t.pinned_offset)
        .min()
        .unwrap_or(0) as usize
}

fn alloc_device(stream: &Arc<CudaStream>, bytes: u64) -> Result<u64> {
    let slice = stream.alloc_zeros::<u8>(bytes as usize)?;
    Ok(slice.leak())
}

impl Drop for RustEngine {
    fn drop(&mut self) {
        // MUST happen before freeing anything below. `alloc_device`
        // (used for every ring/shared/resident slot) zeroes its buffer
        // via an async memset on `transfer_stream` that `new()` never
        // waits for -- freeing a device pointer while that memset (or
        // any other pending async op targeting it) is still in flight is
        // a real, silent race, not a hypothetical one: reproduced by
        // creating a second `RustEngine` in the same process right after
        // dropping the first, with zero block fetches in between --
        // CUDA_ERROR_ILLEGAL_ADDRESS on the second engine's own init,
        // fixed by exactly this synchronize. Pre-existing gap (present
        // before ResidentPool), only ever surfaced once something
        // (stage 4's correctness test) actually created two engines
        // per process instead of one engine per OS process.
        let _ = self.transfer_stream.synchronize();
        for slot in self
            .slots
            .drain(..)
            .chain(self.shared_slot.take())
            .chain(self.resident.drain().map(|(_, slot)| slot))
        {
            let _ = unsafe { cu_result::free_sync(slot.device_ptr) };
        }
    }
}
