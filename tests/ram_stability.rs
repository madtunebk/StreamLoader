//! Directly tests the concern that motivated this crate's design: the
//! old Python reference streamer's `RamBlockCache` never evicted (grows to
//! hold the whole model) and `PinnedBlockCache` kept a *second*, pinned
//! copy on top of it without freeing the first — so touching blocks
//! across many forward passes silently doubled (or worse) resident RAM.
//!
//! v1 has no such cache at all: `tensor_bytes`/`block_views` hand out
//! borrowed slices into the mmap, and `copy_block` returns a plain
//! caller-owned buffer that nothing internal retains. This is verified
//! here with a byte-counting global allocator (exact and deterministic),
//! not `/proc/self/status` RSS sampling (which is real but noisy/flaky
//! across environments) -- if the loader retained anything pass-to-pass,
//! net allocated bytes after N passes would grow with N; it must not.

mod support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};

use streamloader::{BlockConfig, Model};
use support::{flux_like_shards, write_sharded_model};

/// The counting allocator below is process-global, but Rust's default test
/// harness runs tests in this same binary concurrently on separate
/// threads. Without serializing, one test's unrelated allocations show up
/// as noise in the other's before/after delta. Both tests take this lock
/// first so only one of them is measuring at a time.
static TEST_LOCK: Mutex<()> = Mutex::new(());

struct CountingAllocator;

static NET_ALLOCATED: AtomicI64 = AtomicI64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            NET_ALLOCATED.fetch_add(layout.size() as i64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        NET_ALLOCATED.fetch_sub(layout.size() as i64, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn net_allocated() -> i64 {
    NET_ALLOCATED.load(Ordering::Relaxed)
}

#[test]
fn zero_copy_views_do_not_grow_heap_across_many_passes() {
    let _guard = TEST_LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let index_path = write_sharded_model(dir.path(), &flux_like_shards());
    let model = Model::open(&index_path).unwrap();
    let config = BlockConfig::flux();
    let block = model.block(&config, "transformer_blocks.2").unwrap();

    // Warm up: absorb any one-time allocation from the first touch (e.g.
    // page-table bookkeeping is kernel-side, not ours, but be conservative
    // about anything on our side too).
    for _ in 0..3 {
        let _ = model.block_views(&block).unwrap();
    }

    let before = net_allocated();
    const PASSES: u32 = 40;
    for _pass in 0..PASSES {
        let views = model.block_views(&block).unwrap();
        let mut touched: u64 = 0;
        for (_name, bytes) in &views {
            touched += bytes.len() as u64;
        }
        std::hint::black_box(touched);
        // `views` (a small Vec of borrowed slices) drops here every pass.
    }
    let after = net_allocated();

    assert_eq!(
        after, before,
        "reading the same block {PASSES} times must not grow net heap allocation \
         (simulates {PASSES} forward passes touching the same block; a 40GB model \
         touched 40 times must not cost anywhere near 40x its size in RAM)"
    );
}

#[test]
fn explicit_copy_api_retains_nothing_internally_across_passes() {
    let _guard = TEST_LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let index_path = write_sharded_model(dir.path(), &flux_like_shards());
    let model = Model::open(&index_path).unwrap();
    let config = BlockConfig::flux();
    let block = model.block(&config, "transformer_blocks.2").unwrap();

    for _ in 0..3 {
        let _ = model.copy_block(&block).unwrap();
    }

    let before = net_allocated();
    for _pass in 0..40 {
        let copied = model.copy_block(&block).unwrap();
        std::hint::black_box(copied.bytes_copied);
        // `copied.buffer` is caller-owned and drops here every pass --
        // there is no library-side cache for it to live in.
    }
    let after = net_allocated();

    assert_eq!(
        after, before,
        "Model must not retain a copy of any block internally between calls to copy_block"
    );
}
