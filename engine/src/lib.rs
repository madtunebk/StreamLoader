use std::collections::HashMap;

use cudarc::driver::sys as cu_sys;
use cudarc::driver::{CudaContext, result as cu_result};
use pyo3::prelude::*;

mod dlpack;
mod engine;
mod error;

use engine::RustEngine;

/// Python-facing handle. Construction can fail loudly (unsupported dtype,
/// budget exceeded, no CUDA device, ...) -- there is deliberately no
/// fallback path here: if this raises, the caller must not proceed with
/// some other (e.g. plain-diffusers) loading path silently.
#[pyclass]
struct Engine {
    inner: RustEngine,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (path, budget_bytes, vram_slots=2, device_ordinal=0, trust_root=None, block_families=None, resident_budget_bytes=0))]
    fn new(
        path: &str,
        budget_bytes: u64,
        vram_slots: usize,
        device_ordinal: usize,
        trust_root: Option<&str>,
        block_families: Option<Vec<String>>,
        resident_budget_bytes: u64,
    ) -> PyResult<Self> {
        Ok(Engine {
            inner: RustEngine::new(
                path,
                budget_bytes,
                vram_slots,
                device_ordinal,
                trust_root,
                block_families,
                resident_budget_bytes,
            )?,
        })
    }

    fn block_ids(&self) -> Vec<String> {
        self.inner.block_ids()
    }

    /// Block ids the static ResidentPool selected at init (stage 3).
    /// Empty when `resident_budget_bytes` was 0 (the default).
    fn resident_block_ids(&self) -> Vec<String> {
        self.inner.resident_block_ids()
    }

    /// (block_id, cyclic_distance) for every non-shared block, relative
    /// to `from_block_id` -- the static farthest-next-use table a future
    /// ResidentPool eviction decision would consult. Purely a query over
    /// `block_order`; does not touch VRAM or issue any transfer.
    fn next_use_distances(&self, from_block_id: &str) -> PyResult<Vec<(String, u64)>> {
        Ok(self.inner.next_use_distances(from_block_id)?)
    }

    fn prefetch(&mut self, block_id: &str) -> PyResult<()> {
        self.inner.prefetch(block_id)?;
        Ok(())
    }

    fn get_block(
        &mut self,
        py: Python<'_>,
        block_id: &str,
        compute_stream_ptr: u64,
    ) -> PyResult<HashMap<String, Py<PyAny>>> {
        Ok(self.inner.get_block(py, block_id, compute_stream_ptr)?)
    }

    fn get_shared(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
        Ok(self.inner.get_shared(py)?)
    }

    fn mark_block_done(&mut self, block_id: &str, compute_stream_ptr: u64) -> PyResult<()> {
        self.inner.mark_block_done(block_id, compute_stream_ptr)?;
        Ok(())
    }

    fn stats(&self) -> HashMap<String, u64> {
        let s = self.inner.stats();
        HashMap::from([
            ("bytes_h2d".to_string(), s.bytes_h2d),
            ("transfer_count".to_string(), s.transfer_count),
            ("cache_hits".to_string(), s.cache_hits),
            ("pinned_bytes".to_string(), s.pinned_bytes),
            ("vram_bytes".to_string(), s.vram_bytes),
            ("resident_bytes".to_string(), s.resident_bytes),
            ("resident_hits".to_string(), s.resident_hits),
        ])
    }

    /// Per-transfer (block_id, bytes, resident, elapsed_ms) for every
    /// real H2D transfer issued so far this engine's lifetime. Blocks
    /// until all recorded transfers have completed (see
    /// `RustEngine::block_timings_ms`) -- call after a run, not inside a
    /// per-step loop.
    fn block_timings(&self) -> PyResult<Vec<(String, u64, bool, f64)>> {
        Ok(self.inner.block_timings_ms()?)
    }
}

/// Standalone correctness proof for the cross-stream handoff mechanism
/// the real engine depends on (see `test_dlpack.py`). Kept as a public
/// function so this specific mechanism can always be re-verified in
/// isolation, independent of the full engine.
#[pyfunction]
fn debug_dlpack_roundtrip(
    py: Python<'_>,
    fill_value: f32,
    n: usize,
    compute_stream_ptr: u64,
) -> PyResult<Py<PyAny>> {
    let ctx = CudaContext::new(0).map_err(to_pyerr)?;
    let transfer_stream = ctx.new_stream().map_err(to_pyerr)?;

    let mut pinned = unsafe { ctx.alloc_pinned::<f32>(n) }.map_err(to_pyerr)?;
    pinned.as_mut_slice().map_err(to_pyerr)?.fill(fill_value);

    let mut dev = transfer_stream.alloc_zeros::<f32>(n).map_err(to_pyerr)?;
    transfer_stream
        .memcpy_htod(&pinned, &mut dev)
        .map_err(to_pyerr)?;

    let ready_event = ctx.new_event(None).map_err(to_pyerr)?;
    ready_event.record(&transfer_stream).map_err(to_pyerr)?;

    let compute_stream = compute_stream_ptr as cu_sys::CUstream;
    unsafe {
        cu_result::stream::wait_event(
            compute_stream,
            ready_event.cu_event(),
            cu_sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
        )
    }
    .map_err(to_pyerr)?;

    let device_ptr = dev.leak();
    let shape = [n];
    dlpack::make_cuda_capsule(
        py,
        device_ptr,
        ctx.ordinal() as i32,
        &shape,
        dlpack::DlDtype::F32,
    )
}

fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

/// Isolated H2D bandwidth measurement: repeated pinned-host -> device
/// copies of `gb_size` GB, `iterations` times, with NO concurrent compute
/// -- unlike the engine's own `bytes_h2d / generation_time` figure (which
/// includes time spent overlapped with real transformer compute and is
/// therefore a lower bound, not a hardware measurement), this isolates
/// the transfer itself. Timed with CUDA events (GPU-side, not a CPU wall
/// clock wrapping an async call) for accuracy. Returns achieved GB/s.
#[pyfunction]
fn bench_h2d(gb_size: f64, iterations: u32) -> PyResult<f64> {
    let ctx = CudaContext::new(0).map_err(to_pyerr)?;
    let stream = ctx.new_stream().map_err(to_pyerr)?;
    let n_bytes = (gb_size * 1e9) as usize;

    let mut pinned = unsafe { ctx.alloc_pinned::<u8>(n_bytes) }.map_err(to_pyerr)?;
    pinned.as_mut_slice().map_err(to_pyerr)?.fill(0xAB);
    let mut dev = stream.alloc_zeros::<u8>(n_bytes).map_err(to_pyerr)?;

    // Warm-up copy, not timed (first transfer can include one-time driver
    // page-locking/mapping overhead that isn't representative).
    stream.memcpy_htod(&pinned, &mut dev).map_err(to_pyerr)?;
    stream.synchronize().map_err(to_pyerr)?;

    let timing_flags = Some(cu_sys::CUevent_flags::CU_EVENT_DEFAULT);
    let start = ctx.new_event(timing_flags).map_err(to_pyerr)?;
    let end = ctx.new_event(timing_flags).map_err(to_pyerr)?;

    start.record(&stream).map_err(to_pyerr)?;
    for _ in 0..iterations {
        stream.memcpy_htod(&pinned, &mut dev).map_err(to_pyerr)?;
    }
    end.record(&stream).map_err(to_pyerr)?;
    end.synchronize().map_err(to_pyerr)?;

    let elapsed_ms = start.elapsed_ms(&end).map_err(to_pyerr)?;
    let total_gb = gb_size * iterations as f64;
    Ok(total_gb / (elapsed_ms as f64 / 1000.0))
}

#[pymodule]
fn streamloader_engine(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Engine>()?;
    m.add_function(pyo3::wrap_pyfunction!(debug_dlpack_roundtrip, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(bench_h2d, m)?)?;
    Ok(())
}
