//! Minimal DLPack producer: builds a `DLManagedTensor` over a CUDA device
//! pointer we own and hands it to Python as a capsule, following the
//! exact ownership-handoff convention every DLPack producer (numpy, jax,
//! torch itself) uses, so `torch.utils.dlpack.from_dlpack(capsule)` reads
//! it correctly.
//!
//! # The capsule dance (why this isn't just "wrap a pointer")
//!
//! A DLPack capsule is created named `"dltensor"`. If `from_dlpack`
//! successfully imports it, it renames the capsule to `"used_dltensor"`
//! and takes over responsibility for eventually calling
//! `DLManagedTensor::deleter` itself (when the resulting `torch.Tensor`
//! is garbage collected). If the capsule is instead dropped *without*
//! ever being imported (e.g. an exception between creating it and
//! calling `from_dlpack`), nothing else will ever call the deleter, so
//! the capsule's own destructor must call it itself — but only in that
//! case. Every producer distinguishes these two cases by checking the
//! capsule's *name* in its own destructor: still `"dltensor"` means never
//! consumed (we must clean up); renamed to `"used_dltensor"` means torch
//! now owns cleanup (we must not double-free). Getting this wrong is a
//! real double-free/leak bug, not a style nitpick, which is why it's
//! implemented against the raw `pyo3_ffi::PyCapsule_*` functions rather
//! than a higher-level wrapper that doesn't expose this distinction.
use std::ffi::{CStr, c_void};

use pyo3::PyErr;
use pyo3::prelude::*;

const DLTENSOR_NAME: &CStr = c"dltensor";

const DL_CUDA: i32 = 2;

#[repr(C)]
struct DlDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
struct DlDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DlTensor {
    data: *mut c_void,
    device: DlDevice,
    ndim: i32,
    dtype: DlDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DlManagedTensor {
    dl_tensor: DlTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DlManagedTensor)>,
}

/// What actually gets freed when a capsule/tensor is done with this
/// export: the heap-allocated shape array and the `DlManagedTensor`
/// itself (strides is always null -- every tensor this engine produces
/// is compact row-major, so there's no strides array to own). The
/// underlying device memory is NOT part of this -- it's owned by the
/// engine's `VramSlot` for the process lifetime, matching the
/// reuse-across-blocks design; a DLPack export is a *view*, never a
/// transfer of device-memory ownership.
struct Owned {
    shape: Box<[i64]>,
}

unsafe extern "C" fn deleter(managed: *mut DlManagedTensor) {
    if managed.is_null() {
        return;
    }
    let managed = unsafe { Box::from_raw(managed) };
    if !managed.manager_ctx.is_null() {
        drop(unsafe { Box::from_raw(managed.manager_ctx as *mut Owned) });
    }
}

unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let name_ptr = unsafe { pyo3::ffi::PyCapsule_GetName(capsule) };
    // If the name is still "dltensor", `from_dlpack` never consumed this
    // capsule (torch renames it to "used_dltensor" on successful import),
    // so nothing else will ever call our deleter -- we must call it here.
    // If it WAS renamed, torch now owns that call; calling it again here
    // would be a double free.
    let still_unclaimed = if name_ptr.is_null() {
        false
    } else {
        let name = unsafe { CStr::from_ptr(name_ptr) };
        name == DLTENSOR_NAME
    };
    if still_unclaimed {
        let ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, DLTENSOR_NAME.as_ptr()) };
        if !ptr.is_null() {
            unsafe { deleter(ptr as *mut DlManagedTensor) };
        }
    }
}

/// Element dtype for a DLPack export. Mirrors exactly the SafeTensors
/// dtypes this engine accepts -- anything else is rejected explicitly by
/// the caller before reaching here (see `engine.rs`), never silently
/// reinterpreted.
#[derive(Clone, Copy, Debug)]
pub enum DlDtype {
    F32,
    F16,
    Bf16,
    F64,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl DlDtype {
    fn code_bits(self) -> (u8, u8) {
        match self {
            DlDtype::F32 => (2, 32),
            DlDtype::F16 => (2, 16),
            DlDtype::Bf16 => (4, 16),
            DlDtype::F64 => (2, 64),
            DlDtype::I64 => (0, 64),
            DlDtype::I32 => (0, 32),
            DlDtype::I16 => (0, 16),
            DlDtype::I8 => (0, 8),
            DlDtype::U8 => (1, 8),
            DlDtype::Bool => (6, 8),
        }
    }
}

/// Build a DLPack capsule viewing `nbytes` bytes at `device_ptr` (a raw
/// CUDA device pointer, valid on `device_id`) as a tensor of `shape` and
/// `dtype`. `device_ptr`/the backing allocation must stay valid and must
/// not be reused for anything else until the caller-side CUDA event
/// protocol says it's safe (see `engine.rs`) -- this function only
/// handles the DLPack/Python object-lifetime side of ownership, not the
/// CUDA-stream-ordering side, which is a separate, explicit contract.
pub fn make_cuda_capsule(
    py: Python<'_>,
    device_ptr: u64,
    device_id: i32,
    shape: &[usize],
    dtype: DlDtype,
) -> PyResult<Py<PyAny>> {
    let shape_vec: Box<[i64]> = shape.iter().map(|&d| d as i64).collect();
    let (code, bits) = dtype.code_bits();
    let shape_ptr = shape_vec.as_ptr() as *mut i64;
    let owned = Box::new(Owned { shape: shape_vec });
    let manager_ctx = Box::into_raw(owned) as *mut c_void;

    let managed = Box::new(DlManagedTensor {
        dl_tensor: DlTensor {
            data: device_ptr as *mut c_void,
            device: DlDevice {
                device_type: DL_CUDA,
                device_id,
            },
            ndim: shape.len() as i32,
            dtype: DlDataType {
                code,
                bits,
                lanes: 1,
            },
            shape: shape_ptr,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx,
        deleter: Some(deleter),
    });
    let managed_ptr = Box::into_raw(managed);

    let capsule_ptr = unsafe {
        pyo3::ffi::PyCapsule_New(
            managed_ptr as *mut c_void,
            DLTENSOR_NAME.as_ptr(),
            Some(capsule_destructor),
        )
    };
    if capsule_ptr.is_null() {
        // Construction failed -- nothing owns `managed`/`manager_ctx` yet,
        // so we must free them ourselves rather than leak.
        unsafe { deleter(managed_ptr) };
        return Err(PyErr::fetch(py));
    }
    Ok(unsafe { Py::from_owned_ptr(py, capsule_ptr) })
}
