use pyo3::PyErr;
use pyo3::exceptions::PyRuntimeError;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Loader(#[from] streamloader::LoaderError),

    #[error("CUDA driver error: {0}")]
    Cuda(#[from] cudarc::driver::result::DriverError),

    #[error(
        "checkpoint needs {needed} pinned bytes but the configured budget is only {budget} bytes -- \
         refusing to start rather than silently caching only part of the model"
    )]
    BudgetExceeded { needed: u64, budget: u64 },

    #[error("tensor {name:?} has unsupported dtype {dtype:?} for GPU export")]
    UnsupportedDtype {
        name: String,
        dtype: streamloader::Dtype,
    },

    #[error("no VRAM slot currently holds block {0:?}")]
    BlockNotResident(String),

    #[error("unknown block id {0:?}")]
    UnknownBlock(String),

    #[error(transparent)]
    Python(#[from] PyErr),
}

pub type Result<T> = std::result::Result<T, EngineError>;

impl From<EngineError> for PyErr {
    fn from(e: EngineError) -> Self {
        PyRuntimeError::new_err(e.to_string())
    }
}
