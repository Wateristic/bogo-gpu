// src/gpu.rs — thin shim that exposes GpuContext backed by the CUDA compute module.
// Drop this file in src/ alongside the existing src/compute/ directory.

pub use crate::compute::BatchResult;

pub struct GpuContext(crate::compute::Backend);

impl GpuContext {
    pub fn new() -> anyhow::Result<Self> {
        Ok(GpuContext(crate::compute::Backend::new_default()?))
    }

    pub fn run_batch(&self, seed: u64, base_index: u64, count: u32) -> anyhow::Result<BatchResult> {
        self.0.run_batch(seed, base_index, count)
    }
}
