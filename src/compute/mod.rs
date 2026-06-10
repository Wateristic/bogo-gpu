// src/compute/mod.rs

#[cfg(feature = "cuda")]
pub mod gpu;

#[cfg(feature = "hip")]
pub mod amd;

use anyhow::Result;
use crate::rng::{Rng, count_fixed_points};

// ── BatchResult ───────────────────────────────────────────────────────────────

pub struct BatchResult {
    pub best_correct: u32,
    pub best_arr:     [u8; 25],
    pub best_index:   u64,
}

// ── Backend ───────────────────────────────────────────────────────────────────

pub enum Backend {
    #[cfg(feature = "cuda")]
    Cuda(gpu::CudaWorker),

    #[cfg(feature = "hip")]
    Hip(amd::HipWorker),

    Cpu,
}

impl Backend {
    pub fn new_default() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            tracing::info!("Initialising CUDA backend");
            return Ok(Backend::Cuda(gpu::CudaWorker::new(2048, 256, 4096)?));
        }

        #[cfg(feature = "hip")]
        {
            tracing::info!("Initialising AMD HIP backend");
            return Ok(Backend::Hip(amd::HipWorker::new(2048, 256, 4096)?));
        }

        #[allow(unreachable_code)]
        {
            tracing::warn!("No GPU backend compiled in — falling back to CPU (rayon)");
            Ok(Backend::Cpu)
        }
    }

    pub fn run_batch(&mut self, seed: u64, base_index: u64, count: u32) -> Result<BatchResult> {
        match self {
            #[cfg(feature = "cuda")]
            Backend::Cuda(w) => {
                let (best_correct, best_index) =
                    w.run_batch(seed, base_index, count)?;
                let best_arr = Rng::new(seed, best_index).shuffle();
                Ok(BatchResult { best_correct, best_arr, best_index })
            }

            #[cfg(feature = "hip")]
            Backend::Hip(w) => {
                let (best_correct, best_index) = w.run_batch(seed, base_index, count)?;
                let best_arr = Rng::new(seed, best_index).shuffle();
                Ok(BatchResult { best_correct, best_arr, best_index })
            }

            Backend::Cpu => cpu_batch(seed, base_index, count),
        }
    }
}

// ── CPU fallback ──────────────────────────────────────────────────────────────

fn cpu_batch(seed: u64, base_index: u64, count: u32) -> Result<BatchResult> {
    use rayon::prelude::*;

    let (best_correct, best_index) = (0..count as u64)
        .into_par_iter()
        .map(|offset| {
            let i   = base_index + offset;
            let arr = Rng::new(seed, i).shuffle();
            (count_fixed_points(&arr), i)
        })
        .reduce(
            || (0u32, base_index),
            |a, b| if b.0 > a.0 { b } else { a },
        );

    let best_arr = Rng::new(seed, best_index).shuffle();
    Ok(BatchResult { best_correct, best_arr, best_index })
}
