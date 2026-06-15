// src/compute/amd/mod.rs  —  HIP/ROCm backend (--features hip)

use anyhow::{anyhow, Result};

#[cfg(feature = "hip")]
static HSACO: &[u8] = include_bytes!(env!("KERNEL_HSACO_PATH"));

#[allow(non_camel_case_types)] type hipError_t  = i32;
#[allow(non_camel_case_types)] type hipDevice_t = i32;
#[repr(C)]
#[derive(Clone, Copy)]
struct hipDeviceProp_t {
    name: [i8; 256],

    // pad out the rest of the struct
    rest: [u8; 4096],
}

#[repr(C)] #[derive(Clone, Copy)] struct hipDeviceptr_t(*mut std::ffi::c_void);
unsafe impl Send for hipDeviceptr_t {}
unsafe impl Sync for hipDeviceptr_t {}
#[repr(C)] #[derive(Clone, Copy)] struct hipModule_t(*mut std::ffi::c_void);
unsafe impl Send for hipModule_t {}
#[repr(C)] #[derive(Clone, Copy)] struct hipFunction_t(*mut std::ffi::c_void);
unsafe impl Send for hipFunction_t {}
#[repr(C)] #[derive(Clone, Copy)] struct hipStream_t(*mut std::ffi::c_void);
unsafe impl Send for hipStream_t {}

const HIP_SUCCESS: hipError_t = 0;

#[link(name = "amdhip64")]
extern "C" {
    fn hipInit(flags: u32) -> hipError_t;

    fn hipGetDeviceCount(count: *mut i32) -> hipError_t;

    fn hipSetDevice(device: i32) -> hipError_t;

    fn hipGetDeviceProperties(
        props: *mut hipDeviceProp_t,
        device: i32,
    ) -> hipError_t;

    fn hipMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> hipError_t;
    fn hipFree(ptr: *mut std::ffi::c_void) -> hipError_t;
    fn hipMemcpyDtoH(dst: *mut std::ffi::c_void, src: *mut std::ffi::c_void, size: usize) -> hipError_t;
    fn hipMemset(ptr: *mut std::ffi::c_void, value: i32, size: usize) -> hipError_t;
    fn hipModuleLoadData(module: *mut hipModule_t, image: *const std::ffi::c_void) -> hipError_t;
    fn hipModuleGetFunction(func: *mut hipFunction_t, module: hipModule_t, name: *const i8) -> hipError_t;
    fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_x: u32, grid_y: u32, grid_z: u32,
        block_x: u32, block_y: u32, block_z: u32,
        shared_mem: u32, stream: hipStream_t,
        kernel_params: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void,
    ) -> hipError_t;
    fn hipDeviceSynchronize() -> hipError_t;
}

macro_rules! hip_check {
    ($expr:expr) => {{
        let err = $expr;
        if err != HIP_SUCCESS {
            return Err(anyhow!("HIP error {} in {}", err, stringify!($expr)));
        }
    }};
}

pub struct HipWorker {
    module:         hipModule_t,
    func:           hipFunction_t,
    pub blocks:     u32,
    pub threads:    u32,
    pub chunk_size: u32,
    // Moving average: ring buffer of recent best scores for min_threshold
    // (mirrors CudaWorker so AMD gets the same pruning benefit)
    score_history:  [u32; 32],
    history_pos:    usize,
    history_count:  usize,
}

impl HipWorker {
    pub fn new(blocks: u32, threads: u32, chunk_size: u32) -> Result<Self> {
        unsafe {
            // These prints are intentionally unconditional (not gated behind
            // a verbose flag) — initialization only happens once, and if it
            // ever hangs, this is the only way to tell *which* HIP call is
            // stuck. Common culprits:
            //   - hipInit hanging: missing /dev/kfd or /dev/dri access
            //     (containers need --device=/dev/kfd --device=/dev/dri and
            //     the user in the `render`/`video` groups), or a broken
            //     amdgpu/ROCm driver install.
            //   - hipModuleLoadData hanging or taking minutes: the embedded
            //     .hsaco wasn't built for this GPU's exact gfx target, so
            //     ROCm's comgr falls back to an on-the-fly JIT recompile of
            //     the (heavily-unrolled) kernel. Check `rocminfo | grep gfx`
            //     vs. the --offload-arch the .hsaco was built with, and try
            //     `HSA_OVERRIDE_GFX_VERSION` if the GPU is close-but-unlisted.
            eprintln!("[hip] hipInit...");
hip_check!(hipInit(0));
eprintln!("[hip] hipInit ok");

let mut count = 0;

hip_check!(hipGetDeviceCount(&mut count));

println!("\nAvailable HIP devices:");

for i in 0..count {

    let mut props: hipDeviceProp_t =
        std::mem::zeroed();

    hip_check!(hipGetDeviceProperties(
        &mut props,
        i,
    ));

    let name =
        std::ffi::CStr::from_ptr(
            props.name.as_ptr()
        );

    println!(
        "  [{}] {}",
        i,
        name.to_string_lossy()
    );
}

use std::io::{self, Write};

let choice = loop {

    print!("Select device: ");

    io::stdout().flush().unwrap();

    let mut line = String::new();

    io::stdin()
        .read_line(&mut line)
        .unwrap();

    match line.trim().parse::<i32>() {

        Ok(n) if n >= 0 && n < count => {

            break n;
        }

        _ => {

            println!(
                "Please enter a number from 0 to {}",
                count - 1
            );
        }
    }
};

hip_check!(hipSetDevice(choice));

println!(
    "[hip] using device {}",
    choice,
);

eprintln!(
    "[hip] hipModuleLoadData ({} byte HSACO)...",
    HSACO.len()
);
            let mut module = hipModule_t(std::ptr::null_mut());
            hip_check!(hipModuleLoadData(&mut module, HSACO.as_ptr() as *const std::ffi::c_void));
            eprintln!("[hip] module loaded");

            let fn_name = b"bogo_shuffle_kernel\0";
            let mut func = hipFunction_t(std::ptr::null_mut());
            eprintln!("[hip] hipModuleGetFunction...");
            hip_check!(hipModuleGetFunction(&mut func, module, fn_name.as_ptr() as *const i8));
            eprintln!("[hip] kernel function ready");

            Ok(HipWorker {
                module, func, blocks, threads, chunk_size,
                score_history: [0u32; 32], history_pos: 0, history_count: 0,
            })
        }
    }

    fn record_score(&mut self, score: u32) {
        if score == 0 { return; }
        self.score_history[self.history_pos] = score;
        self.history_pos = (self.history_pos + 1) % 32;
        if self.history_count < 32 { self.history_count += 1; }
    }

    fn min_threshold(&self) -> u32 {
        if self.history_count < 8 { return 0; }
        let sum: u32 = self.score_history[..self.history_count].iter().sum();
        let avg = sum / self.history_count as u32;
        avg.saturating_sub(1)
    }

    pub fn run_batch(&mut self, seed: u64, base_index: u64, count: u32) -> Result<(u32, u64)> {
        let blocks = self.blocks as usize;
        // FIX 3: Compute min_threshold before entering unsafe so we can use &mut self freely.
        let min_threshold = self.min_threshold();

        let best_correct;
        let best_index_out;

        unsafe {
            let mut d_best_bid_raw:      *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_indices_raw:       *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_arrays_raw:        *mut std::ffi::c_void = std::ptr::null_mut();
            // FIX 1: allocate global_threshold device buffer (4 bytes).
            let mut d_global_thresh_raw: *mut std::ffi::c_void = std::ptr::null_mut();

            hip_check!(hipMalloc(&mut d_best_bid_raw,      8));              // 1 x u64 packed (score<<32)|blk
            hip_check!(hipMalloc(&mut d_indices_raw,       blocks * 8));     // blocks x u64
            hip_check!(hipMalloc(&mut d_arrays_raw,        blocks * 25));    // blocks x 25 x u8
            hip_check!(hipMalloc(&mut d_global_thresh_raw, 4));              // 1 x u32

            hip_check!(hipMemset(d_best_bid_raw,      0, 8));
            hip_check!(hipMemset(d_indices_raw,       0, blocks * 8));
            hip_check!(hipMemset(d_arrays_raw,        0, blocks * 25));
            // Initialise global_threshold to min_threshold so all blocks start pruning immediately.
            // hipMemset fills byte-by-byte; for values other than 0 we write via a host copy.
            // Since min_threshold is 0 when history is sparse, hipMemset(0) is correct for
            // that case. For non-zero values we do a 4-byte DtoH-style write via a tmp host var.
            if min_threshold == 0 {
                hip_check!(hipMemset(d_global_thresh_raw, 0, 4));
            } else {
                // hipMemcpyHtoD is the correct direction here (host → device).
                // We declare it inline to avoid polluting the extern block above.
                extern "C" {
                    fn hipMemcpyHtoD(
                        dst: *mut std::ffi::c_void,
                        src: *const std::ffi::c_void,
                        size: usize,
                    ) -> hipError_t;
                }
                let thresh_host = min_threshold;
                hip_check!(hipMemcpyHtoD(
                    d_global_thresh_raw,
                    &thresh_host as *const u32 as *const _,
                    4,
                ));
            }

            let mut seed_arg         = seed;
            let mut base_arg         = base_index;
            let mut count_arg        = count;
            let mut cs_arg           = self.chunk_size;
            let mut best_bid_arg     = d_best_bid_raw;
            let mut indices_arg      = d_indices_raw;
            let mut arrays_arg       = d_arrays_raw;
            // FIX 1: pass min_threshold and global_threshold pointer to match the
            // updated 9-argument kernel signature.
            let mut min_thresh_arg   = min_threshold;
            let mut global_thresh_arg = d_global_thresh_raw;

            let mut args: [*mut std::ffi::c_void; 9] = [
                &mut seed_arg          as *mut _ as *mut _,
                &mut base_arg          as *mut _ as *mut _,
                &mut count_arg         as *mut _ as *mut _,
                &mut cs_arg            as *mut _ as *mut _,
                &mut best_bid_arg      as *mut _ as *mut _,
                &mut indices_arg       as *mut _ as *mut _,
                &mut arrays_arg        as *mut _ as *mut _,
                &mut min_thresh_arg    as *mut _ as *mut _,
                &mut global_thresh_arg as *mut _ as *mut _,
            ];

            hip_check!(hipModuleLaunchKernel(
                self.func, self.blocks, 1, 1, self.threads, 1, 1,
                0, hipStream_t(std::ptr::null_mut()), args.as_mut_ptr(), std::ptr::null_mut(),
            ));
            hip_check!(hipDeviceSynchronize());

            // Read packed (score << 32) | block_id
            let mut best_and_bid = 0u64;
            hip_check!(hipMemcpyDtoH(
                &mut best_and_bid as *mut u64 as *mut _,
                d_best_bid_raw,
                8,
            ));

            best_correct          = (best_and_bid >> 32) as u32;
            let best_block        = (best_and_bid & 0xFFFF_FFFF) as usize;

            // Read just the winning block's index.
            let mut best_index = base_index;
            hip_check!(hipMemcpyDtoH(
                &mut best_index as *mut u64 as *mut _,
                (d_indices_raw as *mut u8).add(best_block * 8) as *mut _,
                8,
            ));
            best_index_out = best_index;

            hip_check!(hipFree(d_best_bid_raw));
            hip_check!(hipFree(d_indices_raw));
            hip_check!(hipFree(d_arrays_raw));
            hip_check!(hipFree(d_global_thresh_raw));
        }

        // Record score for the moving-average threshold (safe to call now, outside unsafe).
        self.record_score(best_correct);

        Ok((best_correct, best_index_out))
    }
}
