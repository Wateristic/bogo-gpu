// src/compute/amd/mod.rs  —  HIP/ROCm backend (--features hip)
 
use anyhow::{anyhow, Result};
 
#[cfg(feature = "hip")]
static HSACO: &[u8] = include_bytes!(env!("KERNEL_HSACO_PATH"));
 
#[allow(non_camel_case_types)] type hipError_t  = i32;
#[allow(non_camel_case_types)] type hipDevice_t = i32;
 
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
}
 
impl HipWorker {
    pub fn new(blocks: u32, threads: u32, chunk_size: u32) -> Result<Self> {
        unsafe {
            hip_check!(hipInit(0));
            let mut module = hipModule_t(std::ptr::null_mut());
            hip_check!(hipModuleLoadData(&mut module, HSACO.as_ptr() as *const std::ffi::c_void));
            let fn_name = b"bogo_shuffle_kernel\0";
            let mut func = hipFunction_t(std::ptr::null_mut());
            hip_check!(hipModuleGetFunction(&mut func, module, fn_name.as_ptr() as *const i8));
            Ok(HipWorker { module, func, blocks, threads, chunk_size })
        }
    }
 
    pub fn run_batch(&self, seed: u64, base_index: u64, count: u32) -> Result<(u32, u64)> {
        let blocks = self.blocks as usize;
        unsafe {
            let mut d_best_bid_raw: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_indices_raw:  *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_arrays_raw:   *mut std::ffi::c_void = std::ptr::null_mut();
 
            hip_check!(hipMalloc(&mut d_best_bid_raw, 8));              // 1 x u64 packed (score<<32)|blk
            hip_check!(hipMalloc(&mut d_indices_raw,  blocks * 8));     // blocks x u64
            hip_check!(hipMalloc(&mut d_arrays_raw,   blocks * 25));    // blocks x 25 x u8
 
            hip_check!(hipMemset(d_best_bid_raw, 0, 8));
            hip_check!(hipMemset(d_indices_raw,  0, blocks * 8));
            hip_check!(hipMemset(d_arrays_raw,   0, blocks * 25));
 
            let mut seed_arg     = seed;
            let mut base_arg     = base_index;
            let mut count_arg    = count;
            let mut cs_arg       = self.chunk_size;
            let mut best_bid_arg = d_best_bid_raw;
            let mut indices_arg  = d_indices_raw;
            let mut arrays_arg   = d_arrays_raw;
 
            let mut args: [*mut std::ffi::c_void; 7] = [
                &mut seed_arg     as *mut _ as *mut _,
                &mut base_arg     as *mut _ as *mut _,
                &mut count_arg    as *mut _ as *mut _,
                &mut cs_arg       as *mut _ as *mut _,
                &mut best_bid_arg as *mut _ as *mut _,
                &mut indices_arg  as *mut _ as *mut _,
                &mut arrays_arg   as *mut _ as *mut _,
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
 
            let best_correct = (best_and_bid >> 32) as u32;
            let best_block   = (best_and_bid & 0xFFFF_FFFF) as usize;
 
            // Read just the winning block's index
            let mut best_index = base_index;
            hip_check!(hipMemcpyDtoH(
                &mut best_index as *mut u64 as *mut _,
                (d_indices_raw as *mut u8).add(best_block * 8) as *mut _,
                8,
            ));
 
            hip_check!(hipFree(d_best_bid_raw));
            hip_check!(hipFree(d_indices_raw));
            hip_check!(hipFree(d_arrays_raw));
 
            Ok((best_correct, best_index))
        }
    }
}
