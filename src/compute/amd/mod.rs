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
        let total = (self.blocks * self.threads) as usize;
        unsafe {
            let mut d_correct_raw:  *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_index_lo_raw: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_index_hi_raw: *mut std::ffi::c_void = std::ptr::null_mut();
            hip_check!(hipMalloc(&mut d_correct_raw,  total * 4));
            hip_check!(hipMalloc(&mut d_index_lo_raw, total * 4));
            hip_check!(hipMalloc(&mut d_index_hi_raw, total * 4));
            hip_check!(hipMemset(d_correct_raw,  0, total * 4));
            hip_check!(hipMemset(d_index_lo_raw, 0, total * 4));
            hip_check!(hipMemset(d_index_hi_raw, 0, total * 4));
            let mut seed_arg     = seed;
            let mut base_arg     = base_index;
            let mut count_arg    = count;
            let mut cs_arg       = self.chunk_size;
            let mut correct_arg  = d_correct_raw;
            let mut index_lo_arg = d_index_lo_raw;
            let mut index_hi_arg = d_index_hi_raw;
            let mut args: [*mut std::ffi::c_void; 7] = [
                &mut seed_arg     as *mut _ as *mut _,
                &mut base_arg     as *mut _ as *mut _,
                &mut count_arg    as *mut _ as *mut _,
                &mut cs_arg       as *mut _ as *mut _,
                &mut correct_arg  as *mut _ as *mut _,
                &mut index_lo_arg as *mut _ as *mut _,
                &mut index_hi_arg as *mut _ as *mut _,
            ];
            hip_check!(hipModuleLaunchKernel(
                self.func, self.blocks, 1, 1, self.threads, 1, 1,
                0, hipStream_t(std::ptr::null_mut()), args.as_mut_ptr(), std::ptr::null_mut(),
            ));
            hip_check!(hipDeviceSynchronize());
            let mut correct  = vec![0u32; total];
            let mut index_lo = vec![0u32; total];
            let mut index_hi = vec![0u32; total];
            hip_check!(hipMemcpyDtoH(correct.as_mut_ptr()  as *mut _, d_correct_raw,  total * 4));
            hip_check!(hipMemcpyDtoH(index_lo.as_mut_ptr() as *mut _, d_index_lo_raw, total * 4));
            hip_check!(hipMemcpyDtoH(index_hi.as_mut_ptr() as *mut _, d_index_hi_raw, total * 4));
            hip_check!(hipFree(d_correct_raw));
            hip_check!(hipFree(d_index_lo_raw));
            hip_check!(hipFree(d_index_hi_raw));
            let mut best_correct = 0u32;
            let mut best_index   = base_index;
            for i in 0..total {
                if correct[i] > best_correct {
                    best_correct = correct[i];
                    best_index   = (index_lo[i] as u64) | ((index_hi[i] as u64) << 32);
                }
            }
            Ok((best_correct, best_index))
        }
    }
}
