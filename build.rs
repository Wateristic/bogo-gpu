fn main() {
    #[cfg(feature = "cuda")]
    cuda::compile();

    #[cfg(feature = "hip")]
    hip::compile();
}

// ─── CUDA ────────────────────────────────────────────────────────────────────
#[cfg(feature = "cuda")]
mod cuda {
    use std::path::PathBuf;
    use std::process::Command;

    pub fn compile() {
        let kernel_src = PathBuf::from("src/compute/gpu/kernel.cu");
        let out_dir    = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let ptx_out    = out_dir.join("kernel.ptx");

        println!("cargo:rerun-if-changed=src/compute/gpu/kernel.cu");
        println!("cargo:rerun-if-env-changed=CUDA_ARCH");

        if !kernel_src.exists() {
            panic!(
                "src/compute/gpu/kernel.cu not found. \
                 Create the file or switch to --features hip / --features wgpu."
            );
        }

        let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_86".into());
        println!("cargo:warning=compiling kernel.cu with -arch={arch}");

        let output = Command::new("nvcc")
            .args([
                "--ptx",
                &format!("-arch={arch}"),
                "-O3",
                "--use_fast_math",
                kernel_src.to_str().unwrap(),
                "-o",
                ptx_out.to_str().unwrap(),
            ])
            .output()
            .expect("nvcc not found — install the CUDA Toolkit and make sure nvcc is on PATH");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("nvcc PTX compilation failed:\n{stderr}");
        }

        println!("cargo:warning=PTX written to {}", ptx_out.display());
        println!("cargo:rustc-env=KERNEL_PTX_PATH={}", ptx_out.display());
    }
}

// ─── HIP / ROCm ──────────────────────────────────────────────────────────────
#[cfg(feature = "hip")]
mod hip {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    pub fn compile() {
        let kernel_src = PathBuf::from("src/compute/amd/kernel.hip");
        let out_dir    = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let hsaco_out  = out_dir.join("kernel.hsaco");

        println!("cargo:rerun-if-changed=src/compute/amd/kernel.hip");
        println!("cargo:rerun-if-env-changed=HIP_ARCH");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");

        if !kernel_src.exists() {
            panic!(
                "src/compute/amd/kernel.hip not found. \
                 Create the file or switch to --features cuda / --features wgpu."
            );
        }

        let arch = std::env::var("HIP_ARCH").unwrap_or_else(|_| "gfx1201".into());
        println!("cargo:warning=compiling kernel.hip for --offload-arch={arch}");

        let output = Command::new("hipcc")
            .args([
                "--genco",
                &format!("--offload-arch={arch}"),
                "-O3",
                "-ffast-math",
                "-mno-wavefrontsize64",
                "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
                kernel_src.to_str().unwrap(),
                "-o",
                hsaco_out.to_str().unwrap(),
            ])
            .output()
            .expect("hipcc not found — install ROCm and make sure hipcc is on PATH");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("hipcc HSACO compilation failed:\n{stderr}");
        }

        // Link libamdhip64
        let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        if let Ok(hip_lib_dir) = std::env::var("HIP_LIB_DIR") {
            println!("cargo:rustc-link-search=native={hip_lib_dir}");
        } else {
            let rocm_lib64 = format!("{rocm_path}/lib64");
            let rocm_lib   = format!("{rocm_path}/lib");
            if Path::new(&rocm_lib64).join("libamdhip64.so").exists() {
                println!("cargo:rustc-link-search=native={rocm_lib64}");
            } else {
                println!("cargo:rustc-link-search=native={rocm_lib}");
            }
        }
        println!("cargo:rustc-link-lib=amdhip64");

        println!("cargo:warning=HSACO written to {}", hsaco_out.display());
        println!("cargo:rustc-env=KERNEL_HSACO_PATH={}", hsaco_out.display());
    }
}
