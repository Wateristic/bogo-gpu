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

        // Link amdhip64.
        // Priority: HIP_LIB_DIR env var (explicit override) > ROCM_PATH env var > OS defaults.
        if let Ok(hip_lib_dir) = std::env::var("HIP_LIB_DIR") {
            // Explicit override always wins, on any OS.
            println!("cargo:rustc-link-search=native={hip_lib_dir}");
        } else if cfg!(target_os = "windows") {
            // On Windows the ROCm installer sets ROCM_PATH (e.g. C:\Program Files\AMD\ROCm\6.2).
            // The import library lives in <ROCM_PATH>\lib.  Fall back to a glob of the
            // standard install root in case ROCM_PATH wasn't set in the build environment.
            let rocm_path = std::env::var("ROCM_PATH")
                .unwrap_or_else(|_| r"C:\Program Files\AMD\ROCm\6.2".into());
            let lib_dir = format!(r"{rocm_path}\lib");
            if Path::new(&lib_dir).join("amdhip64.lib").exists() {
                println!("cargo:rustc-link-search=native={lib_dir}");
            } else {
                // Last-ditch: try every versioned ROCm directory under Program Files\AMD\ROCm.
                let base = Path::new(r"C:\Program Files\AMD\ROCm");
                let found = base.read_dir().ok().and_then(|mut rd| {
                    rd.find_map(|e| {
                        let e = e.ok()?;
                        let candidate = e.path().join("lib");
                        if candidate.join("amdhip64.lib").exists() {
                            Some(candidate.display().to_string())
                        } else {
                            None
                        }
                    })
                });
                match found {
                    Some(p) => println!("cargo:rustc-link-search=native={p}"),
                    None => panic!(
                        "amdhip64.lib not found. Install ROCm for Windows or set \
                         HIP_LIB_DIR to the directory containing amdhip64.lib."
                    ),
                }
            }
        } else {
            // Linux / macOS: prefer lib64 if it exists, otherwise lib.
            let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
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
