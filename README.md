# bogo-gpu

A distributed, GPU-accelerated shuffle worker that connects to `bogo.swapjs.dev` and searches for shuffle configurations with the maximum number of fixed points. Work is assigned by the server as leases; your GPU crunches them and reports results back over WebSocket.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Building](#building)
  - [NVIDIA CUDA (default)](#nvidia-cuda-default)
  - [AMD HIP / ROCm](#amd-hip--rocm)
  - [CPU-only (no GPU)](#cpu-only-no-gpu)
- [Running](#running)
- [First-Run Setup](#first-run-setup)
- [Configuration](#configuration)
  - [Config file location](#config-file-location)
  - [Full config reference](#full-config-reference)
- [Tuning for Your GPU](#tuning-for-your-gpu)
  - [NVIDIA](#nvidia)
  - [AMD](#amd)
- [Architecture Overview](#architecture-overview)
- [Troubleshooting](#troubleshooting)

---

## How It Works

1. The worker connects to `wss://bogo.swapjs.dev/ws` and identifies itself with your UUID, nickname, and code.
2. The server sends a **job** (a seed + count of shuffle indices to evaluate).
3. The GPU kernel runs xoshiro128++ RNG and Fisher-Yates shuffle for each index, counting fixed points, and returns the best result per block via an `atomicMax` reduction — O(1) host-side work per batch.
4. Results are streamed back to the server every second while a job is running.
5. The process repeats; the connection auto-reconnects with exponential backoff if it drops.

---

## Prerequisites

### All backends
- Rust toolchain: **1.75+** (`rustup update stable`)
- A registered account on `bogo.swapjs.dev` — you need a **UUID**, **nickname**, and **code** to authenticate

### NVIDIA CUDA
- NVIDIA GPU with compute capability **sm_75** (Turing) or newer
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) — `nvcc` must be on your `PATH`
- Verify with: `nvcc --version`

### AMD HIP / ROCm
- AMD GPU supported by ROCm (see [ROCm GPU support list](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html))
- [ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) installed — `hipcc` must be on your `PATH`
- Verify with: `hipcc --version`

---

## Building

### NVIDIA CUDA (default)

The default build targets NVIDIA GPUs using CUDA. `nvcc` compiles the kernel to PTX at build time; the PTX is embedded directly in the binary.

Build using the x64 Native Tools Command Prompt for VS 2022

```sh
# Build for the default GPU arch (sm_86, Ampere / RTX 30-series)
cargo build --release

# Override the GPU architecture if needed (see table below)
CUDA_ARCH=sm_89 cargo build --release   # Ada Lovelace — RTX 40-series
CUDA_ARCH=sm_86 cargo build --release   # Ampere — RTX 30-series (default)
CUDA_ARCH=sm_80 cargo build --release   # Ampere — A100
CUDA_ARCH=sm_75 cargo build --release   # Turing — RTX 20-series / T4
```

| GPU generation | Example cards | `CUDA_ARCH` |
|---|---|---|
| Ada Lovelace | RTX 4090, 4080, 4070 | `sm_89` |
| Ampere | RTX 3090, 3080, A100 | `sm_86` / `sm_80` |
| Turing | RTX 2080, T4 | `sm_75` |
| Volta | V100 | `sm_70` |

If you're unsure of your arch, run `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` and convert `X.Y` → `smXY`.

### AMD HIP / ROCm

The HIP build embeds a compiled HSACO binary. You must specify your GPU's architecture with `HIP_ARCH`.

```sh
# Disable the default CUDA feature and enable HIP
HIP_ARCH=gfx1201 ROCM_PATH=/opt/rocm \
  cargo build --release --no-default-features --features hip
```

If ROCm is installed to `/usr` instead of `/opt/rocm`:
```sh
HIP_ARCH=gfx1201 ROCM_PATH=/usr RUSTFLAGS="-L native=/usr/lib64" \
  cargo build --release --no-default-features --features hip
```

Common `HIP_ARCH` values:

| GPU generation | Example cards | `HIP_ARCH` |
|---|---|---|
| RDNA 4 | RX 9070 XT | `gfx1201` |
| RDNA 3 | RX 7900 XTX, 7800 XT | `gfx1100` / `gfx1103` |
| RDNA 2 | RX 6900 XT, 6700 XT | `gfx1030` |
| Vega 20 | Radeon VII, MI50 | `gfx906` |

Not listed? To find your exact arch: `rocminfo | grep gfx`

To bake the HIP paths into the project so you don't have to set env vars every time, uncomment and edit `.cargo/config.toml`:

```toml
[env]
HIP_ARCH  = "gfx1201"
ROCM_PATH = "/opt/rocm"

[target.x86_64-unknown-linux-gnu]
rustflags = ["-L", "native=/opt/rocm/lib"]
```

### CPU-only (no GPU)

Falls back to a parallel CPU implementation using Rayon. Much slower, but requires no GPU or driver.

```sh
cargo build --release --no-default-features
```

---

## Running

For CUDA:
```sh
cargo run --release
```
For AMD:
```sh
cargo run --release --no-default-features --features hip
```
Or run the compiled binary directly:

```sh
./target/release/bogo-gpu
```

On first launch you'll be prompted for your credentials (see below). After that the worker connects automatically on every subsequent run.

---

## First-Run Setup

The first time you run `bogo-gpu` with no saved config you'll see:

```
UUID: 
Nickname: 
Code: 
```

Enter the UUID, nickname, and code from your `bogo.swapjs.dev` account. These are saved to `~/.config/bogo-gpu/config.toml` and loaded automatically on all future runs.

---

## Configuration

### Config file location

| OS | Path |
|---|---|
| Linux | `~/.config/bogo-gpu/config.toml` |
| macOS | `~/Library/Application Support/bogo-gpu/config.toml` |
| Windows | `%APPDATA%\bogo-gpu\config.toml` |

### Full config reference

```toml
[identity]
uuid     = "your-uuid-here"
nickname = "your-nickname"
code     = "your-code-here"

[compute]
# GPU grid dimensions — total threads per launch = gpu_blocks × gpu_threads_per_block
gpu_blocks            = 1024   # number of CUDA/HIP blocks
gpu_threads_per_block = 256    # threads per block (must be a multiple of 32)
gpu_chunk_size        = 2048   # shuffles each thread evaluates per kernel launch

# CPU fallback chunk size (only used when no GPU backend is compiled in)
cpu_chunk_size        = 50000000
```

Only `[identity]` is required. The `[compute]` section is optional — defaults shown above are reasonable for most GPUs.

---

## Tuning for Your GPU

The three parameters that control throughput are `gpu_blocks`, `gpu_threads_per_block`, and `gpu_chunk_size`. Total shuffles per kernel launch = `gpu_blocks × gpu_threads_per_block × gpu_chunk_size`.

### NVIDIA

- **gpu_threads_per_block**: Keep at 256 (4 warps × 32 threads). Going higher rarely helps and wastes registers.
- **gpu_blocks**: Set to 2–4× your GPU's SM count. Find SM count with `nvidia-smi --query-gpu=name --format=csv` and look it up, or use `deviceQuery` from the CUDA samples.
- **gpu_chunk_size**: Increase this (e.g. 4096–8192) if your GPU is fast enough that kernel launch overhead becomes visible. Decrease it if you want more responsive reporting.

Example for an RTX 3090 (82 SMs):
```toml
gpu_blocks            = 328   # 4× SM count
gpu_threads_per_block = 256
gpu_chunk_size        = 4096
```

Example for an RTX 4090 (128 SMs):
```toml
gpu_blocks            = 512
gpu_threads_per_block = 256
gpu_chunk_size        = 8192
```

### AMD

The same logic applies. Find your CU count with `rocminfo | grep "Compute Unit"`.

Example for an RX 7900 XTX (96 CUs):
```toml
gpu_blocks            = 384   # 4× CU count
gpu_threads_per_block = 256
gpu_chunk_size        = 4096
```

---

## Architecture Overview

```
main.rs
  └── Worker::run()
        ├── Backend::new_default()        — init GPU once at startup
        ├── NetClient (async task)        — WebSocket to bogo.swapjs.dev
        │     receives: job leases
        │     sends:    result reports
        ├── Scheduler (async task)        — lease → chunk → report pipeline
        │     breaks leases into 2^31-shuffle chunks
        │     sends reports every 1 second
        └── run_compute_worker (blocking thread)
              drives GPU batch loop
              Backend::run_batch() per chunk
```

**Backends** (`src/compute/`):

- `gpu/` — CUDA via the `cust` crate. Triple-buffered kernel dispatch (one running, one doing async DMA, one being read). AtomicMax reduction means only one u64 read host-side to find the winning block per batch.
- `amd/` — HIP via raw `extern "C"` FFI against `libamdhip64`. Simpler single-launch design; host-side linear scan to find the best thread result.
- CPU — Rayon parallel iterator fallback.

**RNG** (`src/rng.rs`): Each shuffle index gets an independent seed via splitmix64 seed-expansion, then runs xoshiro128++. Bounded random uses rejection sampling (not Lemire's method) — this is required to match the server's verifier exactly. Do not change this.

---

## Troubleshooting

**`nvcc not found`** — install the CUDA Toolkit and ensure `nvcc` is on `PATH`. On Linux: `export PATH=/usr/local/cuda/bin:$PATH`.

**`hipcc not found`** — install ROCm. On Ubuntu: `sudo apt install rocm-dev`. Ensure `/opt/rocm/bin` is on `PATH`.

**Wrong GPU arch / PTX JIT errors at runtime** — set `CUDA_ARCH` correctly for your card before building (see table above). Run `nvidia-smi` to confirm the GPU model.

**`HIP error` at launch** — confirm `HIP_ARCH` matches your GPU exactly (`rocminfo | grep gfx`) and that `libamdhip64.so` is findable (`ldconfig -p | grep amdhip64`).

**`verify_mismatch` rejections from server** — the RNG in `src/rng.rs` has been modified. The server's verifier is exact. Do not change the RNG or shuffle logic.

**Connection drops / reconnects** — normal; the worker reconnects automatically with exponential backoff (2s → 4s → 8s → … → 30s max). Check your internet connection if it never reconnects.

**Low shuffle rate** — increase `gpu_chunk_size` to keep the GPU saturated. Watch GPU utilization with `nvidia-smi dmon` (NVIDIA) or `rocm-smi` (AMD); aim for >95% GPU busy.
