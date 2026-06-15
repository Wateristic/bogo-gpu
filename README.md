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
- [Settings](#settings)
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
 
> Note: the `CUDA_ARCH` build-time flag controls how the kernel is *compiled*. The `gpu_arch` value in the settings/config (below) is stored for reference and future use by the GUI — make sure it matches whatever you build with.
 
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
 
To find your exact arch: `rocminfo | grep gfx`
 
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
 
```sh
cargo run --release
```
 
Or run the compiled binary directly:
 
```sh
./target/release/bogo-gpu
```
 
This launches the GUI dashboard. On first run, with no saved config, you'll be prompted on the command line for your credentials (see below). On every subsequent run, the saved config is loaded automatically and the worker connects on its own — hit **▶ Start** in the dashboard to begin.
 
### Headless mode
 
For servers, containers, or anywhere a GUI isn't available (or wanted), pass `--headless` (or `-H`):
 
```sh
cargo run --release -- --headless
# or
./target/release/bogo-gpu --headless
```
 
A saved config is required (run once normally, or fill out `config.toml` by hand, before using headless mode — there's no GUI to prompt for credentials). In headless mode the worker connects and starts crunching immediately, with no Start button needed, and logs a status line (connection state, shuffles/sec, session/all-time bests, total shuffles) every 5 seconds. Press **Ctrl+C** to stop cleanly — it sends a `stop` message to the server before exiting.
 
Run `bogo-gpu --help` for a quick reference of all flags.
 
---
 
## First-Run Setup
 
The first time you run `bogo-gpu` with no saved config you'll see:
 
```
UUID: 
Nickname: 
Code: 
```
 
Enter the UUID, nickname, and code from your `bogo.swapjs.dev` account. These — along with default compute settings — are saved to your config file and loaded automatically on all future runs. You can change any of these later from the **Settings** tab in the GUI (see below).
 
---
 
## Settings
 
All configuration — identity and compute tuning — now lives in one place: the **Settings** tab in the GUI. Open it from the top bar, edit any field, and click **Save & Restart**:
 
- If the worker is currently running, it stops, saves your changes, and restarts immediately with the new settings.
- If the worker is stopped, your changes are just saved — press **▶ Start** when you're ready.
- Hit **Reset** to discard unsaved edits and revert the fields to the last-saved config.
The fields are pre-filled from your current config (or sensible defaults if a field was never set), so you can see exactly what's active before changing anything.
 
Editing `config.toml` by hand still works too — the GUI just reads/writes the same file.
 
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
backend     = "gpu"       # "gpu" or "cpu"
gpu_arch    = "gfx1201"    # informational — must match the arch you built with
                            # (e.g. "sm_86" for CUDA Ampere, "gfx1201" for RDNA4)
blocks      = 256          # number of GPU blocks (CUDA blocks / HIP blocks)
threads     = 256          # threads per block (must be a multiple of 32)
cpu_threads = 0             # CPU fallback thread count (0 = use all cores)
```
 
Only `[identity]` is required. The `[compute]` section is optional — if omitted, the defaults shown above (`backend = "gpu"`, `gpu_arch = "gfx1201"`, `blocks = 256`, `threads = 256`, `cpu_threads = 0`) are used.
 
`blocks` and `threads` only apply to the GPU backend. `cpu_threads` only applies when `backend = "cpu"` (or when no GPU backend was compiled in).
 
---
 
## Tuning for Your GPU
 
The two parameters that control GPU throughput are `blocks` and `threads` (total threads launched per kernel = `blocks × threads`). Edit these from the **Settings** tab and click **Save & Restart** — the dashboard's "shuffles / sec" readout updates within a few seconds, so it's easy to A/B different values live.
 
### NVIDIA
 
- **threads**: 256 is a solid default (4 warps × 32 threads). Going much higher (512+) increases register pressure and can *reduce* occupancy for this kernel — try it, but don't expect gains above 256.
- **blocks**: Aim for roughly 1–4× your GPU's SM count. Find your SM count via `nvidia-smi --query-gpu=name --format=csv` (look up the model) or `deviceQuery` from the CUDA samples.
Example for an RTX 3070 Ti (48 SMs):
```toml
[compute]
backend  = "gpu"
gpu_arch = "sm_86"
blocks   = 128   # ~2-3x SM count
threads  = 256
```
Other values worth A/B testing on a 3070 Ti: `blocks = 48, threads = 128` (1 block/SM, minimal overhead) or `blocks = 192, threads = 256` (aggressive latency hiding).
 
Example for an RTX 3090 (82 SMs):
```toml
[compute]
backend  = "gpu"
gpu_arch = "sm_86"
blocks   = 328   # 4x SM count
threads  = 256
```
 
Example for an RTX 4090 (128 SMs):
```toml
[compute]
backend  = "gpu"
gpu_arch = "sm_89"
blocks   = 512
threads  = 256
```
 
### AMD
 
The same logic applies. Find your CU count with `rocminfo | grep "Compute Unit"`.
 
Example for an RX 7900 XTX (96 CUs):
```toml
[compute]
backend  = "gpu"
gpu_arch = "gfx1100"
blocks   = 384   # 4x CU count
threads  = 256
```
 
---
 
## Architecture Overview
 
```
main.rs
  └── Worker::run()
        ├── waits for a Start command from the GUI (with current Config)
        ├── Backend::new_default()        — init GPU once per session
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
 
The GUI (`gui.rs`) and worker (`worker.rs`) communicate via `GuiStats`: the worker writes live stats (shuffles/sec, history, status) that the GUI reads every frame, and the GUI sends `WorkerCmd::Start(config)` / `WorkerCmd::Stop` back to the worker — this is what powers the Settings tab's "Save & Restart" and the dashboard's Start/Stop buttons.
 
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
 
**Low shuffle rate** — try increasing `blocks` in Settings and hitting Save & Restart, watching the "shuffles / sec" readout. Also watch GPU utilization with `nvidia-smi dmon` (NVIDIA) or `rocm-smi` (AMD); aim for >95% GPU busy.
 
**Settings won't save** — check the feedback message below the Save/Reset buttons in the Settings tab; it'll tell you which field failed validation (e.g. non-numeric Blocks/Threads, or an empty UUID/Nickname/Code).
