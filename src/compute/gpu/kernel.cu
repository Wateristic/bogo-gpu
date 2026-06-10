#include <stdint.h>
#include <cuda_runtime.h>

// RNG

__device__ __forceinline__ uint64_t splitmix64(uint64_t *z) {
    *z += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t v = *z;
    v = (v ^ (v >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)) * UINT64_C(0x94d049bb133111eb);
    return v ^ (v >> 31);
}

__device__ __forceinline__ void make_rng(uint64_t seed, uint64_t index, uint32_t s[4]) {
    uint64_t z = seed + index * UINT64_C(0x9e3779b97f4a7c15);
    uint64_t a = splitmix64(&z);
    uint64_t b = splitmix64(&z);
    s[0] = (uint32_t)a;        s[1] = (uint32_t)(a >> 32);
    s[2] = (uint32_t)b;        s[3] = (uint32_t)(b >> 32);
    if (!s[0] && !s[1] && !s[2] && !s[3]) s[0] = 1;
}

__device__ __forceinline__ uint32_t rng_next(uint32_t s[4]) {
    const uint32_t result = __funnelshift_l(s[0] + s[3], s[0] + s[3], 7) + s[0];
    const uint32_t t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1];
    s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = __funnelshift_l(s[3], s[3], 11);
    return result;
}

// IMPORTANT: This must stay as rejection sampling to match the server's verifier.
// Do NOT replace with Lemire's method. Any RNG difference causes verify_mismatch.
__device__ __forceinline__ uint32_t rng_bounded(uint32_t s[4], const uint32_t n) {
    const uint32_t threshold = (uint32_t)(((uint64_t)1 << 32) % (uint64_t)n);
    uint32_t val;
    do { val = rng_next(s); } while (val < threshold);
    return val % n;
}

// Reduction

__device__ __forceinline__ void warp_reduce(
    uint32_t &best_c, uint64_t &best_i
) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        uint32_t oc = __shfl_xor_sync(0xffffffff, best_c, off);
        uint64_t oi = __shfl_xor_sync(0xffffffff, best_i, off);
        if (oc > best_c) { best_c = oc; best_i = oi; }
    }
}

// Kernel
//
// Outputs:
//   out_best_and_bid [1]         atomicMax target: (score << 32) | block_id
//   out_indices      [gridDim.x] best absolute shuffle index per block
//
// Host reads out_best_and_bid[0] once to get score + block_id, then indexes
// out_indices[block_id]. The host reconstructs the deck from seed + index.

extern "C" __global__
__launch_bounds__(256, 5)
void bogo_shuffle_kernel(
    uint64_t           seed,
    uint64_t           base_index,
    uint32_t           count,
    uint32_t           chunk_size,
    unsigned long long *out_best_and_bid,
    uint64_t           *out_indices
) {
    __shared__ uint32_t fys[25 * 256];
    __shared__ uint32_t block_threshold;

    if (threadIdx.x == 0) block_threshold = 0;
    __syncthreads();

    // Column-major: arr[p] == fys[p * blockDim.x + threadIdx.x].
    // For the current 256-thread launch this avoids 32-wide warp bank conflicts.
    uint32_t *arr = fys + threadIdx.x;

    const uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x  * blockDim.x;
    const uint64_t end    = base_index + (uint64_t)count;
    const uint32_t lane   = threadIdx.x & 31;
    const uint32_t warpid = threadIdx.x >> 5;

    uint32_t best_c = 0;
    uint64_t best_i = base_index + tid;
    const uint64_t t_start  = base_index + (uint64_t)tid;
    const uint32_t my_iters = (t_start >= end) ? 0u :
        (uint32_t)min((uint64_t)chunk_size,
                      (end - t_start + (uint64_t)stride - 1) / (uint64_t)stride);

    for (uint32_t c = 0; c < my_iters; c++) {
        uint32_t threshold = block_threshold;
        if (best_c > threshold) threshold = best_c;
        if (threshold >= 25) break;

        const uint64_t idx = t_start + (uint64_t)c * stride;

        uint32_t s[4];
        make_rng(seed, idx, s);

        #pragma unroll
        for (int p = 0; p < 25; p++) arr[p * 256] = (uint32_t)(p + 1);

        uint32_t fixed = 0;
        uint32_t active_mask = 0x01ffffffu;
        bool aborted = false;

        #pragma unroll
        for (int i = 24; i >= 1; i--) {
            const uint32_t j      = rng_bounded(s, (uint32_t)(i + 1));
            const uint32_t tmp    = arr[i * 256];
            const uint32_t placed = arr[j * 256];

            // Position i is finalized after this Fisher-Yates step and will
            // never be read again, so only the still-active position j needs
            // to be updated. The host reconstructs the winning deck from the
            // seed and index, so finalized deck slots do not need storing.
            arr[j * 256] = tmp;

            fixed += (placed == (uint32_t)(i + 1));
            active_mask &= ~(1u << (placed - 1u));

            // Only values still present in the active prefix can become future
            // fixed points. If even that upper bound cannot beat the current
            // block-local best, this candidate cannot affect the exact result.
            const uint32_t future_mask = (1u << (uint32_t)i) - 1u;
            const uint32_t possible_future = __popc(active_mask & future_mask);
            if (fixed + possible_future <= threshold) {
                aborted = true;
                break;
            }
        }

        if (aborted) continue;

        const uint32_t fp = fixed + (arr[0] == 1u);
        if (fp > best_c) {
            best_c = fp;
            best_i = idx;
            atomicMax(&block_threshold, fp);
            if (fp == 25) break;
        }
    }

    __syncthreads();

    warp_reduce(best_c, best_i);

    // Reuse fys for inter-warp staging.
    uint32_t *sh_c = fys;
    uint64_t *sh_i = (uint64_t *)(sh_c + 8);

    if (lane == 0) {
        sh_c[warpid] = best_c;
        sh_i[warpid] = best_i;
    }
    __syncthreads();

    if (warpid != 0) return;

    best_c = (lane < 8) ? sh_c[lane] : 0;
    best_i = (lane < 8) ? sh_i[lane] : base_index;
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        uint32_t oc = __shfl_xor_sync(0xffffffff, best_c, off);
        uint64_t oi = __shfl_xor_sync(0xffffffff, best_i, off);
        if (oc > best_c) { best_c = oc; best_i = oi; }
    }

    if (lane == 0) {
        const uint32_t blk = blockIdx.x;
        out_indices[blk] = best_i;
        atomicMax(out_best_and_bid,
            ((unsigned long long)best_c << 32) | (unsigned long long)blk);
    }
}
