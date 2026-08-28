#pragma once
#include <cuda_runtime.h>

// Warp/block reduction helpers shared by the fused kernels.
// Reducing N accumulators together (instead of N separate calls) keeps the
// shared-memory traffic and the number of __syncthreads() barriers constant
// in N, which matters because the fused ops below reduce 3-5 scalars per row.

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

template <typename T, int N>
__device__ __forceinline__ void warpReduceSumN(T (&val)[N]) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            val[i] += __shfl_down_sync(kFullMask, val[i], offset);
        }
    }
}

// `shared` must hold N * (blockDim.x / kWarpSize) elements.
// On return every thread in the block holds the block-wide sums.
template <typename T, int N>
__device__ __forceinline__ void blockReduceSumN(T (&val)[N], T* shared) {
    const int lane = threadIdx.x % kWarpSize;
    const int wid = threadIdx.x / kWarpSize;
    const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

    warpReduceSumN<T, N>(val);

    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < N; ++i) shared[i * num_warps + wid] = val[i];
    }
    __syncthreads();

    // First warp folds the per-warp partials.
    if (wid == 0) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            val[i] = (lane < num_warps) ? shared[i * num_warps + lane] : T(0);
        }
        warpReduceSumN<T, N>(val);
        if (lane == 0) {
#pragma unroll
            for (int i = 0; i < N; ++i) shared[i * num_warps] = val[i];
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < N; ++i) val[i] = shared[i * num_warps];
    __syncthreads();
}
