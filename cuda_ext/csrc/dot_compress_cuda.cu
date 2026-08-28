// Fused DotCompress interaction kernel.
//
// The reference implementation (sparsenn.py::DotCompressScoringModel) computes,
// for a batch of user/item embedding pairs (u, v) in R^D with W in R^{2 x H},
// bias b in R^H and H = D/2:
//
//     E = stack(u, v)                       [2, D]
//     M = E^T @ W + b                       [D, H]     <-- materialised, O(D*H)
//     y = flatten(E @ M)                    [2H] = [D]
//
// The [B, D, H] intermediate M dominates both runtime and memory: for D=192 it
// is 18,432 floats *per sample*, i.e. 36 MB at batch 512 in fp32, and it is
// written and immediately re-read. But M is rank-structured, and expanding the
// product collapses it entirely:
//
//     y[0, j] = sum_d u_d * (u_d W[0,j] + v_d W[1,j] + b_j)
//             = (u.u) W[0,j] + (u.v) W[1,j] + (sum u) b_j
//     y[1, j] = (u.v) W[0,j] + (v.v) W[1,j] + (sum v) b_j
//
// So the entire op is five per-row scalars -- p=u.u, q=u.v, r=v.v, su=sum(u),
// sv=sum(v) -- followed by a rank-2 broadcast. That drops the arithmetic from
// O(B*D*H) to O(B*D) and removes the intermediate tensor outright. This file
// implements that identity as a single fused kernel plus a hand-derived
// backward pass. Correctness against the reference is asserted in
// tests/test_ops.py (allclose + torch.autograd.gradcheck in float64).

#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "reduce.cuh"

namespace {

constexpr int kFwdBlock = 256;
constexpr int kBwdBlock = 256;

// One block per batch row. Computes the five reduction scalars over D, caches
// them for backward, then broadcasts them across the H output columns.
template <typename scalar_t>
__global__ void dot_compress_forward_kernel(
    const scalar_t* __restrict__ U,      // [B, D]
    const scalar_t* __restrict__ V,      // [B, D]
    const scalar_t* __restrict__ W,      // [2, H]
    const scalar_t* __restrict__ bias,   // [H]
    scalar_t* __restrict__ out,          // [B, 2H]
    scalar_t* __restrict__ stats,        // [B, 5] -> (p, q, r, su, sv)
    const int B, const int D, const int H) {
    using acc_t = at::acc_type<scalar_t, true>;

    const int row = blockIdx.x;
    if (row >= B) return;

    const scalar_t* u = U + static_cast<long>(row) * D;
    const scalar_t* v = V + static_cast<long>(row) * D;

    acc_t acc[5] = {0, 0, 0, 0, 0};
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const acc_t ud = static_cast<acc_t>(u[d]);
        const acc_t vd = static_cast<acc_t>(v[d]);
        acc[0] += ud * ud;   // p
        acc[1] += ud * vd;   // q
        acc[2] += vd * vd;   // r
        acc[3] += ud;        // su
        acc[4] += vd;        // sv
    }

    __shared__ acc_t smem[5 * (kFwdBlock / kWarpSize)];
    blockReduceSumN<acc_t, 5>(acc, smem);

    const acc_t p = acc[0], q = acc[1], r = acc[2], su = acc[3], sv = acc[4];

    // Written by one thread with compile-time indices: dynamic indexing into
    // `acc` would spill the whole array from registers to local memory.
    if (threadIdx.x == 0) {
        scalar_t* s = stats + static_cast<long>(row) * 5;
        s[0] = static_cast<scalar_t>(p);
        s[1] = static_cast<scalar_t>(q);
        s[2] = static_cast<scalar_t>(r);
        s[3] = static_cast<scalar_t>(su);
        s[4] = static_cast<scalar_t>(sv);
    }

    scalar_t* o = out + static_cast<long>(row) * 2 * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        const acc_t w0 = static_cast<acc_t>(W[j]);
        const acc_t w1 = static_cast<acc_t>(W[H + j]);
        const acc_t bj = static_cast<acc_t>(bias[j]);
        o[j]     = static_cast<scalar_t>(p * w0 + q * w1 + su * bj);
        o[H + j] = static_cast<scalar_t>(q * w0 + r * w1 + sv * bj);
    }
}

// Backward w.r.t. the activations. One block per row; reduces the incoming
// gradient against W and b to recover d(p,q,r,su,sv), then applies the chain
// rule elementwise. Deterministic -- no atomics.
template <typename scalar_t>
__global__ void dot_compress_backward_input_kernel(
    const scalar_t* __restrict__ gout,   // [B, 2H]
    const scalar_t* __restrict__ U,      // [B, D]
    const scalar_t* __restrict__ V,      // [B, D]
    const scalar_t* __restrict__ W,      // [2, H]
    const scalar_t* __restrict__ bias,   // [H]
    scalar_t* __restrict__ gU,           // [B, D]
    scalar_t* __restrict__ gV,           // [B, D]
    const int B, const int D, const int H) {
    using acc_t = at::acc_type<scalar_t, true>;

    const int row = blockIdx.x;
    if (row >= B) return;

    const scalar_t* g = gout + static_cast<long>(row) * 2 * H;

    acc_t acc[5] = {0, 0, 0, 0, 0};
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        const acc_t g0 = static_cast<acc_t>(g[j]);
        const acc_t g1 = static_cast<acc_t>(g[H + j]);
        const acc_t w0 = static_cast<acc_t>(W[j]);
        const acc_t w1 = static_cast<acc_t>(W[H + j]);
        const acc_t bj = static_cast<acc_t>(bias[j]);
        acc[0] += g0 * w0;              // dp
        acc[1] += g0 * w1 + g1 * w0;    // dq
        acc[2] += g1 * w1;              // dr
        acc[3] += g0 * bj;              // dsu
        acc[4] += g1 * bj;              // dsv
    }

    __shared__ acc_t smem[5 * (kBwdBlock / kWarpSize)];
    blockReduceSumN<acc_t, 5>(acc, smem);

    const acc_t dp = acc[0], dq = acc[1], dr = acc[2], dsu = acc[3], dsv = acc[4];

    const scalar_t* u = U + static_cast<long>(row) * D;
    const scalar_t* v = V + static_cast<long>(row) * D;
    scalar_t* du = gU + static_cast<long>(row) * D;
    scalar_t* dv = gV + static_cast<long>(row) * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const acc_t ud = static_cast<acc_t>(u[d]);
        const acc_t vd = static_cast<acc_t>(v[d]);
        du[d] = static_cast<scalar_t>(acc_t(2) * dp * ud + dq * vd + dsu);
        dv[d] = static_cast<scalar_t>(acc_t(2) * dr * vd + dq * ud + dsv);
    }
}

// Backward w.r.t. the parameters. One block per output column j, reducing over
// the batch. Also deterministic, and avoids B*3H atomicAdds into a tiny
// parameter tensor (which is where the naive version serialises).
template <typename scalar_t>
__global__ void dot_compress_backward_param_kernel(
    const scalar_t* __restrict__ gout,   // [B, 2H]
    const scalar_t* __restrict__ stats,  // [B, 5]
    scalar_t* __restrict__ gW,           // [2, H]
    scalar_t* __restrict__ gb,           // [H]
    const int B, const int H) {
    using acc_t = at::acc_type<scalar_t, true>;

    const int j = blockIdx.x;
    if (j >= H) return;

    acc_t acc[3] = {0, 0, 0};
    for (int row = threadIdx.x; row < B; row += blockDim.x) {
        const scalar_t* g = gout + static_cast<long>(row) * 2 * H;
        const scalar_t* s = stats + static_cast<long>(row) * 5;
        const acc_t g0 = static_cast<acc_t>(g[j]);
        const acc_t g1 = static_cast<acc_t>(g[H + j]);
        const acc_t p = static_cast<acc_t>(s[0]);
        const acc_t q = static_cast<acc_t>(s[1]);
        const acc_t r = static_cast<acc_t>(s[2]);
        const acc_t su = static_cast<acc_t>(s[3]);
        const acc_t sv = static_cast<acc_t>(s[4]);
        acc[0] += g0 * p + g1 * q;      // dW[0, j]
        acc[1] += g0 * q + g1 * r;      // dW[1, j]
        acc[2] += g0 * su + g1 * sv;    // db[j]
    }

    __shared__ acc_t smem[3 * (kBwdBlock / kWarpSize)];
    blockReduceSumN<acc_t, 3>(acc, smem);

    if (threadIdx.x == 0) {
        gW[j] = static_cast<scalar_t>(acc[0]);
        gW[H + j] = static_cast<scalar_t>(acc[1]);
        gb[j] = static_cast<scalar_t>(acc[2]);
    }
}

}  // namespace

std::vector<torch::Tensor> dot_compress_forward_cuda(
    torch::Tensor U, torch::Tensor V, torch::Tensor W, torch::Tensor bias) {
    const at::cuda::CUDAGuard guard(U.device());

    const int B = U.size(0);
    const int D = U.size(1);
    const int H = W.size(1);
    TORCH_CHECK(2 * H == D, "DotCompress expects W of shape [2, D/2]; got D=", D, " H=", H);

    auto out = torch::empty({B, 2 * H}, U.options());
    auto stats = torch::empty({B, 5}, U.options());
    if (B == 0) return {out, stats};

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(U.scalar_type(), "dot_compress_forward", [&] {
        dot_compress_forward_kernel<scalar_t><<<B, kFwdBlock, 0, stream>>>(
            U.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(), W.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), stats.data_ptr<scalar_t>(),
            B, D, H);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, stats};
}

std::vector<torch::Tensor> dot_compress_backward_cuda(
    torch::Tensor gout, torch::Tensor U, torch::Tensor V, torch::Tensor W,
    torch::Tensor bias, torch::Tensor stats) {
    const at::cuda::CUDAGuard guard(U.device());

    const int B = U.size(0);
    const int D = U.size(1);
    const int H = W.size(1);

    auto gU = torch::empty_like(U);
    auto gV = torch::empty_like(V);
    auto gW = torch::empty_like(W);
    auto gb = torch::empty_like(bias);
    if (B == 0) {
        gW.zero_();
        gb.zero_();
        return {gU, gV, gW, gb};
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(U.scalar_type(), "dot_compress_backward", [&] {
        dot_compress_backward_input_kernel<scalar_t><<<B, kBwdBlock, 0, stream>>>(
            gout.data_ptr<scalar_t>(), U.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(),
            W.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
            gU.data_ptr<scalar_t>(), gV.data_ptr<scalar_t>(), B, D, H);
        dot_compress_backward_param_kernel<scalar_t><<<H, kBwdBlock, 0, stream>>>(
            gout.data_ptr<scalar_t>(), stats.data_ptr<scalar_t>(),
            gW.data_ptr<scalar_t>(), gb.data_ptr<scalar_t>(), B, H);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gU, gV, gW, gb};
}
