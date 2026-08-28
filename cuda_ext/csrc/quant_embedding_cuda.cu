// Fused INT8 embedding gather + dequantise + concat.
//
// The item tower (sparsenn.py::TrackSparseNNItemModel) does four sparse lookups
// and concatenates them:
//
//     cat([emb_id(i), emb_artist(a), emb_tag(t), dense(names)], dim=1)
//
// In eager PyTorch that is four gather kernels, four intermediate [B, d]
// tensors, and a fifth kernel for the cat that re-reads and re-writes every
// byte. Embedding lookup is pure memory movement, so the extra pass is close to
// a 2x tax on the dominant cost of the tower.
//
// This kernel does all of it in one launch, writing each dequantised row
// directly into its final column slice of the output. Tables are stored INT8
// with a per-row (per-embedding-vector) fp32 scale, which is the right
// granularity here: embedding rows are independently learned and their dynamic
// ranges differ by orders of magnitude between head and tail items, so a single
// per-tensor scale would crush the tail -- exactly the cold-start items the
// paper cares about. Per-row scaling cuts table memory 4x and, because the op
// is bandwidth-bound, buys most of that back as speed.
//
// Inference only: quantisation is applied post-training to a trained
// checkpoint, so no backward pass is defined.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>

namespace {

constexpr int kMaxTables = 8;
constexpr int kBlock = 256;

struct TableDesc {
    const int8_t* data;    // [num_rows, dim], row-major
    const float* scale;    // [num_rows]
    const int64_t* idx;    // [B] row ids to gather
    int dim;
    int col_offset;        // destination column in the concatenated output
};

struct TableDescPack {
    TableDesc t[kMaxTables];
};

template <typename scalar_t>
__global__ void gather_dequant_concat_kernel(
    const TableDescPack pack,
    scalar_t* __restrict__ out,   // [B, out_stride]
    const int B, const int out_stride) {
    const TableDesc desc = pack.t[blockIdx.y];
    const long total = static_cast<long>(B) * desc.dim;

    for (long e = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         e < total; e += static_cast<long>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(e / desc.dim);
        const int d = static_cast<int>(e - static_cast<long>(row) * desc.dim);
        const int64_t src = desc.idx[row];
        const float s = desc.scale[src];
        const int8_t q = desc.data[src * desc.dim + d];
        out[static_cast<long>(row) * out_stride + desc.col_offset + d] =
            static_cast<scalar_t>(s * static_cast<float>(q));
    }
}

// dim % 4 == 0 fast path: one char4 load per thread instead of four scalar
// loads, which is what actually saturates the memory pipe on a gather.
template <typename scalar_t>
__global__ void gather_dequant_concat_vec4_kernel(
    const TableDescPack pack,
    scalar_t* __restrict__ out,
    const int B, const int out_stride) {
    const TableDesc desc = pack.t[blockIdx.y];
    const int dim4 = desc.dim >> 2;
    const long total = static_cast<long>(B) * dim4;

    for (long e = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         e < total; e += static_cast<long>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(e / dim4);
        const int d4 = static_cast<int>(e - static_cast<long>(row) * dim4);
        const int64_t src = desc.idx[row];
        const float s = desc.scale[src];
        const char4 q = reinterpret_cast<const char4*>(desc.data + src * desc.dim)[d4];
        scalar_t* dst = out + static_cast<long>(row) * out_stride + desc.col_offset + (d4 << 2);
        dst[0] = static_cast<scalar_t>(s * static_cast<float>(q.x));
        dst[1] = static_cast<scalar_t>(s * static_cast<float>(q.y));
        dst[2] = static_cast<scalar_t>(s * static_cast<float>(q.z));
        dst[3] = static_cast<scalar_t>(s * static_cast<float>(q.w));
    }
}

}  // namespace

// tables/scales/indices are parallel lists, one entry per embedding table.
torch::Tensor fused_quant_embedding_concat_cuda(
    std::vector<torch::Tensor> tables,
    std::vector<torch::Tensor> scales,
    std::vector<torch::Tensor> indices,
    c10::ScalarType out_dtype) {
    TORCH_CHECK(!tables.empty(), "need at least one table");
    TORCH_CHECK(tables.size() <= kMaxTables, "at most ", kMaxTables, " tables per launch");
    TORCH_CHECK(tables.size() == scales.size() && tables.size() == indices.size(),
                "tables/scales/indices must be the same length");

    const at::cuda::CUDAGuard guard(tables[0].device());
    const int B = indices[0].size(0);

    TableDescPack pack;
    int out_stride = 0;
    bool all_vec4 = true;
    for (size_t i = 0; i < tables.size(); ++i) {
        TORCH_CHECK(tables[i].scalar_type() == torch::kChar, "table ", i, " must be int8");
        TORCH_CHECK(tables[i].is_contiguous() && scales[i].is_contiguous(), "table ", i, " must be contiguous");
        TORCH_CHECK(indices[i].scalar_type() == torch::kLong, "indices ", i, " must be int64");
        TORCH_CHECK(indices[i].size(0) == B, "all index tensors must share batch size");
        const int dim = tables[i].size(1);
        pack.t[i] = TableDesc{tables[i].data_ptr<int8_t>(), scales[i].data_ptr<float>(),
                              indices[i].data_ptr<int64_t>(), dim, out_stride};
        out_stride += dim;
        all_vec4 &= (dim % 4 == 0);
    }
    // Unused slots still get dereferenced-free descriptors; grid.y bounds them out.
    for (size_t i = tables.size(); i < kMaxTables; ++i) {
        pack.t[i] = TableDesc{nullptr, nullptr, nullptr, 0, 0};
    }

    auto out = torch::empty({B, out_stride}, tables[0].options().dtype(out_dtype));
    if (B == 0) return out;

    int max_elems = 0;
    for (size_t i = 0; i < tables.size(); ++i) {
        max_elems = std::max(max_elems, B * pack.t[i].dim);
    }
    const int grid_x = std::min((max_elems + kBlock - 1) / kBlock, 65535);
    const dim3 grid(std::max(grid_x, 1), static_cast<unsigned>(tables.size()));

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(out_dtype, "fused_quant_embedding_concat", [&] {
        if (all_vec4) {
            gather_dequant_concat_vec4_kernel<scalar_t><<<grid, kBlock, 0, stream>>>(
                pack, out.data_ptr<scalar_t>(), B, out_stride);
        } else {
            gather_dequant_concat_kernel<scalar_t><<<grid, kBlock, 0, stream>>>(
                pack, out.data_ptr<scalar_t>(), B, out_stride);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
