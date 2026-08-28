#include <torch/extension.h>

std::vector<torch::Tensor> dot_compress_forward_cuda(
    torch::Tensor U, torch::Tensor V, torch::Tensor W, torch::Tensor bias);

std::vector<torch::Tensor> dot_compress_backward_cuda(
    torch::Tensor gout, torch::Tensor U, torch::Tensor V, torch::Tensor W,
    torch::Tensor bias, torch::Tensor stats);

torch::Tensor fused_quant_embedding_concat_cuda(
    std::vector<torch::Tensor> tables,
    std::vector<torch::Tensor> scales,
    std::vector<torch::Tensor> indices,
    c10::ScalarType out_dtype);

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_CONTIG(x); } while (0)

std::vector<torch::Tensor> dot_compress_forward(
    torch::Tensor U, torch::Tensor V, torch::Tensor W, torch::Tensor bias) {
    CHECK_INPUT(U); CHECK_INPUT(V); CHECK_INPUT(W); CHECK_INPUT(bias);
    TORCH_CHECK(U.dim() == 2 && V.dim() == 2, "U and V must be 2-D [B, D]");
    TORCH_CHECK(U.sizes() == V.sizes(), "U and V must have the same shape");
    TORCH_CHECK(W.dim() == 2 && W.size(0) == 2, "W must be [2, D/2]");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == W.size(1), "bias must be [D/2]");
    TORCH_CHECK(U.scalar_type() == W.scalar_type() && U.scalar_type() == bias.scalar_type(),
                "U, V, W, bias must share a dtype");
    return dot_compress_forward_cuda(U, V, W, bias);
}

std::vector<torch::Tensor> dot_compress_backward(
    torch::Tensor gout, torch::Tensor U, torch::Tensor V, torch::Tensor W,
    torch::Tensor bias, torch::Tensor stats) {
    CHECK_INPUT(gout); CHECK_INPUT(U); CHECK_INPUT(V);
    CHECK_INPUT(W); CHECK_INPUT(bias); CHECK_INPUT(stats);
    return dot_compress_backward_cuda(gout, U, V, W, bias, stats);
}

torch::Tensor fused_quant_embedding_concat(
    std::vector<torch::Tensor> tables,
    std::vector<torch::Tensor> scales,
    std::vector<torch::Tensor> indices,
    at::ScalarType out_dtype) {
    for (auto& t : tables) CHECK_INPUT(t);
    for (auto& s : scales) CHECK_INPUT(s);
    for (auto& i : indices) CHECK_INPUT(i);
    return fused_quant_embedding_concat_cuda(tables, scales, indices, out_dtype);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_compress_forward", &dot_compress_forward, "Fused DotCompress forward (CUDA)");
    m.def("dot_compress_backward", &dot_compress_backward, "Fused DotCompress backward (CUDA)");
    m.def("fused_quant_embedding_concat", &fused_quant_embedding_concat,
          "Fused INT8 embedding gather + dequantise + concat (CUDA)");
}
