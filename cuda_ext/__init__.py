"""GPU acceleration for the RecSetters two-tower music recommender.

Three pieces, in the order they matter for evaluation throughput:

  ops       fused CUDA kernels -- the DotCompress interaction (rewritten via a
            rank-2 identity that removes an O(B*D^2/2) intermediate) and an
            INT8 embedding gather/dequantise/concat.
  quant     post-training per-row INT8 quantisation of the embedding tables.
  cascade   draft-and-verify ranking: a cheap inner-product draft proposes
            candidates, the expensive scoring MLP verifies only those.
  evaluate  GPU HR@k/NDCG@k plus an A/B harness that reports every speedup
            next to its metric delta.

Everything degrades gracefully: with no CUDA extension built, the same maths
runs through PyTorch fallbacks, so the repo still works on a laptop.
"""

from . import cascade, evaluate, ops, quant
from .ops import (
    dot_compress,
    dot_compress_reference,
    dot_compress_torch_fused,
    extension_available,
    extension_error,
    quant_embedding_concat,
    quant_embedding_concat_reference,
)
from .quant import quantize_model, quantize_rowwise, quantization_error, model_footprint
from .cascade import speculative_topk, exhaustive_topk, acceptance_rate, tune_gamma
from .evaluate import compute_ranking_metrics, compare_configurations, format_comparison

__all__ = [
    "cascade", "evaluate", "ops", "quant",
    "dot_compress", "dot_compress_reference", "dot_compress_torch_fused",
    "extension_available", "extension_error",
    "quant_embedding_concat", "quant_embedding_concat_reference",
    "quantize_model", "quantize_rowwise", "quantization_error", "model_footprint",
    "speculative_topk", "exhaustive_topk", "acceptance_rate", "tune_gamma",
    "compute_ranking_metrics", "compare_configurations", "format_comparison",
]
