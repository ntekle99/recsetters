"""Python entry points for the fused CUDA ops.

Every op here has three implementations that must agree numerically:

  * ``*_reference``    -- the original eager PyTorch code, kept verbatim so the
                          tests have something independent to compare against.
  * ``*_torch_fused``  -- the same algebraic identity the CUDA kernel uses,
                          written in PyTorch. Runs anywhere (CPU, MPS, ROCm) and
                          is the fallback when the extension is not built.
  * the CUDA kernel    -- selected automatically when inputs are on a GPU and
                          the extension imported cleanly.

Keeping the pure-PyTorch fused path around is deliberate: it isolates "is the
math right" from "is the kernel right" when something goes wrong, and it means
the repo still runs on a laptop.
"""

import os
import warnings

import torch
from torch.autograd import Function

_EXT = None
_EXT_ERROR = None


def _load_extension():
    """Import the compiled extension, falling back to a JIT build."""
    global _EXT, _EXT_ERROR
    if _EXT is not None or _EXT_ERROR is not None:
        return _EXT

    try:
        import recsetters_cuda  # built via `pip install -e cuda/`
        _EXT = recsetters_cuda
        return _EXT
    except ImportError:
        pass

    if os.environ.get("RECSETTERS_NO_JIT", "0") == "1" or not torch.cuda.is_available():
        _EXT_ERROR = "extension not installed and JIT build disabled/unavailable"
        return None

    try:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        _EXT = load(
            name="recsetters_cuda",
            sources=[
                os.path.join(here, "csrc", "bindings.cpp"),
                os.path.join(here, "csrc", "dot_compress_cuda.cu"),
                os.path.join(here, "csrc", "quant_embedding_cuda.cu"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        return _EXT
    except Exception as exc:  # noqa: BLE001 - any build failure means "use fallback"
        _EXT_ERROR = str(exc)
        return None


def extension_available():
    return _load_extension() is not None


def extension_error():
    _load_extension()
    return _EXT_ERROR


# --------------------------------------------------------------------------- #
# DotCompress interaction
# --------------------------------------------------------------------------- #

def dot_compress_reference(u, v, weight, bias):
    """The original implementation, materialising the [B, D, H] intermediate."""
    e = torch.stack([u, v], dim=1)
    return torch.matmul(e, torch.matmul(e.transpose(1, 2), weight) + bias).flatten(1)


def dot_compress_torch_fused(u, v, weight, bias):
    """Same result via the rank-2 identity; no [B, D, H] intermediate."""
    p = (u * u).sum(dim=1, keepdim=True)
    q = (u * v).sum(dim=1, keepdim=True)
    r = (v * v).sum(dim=1, keepdim=True)
    su = u.sum(dim=1, keepdim=True)
    sv = v.sum(dim=1, keepdim=True)
    w0, w1 = weight[0], weight[1]
    row0 = p * w0 + q * w1 + su * bias
    row1 = q * w0 + r * w1 + sv * bias
    return torch.cat([row0, row1], dim=1)


class _DotCompressCUDA(Function):
    @staticmethod
    def forward(ctx, u, v, weight, bias):
        ext = _load_extension()
        out, stats = ext.dot_compress_forward(
            u.contiguous(), v.contiguous(), weight.contiguous(), bias.contiguous()
        )
        ctx.save_for_backward(u, v, weight, bias, stats)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ext = _load_extension()
        u, v, weight, bias, stats = ctx.saved_tensors
        gu, gv, gw, gb = ext.dot_compress_backward(
            grad_out.contiguous(), u.contiguous(), v.contiguous(),
            weight.contiguous(), bias.contiguous(), stats,
        )
        return gu, gv, gw, gb


def dot_compress(u, v, weight, bias, force_impl=None):
    """Fused DotCompress interaction.

    Args:
        u, v: ``[B, D]`` user and item representations.
        weight: ``[2, D // 2]``.
        bias: ``[D // 2]``.
        force_impl: ``None`` (auto), ``"cuda"``, ``"torch"`` or ``"reference"``.
    """
    if force_impl == "reference":
        return dot_compress_reference(u, v, weight, bias)
    if force_impl == "torch":
        return dot_compress_torch_fused(u, v, weight, bias)

    use_cuda = u.is_cuda and extension_available()
    if force_impl == "cuda":
        if not use_cuda:
            raise RuntimeError(f"CUDA kernel unavailable: {extension_error()}")
        return _DotCompressCUDA.apply(u, v, weight, bias)

    if use_cuda:
        return _DotCompressCUDA.apply(u, v, weight, bias)
    return dot_compress_torch_fused(u, v, weight, bias)


# --------------------------------------------------------------------------- #
# INT8 embedding gather + dequantise + concat
# --------------------------------------------------------------------------- #

def quant_embedding_concat_reference(tables, scales, indices, out_dtype=torch.float32):
    """Eager equivalent: one gather + one dequant per table, then a cat."""
    parts = []
    for table, scale, idx in zip(tables, scales, indices):
        rows = table[idx].to(out_dtype)
        parts.append(rows * scale[idx].to(out_dtype).unsqueeze(1))
    return torch.cat(parts, dim=1)


def quant_embedding_concat(tables, scales, indices, out_dtype=torch.float32, force_impl=None):
    """Gather INT8 embedding rows, dequantise, and concat -- in one kernel."""
    if force_impl == "reference":
        return quant_embedding_concat_reference(tables, scales, indices, out_dtype)

    on_cuda = all(t.is_cuda for t in tables)
    if force_impl == "cuda" and not (on_cuda and extension_available()):
        raise RuntimeError(f"CUDA kernel unavailable: {extension_error()}")
    if on_cuda and extension_available():
        ext = _load_extension()
        return ext.fused_quant_embedding_concat(
            [t.contiguous() for t in tables],
            [s.contiguous().float() for s in scales],
            [i.to(torch.int64).contiguous() for i in indices],
            out_dtype,
        )
    return quant_embedding_concat_reference(tables, scales, indices, out_dtype)


def warn_if_fallback(context=""):
    if not extension_available():
        warnings.warn(
            f"recsetters CUDA extension unavailable{': ' + context if context else ''} "
            f"({extension_error()}); using the PyTorch fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
