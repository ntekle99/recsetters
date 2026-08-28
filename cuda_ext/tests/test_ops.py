"""Correctness tests for the fused ops, quantisation and cascade ranking.

Run with ``pytest cuda_ext/tests`` or directly: ``python cuda_ext/tests/test_ops.py``.

The CPU tests are the ones that pin down the *maths* and run anywhere. The
CUDA-marked tests pin down the *kernel* and skip cleanly on a machine without a
GPU, so a laptop run still tells you whether the identity and the cascade
invariants hold.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cuda_ext import cascade, evaluate, ops, quant  # noqa: E402
from sparsenn import TrackSparseNN  # noqa: E402

HAS_CUDA = torch.cuda.is_available() and ops.extension_available()
requires_cuda = pytest.mark.skipif(
    not HAS_CUDA, reason=f"CUDA extension unavailable: {ops.extension_error()}"
)


def _dot_compress_inputs(B=17, D=96, dtype=torch.float64, device="cpu", grad=False):
    g = torch.Generator(device="cpu").manual_seed(0)
    mk = lambda *s: torch.randn(*s, generator=g, dtype=dtype).to(device).requires_grad_(grad)
    return mk(B, D), mk(B, D), mk(2, D // 2), mk(D // 2)


# --------------------------------------------------------------------------- #
# DotCompress: the algebraic identity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("D", [4, 96, 192])
def test_torch_fused_matches_reference(D):
    u, v, w, b = _dot_compress_inputs(D=D)
    torch.testing.assert_close(
        ops.dot_compress_torch_fused(u, v, w, b),
        ops.dot_compress_reference(u, v, w, b),
    )


def test_torch_fused_gradcheck():
    u, v, w, b = _dot_compress_inputs(B=5, D=8, grad=True)
    assert torch.autograd.gradcheck(ops.dot_compress_torch_fused, (u, v, w, b))


def test_fused_allocates_no_bd2_intermediate():
    """The point of the rewrite: peak memory must not scale with D^2."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device to read peak allocation")
    u, v, w, b = _dot_compress_inputs(B=512, D=192, dtype=torch.float32, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    ops.dot_compress_reference(u, v, w, b)
    torch.cuda.synchronize()
    ref_peak = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    ops.dot_compress(u, v, w, b)
    torch.cuda.synchronize()
    fused_peak = torch.cuda.max_memory_allocated()

    assert fused_peak < ref_peak / 4, (fused_peak, ref_peak)


@requires_cuda
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-4), (torch.float64, 1e-10)])
def test_cuda_forward_matches_reference(dtype, tol):
    u, v, w, b = _dot_compress_inputs(B=129, D=192, dtype=dtype, device="cuda")
    got = ops.dot_compress(u, v, w, b, force_impl="cuda")
    want = ops.dot_compress_reference(u, v, w, b)
    torch.testing.assert_close(got, want, rtol=tol, atol=tol)


@requires_cuda
def test_cuda_gradcheck():
    u, v, w, b = _dot_compress_inputs(B=5, D=8, dtype=torch.float64, device="cuda", grad=True)
    fn = lambda *args: ops.dot_compress(*args, force_impl="cuda")
    assert torch.autograd.gradcheck(fn, (u, v, w, b))


@requires_cuda
def test_cuda_backward_matches_reference():
    for impl in ("cuda", "reference"):
        u, v, w, b = _dot_compress_inputs(B=64, D=192, dtype=torch.float64,
                                          device="cuda", grad=True)
        out = ops.dot_compress(u, v, w, b, force_impl=impl)
        out.sum().backward()
        grads = [t.grad.clone() for t in (u, v, w, b)]
        if impl == "cuda":
            cuda_grads = grads
        else:
            for g_cuda, g_ref in zip(cuda_grads, grads):
                torch.testing.assert_close(g_cuda, g_ref, rtol=1e-10, atol=1e-10)


@requires_cuda
def test_cuda_empty_batch():
    u, v, w, b = _dot_compress_inputs(B=0, D=96, dtype=torch.float32, device="cuda")
    assert ops.dot_compress(u, v, w, b, force_impl="cuda").shape == (0, 96)


# --------------------------------------------------------------------------- #
# INT8 quantisation
# --------------------------------------------------------------------------- #

def test_rowwise_quantisation_error_bound():
    """Per-row symmetric int8 bounds elementwise error at half a step."""
    w = torch.randn(500, 96) * torch.logspace(-3, 2, 500).unsqueeze(1)  # wildly varying rows
    q, scale = quant.quantize_rowwise(w)
    deq = q.float() * scale.unsqueeze(1)
    assert ((deq - w).abs() <= scale.unsqueeze(1) / 2 + 1e-6).all()
    assert q.dtype == torch.int8

    stats = quant.quantization_error(w)
    assert stats["cos_min"] > 0.999, stats
    assert stats["bytes_int8"] < stats["bytes_fp32"] / 3.5


def test_zero_row_quantises_exactly():
    w = torch.zeros(4, 8)
    w[1] = torch.randn(8)
    q, scale = quant.quantize_rowwise(w)
    deq = q.float() * scale.unsqueeze(1)
    assert torch.isfinite(deq).all()
    torch.testing.assert_close(deq[0], w[0])


@pytest.mark.parametrize("dims", [(96, 96, 96), (96, 64, 32)])
def test_quant_embedding_concat_matches_reference(dims):
    device = "cuda" if HAS_CUDA else "cpu"
    tables, scales, indices = [], [], []
    for d in dims:
        w = torch.randn(200, d)
        q, s = quant.quantize_rowwise(w)
        tables.append(q.to(device))
        scales.append(s.to(device))
        indices.append(torch.randint(0, 200, (33,), device=device))

    got = ops.quant_embedding_concat(tables, scales, indices)
    want = ops.quant_embedding_concat_reference(tables, scales, indices)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    assert got.shape == (33, sum(dims))


def test_quantized_tower_tracks_fp_tower():
    model = _small_model()
    ids = torch.randint(0, 50, (16,))
    artists = torch.randint(0, 7, (16,))
    tags = torch.randint(0, 5, (16,))
    names = torch.randn(16, 384)

    with torch.no_grad():
        fp = model.item_embedding_model(ids, artists, tags, names)
    qt = quant.QuantizedItemTower(model.item_embedding_model)
    q = qt(ids, artists, tags, names)

    cos = torch.nn.functional.cosine_similarity(fp, q, dim=1)
    assert cos.min() > 0.99, cos.min().item()


def test_quantize_model_shrinks_footprint():
    model = _small_model()
    before = quant.model_footprint(model)
    after = quant.model_footprint(quant.quantize_model(model))
    assert after < before


# --------------------------------------------------------------------------- #
# Cascade ranking
# --------------------------------------------------------------------------- #

def _small_model(seed=0):
    torch.manual_seed(seed)
    return TrackSparseNN(
        num_user_ids=50, num_user_countries=10, num_user_names=50,
        num_track_ids=50, num_track_artists=7, num_track_tags=5, num_track_names=50,
    ).eval()


def test_full_speculation_is_exact():
    """When the draft proposes every candidate, the cascade *must* be exact.

    This is the invariant that makes the approximation auditable: any deviation
    at large gamma is a bug in the verify path, not an accuracy/speed tradeoff.
    """
    model = _small_model()
    torch.manual_seed(1)
    users = torch.randn(12, model.user_embedding_model.output_embed_dim)
    items = torch.randn(40, model.item_embedding_model.output_embed_dim)

    spec = cascade.speculative_topk(model, users, items, k=10, gamma=40)  # gamma*k >= N
    exact = cascade.exhaustive_topk(model, users, items, k=10)
    torch.testing.assert_close(spec, exact)
    assert cascade.acceptance_rate(spec, exact) == 1.0


def test_acceptance_is_monotone_in_gamma():
    model = _small_model()
    torch.manual_seed(2)
    users = torch.randn(32, model.user_embedding_model.output_embed_dim)
    items = torch.randn(300, model.item_embedding_model.output_embed_dim)
    exact = cascade.exhaustive_topk(model, users, items, k=10)

    rates = [
        cascade.acceptance_rate(
            cascade.speculative_topk(model, users, items, k=10, gamma=g), exact)
        for g in (1, 2, 4, 8, 30)
    ]
    assert rates[-1] == 1.0
    assert all(a <= b + 1e-9 for a, b in zip(rates, rates[1:])), rates


def test_cascade_respects_candidate_pools():
    model = _small_model()
    torch.manual_seed(3)
    users = torch.randn(8, model.user_embedding_model.output_embed_dim)
    items = torch.randn(100, model.item_embedding_model.output_embed_dim)
    pools = torch.stack([torch.randperm(100)[:25] for _ in range(8)])

    spec = cascade.speculative_topk(model, users, items, k=5, gamma=25, candidate_ids=pools)
    exact = cascade.exhaustive_topk(model, users, items, k=5, candidate_ids=pools)
    torch.testing.assert_close(spec, exact)
    # Every returned id must come from that user's own pool.
    assert (spec.unsqueeze(2) == pools.unsqueeze(1)).any(dim=2).all()


def test_sufficient_statistic_determines_target_score():
    """Two pairs sharing (p, q, r, Su, Sv) must get the identical target score.

    This is what licenses the distilled draft: the statistic is sufficient, so
    a model on it is distilling an exactly-learnable function.
    """
    model = _small_model()
    d = model.user_embedding_model.output_embed_dim
    torch.manual_seed(11)
    u = torch.randn(1, d)
    v = torch.randn(1, d)
    # A rotation about u preserves u.v, v.v -- but not Sv, so rotate the sum in
    # too by permuting coordinates of both vectors identically.
    perm = torch.randperm(d)
    u2, v2 = u[:, perm], v[:, perm]

    stats1 = cascade.sufficient_statistics(u, v).flatten()
    stats2 = cascade.sufficient_statistics(u2, v2).flatten()
    torch.testing.assert_close(stats1, stats2)

    with torch.no_grad():
        s1 = model(user_embeddings_precomputed=u, item_embeddings_precomputed=v)
        s2 = model(user_embeddings_precomputed=u2, item_embeddings_precomputed=v2)
    torch.testing.assert_close(s1, s2, rtol=1e-5, atol=1e-5)


def test_distilled_draft_beats_inner_product():
    """The distilled draft should track the target far better than u.v."""
    model = _small_model()
    torch.manual_seed(12)
    d = model.user_embedding_model.output_embed_dim
    users = torch.randn(64, d)
    items = torch.randn(400, d)
    exact = cascade.exhaustive_topk(model, users, items, k=10)

    draft = cascade.DistilledDraft(hidden=8, layers=1)
    draft.fit(model, users, items, num_pairs=20_000, steps=300)
    draft.eval()

    acc_ip = cascade.acceptance_rate(
        cascade.speculative_topk(model, users, items, k=10, gamma=4), exact)
    acc_dd = cascade.acceptance_rate(
        cascade.speculative_topk(model, users, items, k=10, gamma=4, draft=draft), exact)
    assert acc_dd > acc_ip, (acc_dd, acc_ip)
    assert acc_dd > 0.9, acc_dd


def test_tune_gamma_returns_usable_setting():
    model = _small_model()
    torch.manual_seed(4)
    users = torch.randn(24, model.user_embedding_model.output_embed_dim)
    items = torch.randn(200, model.item_embedding_model.output_embed_dim)

    gamma, measured = cascade.tune_gamma(model, users, items, k=10,
                                         target_acceptance=0.5, candidates=(1, 2, 20),
                                         sample_users=24)
    assert gamma in measured
    assert measured[20] >= measured[1]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_metrics_match_numpy_reference():
    """Must agree with main.py::compute_ranking_metrics on the same rankings."""
    import numpy as np

    torch.manual_seed(5)
    u, k, n = 200, 10, 60
    ranked = torch.stack([torch.randperm(n)[:k] for _ in range(u)])
    # Half the users have their true item somewhere in the list, half do not.
    true = ranked[torch.arange(u), torch.randint(0, k, (u,))].clone()
    true[u // 2:] = n + 1

    hr, ndcg = evaluate.compute_ranking_metrics(ranked, true, k=k)

    topk_np, true_np = ranked.numpy(), true.numpy()
    membership = (topk_np == true_np.reshape(-1, 1)).any(axis=1)
    hr_np = 100 * (membership.sum() / membership.shape[0])
    denoms = np.log2(np.argwhere(topk_np == true_np.reshape(-1, 1))[:, 1] + 2)
    ndcg_np = 100 * np.sum(1 / denoms) / true_np.shape[0]

    assert abs(hr - hr_np) < 1e-4, (hr, hr_np)
    assert abs(ndcg - ndcg_np) < 1e-4, (ndcg, ndcg_np)


def test_metrics_perfect_and_empty():
    ranked = torch.arange(10).unsqueeze(0).repeat(4, 1)
    hr, ndcg = evaluate.compute_ranking_metrics(ranked, torch.zeros(4, dtype=torch.long), k=10)
    assert hr == 100.0 and abs(ndcg - 100.0) < 1e-4
    hr, ndcg = evaluate.compute_ranking_metrics(ranked, torch.full((4,), 99), k=10)
    assert hr == 0.0 and ndcg == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
