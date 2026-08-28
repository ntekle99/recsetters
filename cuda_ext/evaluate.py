"""GPU evaluation: HR@k / NDCG@k, and an A/B harness for the fast paths.

``compute_ranking_metrics`` reproduces main.py's definitions exactly -- 100 *
hit rate, and NDCG credited only on hits as ``1 / log2(rank + 2)`` averaged over
*all* users -- but on ranked id tensors that never leave the GPU. main.py's
version round-trips every user through NumPy inside a Python loop, which is
where a lot of evaluation wall-clock actually goes.

``compare_configurations`` is the number to put in a paper or a resume bullet:
it runs the fp32 exhaustive baseline and each fast configuration over the same
representations and reports speedup *next to* the metric delta, so a speedup
that quietly cost you 3 points of HR@10 cannot be reported as a win.
"""

import torch

from . import cascade, quant


@torch.no_grad()
def compute_ranking_metrics(ranked_ids, true_ids, k=10):
    """HR@k and NDCG@k, matching main.py::compute_ranking_metrics.

    Args:
        ranked_ids: ``[U, >=k]`` item ids, best first.
        true_ids: ``[U]`` the held-out item id for each user.
    """
    ranked = ranked_ids[:, :k]
    hits = ranked == true_ids.unsqueeze(1)                    # [U, k]
    num_users = ranked.size(0)

    hit_rate = 100.0 * hits.any(dim=1).float().mean()

    positions = torch.arange(k, device=ranked.device, dtype=torch.float32)
    gains = 1.0 / torch.log2(positions + 2.0)                 # [k]
    ndcg = 100.0 * (hits.float() * gains).sum() / num_users
    return hit_rate.item(), ndcg.item()


@torch.no_grad()
def rank_exhaustive(model, user_repr, item_repr, candidate_ids=None, k=10, user_chunk=16):
    return cascade.exhaustive_topk(model, user_repr, item_repr, k=k,
                                   candidate_ids=candidate_ids, user_chunk=user_chunk)


@torch.no_grad()
def rank_cascade(model, user_repr, item_repr, candidate_ids=None, k=10, gamma=8):
    return cascade.speculative_topk(model, user_repr, item_repr, k=k, gamma=gamma,
                                    candidate_ids=candidate_ids)


@torch.no_grad()
def compare_configurations(model, user_repr, item_repr, true_ids, candidate_ids=None,
                           k=10, gammas=(2, 4, 8, 16), quantized_reprs=None):
    """Speedup vs. metric delta for each fast path, against the fp32 baseline.

    ``quantized_reprs`` is an optional ``(user_repr, item_repr)`` pair recomputed
    from an INT8 model (see ``quant.quantize_model``); when given, each gamma is
    also evaluated on those representations so the quantisation and cascade
    effects can be read separately and jointly.

    Returns a list of dicts, baseline first.
    """
    import time

    def timed(fn):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out, (time.perf_counter() - t0) * 1e3

    baseline_ids, baseline_ms = timed(
        lambda: rank_exhaustive(model, user_repr, item_repr, candidate_ids, k))
    base_hr, base_ndcg = compute_ranking_metrics(baseline_ids, true_ids, k)

    rows = [{
        "config": "fp32 exhaustive (baseline)",
        "ms": baseline_ms, "speedup": 1.0, "acceptance": 1.0,
        "hr": base_hr, "ndcg": base_ndcg, "hr_delta": 0.0, "ndcg_delta": 0.0,
    }]

    variants = [("fp32", user_repr, item_repr)]
    if quantized_reprs is not None:
        variants.append(("int8", quantized_reprs[0], quantized_reprs[1]))

    for precision, u_repr, i_repr in variants:
        for gamma in gammas:
            ids, ms = timed(
                lambda: rank_cascade(model, u_repr, i_repr, candidate_ids, k, gamma))
            hr, ndcg = compute_ranking_metrics(ids, true_ids, k)
            rows.append({
                "config": f"{precision} cascade gamma={gamma}",
                "ms": ms,
                "speedup": baseline_ms / ms,
                "acceptance": cascade.acceptance_rate(ids, baseline_ids),
                "hr": hr, "ndcg": ndcg,
                "hr_delta": hr - base_hr, "ndcg_delta": ndcg - base_ndcg,
            })
    return rows


def format_comparison(rows):
    header = (f"{'config':<30}{'time':>11}{'speedup':>10}{'accept':>9}"
              f"{'HR@10':>9}{'dHR':>8}{'NDCG@10':>10}{'dNDCG':>8}")
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['config']:<30}{r['ms']:9.1f}ms{r['speedup']:9.2f}x{r['acceptance']:9.3f}"
            f"{r['hr']:9.4f}{r['hr_delta']:+8.4f}{r['ndcg']:10.4f}{r['ndcg_delta']:+8.4f}")
    return "\n".join(lines)


@torch.no_grad()
def quantized_representations(model, user_batches, item_batches, device):
    """Recompute both towers' representations under INT8 tables.

    ``*_batches`` are iterables of the argument tuples each tower's forward
    takes, i.e. exactly what main.py's inference loops already build.
    """
    qmodel = quant.quantize_model(model)
    user_out = torch.cat([qmodel.user_embedding_model(*[t.to(device) for t in b])
                          for b in user_batches], dim=0)
    item_out = torch.cat([qmodel.item_embedding_model(*[t.to(device) for t in b])
                          for b in item_batches], dim=0)
    return qmodel, user_out, item_out
