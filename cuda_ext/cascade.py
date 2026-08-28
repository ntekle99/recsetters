"""Draft-and-verify cascade ranking -- speculative decoding applied to retrieval.

The analogy, and its limits, stated up front:

  Speculative decoding runs a cheap draft model to propose gamma tokens, then
  verifies all of them in a single batched forward of the expensive target
  model. It wins because the target's cost is dominated by sequential launches,
  not arithmetic, so verifying gamma proposals at once costs about what
  verifying one costs. A rejection-sampling correction makes the output
  distribution *provably identical* to sampling from the target directly.

  Here the expensive model is the scoring MLP in ``DotCompressScoringModel``,
  which must run once per (user, candidate) pair -- at 8,000 users x ~6,800
  items that is 54M MLP evaluations per epoch of evaluation. The cheap draft is
  the plain inner product between the two tower outputs, which is a single GEMM
  and needs no MLP at all. The draft proposes ``gamma * k`` candidates; the
  target scores only those and picks the final top-k.

  What does NOT carry over is the exactness guarantee. Rejection sampling works
  because token proposals are draws from a distribution over a shared support;
  a top-k over ranked scores has no such correction unless the draft score
  bounds the target score, which an unconstrained MLP does not. So the
  guarantee here is *empirical*, and this module measures it rather than
  assuming it: ``acceptance_rate`` compares the cascade's top-k against the
  exhaustive top-k, and ``tune_gamma`` picks the smallest gamma that clears a
  target acceptance. Report the acceptance number alongside any speedup -- a
  cascade whose acceptance you did not measure is just an approximation you got
  away with.
"""

import torch


@torch.no_grad()
def draft_scores(user_repr, item_repr):
    """Inner-product proposal score. ``[U, D] x [N, D] -> [U, N]``.

    WARNING: this is a *bad* draft for this architecture, kept only as a
    baseline. Measured Spearman correlation against the true scoring head is
    -0.44 on an untrained model -- it is anti-correlated, so it proposes
    precisely the items the target ranks lowest and acceptance collapses to 0.

    The reason is visible in the DotCompress identity: the target sees the
    interaction only through ``u.v``, scaled by ``W[0,j]`` and ``W[1,j]``, which
    are Xavier-initialised and freely negative. Nothing makes the target
    increasing in ``u.v``. Use ``DistilledDraft`` instead.
    """
    return user_repr @ item_repr.t()


def sufficient_statistics(user_repr, item_repr):
    """The 5 scalars the scoring head's output depends on. ``-> [U, N, 5]``.

    From the DotCompress identity, the MLP's input for a pair (u, v) is

        row0 = p*W[0] + q*W[1] + su*b        row1 = q*W[0] + r*W[1] + sv*b

    which is fully determined by ``(p, q, r, su, sv) = (u.u, u.v, v.v, Su, Sv)``.
    So these five scalars are a *sufficient statistic* for the target score: a
    draft model built on them is not a heuristic proxy but can in principle
    reproduce the target exactly. Four of the five are per-user or per-item
    constants computed once; only ``q`` needs a GEMM.
    """
    u_n, i_n = user_repr.size(0), item_repr.size(0)
    p = (user_repr * user_repr).sum(1)          # [U]
    su = user_repr.sum(1)                       # [U]
    r = (item_repr * item_repr).sum(1)          # [N]
    sv = item_repr.sum(1)                       # [N]
    q = user_repr @ item_repr.t()               # [U, N]

    return torch.stack([
        p.unsqueeze(1).expand(u_n, i_n),
        q,
        r.unsqueeze(0).expand(u_n, i_n),
        su.unsqueeze(1).expand(u_n, i_n),
        sv.unsqueeze(0).expand(u_n, i_n),
    ], dim=2)


class DistilledDraft(torch.nn.Module):
    """A small model distilled from the scoring head -- the actual draft model.

    Takes the 5-D sufficient statistic and predicts the target's score. Because
    the statistic is sufficient, this is distillation of an exactly-learnable
    function, not an approximation of an unrelated signal.

    Cost per (user, item): ~5x32 + 32 multiply-adds, against the target head's
    192->128->64->1 MLP -- roughly two orders of magnitude cheaper, which is
    what makes the draft worth running over the full catalogue.
    """

    def __init__(self, hidden=16, layers=1):
        super().__init__()
        mods, dim = [], 5
        for _ in range(layers):
            mods += [torch.nn.Linear(dim, hidden), torch.nn.GELU()]
            dim = hidden
        mods.append(torch.nn.Linear(dim, 1))
        self.net = torch.nn.Sequential(*mods)
        self.register_buffer("feat_mean", torch.zeros(5))
        self.register_buffer("feat_std", torch.ones(5))

    def forward(self, feats):
        return self.net((feats - self.feat_mean) / self.feat_std).squeeze(-1)

    @torch.no_grad()
    def _sample_pairs(self, user_repr, item_repr, num_pairs, generator):
        u_idx = torch.randint(0, user_repr.size(0), (num_pairs,),
                              device=user_repr.device, generator=generator)
        i_idx = torch.randint(0, item_repr.size(0), (num_pairs,),
                              device=item_repr.device, generator=generator)
        u, v = user_repr[u_idx], item_repr[i_idx]
        feats = torch.stack([
            (u * u).sum(1), (u * v).sum(1), (v * v).sum(1), u.sum(1), v.sum(1)
        ], dim=1)
        return feats, u, v

    def fit(self, model, user_repr, item_repr, num_pairs=200_000, steps=600,
            lr=3e-3, generator=None, verbose=False):
        """Distil against the target head on randomly sampled pairs."""
        feats, u, v = self._sample_pairs(user_repr, item_repr, num_pairs, generator)
        with torch.no_grad():
            targets = _target_scores(model, u, v)

        self.feat_mean.copy_(feats.mean(0))
        self.feat_std.copy_(feats.std(0).clamp_min(1e-6))
        # Standardise the regression target too; only the ranking matters.
        t_mean, t_std = targets.mean(), targets.std().clamp_min(1e-6)
        targets = (targets - t_mean) / t_std

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        batch = min(16384, num_pairs)
        for step in range(steps):
            idx = torch.randint(0, num_pairs, (batch,), device=feats.device,
                                generator=generator)
            loss = torch.nn.functional.mse_loss(self(feats[idx]), targets[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if verbose and step % 100 == 0:
                print(f"  draft distil step {step:4d} mse {loss.item():.5f}", flush=True)
        return self


@torch.no_grad()
def distilled_draft_scores(draft, user_repr, item_repr, user_chunk=256):
    """``[U, N]`` draft scores, chunked so the [chunk, N, 5] feature tensor fits."""
    out = []
    for start in range(0, user_repr.size(0), user_chunk):
        feats = sufficient_statistics(user_repr[start:start + user_chunk], item_repr)
        out.append(draft(feats))
    return torch.cat(out, dim=0)


@torch.no_grad()
def _target_scores(model, user_repr, item_repr, batch_size=1 << 20):
    """Full scoring model over paired rows. ``[M, D], [M, D] -> [M]``."""
    outputs = []
    for start in range(0, user_repr.size(0), batch_size):
        stop = start + batch_size
        out = model(
            user_embeddings_precomputed=user_repr[start:stop],
            item_embeddings_precomputed=item_repr[start:stop],
        )
        outputs.append(out.flatten())
    return torch.cat(outputs) if len(outputs) > 1 else outputs[0]


@torch.no_grad()
def speculative_topk(model, user_repr, item_repr, k=10, gamma=8,
                     candidate_ids=None, return_scores=False, draft=None):
    """Draft-then-verify top-k.

    Args:
        model: a ``TrackSparseNN``; only its scoring head is used.
        user_repr: ``[U, D]`` precomputed user representations.
        item_repr: ``[N, D]`` precomputed item representations.
        k: final list length.
        gamma: speculation factor. The draft proposes ``gamma * k`` candidates.
        draft: a fitted ``DistilledDraft``. If ``None``, falls back to the raw
            inner product, which is a poor draft here -- see ``draft_scores``.
        candidate_ids: optional ``[U, C]`` per-user candidate pool (indices into
            ``item_repr``). When ``None`` every item is a candidate.

    Returns:
        ``[U, k]`` indices into ``item_repr`` (or into ``candidate_ids`` columns
        mapped back to item ids), optionally with their target scores.
    """
    num_users = user_repr.size(0)

    def _draft_all():
        if draft is None:
            return draft_scores(user_repr, item_repr)
        return distilled_draft_scores(draft, user_repr, item_repr)

    if candidate_ids is None:
        pool_scores = _draft_all()                                 # [U, N]
        pool_size = item_repr.size(0)
        pool_ids = None
    else:
        pool_ids = candidate_ids                                   # [U, C]
        pool_size = pool_ids.size(1)
        # Gather from the [U, N] draft matrix rather than building a [U, C, D]
        # tensor of candidate embeddings -- same result, D times less memory.
        pool_scores = torch.gather(_draft_all(), 1, pool_ids)

    num_proposals = min(gamma * k, pool_size)
    proposals = torch.topk(pool_scores, k=num_proposals, dim=1).indices  # [U, P]

    # Verify: one batched pass of the expensive head over U*P pairs.
    item_ids = proposals if pool_ids is None else torch.gather(pool_ids, 1, proposals)
    verify_items = item_repr[item_ids.reshape(-1)]                  # [U*P, D]
    verify_users = user_repr.unsqueeze(1).expand(num_users, num_proposals, -1).reshape(
        num_users * num_proposals, -1)

    scores = _target_scores(model, verify_users, verify_items).view(num_users, num_proposals)
    top = torch.topk(scores, k=min(k, num_proposals), dim=1)
    final_ids = torch.gather(item_ids, 1, top.indices)

    return (final_ids, top.values) if return_scores else final_ids


@torch.no_grad()
def exhaustive_topk(model, user_repr, item_repr, k=10, candidate_ids=None,
                    user_chunk=16, return_scores=False):
    """Ground truth: run the full scoring model over every candidate.

    Chunked over users so the U*N pair tensor never has to fit in memory at once.
    """
    all_ids, all_scores = [], []
    for start in range(0, user_repr.size(0), user_chunk):
        stop = min(start + user_chunk, user_repr.size(0))
        chunk_users = user_repr[start:stop]
        u = chunk_users.size(0)

        if candidate_ids is None:
            ids = torch.arange(item_repr.size(0), device=item_repr.device).expand(u, -1)
        else:
            ids = candidate_ids[start:stop]
        c = ids.size(1)

        pair_items = item_repr[ids.reshape(-1)]
        pair_users = chunk_users.unsqueeze(1).expand(u, c, -1).reshape(u * c, -1)
        scores = _target_scores(model, pair_users, pair_items).view(u, c)

        top = torch.topk(scores, k=min(k, c), dim=1)
        all_ids.append(torch.gather(ids, 1, top.indices))
        all_scores.append(top.values)

    ids = torch.cat(all_ids, dim=0)
    return (ids, torch.cat(all_scores, dim=0)) if return_scores else ids


@torch.no_grad()
def acceptance_rate(speculative_ids, exact_ids):
    """Mean fraction of the exhaustive top-k recovered by the cascade.

    This is the cascade's analogue of the acceptance rate in speculative
    decoding, and the number to quote next to any speedup claim. 1.0 means the
    cascade reproduced the exhaustive ranking's membership exactly.
    """
    k = exact_ids.size(1)
    hits = (speculative_ids.unsqueeze(2) == exact_ids.unsqueeze(1)).any(dim=1)
    return (hits.sum(dim=1).float() / k).mean().item()


@torch.no_grad()
def tune_gamma(model, user_repr, item_repr, k=10, target_acceptance=0.99,
               candidates=(2, 4, 8, 16, 32), sample_users=256, candidate_ids=None,
               generator=None, draft=None):
    """Smallest gamma whose acceptance clears ``target_acceptance`` on a sample.

    Returns ``(gamma, {gamma: acceptance})``. If no candidate clears the bar,
    returns the best one tried, so callers always get a usable setting.
    """
    num_users = user_repr.size(0)
    if sample_users < num_users:
        idx = torch.randperm(num_users, device=user_repr.device, generator=generator)[:sample_users]
    else:
        idx = torch.arange(num_users, device=user_repr.device)

    sample_users_repr = user_repr[idx]
    sample_candidates = None if candidate_ids is None else candidate_ids[idx]
    exact = exhaustive_topk(model, sample_users_repr, item_repr, k=k,
                            candidate_ids=sample_candidates)

    measured = {}
    chosen = None
    for gamma in sorted(candidates):
        spec = speculative_topk(model, sample_users_repr, item_repr, k=k, gamma=gamma,
                                candidate_ids=sample_candidates, draft=draft)
        measured[gamma] = acceptance_rate(spec, exact)
        if chosen is None and measured[gamma] >= target_acceptance:
            chosen = gamma

    if chosen is None:
        chosen = max(measured, key=measured.get)
    return chosen, measured
