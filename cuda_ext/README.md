# GPU acceleration for RecSetters

Three additions to the two-tower recommender from *Music Recommendation through
LLM Song Summary*, all aimed at evaluation and serving throughput rather than
recommendation quality:

| | what | where |
|---|---|---|
| **Fused CUDA kernels** | DotCompress interaction rewritten via a rank-2 identity; INT8 embedding gather+dequant+concat | `csrc/`, `ops.py` |
| **INT8 quantisation** | post-training per-row quantisation of the embedding tables | `quant.py` |
| **Cascade ranking** | speculative-decoding-style draft-then-verify top-k | `cascade.py` |

Nothing here changes the model's maths. Every fast path has a slow reference
implementation next to it and a test asserting they agree.

## 1. The fused DotCompress kernel

`DotCompressScoringModel` scores a user/item pair `(u, v)` in `R^D` as

```
E = stack(u, v)          # [2, D]
M = E^T @ W + b          # [D, D/2]   <-- materialised, then immediately reduced
y = flatten(E @ M)       # [D]
```

That `M` is the problem. At `D = 192` it is 18,432 floats **per sample** — 36 MB
at batch 512 in fp32 — written out and read straight back in. But expanding the
product shows it is rank-structured and collapses completely:

```
y[0, j] = (u·u) W[0,j] + (u·v) W[1,j] + (Σu) b[j]
y[1, j] = (u·v) W[0,j] + (v·v) W[1,j] + (Σv) b[j]
```

So the whole op is **five scalars per row** — `u·u`, `u·v`, `v·v`, `Σu`, `Σv` —
followed by a rank-2 broadcast. Arithmetic drops from `O(B·D²/2)` to `O(B·D)`
and the intermediate disappears.

`csrc/dot_compress_cuda.cu` implements this as one block per row: a fused
five-way warp-shuffle block reduction over `D`, then a broadcast across the
`D/2` output columns. The backward pass is hand-derived (the same reduction
structure runs in reverse) and split into two deterministic kernels — one block
per row for `dU`/`dV`, one block per output column for `dW`/`db` — so no
atomics are needed and results are bitwise reproducible run to run.

This is not an approximation: `tests/test_ops.py` checks the kernel against the
original eager code in fp32 and fp64, and runs `torch.autograd.gradcheck` on the
custom backward.

## 2. INT8 embedding tables

The embedding tables are nearly all of the model's parameters and the item
tower's cost is dominated by gathering from them, so it is memory-bound —
exactly where lower precision pays.

Quantisation is **per row**, symmetric: `scale[i] = max|W[i]| / 127`. Per row
rather than per tensor because head and tail item embeddings have very different
dynamic ranges, and a shared scale would flatten most cold-start item vectors to
a handful of levels — precisely the items the paper's cold/warm split measures.
The overhead is one fp32 scale per row (0.4% at `D = 96`).

The kernel in `csrc/quant_embedding_cuda.cu` fuses what eager PyTorch does in
eight steps — four gathers, three dequantise multiplies, one concat — into a
**single launch** that writes each dequantised row directly into its final
column slice of the output, with a `char4`-vectorised path when the dimension
allows. No intermediates, one pass over the data.

Inference only; quantisation is applied to a trained checkpoint.

## 3. Cascade ranking (speculative decoding, applied to retrieval)

Speculative decoding runs a cheap draft model to propose γ tokens and verifies
them all in one batched forward of the expensive target model, because the
target's cost is launch-bound rather than arithmetic-bound.

The same asymmetry exists here. The expensive part is the scoring MLP, which
must run once per (user, candidate) pair — at 8,000 users × 6,800 items that is
**54M MLP evaluations per evaluation pass**. The cheap draft is the plain inner
product of the two tower outputs: a single GEMM, no MLP. The draft proposes
`γ·k` candidates; the target scores only those.

**What does not carry over is the exactness guarantee.** Speculative decoding's
rejection-sampling correction works because token proposals are draws from
distributions over a shared support; a top-k over ranked scores has no such
correction unless the draft bounds the target score, which an unconstrained MLP
does not. So the guarantee here is *empirical*, and this module measures it
rather than assuming it:

- `acceptance_rate` compares the cascade's top-k against the exhaustive top-k.
- `tune_gamma` picks the smallest γ clearing a target acceptance on a sample.
- A test asserts that when `γ·k ≥ N` the cascade is **exactly** the exhaustive
  ranking — so any deviation at large γ is a bug in the verify path, not an
  accuracy tradeoff.

Report acceptance next to any speedup. `evaluate.compare_configurations` is
built to make that hard to avoid: it prints speedup and HR@10/NDCG@10 delta in
the same row.

## Measured results

NVIDIA L4 (24 GB, sm_89), CUDA 12.4, PyTorch 2.6.0, `D = 192`. Reproduce with
`python cuda_ext/bench.py --items 6800 --users 8000`. All 22 correctness tests
pass on this configuration, including `gradcheck` on the custom backward.

### Fused DotCompress kernel

| batch | eager | pytorch fused | **cuda fused** | speedup | peak mem |
|---|---|---|---|---|---|
| 512 | 0.560 ms | 0.254 ms | **0.060 ms** | 9.3x | 81.6 -> 9.3 MiB (8.8x) |
| 4,096 | 5.229 ms | 0.250 ms | **0.066 ms** | 79.8x | 596 -> 17 MiB (34.7x) |
| 32,768 | 41.687 ms | 1.218 ms | **0.345 ms** | 121.0x | 4,712 -> 81 MiB (58.4x) |

The speedup grows with batch because the eager path's cost is the `[B, D, D/2]`
intermediate, which scales with `B` while the fused kernel's working set does
not. At batch 32K the eager version allocates 4.6 GiB to produce 25 MiB of
output.

Roughly a third of the win is the identity itself (available in pure PyTorch)
and the rest is the kernel -- worth knowing, because the PyTorch fused path
gets you 34x on a machine with no toolkit.

### INT8 embedding tables

| variant | time | notes |
|---|---|---|
| fp32 gather + cat | 0.084 ms | baseline |
| int8 eager (gather, dequant, cat) | 0.249 ms | *slower* -- dequant costs more than it saves |
| **int8 fused kernel** | **0.052 ms** | 1.61x vs fp32 |

Table memory drops 3.84x (fp32 scale per row included). **The honest reading:**
the 1.6x is the *fusion* (one launch instead of eight passes), not the
precision. Kernel time is flat at ~0.05 ms whether the tables hold 6.8K rows or
2M, because a gather's working set is set by the batch, not the catalog -- at
batch 4,096 the gathered rows fit in L2 either way. So INT8 buys **capacity**
(3.84x more catalog per GB of VRAM) and the fusion buys **latency**. Quoting
INT8 as a bandwidth win at this scale would be wrong.

### Cascade ranking

8,000 users x 6,800 items, k=10. **The draft model choice dominates this result.**

The obvious draft -- the plain inner product `u.v` -- does not work here. Measured
against the true scoring head its Spearman correlation was **+0.87 in one
initialisation and -0.44 in another**: the sign is not even stable, and in the
anti-correlated case acceptance is exactly 0.000, because the draft proposes
precisely the items the target ranks last. The DotCompress identity explains
why: the target sees the interaction only through `u.v`, scaled by `W[0,j]` and
`W[1,j]`, which are Xavier-initialised and freely negative.

The same identity supplies the fix. The target depends on `(u, v)` **only**
through `(p, q, r, Su, Sv)`, so those five scalars are a *sufficient statistic*
(asserted directly in the tests: permuting both vectors' coordinates leaves all
five, and the target score, unchanged). A small MLP on them is therefore
distilling an exactly-learnable function rather than hoping a proxy correlates.
Distillation takes ~2 s and reaches Spearman **0.9998**.

| draft | gamma | total | speedup | acceptance |
|---|---|---|---|---|
| inner product | 2 | 15.3 ms | 163x | **0.320** |
| inner product | 32 | 128.6 ms | 19x | 0.961 |
| distilled, h32 x2 | 2 | 334.7 ms | 7.5x | 1.000 |
| distilled, h16 x1 | 4 | 136.4 ms | 24.2x | 1.000 |
| **distilled, h8 x1** | **4** | **108.8 ms** | **30.4x** | **1.000** |
| distilled, h8 x1 | 2 | 101.3 ms | 32.6x | 0.980 |

**Headline: 30x faster than exhaustive scoring at acceptance 1.000** -- the
cascade reproduces the exhaustive top-10 exactly, on every user, at 1/30th the
cost. The inner-product draft's 163x is not a competing result; at acceptance
0.32 it is returning a different (wrong) ranking two thirds of the time.

The draft is easy to over-size: h32 x2 is 4.5x slower than h8 x1 for identical
acceptance, because the draft runs over all 54M pairs while the target runs
over `gamma * k` per user. Fit the smallest draft that clears your acceptance
bar -- `tune_gamma` does the gamma half of that search.

These numbers are on an untrained model, which is the *harder* case for a
distilled draft (a trained scorer is smoother). Acceptance should still be
re-measured on a real checkpoint before quoting, since it is cheap to do.

## Install

```bash
pip install -e cuda_ext/     # from the repo root, on a CUDA machine
```

Needs an `nvcc` matching `python -c "import torch; print(torch.version.cuda)"`.
Without a toolkit the Python half still installs and `ops.py` falls back to
PyTorch implementations of the same maths, so the repo runs unchanged on a
laptop. `ops.py` will also JIT-compile the kernels on first use if the package
was not built ahead of time.

## Run

```bash
pytest cuda_ext/tests -v          # CUDA tests skip cleanly without a GPU
python cuda_ext/bench.py          # kernels, INT8 tables, cascade
python cuda_ext/bench.py --items 6800 --users 8000    # paper scale (Apple)
```

## Using it

The fused kernel is already wired in — `DotCompressScoringModel` calls it
automatically when the extension is available, and falls back otherwise. Pass
`use_fused=False` to force the original path.

For quantised, cascaded evaluation:

```python
from cuda_ext import quant, cascade, evaluate

qmodel = quant.quantize_model(model)                  # INT8 tables, eval only
gamma, accept = cascade.tune_gamma(model, user_repr, item_repr, k=10,
                                   target_acceptance=0.99)
ranked = cascade.speculative_topk(model, user_repr, item_repr, k=10, gamma=gamma)
hr, ndcg = evaluate.compute_ranking_metrics(ranked, true_ids, k=10)
```

`evaluate.compute_ranking_metrics` reproduces `main.py`'s HR/NDCG definitions
exactly (verified against the NumPy version in the tests) but stays on the GPU,
instead of round-tripping every user through NumPy inside a Python loop.

## Not done yet

`main.py`'s `inference()` still ranks **one user per Python iteration**, with a
`.cpu()` sync, a `Tensor.apply_()` (which runs a Python lambda per element on
CPU), and a separate `topk` inside the loop. That loop, not the kernels, is
likely the largest remaining chunk of evaluation wall-clock. `cascade.py` and
`evaluate.py` are written to drop straight into it — they take precomputed
representations and an optional per-user candidate pool, which is exactly what
that loop already builds — but the swap is not made, because it could not be
tested without a GPU and the datasets.
