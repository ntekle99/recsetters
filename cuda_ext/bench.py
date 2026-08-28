"""Benchmarks for the fused kernels, INT8 tables and cascade ranking.

    python cuda_ext/bench.py                 # all three, synthetic data
    python cuda_ext/bench.py --only cascade
    python cuda_ext/bench.py --items 6800 --users 8000   # paper-scale (Apple)

Synthetic by default so it runs without the datasets. Every number is measured
with CUDA events after warmup, with an explicit synchronise, and peak memory is
read from the caching allocator's high-water mark rather than estimated.
"""

import argparse
import statistics
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuda_ext import cascade, ops, quant  # noqa: E402


def timeit(fn, warmup=10, iters=50):
    """Median wall time in ms, plus peak allocator bytes for one call."""
    on_cuda = torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if on_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()

        times = []
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        for _ in range(iters):
            start.record()
            fn()
            stop.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(stop))
        return statistics.median(times), peak

    import time
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times), 0


def _row(name, ms, peak, baseline_ms=None, baseline_peak=None):
    speed = f"{baseline_ms / ms:6.2f}x" if baseline_ms else "     --"
    mem = f"{baseline_peak / peak:6.2f}x" if baseline_peak and peak else "     --"
    print(f"  {name:<34} {ms:9.3f} ms  {peak / 2**20:9.1f} MiB   {speed}  {mem}")


def bench_dot_compress(device, dims=(192,), batches=(512, 4096, 32768)):
    print("\n== Fused DotCompress interaction ==")
    print(f"  {'variant':<34} {'time':>12}  {'peak mem':>13}   speedup  mem-red")
    for d in dims:
        for b in batches:
            u = torch.randn(b, d, device=device)
            v = torch.randn(b, d, device=device)
            w = torch.randn(2, d // 2, device=device)
            bias = torch.randn(d // 2, device=device)
            print(f"\n  B={b}, D={d}")
            ref_ms, ref_peak = timeit(lambda: ops.dot_compress_reference(u, v, w, bias))
            _row("eager (materialises [B,D,D/2])", ref_ms, ref_peak)

            t_ms, t_peak = timeit(lambda: ops.dot_compress_torch_fused(u, v, w, bias))
            _row("pytorch fused identity", t_ms, t_peak, ref_ms, ref_peak)

            if device.type == "cuda" and ops.extension_available():
                c_ms, c_peak = timeit(lambda: ops.dot_compress(u, v, w, bias, force_impl="cuda"))
                _row("cuda fused kernel", c_ms, c_peak, ref_ms, ref_peak)


def bench_quant_embedding(device, num_rows=6800, dims=(96, 96, 96), batch=4096):
    print("\n== INT8 embedding gather + dequantise + concat ==")
    fp_tables, tables, scales, indices = [], [], [], []
    for d in dims:
        w = torch.randn(num_rows, d)
        q, s = quant.quantize_rowwise(w)
        fp_tables.append(w.to(device))
        tables.append(q.to(device))
        scales.append(s.to(device))
        indices.append(torch.randint(0, num_rows, (batch,), device=device))

    fp32_bytes = sum(t.numel() * 4 for t in fp_tables)
    int8_bytes = sum(t.numel() for t in tables) + sum(s.numel() * 4 for s in scales)
    print(f"  table memory: {fp32_bytes / 2**20:.1f} MiB fp32 -> "
          f"{int8_bytes / 2**20:.1f} MiB int8 ({fp32_bytes / int8_bytes:.2f}x smaller)")
    print(f"  {'variant':<34} {'time':>12}  {'peak mem':>13}   speedup  mem-red")

    fp_ms, fp_peak = timeit(lambda: torch.cat([t[i] for t, i in zip(fp_tables, indices)], dim=1))
    _row("fp32 gather + cat", fp_ms, fp_peak)

    ref_ms, ref_peak = timeit(
        lambda: ops.quant_embedding_concat_reference(tables, scales, indices))
    _row("int8 eager (gather,dequant,cat)", ref_ms, ref_peak, fp_ms, fp_peak)

    if device.type == "cuda" and ops.extension_available():
        c_ms, c_peak = timeit(
            lambda: ops.quant_embedding_concat(tables, scales, indices, force_impl="cuda"))
        _row("int8 fused kernel", c_ms, c_peak, fp_ms, fp_peak)


def bench_cascade(device, num_users=1024, num_items=6800, k=10):
    from sparsenn import TrackSparseNN

    print("\n== Draft-and-verify cascade ranking ==")
    torch.manual_seed(0)
    model = TrackSparseNN(
        num_user_ids=num_users, num_user_countries=100, num_user_names=num_users,
        num_track_ids=num_items, num_track_artists=500, num_track_tags=100,
        num_track_names=num_items,
    ).eval().to(device)

    d = model.user_embedding_model.output_embed_dim
    users = torch.randn(num_users, d, device=device)
    items = torch.randn(num_items, d, device=device)

    print(f"  {num_users} users x {num_items} items, k={k}")
    ex_ms, ex_peak = timeit(
        lambda: cascade.exhaustive_topk(model, users, items, k=k), warmup=1, iters=3)
    print(f"  {'variant':<34} {'time':>12}  {'peak mem':>13}   speedup   accept")
    print(f"  {'exhaustive (score every pair)':<34} {ex_ms:9.3f} ms  "
          f"{ex_peak / 2**20:9.1f} MiB        --       1.000")

    exact = cascade.exhaustive_topk(model, users, items, k=k)

    print("  fitting distilled draft on the 5-D sufficient statistic...")
    draft = cascade.DistilledDraft(hidden=8, layers=1).to(device)
    draft.fit(model, users, items)
    draft.eval()

    for name, dr in (("inner-product draft", None), ("distilled draft", draft)):
        for gamma in (2, 4, 8, 16):
            ms, peak = timeit(
                lambda g=gamma, d=dr: cascade.speculative_topk(
                    model, users, items, k=k, gamma=g, draft=d),
                warmup=2, iters=5)
            spec = cascade.speculative_topk(model, users, items, k=k, gamma=gamma, draft=dr)
            acc = cascade.acceptance_rate(spec, exact)
            print(f"  {name + ' g=' + str(gamma):<34} {ms:9.3f} ms  "
                  f"{peak / 2**20:9.1f} MiB  {ex_ms / ms:7.2f}x   {acc:7.3f}")

    gamma, measured = cascade.tune_gamma(model, users, items, k=k, draft=draft,
                                         target_acceptance=0.99, sample_users=256)
    print(f"  tuned gamma for >=0.99 acceptance: {gamma}  (measured: "
          + ", ".join(f"{g}:{a:.3f}" for g, a in sorted(measured.items())) + ")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["dot", "quant", "cascade"], default=None)
    ap.add_argument("--users", type=int, default=1024)
    ap.add_argument("--items", type=int, default=6800)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"cuda extension: {'built' if ops.extension_available() else 'NOT built -- ' + str(ops.extension_error())}")

    if args.only in (None, "dot"):
        bench_dot_compress(device)
    if args.only in (None, "quant"):
        bench_quant_embedding(device, num_rows=args.items)
    if args.only in (None, "cascade"):
        bench_cascade(device, num_users=args.users, num_items=args.items, k=args.k)


if __name__ == "__main__":
    main()
