"""Post-training INT8 quantisation of the two-tower embedding tables.

Why the embedding tables and not the MLPs: in this model the tables are almost
all of the parameters (items x 96 dims x 4 tables, plus users x 64 x 3) while
the MLPs are a few hundred KB. Inference on the item tower is therefore
bandwidth-bound on gathers, and shrinking the tables 4x is the single largest
lever available.

Quantisation is *per row* (per embedding vector), symmetric, int8:

    scale[i] = max(|W[i]|) / 127        q[i] = round(W[i] / scale[i])

Per-row rather than per-tensor because embedding rows for head and tail items
have very different dynamic ranges -- a shared scale would quantise most
cold-start item vectors to a handful of levels, which is precisely the regime
the paper's cold/warm split is measuring. The cost is one fp32 scale per row
(0.4% overhead at D=96) for a much tighter error bound.

Weights only, activations left in fp: the MLPs that consume these embeddings
are small and the gather is the bottleneck, so quantising activations too would
add requantise overhead for little gain.
"""

import torch
import torch.nn as nn

from . import ops


@torch.no_grad()
def quantize_rowwise(weight: torch.Tensor):
    """Symmetric per-row INT8 quantisation. Returns (int8 table, fp32 scales)."""
    assert weight.dim() == 2, "expected a [num_rows, dim] embedding table"
    w = weight.detach().float()
    amax = w.abs().amax(dim=1)
    # An all-zero row (e.g. an unused padding id) has no scale; use 1.0 so
    # dequantisation is exact rather than NaN.
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.round(w / scale.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


@torch.no_grad()
def quantization_error(weight: torch.Tensor):
    """Diagnostics for a single table: relative L2 error and worst-row cosine."""
    q, scale = quantize_rowwise(weight)
    deq = q.float() * scale.unsqueeze(1)
    w = weight.detach().float()
    rel_l2 = (deq - w).norm() / w.norm().clamp_min(1e-12)
    cos = nn.functional.cosine_similarity(deq, w, dim=1)
    return {
        "rel_l2": rel_l2.item(),
        "cos_mean": cos.mean().item(),
        "cos_min": cos.min().item(),
        "bytes_fp32": w.numel() * 4,
        "bytes_int8": q.numel() + scale.numel() * 4,
    }


class QuantizedTables(nn.Module):
    """A group of INT8 tables gathered and concatenated by one fused kernel."""

    def __init__(self, embeddings):
        super().__init__()
        self.num_tables = len(embeddings)
        self.dims = []
        for i, emb in enumerate(embeddings):
            q, scale = quantize_rowwise(emb.weight)
            self.register_buffer(f"table_{i}", q)
            self.register_buffer(f"scale_{i}", scale)
            self.dims.append(q.size(1))

    def forward(self, indices, out_dtype=torch.float32, force_impl=None):
        tables = [getattr(self, f"table_{i}") for i in range(self.num_tables)]
        scales = [getattr(self, f"scale_{i}") for i in range(self.num_tables)]
        return ops.quant_embedding_concat(
            tables, scales, list(indices), out_dtype=out_dtype, force_impl=force_impl
        )


class QuantizedUserTower(nn.Module):
    """INT8 drop-in for ``TrackSparseNNUserModel`` (eval only)."""

    def __init__(self, user_model):
        super().__init__()
        assert user_model.combine_op == "cat", "INT8 path assumes combine_op='cat'"
        self.tables = QuantizedTables([
            user_model.id_embeddings,
            user_model.countries_embeddings,
            user_model.user_name_embeddings,
        ])
        self.output_mlp = user_model.output_mlp
        self.act = user_model.act
        self.output_embed_dim = user_model.output_embed_dim

    @torch.no_grad()
    def forward(self, user_ids, user_countries, user_names):
        combined = self.tables([user_ids, user_countries, user_names])
        return self.act(self.output_mlp(combined))


class QuantizedItemTower(nn.Module):
    """INT8 drop-in for ``TrackSparseNNItemModel`` (eval only).

    The three sparse tables go through the fused kernel; ``dense_transform``
    stays in fp because its input is a dense sentence-transformer vector, not a
    gather, so there is nothing to save by quantising it.
    """

    def __init__(self, item_model):
        super().__init__()
        assert item_model.combine_op == "cat", "INT8 path assumes combine_op='cat'"
        self.tables = QuantizedTables([
            item_model.track_id_embeddings,
            item_model.artists_embeddings,
            item_model.tags_embeddings,
        ])
        self.dense_transform = item_model.dense_transform
        self.output_mlp = item_model.output_mlp
        self.act = item_model.act
        self.output_embed_dim = item_model.output_embed_dim

    @torch.no_grad()
    def forward(self, track_ids, track_artists, track_tags, track_names):
        sparse = self.tables([track_ids, track_artists, track_tags])
        dense = self.act(self.dense_transform(track_names))
        combined = torch.cat([sparse, dense.to(sparse.dtype)], dim=1)
        return self.act(self.output_mlp(combined))


@torch.no_grad()
def quantize_model(model):
    """Return ``model`` with both towers swapped for their INT8 versions.

    Mutates a *copy*-free view: the scoring MLP and its fused interaction are
    untouched, so ranking behaviour changes only through embedding error.
    """
    model.eval()
    model.user_embedding_model = QuantizedUserTower(model.user_embedding_model)
    model.item_embedding_model = QuantizedItemTower(model.item_embedding_model)
    return model


@torch.no_grad()
def model_footprint(model):
    """Parameter + buffer bytes, for before/after reporting."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total
