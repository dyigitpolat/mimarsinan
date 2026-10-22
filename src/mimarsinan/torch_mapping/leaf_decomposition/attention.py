"""Decomposed attention and transformer encoder layers: the leaves' own math, traceable, on the leaves' parameters.

``DecomposedAttention`` is ``nn.MultiheadAttention`` (no masks, no attention weights read, as the registry admits it):
query, key and value projections as three ``nn.Linear`` (views of ``in_proj_weight``/``in_proj_bias``, or of the
separate projection weights), heads split by ``view`` and ``transpose``, the scaled scores and the weighted values as
two products of activations (Contractions at conversion), ``softmax``, and the leaf's own output projection
parameters; ``batch_first`` both ways. ``DecomposedEncoderLayer`` is ``nn.TransformerEncoderLayer`` (pre- or post-norm)
around it, its LayerNorms, Linears, activation and dropouts the layer's own modules; ``DecomposedEncoder`` is the
``nn.TransformerEncoder`` container (itself an FX leaf: a ``torch.nn`` module), its layers decomposed, then its norm.
"""

from __future__ import annotations

import math
from typing import List, Optional, cast

import torch
import torch.nn as nn


def shared_linear(weight: torch.Tensor, bias: Optional[torch.Tensor]) -> nn.Linear:
    """An ``nn.Linear`` computing with ``weight`` and ``bias`` themselves (a parameter, or a view of one: same storage)."""
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device="meta")
    linear.weight = weight if isinstance(weight, nn.Parameter) else nn.Parameter(weight, weight.requires_grad)
    if bias is not None:
        linear.bias = bias if isinstance(bias, nn.Parameter) else nn.Parameter(bias, bias.requires_grad)
    return linear


def _projections(mha: nn.MultiheadAttention) -> List[nn.Linear]:
    """The q, k, v projections of ``mha``, each on views of the leaf's own rows."""
    e = mha.embed_dim
    weights = [mha.in_proj_weight[i * e:(i + 1) * e] for i in range(3)] if mha._qkv_same_embed_dim else [
        mha.q_proj_weight, mha.k_proj_weight, mha.v_proj_weight]
    biases = [None] * 3 if mha.in_proj_bias is None else [mha.in_proj_bias[i * e:(i + 1) * e] for i in range(3)]
    return [shared_linear(w, b) for w, b in zip(weights, biases)]


class DecomposedAttention(nn.Module):
    """``nn.MultiheadAttention`` without masks as its traceable math."""

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = _projections(mha)
        self.out_proj = shared_linear(mha.out_proj.weight, mha.out_proj.bias)
        self.heads, self.batch_first = mha.num_heads, mha.batch_first
        self.head_dim = mha.embed_dim // mha.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _e = x.shape
        return x.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        if not self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        q, k, v = self._split(self.q_proj(query)), self._split(self.k_proj(key)), self._split(self.v_proj(value))
        scores = torch.matmul(q * self.scale, k.transpose(-2, -1))
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        b, _h, n, _d = out.shape
        out = self.out_proj(out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim))
        return (out if self.batch_first else out.transpose(0, 1)), None


class DecomposedEncoderLayer(nn.Module):
    """``nn.TransformerEncoderLayer`` without masks, its attention decomposed."""

    def __init__(self, layer: nn.TransformerEncoderLayer):
        super().__init__()
        self.self_attn = DecomposedAttention(layer.self_attn)
        self.linear1, self.linear2, self.dropout = layer.linear1, layer.linear2, layer.dropout
        self.norm1, self.norm2, self.dropout1, self.dropout2 = layer.norm1, layer.norm2, layer.dropout1, layer.dropout2
        self.activation, self.norm_first = layer.activation, layer.norm_first

    def _attention(self, x):
        return self.dropout1(self.self_attn(x, x, x, need_weights=False)[0])

    def _feed_forward(self, x):
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._attention(self.norm1(x))
            return x + self._feed_forward(self.norm2(x))
        x = self.norm1(x + self._attention(x))
        return self.norm2(x + self._feed_forward(x))


class DecomposedEncoder(nn.Module):
    """``nn.TransformerEncoder`` without masks: its layers decomposed in order, then its final norm."""

    def __init__(self, encoder: nn.TransformerEncoder):
        super().__init__()
        self.layers = nn.ModuleList(DecomposedEncoderLayer(cast(nn.TransformerEncoderLayer, layer))
                                    for layer in encoder.layers)
        self.norm = encoder.norm

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        out = src
        for layer in self.layers:
            out = layer(out)
        return out if self.norm is None else self.norm(out)
