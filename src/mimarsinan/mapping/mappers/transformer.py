"""Transformer / ViT mappers: LayerNorm, GELU, constant prepend/add, dropout, select, MHSA."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.mappers.base import Mapper


class LayerNormMapper(Mapper):
    """
    LayerNorm operation as a ComputeOp.

    Forward: applies the wrapped nn.LayerNorm.
    IR mapping: creates a ComputeOp with frozen weight/bias.
    """

    def __init__(self, source_mapper, layer_norm: nn.LayerNorm, name: str = "LayerNorm"):
        super().__init__(source_mapper)
        self._ln_container = nn.ModuleList([layer_norm])
        self.name = str(name)

    @property
    def layer_norm(self):
        return self._ln_container[0]

    def _map(self, mapping):
        raise NotImplementedError("LayerNormMapper: not supported in SoftCoreMapping.")

    def _map_to_ir(self, ir_mapping):
        input_sources = self.source_mapper.map_to_ir(ir_mapping)
        shape = input_sources.shape
        ln = self.layer_norm
        return ir_mapping.add_compute_op(
            input_sources=input_sources,
            op_type="layer_norm",
            params={
                "weight": ln.weight.detach().cpu().numpy().tolist(),
                "bias": ln.bias.detach().cpu().numpy().tolist(),
                "eps": ln.eps,
                "normalized_shape": list(ln.normalized_shape),
            },
            input_shape=shape,
            output_shape=shape,
            name=self.name,
        )

    def _forward_impl(self, x):
        return self.layer_norm(x)


class GELUMapper(Mapper):
    """
    GELU activation as a ComputeOp.

    Forward: F.gelu.
    IR mapping: ComputeOp node.
    """

    def __init__(self, source_mapper, name: str = "GELU"):
        super().__init__(source_mapper)
        self.name = str(name)

    def _map(self, mapping):
        raise NotImplementedError("GELUMapper: not supported in SoftCoreMapping.")

    def _map_to_ir(self, ir_mapping):
        input_sources = self.source_mapper.map_to_ir(ir_mapping)
        shape = input_sources.shape
        return ir_mapping.add_compute_op(
            input_sources=input_sources,
            op_type="gelu",
            params={},
            input_shape=shape,
            output_shape=shape,
            name=self.name,
        )

    def _forward_impl(self, x):
        return F.gelu(x)


class ConstantPrependMapper(Mapper):
    """
    Prepend a learnable constant along the sequence dimension (dim 0 of mapping sources).
    Used for the CLS token in ViT.

    Forward: (B, S, D) -> (B, S+1, D)  (constant is broadcast over batch)
    IR mapping: ComputeOp with stored constant.
    """

    def __init__(self, source_mapper, constant_param: nn.Parameter, name: str = "CLSPrepend"):
        super().__init__(source_mapper)
        self._constant_container = nn.ParameterList([constant_param])
        self.name = str(name)

    @property
    def constant_param(self):
        return self._constant_container[0]

    def _map(self, mapping):
        raise NotImplementedError("ConstantPrependMapper: not supported in SoftCoreMapping.")

    def _map_to_ir(self, ir_mapping):
        input_sources = self.source_mapper.map_to_ir(ir_mapping)
        S, D = input_sources.shape
        const_np = self.constant_param.detach().cpu().numpy().flatten()[:D].tolist()
        return ir_mapping.add_compute_op(
            input_sources=input_sources,
            op_type="concat_constant",
            params={"constant": const_np, "dim": 0},
            input_shape=(S, D),
            output_shape=(S + 1, D),
            name=self.name,
        )

    def _forward_impl(self, x):
        B = x.shape[0]
        const = self.constant_param.expand(B, -1, -1)
        return torch.cat([const, x], dim=1)


class ConstantAddMapper(Mapper):
    """
    Element-wise addition of a learnable constant (e.g. positional embedding).

    Forward: x + constant  (broadcast over batch)
    IR mapping: ComputeOp with stored constant.
    """

    def __init__(self, source_mapper, constant_param: nn.Parameter, name: str = "PosEmbedAdd"):
        super().__init__(source_mapper)
        self._constant_container = nn.ParameterList([constant_param])
        self.name = str(name)

    @property
    def constant_param(self):
        return self._constant_container[0]

    def _map(self, mapping):
        raise NotImplementedError("ConstantAddMapper: not supported in SoftCoreMapping.")

    def _map_to_ir(self, ir_mapping):
        input_sources = self.source_mapper.map_to_ir(ir_mapping)
        shape = input_sources.shape
        const_np = self.constant_param.detach().cpu().numpy().flatten().tolist()
        return ir_mapping.add_compute_op(
            input_sources=input_sources,
            op_type="add_constant",
            params={"constant": const_np},
            input_shape=shape,
            output_shape=shape,
            name=self.name,
        )

    def _forward_impl(self, x):
        return x + self.constant_param


class DropoutMapper(Mapper):
    """
    Dropout (training-only).  Identity during mapping / inference.
    """

    def __init__(self, source_mapper, p: float = 0.1, name: str = "Dropout"):
        super().__init__(source_mapper)
        self.dropout = nn.Dropout(p)
        self.name = str(name)

    def _map(self, mapping):
        return self.source_mapper.map(mapping)

    def _map_to_ir(self, ir_mapping):
        return self.source_mapper.map_to_ir(ir_mapping)

    def _forward_impl(self, x):
        return self.dropout(x)


class SelectMapper(Mapper):
    """
    Select a single index along the sequence (dim 1 of forward, dim 0 of mapping).

    Forward: (B, S, D) -> (B, D)
    IR mapping: (S, D) -> ComputeOp -> (D,)
    """

    def __init__(self, source_mapper, index: int = 0, name: str = "Select"):
        super().__init__(source_mapper)
        self.index = int(index)
        self.name = str(name)

    def _map(self, mapping):
        return self.source_mapper.map(mapping)[self.index]

    def _map_to_ir(self, ir_mapping):
        input_sources = self.source_mapper.map_to_ir(ir_mapping)
        if len(input_sources.shape) >= 2:
            S, D = input_sources.shape
            return ir_mapping.add_compute_op(
                input_sources=input_sources,
                op_type="select",
                params={"index": self.index},
                input_shape=(S, D),
                output_shape=(D,),
                name=self.name,
            )
        return input_sources

    def _forward_impl(self, x):
        return x[:, self.index]


class MultiHeadAttentionComputeMapper(Mapper):
    """
    Multi-head self-attention as a ComputeOp.

    Takes three source mappers (Q, K, V) and computes scaled dot-product attention.

    Forward:  receives (q, k, v) tuple of (B, S, D) tensors.
    IR mapping: concatenates Q/K/V sources and creates a ComputeOp.
    """

    def __init__(self, source_mappers, num_heads: int, attn_dropout: float = 0.0,
                 name: str = "MHSA"):
        super().__init__()
        self._source_mappers_list = list(source_mappers)
        self.num_heads = int(num_heads)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.name = str(name)

    @property
    def source_mappers(self):
        return self._source_mappers_list

    def get_source_mappers(self):
        return [m for m in self._source_mappers_list if m is not None]

    def _map(self, mapping):
        raise NotImplementedError("MultiHeadAttentionComputeMapper: not supported in SoftCoreMapping.")

    def _map_to_ir(self, ir_mapping):
        q_sources = self._source_mappers_list[0].map_to_ir(ir_mapping)
        k_sources = self._source_mappers_list[1].map_to_ir(ir_mapping)
        v_sources = self._source_mappers_list[2].map_to_ir(ir_mapping)

        assert q_sources.shape == k_sources.shape == v_sources.shape
        S, D = q_sources.shape

        all_sources = np.concatenate([
            q_sources.flatten(), k_sources.flatten(), v_sources.flatten()
        ])
        return ir_mapping.add_compute_op(
            input_sources=all_sources,
            op_type="multi_head_attention",
            params={
                "num_heads": self.num_heads,
                "d_model": D,
                "seq_len": S,
            },
            input_shape=(3, S, D),
            output_shape=(S, D),
            name=self.name,
        )

    def _forward_impl(self, x):
        q, k, v = x
        B, S, D = q.shape
        H = self.num_heads
        d_k = D // H

        q = q.view(B, S, H, d_k).transpose(1, 2)
        k = k.view(B, S, H, d_k).transpose(1, 2)
        v = v.view(B, S, H, d_k).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return out
