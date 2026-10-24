"""Every operand of ``torch.cat`` is carried: activations, parameters and buffers, and their per-batch expansions.

Before, an operand without a mapper was dropped unless it was the first of exactly
two operands at ``dim=1`` (a class token); the probe passed when the dropped operand
never won (a zero floor token under a max on positive inputs).
"""

from __future__ import annotations

import io
import pickle

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mapping_utils import ComputeOpMapper, ConcatMapper
from mimarsinan.mapping.support.compute_modules import ComputeAdapter, _cat_along
from mimarsinan.torch_mapping.converter_handlers.refusals import ConcatenationOperandError
from mimarsinan.torch_mapping.mapper_graph_fx import GenericOpArgumentError

from .harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_refused_by_name,
    assert_value_program_exact,
    convert,
    native,
)


class PadMax(nn.Module):
    """The review's witness: a max over time with a learned floor token appended (zero-initialised)."""

    def __init__(self, floor=0.0):
        super().__init__()
        self.conv = nn.Conv1d(3, 8, 3, padding=1)
        self.floor = nn.Parameter(torch.full((1, 8, 1), floor))
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        y = torch.cat([self.conv(x), self.floor.expand(x.size(0), -1, -1)], dim=2)
        return self.fc(torch.amax(y, dim=2))


class NullLogit(nn.Module):
    """A learned null logit appended before a log-sum-exp (the review's 2.06e-11 witness at -22)."""

    def __init__(self, null=-22.0):
        super().__init__()
        self.fc = nn.Linear(8, 6)
        self.null = nn.Parameter(torch.full((1, 1), null))

    def forward(self, x):
        z = torch.cat([self.fc(x), self.null.expand(x.size(0), -1)], dim=1)
        return torch.logsumexp(z, dim=1, keepdim=True) - z[:, :6]


class Tokens(nn.Module):
    """Tokens (B, N, C) with learned tokens concatenated at ``where`` along ``dim``."""

    def __init__(self, where, dim=1, count=2):
        super().__init__()
        self.where, self.dim = where, dim
        self.embed = nn.Linear(6, 8)
        shape = (1, count, 8) if dim == 1 else (1, 5, count)
        self.extra = nn.Parameter(torch.randn(shape))
        self.second = nn.Linear(6, 8)
        self.head = nn.Linear(8 if dim == 1 else 8 + count, 3)

    def forward(self, x):
        t = self.embed(x)
        e = self.extra.expand(x.shape[0], -1, -1)
        if self.where == "prefix":
            parts = [e, t]
        elif self.where == "suffix":
            parts = [t, e]
        else:
            parts = [t, e, self.second(x)]
        y = torch.cat(parts, dim=self.dim)
        return self.head(y.mean(dim=1)) if self.dim == 1 else self.head(y).mean(dim=1)


class LastAxisSuffix(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)
        self.tail = nn.Parameter(torch.randn(1, 2))
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        return self.head(torch.cat([self.fc(x), self.tail.expand(x.size(0), -1)], dim=-1))


class DerivedToken(nn.Module):
    """A token computed from a parameter only (``2 * p``, then expanded): a batch-free value (coverage W6)."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(6, 8)
        self.extra = nn.Parameter(torch.randn(1, 1, 8))
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        t = self.embed(x)
        token = (2.0 * self.extra).expand(x.shape[0], -1, -1)
        return self.head(torch.cat([t, token], dim=1).mean(dim=1))


class BatchAxisCat(nn.Module):
    def __init__(self, explicit=True):
        super().__init__()
        self.explicit = explicit
        self.a = nn.Linear(6, 4)
        self.b = nn.Linear(6, 4)

    def forward(self, x):
        parts = [self.a(x), self.b(x)]
        return torch.cat(parts, 0) if self.explicit else torch.cat(parts)


class TwoActivations(nn.Module):
    """A channel concatenation of two conv outputs (the layout every IR lowering expresses)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.a = nn.Conv2d(3, 4, 3, padding=1)
        self.b = nn.Conv2d(3, 5, 3, padding=1)
        self.head = nn.Linear(9 * 4 * 4, 3)

    def forward(self, x):
        return self.head(torch.flatten(torch.relu(torch.cat([self.a(x), self.b(x)], dim=self.dim)), 1))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("floor", [0.0, -0.3])
def test_floor_token_under_a_max_is_exact_on_silence_and_negative_inputs(floor, contract):
    assert_converted_exact(native(lambda: PadMax(floor)), (3, 16), contract)


def test_floor_token_value_program_is_exact():
    assert_value_program_exact(native(PadMax), (3, 16), kinds=("zero", "negative", "normal"))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("null", [-22.0, -25.0, 0.5])
def test_null_logit_is_exact(null, contract):
    assert_converted_exact(native(lambda: NullLogit(null)), (8,), contract)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("where,dim", [("prefix", 1), ("suffix", 1), ("middle", 1), ("prefix", 2), ("suffix", 2)])
def test_learned_tokens_at_any_position_and_axis_are_exact(where, dim, contract):
    assert_converted_exact(native(lambda: Tokens(where, dim)), (5, 6), contract)


@pytest.mark.parametrize("where", ["suffix", "middle"])
def test_learned_tokens_value_program_is_exact(where):
    assert_value_program_exact(native(lambda: Tokens(where, 1)), (5, 6))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_parameter_operand_on_the_last_axis_is_exact(contract):
    assert_converted_exact(native(LastAxisSuffix), (8,), contract)


class RandomToken(nn.Module):
    """A token drawn at random at every call (``rand_like`` of a parameter), then expanded: an operand with no
    conversion rule."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(6, 8)
        self.extra = nn.Parameter(torch.randn(1, 1, 8))
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        t = self.embed(x)
        token = torch.rand_like(self.extra).expand(x.shape[0], -1, -1)
        return self.head(torch.cat([t, token], dim=1).mean(dim=1))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_operand_without_a_rule_is_refused_by_name(contract):
    """[changed in coverage W6: the witness was ``DerivedToken``, a token computed from a parameter, which is now a
    batch-free operand (``test_token_computed_from_a_parameter_is_exact``)]"""
    model = native(RandomToken)
    assert_refused_by_name(lambda: convert(model, (5, 6), contract), GenericOpArgumentError, "'cat'", "expand")


@pytest.mark.parametrize("contract", CONTRACTS)
def test_token_computed_from_a_parameter_is_exact(contract):
    """Coverage W6: ``(2 * p).expand(B, -1, -1)`` concatenated, one sample's slice computed at every call from the
    converted model's copy of ``p`` (on the base: ``GenericOpArgumentError`` naming ``cat`` and ``expand``)."""
    assert_converted_exact(native(DerivedToken), (5, 6), contract)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("explicit", [True, False], ids=["dim0", "default_dim"])
def test_concatenation_along_the_batch_axis_is_refused_by_name(explicit, contract):
    model = native(lambda: BatchAxisCat(explicit))
    assert_refused_by_name(lambda: convert(model, (6,), contract), ConcatenationOperandError, "'cat'", "batch axis")


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("dim", [1, -3])
def test_activation_only_concatenation_keeps_its_concat_mapper(dim, contract):
    flow = assert_converted_exact(native(lambda: TwoActivations(dim)), (3, 4, 4), contract)
    cats = [m for m in flow.get_mapper_repr().execution_order() if isinstance(m, ConcatMapper)]
    assert len(cats) == 1 and cats[0].dim == dim


def _tensor_marker(*_):
    raise RuntimeError("canonical bytes only")


class _CanonicalPickler(pickle.Pickler):
    def reducer_override(self, obj):
        if isinstance(obj, torch.Tensor):
            t = obj.detach().cpu().contiguous()
            raw = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
            return (_tensor_marker, (type(obj).__name__, str(t.dtype), tuple(t.shape), bool(obj.requires_grad), raw))
        return NotImplemented


def _canonical_bytes(obj) -> bytes:
    buffer = io.BytesIO()
    _CanonicalPickler(buffer, protocol=4).dump(obj)
    return buffer.getvalue()


class ClassToken(nn.Module):
    """The layout converted exactly before (a ViT class token): its adapter keeps the pre-change bytes."""

    def __init__(self, token_shape):
        super().__init__()
        self.embed = nn.Linear(6, 8)
        self.cls = nn.Parameter(torch.randn(token_shape))
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        t = self.embed(x)
        y = torch.cat([self.cls.expand(x.shape[0], -1, -1), t], dim=1)
        return self.head(y.mean(dim=1))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("token_shape", [(1, 1, 8), (1, 8)])
def test_class_token_prefix_keeps_the_pre_change_adapter_bytes(token_shape, contract):
    model = native(lambda: ClassToken(token_shape))
    flow = assert_converted_exact(model, (5, 6), contract)
    (mapper,) = [m for m in flow.get_mapper_repr().execution_order()
                 if isinstance(m, ComputeOpMapper) and m.name == "cat"]
    prefix = model.cls.detach().clone()
    prefix = prefix.squeeze(0) if prefix.dim() == 3 else prefix
    legacy = ComputeAdapter(_cat_along, bound_tensors=[prefix], kwargs={"dim": 1}).train(mapper.module.training)
    assert _canonical_bytes(mapper.module) == _canonical_bytes(legacy)
    assert len(mapper.sources) == 1


class AllBoundCat(nn.Module):
    """Every operand is a per-batch expansion of a parameter: the concatenation has no activation source."""

    def __init__(self, nested: bool):
        super().__init__()
        self.nested = nested
        self.fc = nn.Linear(6, 6)
        self.a = nn.Parameter(torch.randn(1, 3))
        self.b = nn.Parameter(torch.randn(1, 3))

    def forward(self, x):
        b = x.shape[0]
        tokens = torch.cat([self.a.expand(b, -1), self.b.expand(b, -1)], 1)
        return torch.cat([self.fc(x), tokens], 1) if self.nested else self.fc(x) + tokens


class StaticDimCat(nn.Module):
    """``dim`` computed from a static shape value (``a.dim() - 1``), known at conversion time."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(6, 6)
        self.b = nn.Linear(6, 6)
        self.h = nn.Linear(12, 3)

    def forward(self, x):
        a, b = self.a(x), self.b(x)
        return self.h(torch.cat([a, b], dim=a.dim() - 1))


class StaticChannelDimCat(TwoActivations):
    """The channel concatenation of two conv outputs with ``dim = a.dim() - 3`` (the layout the IR expresses)."""

    def __init__(self):
        super().__init__(1)

    def forward(self, x):
        a, b = self.a(x), self.b(x)
        return self.head(torch.flatten(torch.relu(torch.cat([a, b], dim=a.dim() - 3)), 1))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("nested", [False, True], ids=["added", "concatenated"])
def test_concatenation_of_bound_operands_only_is_refused_by_name(nested, contract):
    model = native(lambda: AllBoundCat(nested))
    assert_refused_by_name(lambda: convert(model, (6,), contract), ConcatenationOperandError, "'cat'", "activation")


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory,shape", [(StaticDimCat, (6,)), (StaticChannelDimCat, (3, 4, 4))],
                         ids=["vector", "channels"])
def test_dim_from_a_static_shape_value_is_exact(factory, shape, contract):
    flow = assert_converted_exact(native(factory), shape, contract)
    (cat,) = [m for m in flow.get_mapper_repr().execution_order() if isinstance(m, ConcatMapper)]
    assert cat.dim == 1


def test_dim_from_a_static_shape_value_program_is_exact():
    """The channel layout (a vector concatenation's IR lowering is open issue 1 of review round 1)."""
    assert_value_program_exact(native(StaticChannelDimCat), (3, 4, 4))


# ----------------------------------------------------------------------------- review round 3: the IR axis of ``dim``
class ActivationCat(nn.Module):
    """Activation-only concatenations the ``ConcatMapper`` carries, by layout and consumer."""

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self.a, self.b, self.b5 = nn.Linear(6, 4), nn.Linear(6, 4), nn.Linear(6, 5)
        self.h, self.h8, self.h4 = nn.Linear(9, 3), nn.Linear(8, 3), nn.Linear(4, 3)
        self.c1, self.c2, self.hc = nn.Conv2d(2, 3, 3, padding=1), nn.Conv2d(2, 4, 3, padding=1), nn.Conv2d(7, 2, 1)

    def forward(self, x):
        k = self.kind
        if k == "vector_into_linear":
            return self.h(torch.cat([self.a(x), self.b5(x)], 1))
        if k == "vector_last_dim":
            return self.h(torch.cat([self.a(x), self.b5(x)], -1))
        if k == "vector_into_host_op":
            return torch.sin(torch.cat([self.a(x), self.b5(x)], 1)) * 2
        if k == "vector_input_slice":
            return self.h(torch.cat([x[:, :4] * 1.0, self.b5(x)], 1))
        if k == "vector_output":
            return torch.cat([self.a(x), self.b(x)], 1)
        if k == "tokens_features_into_linear":
            return self.h8(torch.cat([self.a(x), self.b(x)], 2)).flatten(1)
        if k == "tokens_features_output":
            return torch.cat([self.a(x), self.b(x)], 2)
        if k == "tokens":
            return self.h4(torch.cat([self.a(x), self.b(x)], 1)).flatten(1)
        if k == "channels":
            return self.hc(torch.cat([self.c1(x), self.c2(x)], 1))
        raise ValueError(k)


ACTIVATION_CATS = {
    "vector_into_linear": (6,), "vector_last_dim": (6,), "vector_into_host_op": (6,), "vector_input_slice": (6,),
    "vector_output": (6,), "tokens_features_into_linear": (3, 6), "tokens_features_output": (3, 6),
    "tokens": (3, 6), "channels": (2, 5, 5),
}


@pytest.mark.parametrize("kind", sorted(ACTIVATION_CATS))
def test_an_activation_concatenation_lowers_exactly(kind):
    """The IR concatenates along ``dim`` counted from the end (axis 0 before: vectors and features crashed, and
    ``tokens_features_output`` was a wrong value program at 1.2, the spiking IR the same layout)."""
    assert_value_program_exact(native(lambda: ActivationCat(kind)), ACTIVATION_CATS[kind])


@pytest.mark.parametrize("kind", ["vector_into_linear", "tokens_features_into_linear", "channels"])
def test_an_activation_concatenation_passes_stage_2(kind):
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING

    flow = convert_torch_model(native(lambda: ActivationCat(kind)), ACTIVATION_CATS[kind], 3,
                               packaging=MVM_PACKAGING, value_program_stage=True)
    assert flow.conversion_certificate.certified


def _concat_ir(model, shape):
    """The ConcatMapper's IR sources, and its operands' (from the MVM conversion's identity IR mapping)."""
    from mimarsinan.mapping.ir_mapping_class import IRMapping

    flow = convert(model, shape, "mvm")
    repr_ = flow.get_mapper_repr()
    IRMapping(q_max=1.0, firing_mode="Default", max_axons=None, max_neurons=None,
              allow_coalescing=False, hardware_bias=False).map(repr_)
    (cat,) = [m for m in repr_.execution_order() if isinstance(m, ConcatMapper)]
    return cat._ir_sources, [m._ir_sources for m in cat.get_source_mappers()]


@pytest.mark.parametrize("kind", ["channels", "tokens"])
def test_a_leading_axis_concatenation_keeps_its_ir(kind):
    """Channels of (B, C, H, W) and tokens of (B, N, D): ``dim`` from the end is IR axis 0, as before."""
    import numpy as np

    concatenated, operands = _concat_ir(native(lambda: ActivationCat(kind)), ACTIVATION_CATS[kind])
    legacy = np.concatenate(operands, axis=0)
    assert concatenated.shape == legacy.shape and all(a is b for a, b in zip(concatenated.flat, legacy.flat))


def test_an_output_vector_concatenation_keeps_its_source_order():
    """Declared layout change: (1, 4) + (1, 4) concatenate to (1, 8), not (2, 4); the sources keep their order."""
    import numpy as np

    concatenated, operands = _concat_ir(native(lambda: ActivationCat("vector_output")), (6,))
    legacy = np.concatenate(operands, axis=0)
    assert [o.shape for o in operands] == [(1, 4), (1, 4)] and concatenated.shape == (1, 8)
    assert all(a is b for a, b in zip(concatenated.flat, legacy.flat))


def test_a_concatenation_whose_output_moves_the_batch_axis_is_refused_by_name():
    class _TransposedCat(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 6)

        def forward(self, x):
            y = self.fc(x)
            return torch.cat([y.T, y.T], 1) * 2

    assert_refused_by_name(lambda: convert(native(_TransposedCat), (8,), "spiking"), ConcatenationOperandError,
                           "'cat'", "along dim 1", "axis 0 is not the batch axis")


@pytest.mark.parametrize("dim,rank,axis", [(1, None, 0), (-1, None, 0), (1, 4, -3), (-1, 3, -1)])
def test_the_ir_concatenation_axis_counts_from_the_end_and_a_pickle_without_rank_keeps_axis_0(dim, rank, axis):
    """Review round 4: a pickle from before ``rank`` concatenated along IR axis 0 for every ``dim``, negative too."""
    assert ConcatMapper([], dim=dim, rank=rank).ir_axis() == axis


def test_a_pickle_without_rank_concatenates_along_ir_axis_0():
    import numpy as np

    class _Source:
        def __init__(self, array):
            self.array = array

        def map_to_ir(self, _ir_mapping):
            return self.array

    concat = ConcatMapper([_Source(np.zeros((2, 3))), _Source(np.ones((1, 3)))], dim=-1)
    assert concat.rank is None and concat._map_to_ir(None).shape == (3, 3)
