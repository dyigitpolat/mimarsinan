"""Unit tests for :class:`ComputeAdapter` — display_name, from_fx_node, pickling,
broadcasting bound tensors, argument templates, extra args / kwargs routing.
"""

from __future__ import annotations

import operator
import pickle
from typing import Any

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.support.call_template import (
    ArgTemplate,
    BoundSlot,
    LiteralSlot,
    SourceSlot,
)
from mimarsinan.mapping.support.compute_modules import ComputeAdapter, _cat_along


class TestDisplayName:
    def test_includes_function_name(self):
        # display_name must surface the callable's identity in some readable form.
        assert "add" in ComputeAdapter(operator.add).display_name

    def test_torch_function(self):
        assert "mean" in ComputeAdapter(torch.mean).display_name

    def test_getitem(self):
        assert "getitem" in ComputeAdapter(operator.getitem).display_name

    def test_named_function(self):
        assert "_cat_along" in ComputeAdapter(_cat_along).display_name

    def test_class_callable_falls_back_to_type_name(self):
        class _OpClass:
            def __call__(self, x): return x

        # _OpClass instances have no __name__ on the instance — adapter
        # falls back to type(fn).__name__.
        adapter = ComputeAdapter(_OpClass())
        assert "_OpClass" in adapter.display_name


class TestFromFxNode:
    def _trace(self, mod):
        return fx.symbolic_trace(mod)

    def test_extracts_positional_extra_args(self):
        class _Mod(nn.Module):
            def forward(self, x):
                return torch.flatten(x, 1)

        gm = self._trace(_Mod())
        flatten_node = next(n for n in gm.graph.nodes if n.target is torch.flatten)
        adapter = ComputeAdapter.from_fx_node(flatten_node, torch.flatten)
        assert adapter.extra_args == (1,)

    def test_extracts_kwargs(self):
        class _Mod(nn.Module):
            def forward(self, x):
                return torch.mean(x, dim=1)

        gm = self._trace(_Mod())
        mean_node = next(n for n in gm.graph.nodes if n.target is torch.mean)
        adapter = ComputeAdapter.from_fx_node(mean_node, torch.mean)
        assert adapter.kwargs == {"dim": 1}

    def test_skips_node_args(self):
        class _Mod(nn.Module):
            def forward(self, x):
                return operator.add(x, x)

        gm = self._trace(_Mod())
        add_node = next(
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target is operator.add
        )
        adapter = ComputeAdapter.from_fx_node(add_node, operator.add)
        assert adapter.extra_args == ()


class TestFromFxNodeKeepsOperandOrder:
    def test_literal_left_operand(self):
        class _Mod(nn.Module):
            def forward(self, x):
                return 2.0 - x

        gm = fx.symbolic_trace(_Mod())
        sub_node = next(n for n in gm.graph.nodes if n.target is operator.sub)
        adapter = ComputeAdapter.from_fx_node(sub_node, operator.sub)
        x = torch.tensor([[1.0, 4.0]])
        assert torch.equal(adapter(x), 2.0 - x)


class TestBatchExpansion:
    def test_no_bound_passes_inputs_through(self):
        adapter = ComputeAdapter(operator.add)
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        assert torch.allclose(adapter(a, b), torch.tensor([[4.0, 6.0]]))

    def test_bound_expands_to_input_batch(self):
        const = torch.tensor([10.0, 20.0])
        adapter = ComputeAdapter(operator.add, bound_tensors=[const])

        x = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        out = adapter(x)
        expected = torch.tensor([[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]])
        assert torch.allclose(out, expected)

    def test_higher_rank_bound(self):
        cls_token = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # (1, 4)
        adapter = ComputeAdapter(
            _cat_along, bound_tensors=[cls_token], kwargs={"dim": 1},
        )
        x = torch.randn(7, 3, 4)
        out = adapter(x)
        assert out.shape == (7, 4, 4)
        assert torch.allclose(out[:, 0], torch.full((7, 4), 0.5))

    def test_batch_size_one(self):
        const = torch.tensor([1.0, 2.0, 3.0])
        adapter = ComputeAdapter(operator.add, bound_tensors=[const])
        out = adapter(torch.zeros(1, 3))
        assert torch.allclose(out, const.unsqueeze(0))

    def test_kwargs_forwarded(self):
        adapter = ComputeAdapter(torch.mean, kwargs={"dim": 0})
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = adapter(x)
        assert torch.allclose(out, torch.tensor([2.5, 3.5, 4.5]))

    def test_extra_args_forwarded(self):
        adapter = ComputeAdapter(operator.getitem, extra_args=((slice(None), 1),))
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = adapter(x)
        assert torch.allclose(out, torch.tensor([2.0, 5.0]))


class TestPickleRoundTrip:
    def test_simple_callable_roundtrips(self):
        adapter = ComputeAdapter(operator.add)
        loaded = pickle.loads(pickle.dumps(adapter))
        a, b = torch.tensor([[1.0]]), torch.tensor([[2.0]])
        assert torch.allclose(adapter(a, b), loaded(a, b))

    def test_bound_tensors_survive_pickle(self):
        const = torch.tensor([7.0, 14.0])
        adapter = ComputeAdapter(operator.add, bound_tensors=[const])
        loaded = pickle.loads(pickle.dumps(adapter))

        x = torch.zeros(2, 2)
        assert torch.allclose(adapter(x), loaded(x))
        assert torch.allclose(loaded.bound_0.data, const)

    def test_kwargs_extra_args_survive_pickle(self):
        adapter = ComputeAdapter(
            operator.getitem, extra_args=((slice(None), 0),),
        )
        loaded = pickle.loads(pickle.dumps(adapter))
        assert loaded.extra_args == ((slice(None), 0),)


class TestBoundTensorIsNonTrainable:
    def test_bound_param_requires_grad_false(self):
        const = torch.tensor([1.0, 2.0])
        adapter = ComputeAdapter(operator.add, bound_tensors=[const])
        assert adapter.bound_0.requires_grad is False

    def test_bound_tensors_isolated_from_source(self):
        const = torch.tensor([1.0])
        adapter = ComputeAdapter(operator.add, bound_tensors=[const])
        const.add_(99.0)
        # adapter holds a detached clone — source mutation does not leak.
        assert adapter.bound_0.item() == 1.0


class TestZeroInputDegenerate:
    def test_batch_size_defaults_to_one_when_no_inputs(self):
        # No source inputs: forward should still produce a valid call.
        const = torch.tensor([5.0])

        def _identity(x):
            return x

        adapter = ComputeAdapter(_identity, bound_tensors=[const])
        out = adapter()
        # bound expanded with batch=1 → shape (1, 1)
        assert out.shape == (1, 1)
        assert out.item() == 5.0


# ── D12: bound tensors broadcast against the inputs by trailing alignment ──

def _seeded(*shape):
    return torch.randn(*shape, dtype=torch.float64, generator=torch.Generator().manual_seed(len(shape)))


class TestTrailingBroadcast:
    """A bound tensor is never given a forced leading batch axis (D12)."""

    @pytest.mark.parametrize("batch", [1, 3])
    def test_rank_one_bound_over_tokens(self, batch):
        gamma = _seeded(6)
        x = _seeded(batch, 5, 6)
        adapter = ComputeAdapter(operator.mul, bound_tensors=[gamma])
        assert torch.equal(adapter(x), x * gamma)

    def test_rank_one_bound_when_the_batch_equals_the_token_count(self):
        gamma = _seeded(6)
        x = _seeded(3, 3, 6)
        assert torch.equal(ComputeAdapter(operator.mul, bound_tensors=[gamma])(x), x * gamma)

    @pytest.mark.parametrize("batch", [1, 3])
    def test_rank_zero_bound(self, batch):
        scale = torch.tensor(1.5, dtype=torch.float64)
        x = _seeded(batch, 4)
        assert torch.equal(ComputeAdapter(operator.mul, bound_tensors=[scale])(x), x * scale)

    def test_matrix_operand_of_a_functional_op(self):
        weight = _seeded(4, 6)
        x = _seeded(3, 5, 6)
        adapter = ComputeAdapter(F.linear, bound_tensors=[weight])
        assert torch.equal(adapter(x), F.linear(x, weight))


class TestArgTemplateEvaluation:
    """A template evaluates the payload callable in the traced argument order (D11)."""

    def test_bound_left_operand(self):
        c = _seeded(6)
        x = _seeded(3, 6)
        adapter = ComputeAdapter(
            operator.sub, bound_tensors=[c],
            arg_template=ArgTemplate((BoundSlot(0), SourceSlot(0))),
        )
        assert torch.equal(adapter(x), c - x)

    def test_literal_left_operand(self):
        x = _seeded(3, 6)
        adapter = ComputeAdapter(
            operator.truediv, extra_args=(2.0,),
            arg_template=ArgTemplate((LiteralSlot(0), SourceSlot(0))),
        )
        assert torch.equal(adapter(x), 2.0 / x)

    def test_keyword_literals_join_the_template(self):
        c = _seeded(6)
        x = _seeded(3, 6)
        adapter = ComputeAdapter(
            torch.sub, bound_tensors=[c], kwargs={"alpha": 2},
            arg_template=ArgTemplate((BoundSlot(0), SourceSlot(0))),
        )
        assert torch.equal(adapter(x), torch.sub(c, x, alpha=2))

    def test_keyword_bound_tensors(self):
        w, b = _seeded(6) + 2.0, _seeded(6)
        x = _seeded(3, 5, 6)
        adapter = ComputeAdapter(
            F.layer_norm, bound_tensors=[w, b], extra_args=((6,),),
            arg_template=ArgTemplate(
                (SourceSlot(0), LiteralSlot(0)), (("weight", BoundSlot(0)), ("bias", BoundSlot(1))),
            ),
        )
        assert torch.equal(adapter(x), F.layer_norm(x, (6,), weight=w, bias=b))

    def test_bound_slot_restores_a_stripped_leading_axis(self):
        row = _seeded(1, 6)
        x = _seeded(3, 6)
        adapter = ComputeAdapter(
            F.linear, bound_tensors=[row.squeeze(0)],
            arg_template=ArgTemplate((SourceSlot(0), BoundSlot(0, (1, 6)))),
        )
        assert torch.equal(adapter(x), F.linear(x, row))

    def test_template_survives_pickle(self):
        c = _seeded(6)
        adapter = ComputeAdapter(
            operator.sub, bound_tensors=[c],
            arg_template=ArgTemplate((BoundSlot(0), SourceSlot(0))),
        )
        loaded = pickle.loads(pickle.dumps(adapter))
        x = _seeded(2, 6)
        assert torch.equal(loaded(x), c - x)
        assert loaded.arg_template == adapter.arg_template


class TestCanonicalAdapterState:
    """Adapters in the canonical layout carry no template, so their state is the pre-template state."""

    def test_canonical_adapter_has_no_template_attribute(self):
        adapter = ComputeAdapter(operator.add, bound_tensors=[torch.ones(3)])
        assert "arg_template" not in adapter.__dict__
        assert adapter.arg_template is None

    def test_explicit_none_template_is_not_stored(self):
        adapter = ComputeAdapter(operator.add, arg_template=None)
        assert "arg_template" not in adapter.__dict__

    def test_templated_adapter_stores_its_template(self):
        template = ArgTemplate((BoundSlot(0), SourceSlot(0)))
        adapter = ComputeAdapter(operator.sub, bound_tensors=[torch.ones(3)], arg_template=template)
        assert adapter.__dict__["arg_template"] == template


class TestCatAlongReplicatesThePrefix:
    def test_batch_stripped_prefix_is_replicated_per_sample(self):
        prefix = _seeded(1, 4)
        x = _seeded(7, 3, 4)
        out = _cat_along(x, prefix, dim=1)
        assert out.shape == (7, 4, 4)
        assert torch.equal(out[:, :1], prefix.expand(7, 1, 4))
        assert torch.equal(out[:, 1:], x)
