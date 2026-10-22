"""Positional call templates: an FX call's argument order kept through source, bound and literal slots."""

from __future__ import annotations

import operator
import pickle
from collections import namedtuple

import pytest
import torch
import torch.fx as fx
import torch.nn.functional as F

from mimarsinan.mapping.support.call_template import (
    COMMUTATIVE_BINARY_FNS,
    ArgTemplate,
    BoundArg,
    BoundSlot,
    LiteralArg,
    LiteralSlot,
    SourceArg,
    SourceSlot,
    partition_call,
)

_CONSTANTS = {
    "c": torch.arange(6, dtype=torch.float64),
    "d": torch.arange(6, dtype=torch.float64) + 10.0,
}


def _call(fn, build):
    """One ``call_function`` node over placeholders ``x``, ``y`` and attributes ``c``, ``d``."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    c = graph.get_attr("c")
    d = graph.get_attr("d")
    args, kwargs = build(x, y, c, d)
    return graph.call_function(fn, tuple(args), dict(kwargs))


def _classify(node: fx.Node):
    if node.op == "placeholder":
        return SourceArg(node.name)
    if node.op == "get_attr":
        tensor = _CONSTANTS[str(node.target)]
        return BoundArg(tensor, tuple(tensor.shape))
    return None


def _partition(fn, build):
    node = _call(fn, build)
    return partition_call(fn, node.args, node.kwargs, _classify)


class TestCanonicalLayout:
    """Sources, then bound tensors, then literals, in FX order: no template is stored."""

    def test_source_then_bound(self):
        part = _partition(torch.sub, lambda x, y, c, d: ((x, c), {}))
        assert part.template is None
        assert part.sources == ["x"]
        assert [t is _CONSTANTS["c"] for t in part.bound] == [True]
        assert part.literals == ()

    def test_source_then_literals(self):
        part = _partition(torch.mean, lambda x, y, c, d: ((x, (2, 3)), {}))
        assert part.template is None
        assert part.literals == ((2, 3),)

    def test_literal_kwargs_stay_keyword_literals(self):
        part = _partition(torch.sub, lambda x, y, c, d: ((x, c), {"alpha": 2}))
        assert part.template is None
        assert part.literal_kwargs == {"alpha": 2}

    def test_sources_are_numbered_in_appearance_order(self):
        part = _partition(torch.addcmul, lambda x, y, c, d: ((y, x, x), {}))
        assert part.template is None
        assert part.sources == ["y", "x", "x"]


class TestOriginalOrderTemplate:
    """Any other order is kept slot by slot, so the op evaluates its arguments as traced."""

    def test_bound_left_of_source(self):
        part = _partition(operator.sub, lambda x, y, c, d: ((c, x), {}))
        assert part.template == ArgTemplate((BoundSlot(0), SourceSlot(0)))

    def test_literal_left_of_source(self):
        part = _partition(operator.truediv, lambda x, y, c, d: ((2.0, x), {}))
        assert part.template == ArgTemplate((LiteralSlot(0), SourceSlot(0)))
        assert part.literals == (2.0,)

    def test_bound_left_with_keyword_literal(self):
        part = _partition(torch.sub, lambda x, y, c, d: ((c, x), {"alpha": 2}))
        assert part.template == ArgTemplate((BoundSlot(0), SourceSlot(0)))
        assert part.literal_kwargs == {"alpha": 2}

    def test_two_bound_before_the_source(self):
        part = _partition(torch.addcmul, lambda x, y, c, d: ((d, c, x), {}))
        assert part.template == ArgTemplate((BoundSlot(0), BoundSlot(1), SourceSlot(0)))
        assert part.bound[0] is _CONSTANTS["d"] and part.bound[1] is _CONSTANTS["c"]

    def test_literal_between_source_and_bound(self):
        part = _partition(F.layer_norm, lambda x, y, c, d: ((x, (6,), c, d), {}))
        assert part.template == ArgTemplate(
            (SourceSlot(0), LiteralSlot(0), BoundSlot(0), BoundSlot(1)),
        )

    def test_keyword_bound_tensors(self):
        part = _partition(
            F.layer_norm, lambda x, y, c, d: ((x, (6,)), {"weight": c, "bias": d}),
        )
        assert part.template == ArgTemplate(
            (SourceSlot(0), LiteralSlot(0)),
            (("weight", BoundSlot(0)), ("bias", BoundSlot(1))),
        )
        assert part.literal_kwargs == {}

    def test_keyword_source(self):
        part = _partition(torch.add, lambda x, y, c, d: ((x,), {"other": y}))
        assert part.template == ArgTemplate((SourceSlot(0),), (("other", SourceSlot(1)),))
        assert part.sources == ["x", "y"]

    def test_nested_container_of_sources(self):
        part = _partition(torch.stack, lambda x, y, c, d: (([x, y],), {"dim": 1}))
        assert part.template == ArgTemplate(([SourceSlot(0), SourceSlot(1)],))
        assert part.literal_kwargs == {"dim": 1}
        assert part.literals == ()

    def test_nested_container_keeps_its_literal_leaves(self):
        part = _partition(
            operator.getitem, lambda x, y, c, d: ((x, (slice(None), c)), {}),
        )
        assert part.template == ArgTemplate((SourceSlot(0), (slice(None), BoundSlot(0))))


class TestCommutativeCanonicalization:
    """A commutative binary op with a bound or literal left operand keeps the canonical layout."""

    @pytest.mark.parametrize("fn", [operator.add, operator.mul, torch.add, torch.mul])
    def test_bound_left(self, fn):
        part = _partition(fn, lambda x, y, c, d: ((c, x), {}))
        assert part.template is None
        assert part.sources == ["x"]
        assert part.bound[0] is _CONSTANTS["c"]

    @pytest.mark.parametrize("fn", [operator.add, operator.mul])
    def test_literal_left(self, fn):
        part = _partition(fn, lambda x, y, c, d: ((2.0, x), {}))
        assert part.template is None
        assert part.literals == (2.0,)

    def test_keyword_arguments_disable_the_swap(self):
        part = _partition(torch.add, lambda x, y, c, d: ((c, x), {"alpha": 2}))
        assert part.template == ArgTemplate((BoundSlot(0), SourceSlot(0)))

    @pytest.mark.parametrize("fn", [operator.sub, operator.truediv, operator.pow, torch.sub])
    def test_non_commutative_ops_are_not_swapped(self, fn):
        assert fn not in COMMUTATIVE_BINARY_FNS
        part = _partition(fn, lambda x, y, c, d: ((c, x), {}))
        assert part.template is not None

    @pytest.mark.parametrize("fn", sorted(COMMUTATIVE_BINARY_FNS, key=repr))
    def test_every_listed_op_is_bitwise_commutative(self, fn):
        g = torch.Generator().manual_seed(0)
        a = torch.randn(4, 5, dtype=torch.float64, generator=g)
        b = torch.randn(5, dtype=torch.float64, generator=g)
        assert torch.equal(fn(a, b), fn(b, a))


class TestUnresolvedArguments:
    def test_an_argument_with_no_classification_is_reported(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        derived = graph.call_function(torch.zeros, (6,))
        node = graph.call_function(operator.sub, (derived, x))
        part = partition_call(operator.sub, node.args, node.kwargs, _classify)
        assert part.unresolved == [derived]


class TestResolve:
    def test_positional_slots_resolve_in_template_order(self):
        x, c = torch.ones(2, 3), torch.full((3,), 5.0)
        args, kwargs = ArgTemplate((BoundSlot(0), SourceSlot(0))).resolve((x,), (c,), ())
        assert args[0] is c and args[1] is x
        assert kwargs == {}

    def test_literal_slots_read_the_literal_table(self):
        args, _ = ArgTemplate((LiteralSlot(1), SourceSlot(0), LiteralSlot(0))).resolve(
            ("x",), (), ("a", "b"),
        )
        assert args == ("b", "x", "a")

    def test_keyword_slots_resolve_to_keywords(self):
        _, kwargs = ArgTemplate((SourceSlot(0),), (("weight", BoundSlot(0)),)).resolve(
            ("x",), ("w",), (),
        )
        assert kwargs == {"weight": "w"}

    def test_bound_slot_shape_restores_the_native_shape(self):
        stored = torch.arange(4.0)
        args, _ = ArgTemplate((BoundSlot(0, (1, 4)),)).resolve((), (stored,), ())
        assert args[0].shape == (1, 4)
        assert torch.equal(args[0].reshape(-1), stored)

    def test_nested_containers_keep_their_types(self):
        Pair = namedtuple("Pair", "a b")
        template = ArgTemplate(
            ([SourceSlot(0), (SourceSlot(1), 3)], slice(None, LiteralSlot(0)), Pair(SourceSlot(0), 7),
             {"k": BoundSlot(0)}),
        )
        args, _ = template.resolve(("x", "y"), ("c",), (5,))
        assert args == (["x", ("y", 3)], slice(None, 5), Pair("x", 7), {"k": "c"})
        assert isinstance(args[2], Pair)

    def test_templates_round_trip_through_pickle(self):
        template = ArgTemplate(
            (BoundSlot(0, (1, 6)), [SourceSlot(0), LiteralSlot(0)]), (("alpha", LiteralSlot(1)),),
        )
        assert pickle.loads(pickle.dumps(template)) == template


class TestExplicitTemplate:
    def test_canonical_partition_has_an_explicit_equivalent(self):
        part = _partition(torch.addcmul, lambda x, y, c, d: ((x, c, d, 2), {}))
        assert part.template is None
        assert part.explicit_template() == ArgTemplate(
            (SourceSlot(0), BoundSlot(0), BoundSlot(1), LiteralSlot(0)),
        )

    def test_native_shapes_are_written_only_where_storage_differs(self):
        def classify_stripping_c(node):
            if node.op == "get_attr" and node.target == "c":
                return BoundArg(_CONSTANTS["c"], (1, 6))
            return _classify(node)

        node = _call(torch.addcmul, lambda x, y, c, d: ((c, d, x), {}))
        part = partition_call(torch.addcmul, node.args, node.kwargs, classify_stripping_c)
        assert part.bound_native_shapes == [(1, 6), (6,)]
        assert part.strips_bound_shapes
        assert part.explicit_template(restore_native_shapes=True) == ArgTemplate(
            (BoundSlot(0, (1, 6)), BoundSlot(1), SourceSlot(0)),
        )
        assert part.explicit_template() == ArgTemplate(
            (BoundSlot(0), BoundSlot(1), SourceSlot(0)),
        )

    def test_unstripped_partition_reports_no_stripping(self):
        part = _partition(torch.sub, lambda x, y, c, d: ((c, x), {}))
        assert not part.strips_bound_shapes


class TestLiteralArguments:
    """An FX value the caller resolves at conversion time (a static shape value) is a literal, as if traced as one."""

    @staticmethod
    def _partition(fn, build, literals):
        graph = fx.Graph()
        x = graph.placeholder("x")
        c = graph.get_attr("c")
        values = {name: graph.placeholder(name) for name in literals}
        args, kwargs = build(x, c, values)
        node = graph.call_function(fn, tuple(args), dict(kwargs))

        def classify(arg: fx.Node):
            if arg.name in literals:
                return LiteralArg(literals[arg.name])
            return _classify(arg)

        return partition_call(fn, node.args, node.kwargs, classify)

    def test_positional_literal_node_is_a_canonical_literal(self):
        part = self._partition(operator.mul, lambda x, c, v: ((x, v["n"]), {}), {"n": 6})
        assert part.sources == ["x"] and part.literals == (6,)
        assert part.template is None and not part.unresolved

    def test_keyword_literal_nodes_are_keyword_literals(self):
        part = self._partition(
            torch.clamp, lambda x, c, v: ((x,), {"min": v["lo"], "max": v["hi"]}), {"lo": -6, "hi": 6},
        )
        assert part.literal_kwargs == {"min": -6, "max": 6}
        assert part.template is None and part.slot_kwargs == ()

    def test_literal_node_left_of_a_source_keeps_its_position(self):
        part = self._partition(operator.sub, lambda x, c, v: ((v["n"], x), {}), {"n": 6})
        assert part.template == ArgTemplate((LiteralSlot(0), SourceSlot(0)))
        args, _ = part.template.resolve(("x",), (), part.literals)
        assert args == (6, "x")

    def test_literal_nodes_nested_with_sources_resolve_in_place(self):
        part = self._partition(
            torch.reshape, lambda x, c, v: ((x, [v["a"], -1]), {}), {"a": 4},
        )
        assert part.literals == ([4, -1],)
        assert part.template is None

    def test_a_literal_tuple_value_is_kept_whole(self):
        part = self._partition(torch.reshape, lambda x, c, v: ((x, v["s"]), {}), {"s": (4, 8)})
        assert part.literals == ((4, 8),)

    def test_literal_classification_is_read_once_per_node(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        n = graph.placeholder("n")
        node = graph.call_function(operator.add, ((x, n), n), {"alpha": n})
        calls = []

        def classify(arg):
            calls.append(arg.name)
            return LiteralArg(3) if arg is n else SourceArg(arg.name)

        part = partition_call(torch.add, node.args, node.kwargs, classify)
        assert sorted(calls) == ["n", "x"]
        assert part.literal_kwargs == {"alpha": 3}
