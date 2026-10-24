"""Values Dynamo folds into a captured body's Python (coverage W8, sign-off L8): every one is re-proved against the
helper's own run, or the call stays opaque.

Dynamo answers some calls of a body's Python at capture time with a constant: the symbolic-shape queries of
``torch.fx.experimental.symbolic_shapes`` (``guard_or_false``, ``guard_or_true``, ``statically_known_true``,
``statically_known_false``, ``has_static_value``, ``size_hint``, ...) from torch's symbolic engine, and the
tracing-state functions (``torch.compiler.is_compiling()``) as a compiled frame sees them. Natively the helper runs on
integers and outside Dynamo. On the base (W6, and the merge head), ``x * 2 if statically_known_true(B % 3 == 2) else
x`` and its ``guard_or_false`` and ``guard_or_true`` forms were inlined and the conversion certified, both stages
agreeing, and wrong at batches 2, 5, 8, ... (the static capture at batch 1 sees the same branch, so rule (c) did not
see it), and ``is_compiling()`` was inlined as True, a value mismatch the probe refused. Now each fold is recorded
during the capture (``body_rules/folds.py``, and every ``guard_or`` decision, its default included, in
``body_rules/symbolic_record.py``) and the call stays opaque unless the domain proves the native answer is the
folded one: exact as a host op at every batch, stage 2 refused by name.
"""

from __future__ import annotations

import pytest
import sympy
import torch
import torch.fx
import torch.nn as nn
from torch.fx.experimental import symbolic_shapes as S

from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import OPAQUE_BODY
from mimarsinan.torch_mapping.fx_rewrites.body_rules.folds import Fold, unproved_fold
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from operand_completeness.harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_impure_stage_2_refused,
    assert_value_program_exact,
    native,
)

W = 6
BATCHES = (1, 2, 3, 5, 8)


@torch.fx.wrap
def default_false(x):
    return x * 2.0 if S.guard_or_false(x.size(0) % 3 == 2) else x


@torch.fx.wrap
def default_true(x):
    return x if S.guard_or_true(x.size(0) % 3 != 2) else x * 2.0


@torch.fx.wrap
def known_true(x):
    return x * 2.0 if S.statically_known_true(x.size(0) % 3 == 2) else x


@torch.fx.wrap
def known_false(x):
    return x if S.statically_known_false(x.size(0) % 3 == 2) else x * 2.0


@torch.fx.wrap
def static_value(x):
    return x * 2.0 if S.has_static_value(x.size(0)) else x


@torch.fx.wrap
def hinted(x):
    return x * float(S.size_hint(x.size(0)))


@torch.fx.wrap
def compiling(x):
    return x * 2.0 if torch.compiler.is_compiling() else x


@torch.fx.wrap
def known_static(x):
    """A query of a per-sample extent, a Python integer: the fold is the query's own answer, so the call inlines."""
    return x * 2.0 if S.statically_known_true(x.size(1) == W) else x


@torch.fx.wrap
def provable_default(x):
    """``guard_or_false(B * 4 == 2)`` is False for every batch B >= 1, a fold the domain proves: the call inlines."""
    return x * 2.0 if S.guard_or_false(x.size(0) * 4 == 2) else x


class _Fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(W, W), nn.Linear(W, 3)


def _net(helper):
    namespace: dict = {}
    exec(f"def forward(self, x):\n    return self.fc2({helper.__name__}(self.fc1(x)))\n", globals(), namespace)
    return type(f"Net_{helper.__name__}", (_Fc,), {"forward": namespace["forward"]})


KEPT = {helper.__name__: helper for helper in (default_false, default_true, known_true, known_false, static_value,
                                                hinted, compiling)}
TAKEN = {helper.__name__: helper for helper in (known_static, provable_default)}


@pytest.mark.parametrize("name", sorted(KEPT))
def test_a_fold_the_native_run_does_not_give_keeps_the_call_opaque(name):
    helper = KEPT[name]
    gm = trace_model(native(_net(helper)), (W,))
    calls = [node for node in gm.graph.nodes if node.target is helper]
    assert calls, gm.graph
    reason = calls[0].meta[OPAQUE_BODY]
    assert reason.startswith(("its capture folds ", "its capture decides ")), reason
    assert_converted_exact(native(_net(helper)), (W,), "mvm", batches=BATCHES)
    assert_impure_stage_2_refused(native(_net(helper)), (W,), calls[0].name, name)


@pytest.mark.parametrize("name", sorted(TAKEN))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_fold_the_native_run_gives_is_inlined(name, contract):
    helper = TAKEN[name]
    gm = trace_model(native(_net(helper)), (W,))
    assert not [node for node in gm.graph.nodes if node.target is helper], gm.graph
    assert_converted_exact(native(_net(helper)), (W,), contract, batches=BATCHES)
    assert_value_program_exact(native(_net(helper)), (W,), batches=(2, 5))


B = sympy.Symbol("B", integer=True, positive=True)
s0 = sympy.Symbol("s0", integer=True)


def _fold(name, operands, value):
    function = getattr(S, name, None) or torch.compiler.is_compiling
    return Fold(function, name, operands, value)


FOLDS = {
    "known_true_proved": (_fold("statically_known_true", (sympy.Ne(s0, 0),), True), None),
    "known_true_unproved": (_fold("statically_known_true", (sympy.Eq(s0 % 3, 2),), False), "folds"),
    "known_false_proved": (_fold("statically_known_false", (sympy.Eq(2 * s0, 1),), True), None),
    "default_proved": (_fold("guard_or_false", (sympy.Eq(4 * s0, 2),), False), None),
    "hint": (_fold("size_hint", (s0,), 3), "folds"),
    "static_value": (_fold("has_static_value", (s0,), False), "folds"),
    "constant_query": (_fold("statically_known_false", (False,), False), "the helper's own run gives True"),
    "constant_agrees": (_fold("statically_known_true", (True,), True), None),
    "tracing_state": (Fold(torch.compiler.is_compiling, "torch.compiler.is_compiling", (), True),
                      "the helper's own run gives False"),
}


@pytest.mark.parametrize("name", sorted(FOLDS))
def test_each_fold_is_reproved_against_the_native_answer(name):
    fold, refused = FOLDS[name]
    found = unproved_fold(fold, {s0: B}, B, 1)
    assert (found is None) == (refused is None), found
    assert refused is None or refused in found, found


@torch.fx.wrap
def known_false_static(x):
    """``statically_known_false(B >= 2)``: torch cannot decide it, so the capture answers False, the helper's own answer
    from batch 2 on, which the domain proves; at batch 1, where rule (c) compares a static capture, Dynamo's handler
    returns the constant operand (``1 >= 2``, False) as it is, where the helper's own run gives True. [coverage W8: the
    two captures took the same branch, so the call was inlined and the probe refused a value mismatch at batch 1]"""
    return x if S.statically_known_false(x.size(0) >= 2) else x * 2.0


def test_a_fold_of_a_static_capture_rule_c_compares_is_reproved_too():
    gm = trace_model(native(_net(known_false_static)), (W,))
    calls = [node for node in gm.graph.nodes if node.target is known_false_static]
    assert calls, gm.graph
    reason = calls[0].meta[OPAQUE_BODY]
    assert reason.startswith("its body takes another path at batch 1 (its capture folds statically_known_false(False)")\
        and "the helper's own run gives True" in reason, reason
    assert_converted_exact(native(_net(known_false_static)), (W,), "mvm", batches=BATCHES)
    assert_impure_stage_2_refused(native(_net(known_false_static)), (W,), calls[0].name, "known_false_static")
