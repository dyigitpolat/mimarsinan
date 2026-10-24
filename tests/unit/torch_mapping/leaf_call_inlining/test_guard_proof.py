"""Shape-specialized inlining (coverage W8, sign-off L8): a capture's statements on the batch are proved in the
converter's own affine batch domain.

On the base a guard, runtime assert, replacement or range of Dynamo's capture was shown to hold for every batch B >= 1
by torch's value-range analysis (``bound_sympy``), part of the symbolic engine the owner's shape-proof domain does not
rest on. Now each is translated into the domain (``body_rules.guard_proof.prove``): integers, a symbol's affine form,
sums, products, floor division and modulus by a divisor of the coefficient, ``max``, ``min``, the comparisons and
Boolean connectives, each decided for every B >= 1 by the domain's own integer arithmetic. A statement the domain does
not decide keeps the call opaque, even where it holds: ``B * B >= 1`` and ``(B + 1) // 2 >= 1`` were inlined on the base
and now stay opaque, exact as host ops, with stage 2 refused by name.
"""

from __future__ import annotations

import pytest
import sympy
import torch
import torch.fx
import torch.fx as fx
import torch.nn as nn
from torch.utils._sympy.functions import CleanDiv, FloorDiv, Max, Min, Mod, PythonMod

from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import OPAQUE_BODY
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import program_purity
from mimarsinan.torch_mapping.fx_rewrites.body_rules.guard_proof import prove
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from operand_completeness.harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_impure_stage_2_refused,
    assert_value_program_exact,
    native,
)

B = sympy.Symbol("B", integer=True, positive=True)
s0, s1, s2 = sympy.symbols("s0 s1 s2", integer=True)
FORMS = {s0: B, s1: 16 * B, s2: B - 1}

PROVED = {
    "batch_nonzero": sympy.Ne(s0, 0),
    "batch_at_least_one": sympy.Ge(s0, 1),
    "affine_comparison": sympy.Gt(2 * s0 + 1, 2),
    "replacement": sympy.Eq(s1, 16 * s0),
    "exact_floor": sympy.Ge(FloorDiv(s1, 16), 1),
    "clean_division": sympy.Eq(CleanDiv(s1, 4), 4 * s0),
    "exact_modulus": sympy.Eq(PythonMod(s1, 4), 0),
    "nonnegative_mod": sympy.Eq(Mod(s1, 8), 0),
    "max": sympy.Ge(Max(s0, 1), 1),
    "min": sympy.Le(Min(s0, s1), s1),
    "and": sympy.And(sympy.Ge(s0, 1), sympy.Ne(s1, 3)),
    "or": sympy.Or(sympy.Eq(s0, 0), sympy.Ge(s1, 16)),
    "not": sympy.Not(sympy.Eq(s1, 0)),
    "batch_symbol_itself": sympy.Ge(B, 1),
}

NOT_PROVED = {
    "batch_not_one": sympy.Ne(s0, 1),  # false at B = 1
    "tail_not_one": sympy.Ne(s2, 1),  # B - 1 is 1 at batch 2
    "square": sympy.Ge(s0 * s0, 1),  # holds, but not an affine form
    "inexact_floor": sympy.Ge(FloorDiv(s0 + 1, 2), 1),  # holds; 2 does not divide the coefficient 1
    "parity": sympy.Eq(PythonMod(s0, 2), 0),
    "bounded": sympy.Le(s0, 8),
    "modulus_of_a_negative": sympy.Eq(Mod(s2 - 3, 2), 0),
    "rational": sympy.Ge(s0 / 2, sympy.Rational(1, 2)),
    "power": sympy.Ge(2 ** s0, 2),
    "unknown_symbol": sympy.Ge(sympy.Symbol("u0", integer=True), 0),
    "or_undecided": sympy.Or(sympy.Eq(s0, 1), sympy.Eq(s0, 2)),
}


@pytest.mark.parametrize("name", sorted(PROVED))
def test_a_statement_the_domain_decides_is_proved(name):
    assert prove(PROVED[name], FORMS, B) is True


@pytest.mark.parametrize("name", sorted(NOT_PROVED))
def test_a_statement_the_domain_does_not_decide_is_not_proved(name):
    assert prove(NOT_PROVED[name], FORMS, B) is False


def test_a_symbol_whose_form_is_not_affine_proves_nothing():
    assert prove(sympy.Ge(s0, 0), {s0: B * B}, B) is False


# ----------------------------------------------------------------------------- the rule on captured bodies
@torch.fx.wrap
def guard_square(x):
    """``B * B >= 1`` holds for every B >= 1, but it is no affine form: the domain does not prove it."""
    return x * 2.0 if x.size(0) * x.size(0) >= 1 else x


@torch.fx.wrap
def guard_floor(x):
    """``(B + 1) // 2 >= 1`` holds for every B >= 1; the domain floors only by a divisor of the coefficient."""
    return x * 2.0 if (x.size(0) + 1) // 2 >= 1 else x


@torch.fx.wrap
def guard_parity(x):
    return x * 2.0 if x.size(0) % 2 == 0 else x


@torch.fx.wrap
def guard_affine(x):
    """``2 * B + 1 > 2`` holds for every B >= 1, an affine comparison the domain decides."""
    return x * 2.0 if 2 * x.size(0) + 1 > 2 else x


@torch.fx.wrap
def guard_static(x):
    """A branch on a per-sample extent, which Dynamo specializes."""
    return x * 2.0 if x.size(1) >= 3 else x


@torch.fx.wrap
def guard_ge2(x):
    """``B >= 2``: false at batch 1 only."""
    return x * 2.0 if x.size(0) >= 2 else x


@torch.fx.wrap
def guard_ne3(x):
    """``B != 3``: false at batch 3 only."""
    return x * 2.0 if x.size(0) != 3 else x


@torch.fx.wrap
def guard_lt100(x):
    """``B < 100``: true at every batch a probe samples, false from batch 100 on."""
    return x * 2.0 if x.size(0) < 100 else x


W = 6


class _Fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(W, W), nn.Linear(W, 3)


def _net(helper):
    namespace: dict = {}
    exec(f"def forward(self, x):\n    return self.fc2({helper.__name__}(self.fc1(x)))\n", globals(), namespace)
    return type(f"Net_{helper.__name__}", (_Fc,), {"forward": namespace["forward"]})


def _call(gm: fx.GraphModule, helper) -> list:
    return [node for node in gm.graph.nodes if node.target is helper]


@pytest.mark.parametrize("helper", [guard_square, guard_floor, guard_parity, guard_ge2, guard_ne3, guard_lt100],
                         ids=lambda h: h.__name__)
def test_a_guard_the_domain_does_not_prove_keeps_the_call_opaque(helper):
    factory = _net(helper)
    gm = trace_model(native(factory), (W,))
    calls = _call(gm, helper)
    assert calls, gm.graph
    reason = calls[0].meta[OPAQUE_BODY]
    assert reason.startswith(("its body decides on the batch: Dynamo's guard ", "its capture computes ",
                              "its capture decides ")) and "affine batch domain" in reason, reason
    assert_converted_exact(native(factory), (W,), "mvm", batches=(1, 2, 3, 9))
    assert_impure_stage_2_refused(native(factory), (W,), calls[0].name, helper.__name__)


def test_a_guard_false_beyond_the_sampled_batches_stays_opaque_and_exact_there():
    """``B < 100`` holds at every batch the probe and the comparison sample: only the proof keeps the call opaque, and
    the converted model computes the native value at batch 101 (an inlined branch would double it there)."""
    factory = _net(guard_lt100)
    gm = trace_model(native(factory), (W,))
    assert _call(gm, guard_lt100), gm.graph
    assert_converted_exact(native(factory), (W,), "mvm", batches=(3, 101))


@pytest.mark.parametrize("helper", [guard_affine, guard_static], ids=lambda h: h.__name__)
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_guard_the_domain_proves_is_inlined(helper, contract):
    factory = _net(helper)
    gm = trace_model(native(factory), (W,))
    assert not _call(gm, helper) and program_purity(gm).pure, gm.graph
    assert_converted_exact(native(factory), (W,), contract, batches=(1, 2, 3, 9))
    assert_value_program_exact(native(factory), (W,), batches=(2, 9))


# ----------------------------------------------------------------------------- torch's folds, without a guard
@torch.fx.wrap
def max_mod(x):
    """Torch's symbolic engine builds ``max(6, 2*B - 12) % 6`` as 0 (sympy's simplification), so Dynamo's capture
    takes the doubling branch with no guard; natively it is taken up to batch 9 and from batch 10 on only where
    ``2*B - 12`` is a multiple of 6. [coverage W8: on the merge head the call was inlined and the conversion certified,
    both stages agreeing, and wrong from batch 10]"""
    return x * 2.0 if max(6, 2 * x.size(0) - 12) % 6 == 0 else x


@torch.fx.wrap
def max_mod_extent(x):
    """The same fold read as a number, a scale of the output."""
    return x * float(max(6, 2 * x.size(0) - 12) % 6 + 1)


@pytest.mark.parametrize("helper", [max_mod, max_mod_extent], ids=lambda h: h.__name__)
def test_a_fold_the_domain_does_not_give_keeps_the_call_opaque(helper):
    factory = _net(helper)
    gm = trace_model(native(factory), (W,))
    calls = _call(gm, helper)
    assert calls, gm.graph
    reason = calls[0].meta[OPAQUE_BODY]
    assert reason.startswith("its capture computes mod(Max(6, ") and "affine batch domain" in reason, reason
    assert_converted_exact(native(factory), (W,), "mvm", batches=(1, 2, 9, 10, 11, 13, 20))
    assert_impure_stage_2_refused(native(factory), (W,), calls[0].name, helper.__name__)


# ----------------------------------------------------------------------------- the records, one by one
from torch.utils._sympy.functions import Max as TorchMax  # noqa: E402

from mimarsinan.torch_mapping.fx_rewrites.body_rules.symbolic_record import Record, unproved_record  # noqa: E402

RECORDS = {
    "affine_operation": (Record("operation", "mul", (s0, sympy.Integer(16)), 16 * s0), None),
    "exact_floor": (Record("operation", "floordiv", (16 * s0, sympy.Integer(16)), s0), None),
    "unfolded_outside": (Record("operation", "sym_max", (2 * s0 - 12, sympy.Integer(6)), TorchMax(6, 2 * s0 - 12)),
                         None),
    "folded_outside": (Record("operation", "mod", (TorchMax(6, 2 * s0 - 12), sympy.Integer(6)), sympy.Integer(0)),
                       "computes"),
    "wrong_value": (Record("operation", "add", (s0, sympy.Integer(1)), s0 + 2), "computes"),
    "true_comparison": (Record("operation", "ge", (16 * s0, sympy.Integer(1)), sympy.true), None),
    "assumed_comparison": (Record("operation", "eq", (s2, sympy.Integer(0)), sympy.false), None),  # from B0 = 3
    "reflexive": (Record("operation", "eq", (s0 * s0, s0 * s0), sympy.true), None),
    "connective": (Record("operation", "or", (sympy.Eq(s0, 1), sympy.true), sympy.true), None),
    "static_decision": (Record("static", "_maybe_evaluate_static", (sympy.Ne(16 * s0, 1),), sympy.true), None),
    "static_square": (Record("static", "_maybe_evaluate_static", (sympy.Ge(s0 * s0, 1),), sympy.true), "decides"),
    "static_on_hints": (Record("static", "_maybe_evaluate_static", (sympy.Ge(s0, 3),), sympy.true,
                               (("compute_hint", True),)), "hints"),
}


@pytest.mark.parametrize("name", sorted(RECORDS))
def test_each_record_is_reproved_in_the_domain(name):
    record, refused = RECORDS[name]
    found = unproved_record([record], {s0: B, s1: 16 * B, s2: B - 1}, B, [1, 2])
    assert (found is None) == (refused is None), found
    assert refused is None or refused in found, found


def test_records_are_proved_from_batch_1_when_rule_c_does_not_cover_the_batches_below():
    record = Record("operation", "eq", (s2, sympy.Integer(0)), sympy.false)  # B - 1 == 0 is True at batch 1
    assert unproved_record([record], {s2: B - 1}, B, [1, 2]) is None
    assert unproved_record([record], {s2: B - 1}, B, [1]) is not None
