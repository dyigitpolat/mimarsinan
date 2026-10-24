"""N2 design §4.1: the two product verdicts of the attention mappability table are conditional on a declared
operand core type; every other verdict, and both without one, stay as they were."""

from __future__ import annotations

import pytest

from mimarsinan.mapping.onchip_attention.attention_mappability import AffineRealizability, affine_realizability_report


@pytest.mark.parametrize("op", ["qk_score", "attention_value_matmul"])
def test_a_product_of_activations_is_realizable_on_an_operand_core_type(op):
    assert not affine_realizability_report(op).realizable
    verdict = affine_realizability_report(op, operand_core_type=True)
    assert verdict.realizable and verdict.reason_code == "dynamic_operand_core"


@pytest.mark.parametrize("op", sorted(set(AffineRealizability.KNOWN_OPS) - {"qk_score", "attention_value_matmul"}))
def test_every_other_verdict_ignores_the_operand_core_type(op):
    assert affine_realizability_report(op, operand_core_type=True) == affine_realizability_report(op)
