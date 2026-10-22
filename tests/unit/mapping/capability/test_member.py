"""The C0 verdict: every required probe stage, the refusal by its root name, and a value mismatch never read as one."""

from __future__ import annotations

import pytest
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.capability import member
from mimarsinan.mapping.capability.member import (
    AGREEMENT,
    INCOMPLETE,
    REFUSED,
    VALUE_MISMATCH,
    c0_verdict,
    member_census,
)
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import probe_stages
from mimarsinan.torch_mapping.value_parity.verdict import (
    STAGE_VALUE_PROGRAM,
    ConversionValueParityError,
    ValueParityResult,
)


class _LoweringRefusal(ToolchainRefusal, RuntimeError):
    """Stand-in for a lowering refusal that a probe stage wraps in its value-parity refusal."""


def _tiny():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)).eval()


def _failed(**fields) -> ValueParityResult:
    return ValueParityResult(ok=False, context="test", stage=STAGE_VALUE_PROGRAM, contract="native_eval_identity",
                             bound=1e-12, samples=4, **fields)


def _raising(exc):
    def probe(*args, **kwargs):
        raise exc

    return probe


def test_an_agreement_carries_every_required_stage():
    verdict = c0_verdict(_tiny(), (4,), packaging=MVM_PACKAGING)
    assert verdict.status == AGREEMENT
    assert tuple(stage for stage, _ in verdict.stages) == probe_stages(MVM_PACKAGING)
    assert verdict.missing_stages == () and not verdict.value_mismatch


def test_a_verdict_missing_a_required_stage_is_incomplete_never_an_agreement(monkeypatch):
    real = member.probe_witness_conversion
    monkeypatch.setattr(member, "probe_witness_conversion", lambda *a, **k: real(*a, **k)[:1])
    verdict = c0_verdict(_tiny(), (4,), packaging=MVM_PACKAGING)
    assert verdict.status == INCOMPLETE
    assert verdict.missing_stages == (STAGE_VALUE_PROGRAM,)
    assert verdict.relative_error is None, "no agreement number from stage 1 alone"
    census = member_census(_tiny(), (4,), 2, packaging=MVM_PACKAGING, value_bits=8)
    assert census.verdict.status == INCOMPLETE and census.rows == ()


def test_a_wrapped_stage_failure_is_named_by_its_root_refusal(monkeypatch):
    root = _LoweringRefusal("host op 'getattr_2' has no IR form")
    wrapped = ConversionValueParityError(_failed(reason="IR lowering of the witness failed", error=root))
    monkeypatch.setattr(member, "probe_witness_conversion", _raising(wrapped))
    verdict = c0_verdict(_tiny(), (4,), packaging=MVM_PACKAGING)
    assert verdict.status == REFUSED and not verdict.value_mismatch
    assert verdict.refusal == "ConversionValueParityError"
    assert verdict.root_refusal == "_LoweringRefusal" and verdict.root_refusal_type is _LoweringRefusal


def test_a_measured_value_mismatch_is_an_exactness_failure_not_a_refusal(monkeypatch):
    mismatch = ConversionValueParityError(_failed(relative_error=1e-3, max_abs_error=1e-3, reference_max=1.0))
    monkeypatch.setattr(member, "probe_witness_conversion", _raising(mismatch))
    verdict = c0_verdict(_tiny(), (4,), packaging=MVM_PACKAGING)
    assert verdict.status == VALUE_MISMATCH and verdict.value_mismatch
    assert (verdict.relative_error, verdict.bound) == (1e-3, 1e-12)
    assert verdict.refusal == verdict.root_refusal == "ConversionValueParityError"
    census = member_census(_tiny(), (4,), 2, packaging=MVM_PACKAGING, value_bits=8)
    assert census.verdict.status == VALUE_MISMATCH and census.rows == ()


def test_a_census_reuses_a_verdict_judged_on_its_own_input():
    verdict = c0_verdict(_tiny(), (4,), packaging=MVM_PACKAGING)
    census = member_census(_tiny(), (4,), 2, packaging=MVM_PACKAGING, value_bits=8, verdict=verdict)
    assert census.verdict is verdict and census.rows


@pytest.mark.parametrize("errors, bounds", [((1e-16, 3e-16), (1e-13, 2e-13)), ((3e-16, 1e-16), (2e-13, 1e-13))])
def test_the_agreement_states_the_worst_stage_error_and_bound(monkeypatch, errors, bounds):
    """The column is the max relative error: a later stage's larger error (or bound) is the one printed."""
    stages = probe_stages(MVM_PACKAGING)
    assert len(stages) == 2
    results = [ValueParityResult(ok=True, context="test", stage=stage, contract="native_eval_identity", bound=bound,
                                 samples=4, relative_error=error) for stage, error, bound in zip(stages, errors, bounds)]
    monkeypatch.setattr(member, "probe_witness_conversion", lambda *a, **k: list(results))
    verdict = c0_verdict(_tiny(), (4,), packaging=MVM_PACKAGING)
    assert verdict.status == AGREEMENT
    assert verdict.relative_error == 3e-16 and verdict.bound == 2e-13
    assert verdict.stages == tuple(zip(stages, errors))
