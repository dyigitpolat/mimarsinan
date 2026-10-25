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
    ReferenceUndeterminedError,
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


def test_a_comparison_the_reference_cannot_decide_is_a_refusal_never_a_value_mismatch(monkeypatch):
    """``ReferenceUndeterminedError`` is named apart from a value mismatch, which it is not shown to be (the native
    model differs from itself by more than the bound on the compared inputs): the verdict is refused by that name,
    its measured error kept for the record (the zoo's regnet_y_1_6gf at 224 reads this way)."""
    undetermined = ConversionValueParityError.of(_failed(relative_error=3e-12, max_abs_error=3e-12, reference_max=1.0,
                                                         reference_floor=1e-11))
    assert isinstance(undetermined, ReferenceUndeterminedError)
    monkeypatch.setattr(member, "probe_witness_conversion", _raising(undetermined))
    verdict = c0_verdict(_tiny(), (4,), packaging=MVM_PACKAGING)
    assert verdict.status == REFUSED and not verdict.value_mismatch
    assert verdict.refusal == verdict.root_refusal == "ReferenceUndeterminedError"
    assert (verdict.relative_error, verdict.bound) == (3e-12, 1e-12), "the measured error is kept beside the name"
    census = member_census(_tiny(), (4,), 2, packaging=MVM_PACKAGING, value_bits=8)
    assert census.verdict.status == REFUSED and census.rows == ()


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


class _SdpaBlock(torch.nn.Module):
    """Three Linear projections into ``F.scaled_dot_product_attention`` (a function-form attention leaf), a head."""

    def __init__(self, d=8, n=4):
        super().__init__()
        self.q, self.k, self.v = torch.nn.Linear(d, d), torch.nn.Linear(d, d), torch.nn.Linear(d, d)
        self.head = torch.nn.Linear(n * d, 3)

    def forward(self, x):
        b, n, d = x.shape
        o = torch.nn.functional.scaled_dot_product_attention(self.q(x), self.k(x), self.v(x))
        return self.head(o.reshape(b, n * d))


def _sdpa():
    torch.manual_seed(0)
    return _SdpaBlock().eval()


def test_a_member_census_under_the_decomposition_key_places_the_decomposed_attention():
    """``decompose_leaf_modules`` reaches the verdict and the census conversion alike: the attention call is lowered
    to its two dynamic products and its softmax, so the member gets rows with no attention leaf on the host."""
    census = member_census(_sdpa(), (4, 8), 3, packaging=MVM_PACKAGING, value_bits=8, decompose_leaf_modules=True)
    assert census.verdict.status == AGREEMENT and census.census_refusal is None
    assert census.rows, "the decomposed member gets rows"
    c0 = census.rows[0]
    assert c0.host_kinds.get("dmm") == 2 and c0.host_kinds.get("softmax") == 1 and "leaf" not in c0.host_kinds
    c3 = next(row for row in census.rows if row.level == "C3")
    assert c3.dynamic_payload_elements > 0, "the stationary operands of both products are written on chip at C3"


def test_the_same_member_without_the_key_is_refused_by_name_and_gets_no_rows():
    """The function-form attention leaf is no batch-leading layout for the reshape that reads it (the merge's verify
    probe on ``net_models/vit_sdpa``): without the key the member is refused by name, so the key is what gives it
    rows."""
    census = member_census(_sdpa(), (4, 8), 3, packaging=MVM_PACKAGING, value_bits=8)
    assert census.verdict.status == REFUSED and census.rows == ()
    assert census.verdict.root_refusal_type is not None
    assert issubclass(census.verdict.root_refusal_type, ToolchainRefusal)


class _TorchRecurrent(torch.nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.rnn = cell(8, 8, batch_first=True)
        self.head = torch.nn.Linear(8, 3)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


@pytest.mark.parametrize("cell", [torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM])
def test_a_recurrent_leaf_under_the_decomposition_key_gets_rows_with_its_recurrence_on_the_host(cell):
    """The plan's recurrent rows (CL4): the decomposed leaf's input projection is a core, its recurrence one host op
    at every level, counted by its declared MACs."""
    torch.manual_seed(0)
    census = member_census(_TorchRecurrent(cell).eval(), (5, 8), 3, packaging=MVM_PACKAGING, value_bits=8,
                           decompose_leaf_modules=True)
    assert census.verdict.status == AGREEMENT and census.census_refusal is None
    assert census.rows and all(row.host_kinds.get("recurrence") == 1 for row in census.rows)
    assert all(not row.uncounted_host_products for row in census.rows), "the recurrence declares its MACs"
