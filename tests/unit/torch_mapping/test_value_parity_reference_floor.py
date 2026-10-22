"""The reference's own reassociation floor (``rounding.reference_floor``, ``settle_line``) and the refusal it names
(``ReferenceUndeterminedError``): a comparison the probe cannot decide at its bound because the native model is not
reproducible within it; a real reassociation error is still a value mismatch. Localization records each native node's
own value (W5 triage)."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping import converter
from mimarsinan.torch_mapping.value_parity import (
    NATIVE_EVAL_IDENTITY,
    ConversionValueParityError,
    ReferenceUndeterminedError,
    ValueParityResult,
    WitnessConversion,
    probe_conversion_values,
    reference_floor,
)
from mimarsinan.torch_mapping.value_parity.localization import mapper_node_values, native_node_values, trace_native
from mimarsinan.torch_mapping.value_parity.rounding import (
    EXACTNESS_ACCEPTANCE,
    REFERENCE_FLOOR_MARGIN,
    UNIT_ROUNDOFF,
    gamma,
    reassociation_bound,
    settle_line,
)
from mimarsinan.torch_mapping.value_parity.verdict import STAGE_CONVERTED, STAGE_VALUE_PROGRAM

select = NATIVE_EVAL_IDENTITY.select_output


class _CancellingDifference(nn.Module):
    """``a(x) - b(x)`` with ``b = a - e``, ``a`` about 1e4 times ``e``: the O(1) output is a difference of large dot
    products, so the program's own fp64 value is determined only to about u |a(x)|, far above tau(P)."""

    def __init__(self, width: int = 64, scale: float = 1e4):
        super().__init__()
        self.a = nn.Linear(width, width)
        self.b = nn.Linear(width, width)
        with torch.no_grad():
            self.a.weight.mul_(scale)
            self.a.bias.mul_(scale)
            self.b.weight.copy_(self.a.weight - torch.randn(width, width) / width ** 0.5)
            self.b.bias.copy_(self.a.bias)

    def forward(self, x):
        return self.a(x) - self.b(x)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 16)
        self.fc = nn.Linear(16, 6)

    def forward(self, x):
        return self.fc(torch.relu(self.l1(x)))


def _seeded(cls, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return cls().double().eval()


def _split_reduction(module, args, output):
    """A legal reassociation: the two input halves summed apart, the same function in another fp64 order."""
    x, half = args[0], args[0].shape[-1] // 2
    return F.linear(x[..., :half], module.weight[:, :half]) + F.linear(x[..., half:], module.weight[:, half:],
                                                                        module.bias)


def _fp32_accumulation(module, args, output):
    """A real reassociation error: the dot products summed in fp32."""
    return F.linear(args[0].float(), module.weight.float(), module.bias.float()).double()


def _offset(rel: float):
    """A defect: the output moved by ``rel * max(1, max|out|)``."""
    def hook(module, args, output):
        return output + rel * max(1.0, float(output.abs().max()))
    return hook


def _converting(hook, root_hook=None):
    """A witness conversion: the snapshot itself, every Linear through ``hook``, the whole through ``root_hook``."""
    def convert(snapshot: nn.Module) -> WitnessConversion:
        flow = copy.deepcopy(snapshot).eval()
        for module in flow.modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(hook)
        if root_hook is not None:
            flow.register_forward_hook(root_hook)
        return WitnessConversion(flow, {})
    return convert


def _refusal(model, convert, shape):
    with pytest.raises(ConversionValueParityError) as excinfo:
        probe_conversion_values(copy.deepcopy(model), convert, shape)
    return excinfo.value


class TestReferenceFloor:
    """The floor is the program against its reversed-reduction twin: the same function, another summation order."""

    @pytest.mark.parametrize("layer, shape", [
        (nn.Linear(24, 5), (3, 7, 24)),
        (nn.Conv2d(8, 6, 3, padding=1), (2, 8, 5, 5)),
        (nn.Conv2d(8, 8, 3, padding=1, groups=4), (2, 8, 5, 5)),
        (nn.Conv1d(6, 4, 3, padding=2, dilation=2), (2, 6, 9)),
    ])
    def test_the_twin_is_the_same_function(self, layer, shape):
        torch.manual_seed(0)
        layer = layer.double().eval()
        floor = reference_floor(layer, torch.randn(*shape, dtype=torch.float64), select)
        assert floor is not None and floor <= 64 * UNIT_ROUNDOFF * layer.weight[0].numel()

    def test_the_twin_is_another_summation_order(self):
        model = _seeded(_CancellingDifference)
        floor = reference_floor(model, torch.randn(6, 64, dtype=torch.float64), select)
        assert floor is not None and floor > 1e-13

    def test_a_program_without_a_weight_layer_has_no_floor(self):
        assert reference_floor(nn.ReLU(), torch.randn(3, 4, dtype=torch.float64), select) is None

    def test_the_floor_leaves_the_program_as_it_was(self):
        model = _seeded(_CancellingDifference)
        x = torch.randn(6, 64, dtype=torch.float64)
        before = model(x)
        reference_floor(model, x, select)
        assert all(not m._forward_hooks for m in model.modules())
        assert torch.equal(model(x), before)


class TestSettleLine:
    """A refused comparison: beta(P) only where the capped line can still decide it, and only until one output column
    decides it (the same verdict as beta(P) over every column), then the reference's floor."""

    @staticmethod
    def _refused(error: float) -> ValueParityResult:
        return ValueParityResult(ok=False, context="t", stage=STAGE_VALUE_PROGRAM, contract="c", bound=1e-13,
                                 samples=1, relative_error=error)

    def test_beta_is_not_derived_above_the_acceptance(self):
        def derive(_inputs, _error):
            raise AssertionError("beta(P) derived where the capped line cannot reach the error")

        settled = settle_line(self._refused(2 * EXACTNESS_ACCEPTANCE), nn.ReLU(), torch.ones(1, 2), select, derive)
        assert not settled.ok and settled.reassociation_bound is None and settled.reference_floor is None

    def test_beta_still_widens_the_line_within_the_acceptance(self):
        seen = []

        def derive(inputs, error):
            seen.append(error)
            return 1e-12, 3e-13

        settled = settle_line(self._refused(5e-13), nn.ReLU(), torch.ones(1, 2), select, derive)
        assert settled.ok and settled.bound == 1e-12 and settled.reassociation_bound == 3e-13 and seen == [5e-13]

    @staticmethod
    def _core():
        torch.manual_seed(0)
        core = nn.Linear(12, 6).double().eval()
        x = torch.randn(3, 12, dtype=torch.float64)
        with torch.no_grad():
            columns = 2 * gamma(13) * F.linear(x.abs(), core.weight.abs(), core.bias.abs())
            scale = max(1.0, float(core(x).abs().max()))
        return core, x, columns.amax(0) / scale

    def test_beta_stops_at_the_first_column_that_decides_the_line(self):
        core, x, columns = self._core()
        assert reassociation_bound(core, x, select, enough=0.0) == pytest.approx(float(columns[0]), rel=1e-12)
        enough = float(columns[:3].max()) * (1 - 1e-9)
        first = int(next(j for j in range(6) if float(columns[j]) >= enough))
        assert reassociation_bound(core, x, select, enough=enough) == pytest.approx(float(columns[first]), rel=1e-12)

    def test_beta_is_beta_p_when_no_column_decides(self):
        core, x, columns = self._core()
        full = reassociation_bound(core, x, select)
        assert full == pytest.approx(float(columns.max()), rel=1e-12)
        assert reassociation_bound(core, x, select, enough=2 * full) == full


class TestReferenceUndetermined:
    """A legal reassociation of a reference that is not reproducible within the bound is named the reference's;
    a real reassociation error, or a defect above the margin, stays a value mismatch."""

    def test_a_legal_reassociation_is_named_the_references(self):
        exc = _refusal(_seeded(_CancellingDifference), _converting(_split_reduction), (64,))
        result = exc.result
        assert isinstance(exc, ReferenceUndeterminedError) and result.reference_undetermined
        assert result.reference_floor > result.bound
        assert result.bound < result.relative_error <= REFERENCE_FLOOR_MARGIN * result.reference_floor
        assert "not reproducible within the bound" in str(exc)

    def test_the_real_identity_value_program_is_named_the_references(self):
        with pytest.raises(ReferenceUndeterminedError) as excinfo:
            converter.probe_witness_conversion(_seeded(_CancellingDifference), (64,), packaging=MVM_PACKAGING)
        assert excinfo.value.stage == STAGE_VALUE_PROGRAM
        assert excinfo.value.result.reference_floor > excinfo.value.result.bound

    def test_an_fp32_accumulation_is_a_value_mismatch(self):
        exc = _refusal(_seeded(_CancellingDifference), _converting(_fp32_accumulation), (64,))
        assert not isinstance(exc, ReferenceUndeterminedError) and not exc.result.reference_undetermined
        assert exc.result.relative_error > REFERENCE_FLOOR_MARGIN * exc.result.reference_floor

    def test_a_defect_above_the_margin_is_a_value_mismatch(self):
        model = _seeded(_CancellingDifference)
        floor = reference_floor(model, torch.randn(6, 64, dtype=torch.float64), select)
        exc = _refusal(model, _converting(_split_reduction, _offset(10 * REFERENCE_FLOOR_MARGIN * floor)), (64,))
        assert not isinstance(exc, ReferenceUndeterminedError)
        assert exc.result.relative_error > REFERENCE_FLOOR_MARGIN * exc.result.reference_floor

    def test_a_defect_on_a_well_conditioned_program_is_a_value_mismatch(self):
        exc = _refusal(_seeded(_MLP), _converting(lambda m, a, o: o, _offset(5e-13)), (12,))
        assert exc.stage == STAGE_CONVERTED and not isinstance(exc, ReferenceUndeterminedError)
        assert exc.result.reference_floor is not None and exc.result.reference_floor <= exc.result.bound


class _InfiniteFloorColumn(nn.Module):
    """``_MLP`` beside one output column that is exactly -inf natively (``1 + u - 1`` summed left to right is 0, its
    log -inf) and finite in the reversed summation order (``-1 + u + 1`` is ``u``): the reference floor is infinite."""

    def __init__(self):
        super().__init__()
        self.mlp = _MLP()
        self.register_buffer("c", torch.tensor([1.0, 2.0 ** -53, -1.0]))
        self.cancel = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.cancel.weight.fill_(1.0)

    def forward(self, x):
        h = self.cancel(x[:, :3] * 0.0 + self.c)
        return torch.cat([self.mlp(x), torch.log(torch.relu(h))], dim=1)


def _mlp_offset(rel: float):
    """A conversion with a defect in the MLP's columns only."""
    def convert(snapshot: nn.Module) -> WitnessConversion:
        flow = copy.deepcopy(snapshot).eval()
        flow.mlp.register_forward_hook(lambda module, args, output: output + rel)
        return WitnessConversion(flow, {})
    return convert


class TestUndecidableNeedsFiniteValues:
    """"Not decidable at the line" needs a finite floor and a finite error; anything else is a value mismatch
    [review round 1 of the coverage fixes: an infinite floor made it hold for every error, ``error <= 4 * inf``]."""

    @staticmethod
    def _refused(error: float, floor: float) -> ValueParityResult:
        return ValueParityResult(ok=False, context="t", stage=STAGE_CONVERTED, contract="c", bound=1e-13, samples=1,
                                 relative_error=error, max_abs_error=error, reference_max=1.0, reference_floor=floor)

    @pytest.mark.parametrize("error, floor", [
        (0.5, float("inf")), (float("inf"), float("inf")), (float("inf"), 1e-11), (float("nan"), 1e-11),
        (0.5, float("nan")), (float("nan"), float("inf")),
    ], ids=["inf_floor", "inf_floor_inf_error", "inf_error", "nan_error", "nan_floor", "nan_error_inf_floor"])
    def test_an_infinite_or_undefined_floor_or_error_is_a_value_mismatch(self, error, floor):
        result = self._refused(error, floor)
        assert not result.reference_undetermined
        assert type(ConversionValueParityError.of(result)) is ConversionValueParityError

    def test_a_finite_error_within_the_margin_of_a_finite_floor_is_undecidable(self):
        result = self._refused(2e-11, 1e-11)
        assert result.reference_undetermined
        assert isinstance(ConversionValueParityError.of(result), ReferenceUndeterminedError)

    def test_a_defect_beside_a_column_of_infinite_floor_is_a_value_mismatch(self):
        exc = _refusal(_seeded(_InfiniteFloorColumn), _mlp_offset(0.5), (12,))
        assert not isinstance(exc, ReferenceUndeterminedError) and not exc.result.reference_undetermined, exc
        assert exc.result.relative_error is not None and exc.result.relative_error > 0.1, exc


class TestNativeNodeValues:
    """Localization compares each mapper with the value its native node produced, not a later in-place write."""

    def test_a_node_keeps_its_value_after_an_in_place_successor(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.ReLU(inplace=True), nn.Linear(6, 2))
        model = model.double().eval()
        gm = trace_native(model, (4,))
        modules = dict(gm.named_modules())
        bn = next(n.name for n in gm.graph.nodes
                  if n.op == "call_module" and isinstance(modules[str(n.target)], nn.BatchNorm1d))
        x = torch.randn(5, 4, dtype=torch.float64)
        with torch.no_grad():
            want = model[1](model[0](x))
            values = native_node_values(gm, x)
        assert bool((want < 0).any()) and torch.equal(values[bn], want)

    def test_a_mapper_keeps_its_value_after_an_in_place_successor(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(inplace=True), nn.Linear(6, 2)).double().eval()
        flow = converter.trace_and_convert(copy.deepcopy(model), (4,), "cpu", packaging=MVM_PACKAGING).flow.eval()
        x = torch.randn(5, 4, dtype=torch.float64)
        with torch.no_grad():
            want = model[0](x)
            labelled = dict(mapper_node_values(flow.get_mapper_repr(), x))
        assert bool((want < 0).any()) and torch.equal(labelled["_0"], want)
