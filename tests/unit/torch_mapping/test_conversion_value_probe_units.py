"""Units of the value-parity conversion probe: metric, witness, contract, localization, wiring."""

from __future__ import annotations

import copy
import operator

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.torch_mapping import converter
from mimarsinan.torch_mapping.conversion_probe import ConversionProbeError
from mimarsinan.torch_mapping.converter import (
    TracedConversion,
    convert_torch_model,
    trace_and_convert,
)
from mimarsinan.torch_mapping.value_parity import (
    EXACTNESS_ACCEPTANCE,
    NATIVE_EVAL_IDENTITY,
    STAGE_CONVERTED,
    STAGE_NATIVE,
    ConversionValueParityError,
    FunctionContract,
    WitnessConversion,
    conversion_value_bound,
    declared_rounding_bound,
    probe_conversion_values,
    probe_inputs,
    randomize_batchnorm_statistics,
    randomize_zero_parameters,
    relative_error,
    weight_layer_depth,
)
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model


class _MLP(nn.Module):
    def __init__(self, act: nn.Module | None = None):
        super().__init__()
        self.l1 = nn.Linear(12, 16)
        self.bn = nn.BatchNorm1d(16)
        self.a = act if act is not None else nn.ReLU()
        self.fc = nn.Linear(16, 6)

    def forward(self, x):
        return self.fc(self.a(self.bn(self.l1(x))))


def _fp64(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model).double().eval()


def _witness_converter(packaging=MVM_PACKAGING, shape=(12,)):
    def convert(model: nn.Module) -> WitnessConversion:
        traced = converter.trace_and_convert(model, tuple(shape), "cpu", packaging=packaging)
        return WitnessConversion(traced.flow, traced.report.absorption_plan)

    return convert


# ----------------------------------------------------------------------------- the metric
class TestDeclaredBound:
    """E270's declared rounding bound tau(P) = 64 u max(D(P), 16), never above the 1e-12 acceptance."""

    def test_tau_follows_the_declared_formula(self):
        u = 2.0 ** -53
        assert declared_rounding_bound(18) == 64 * u * 18
        assert declared_rounding_bound(1001) == 64 * u * 1001
        assert declared_rounding_bound(0) == declared_rounding_bound(16) == 64 * u * 16

    def test_tau_values_of_the_specification(self):
        assert declared_rounding_bound(16) == pytest.approx(1.137e-13, rel=1e-3)
        assert declared_rounding_bound(18) == pytest.approx(1.279e-13, rel=1e-3)
        assert declared_rounding_bound(1001) == pytest.approx(7.11e-12, rel=1e-3)

    def test_the_refusal_line_never_exceeds_the_owner_acceptance(self):
        assert EXACTNESS_ACCEPTANCE == 1e-12
        assert conversion_value_bound(140) == declared_rounding_bound(140) < 1e-12
        assert conversion_value_bound(141) == conversion_value_bound(1001) == 1e-12
        assert conversion_value_bound(3) == declared_rounding_bound(16)

    @staticmethod
    def _depth(model, shape):
        return weight_layer_depth(trace_model(model.eval(), shape))

    def test_depth_counts_weight_layers_on_the_longest_path(self):
        assert self._depth(_MLP(), (12,)) == 2
        assert self._depth(nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.Linear(4, 2)), (4,)) == 3

    def test_depth_takes_the_longest_branch(self):
        class Branches(nn.Module):
            def __init__(self):
                super().__init__()
                self.a, self.b, self.c = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)

            def forward(self, x):
                return self.c(self.b(x)) + self.a(x)

        assert self._depth(Branches(), (4,)) == 2

    def test_contractions_without_modules_count(self):
        class Matmuls(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return torch.matmul(x @ self.w, self.w)

        assert self._depth(Matmuls(), (4,)) == 2

    def test_leaf_modules_count_their_weight_matrices(self):
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.mha = nn.MultiheadAttention(4, 2, batch_first=True)

            def forward(self, x):
                return self.mha(x, x, x)[0]

        assert self._depth(Attention(), (3, 4)) == 2


class TestRelativeError:
    def test_matching_nan_and_infinity_agree(self):
        nan, inf = float("nan"), float("inf")
        delta = relative_error(torch.tensor([nan, inf, -inf, 2.0]), torch.tensor([nan, inf, -inf, 2.0]))
        assert delta.relative_error == 0.0 and delta.reference_max == 2.0

    def test_nan_or_infinity_on_one_side_never_agrees(self):
        nan, inf = float("nan"), float("inf")
        assert relative_error(torch.tensor([1.0]), torch.tensor([nan])).relative_error == float("inf")
        assert relative_error(torch.tensor([inf]), torch.tensor([-inf])).relative_error == float("inf")
        assert relative_error(torch.tensor([inf]), torch.tensor([1.0])).relative_error == float("inf")

    def test_small_reference_reads_absolute_error(self):
        delta = relative_error(torch.tensor([0.5, 0.1]), torch.tensor([0.25, 0.1]))
        assert delta.max_abs_error == pytest.approx(0.25)
        assert delta.reference_max == pytest.approx(0.25)
        assert delta.relative_error == pytest.approx(0.25)

    def test_large_reference_normalizes(self):
        delta = relative_error(torch.tensor([1001.0]), torch.tensor([1000.0]))
        assert delta.relative_error == pytest.approx(1e-3)

    def test_identical_tensors_are_zero(self):
        t = torch.randn(4, 3, dtype=torch.float64)
        assert relative_error(t, t.clone()).relative_error == 0.0

    def test_shape_mismatch_is_refused_not_broadcast(self):
        with pytest.raises(ValueError, match="shape"):
            relative_error(torch.zeros(4, 1, 3), torch.zeros(4, 3))

    def test_nan_is_never_within_the_bound(self):
        delta = relative_error(torch.tensor([float("nan")]), torch.tensor([0.0]))
        assert not delta.relative_error <= EXACTNESS_ACCEPTANCE


# ----------------------------------------------------------------------------- the witness
class TestWitness:
    def test_randomizes_every_batchnorm_deterministically(self):
        model = nn.Sequential(nn.BatchNorm1d(5), nn.Linear(5, 5), nn.BatchNorm2d(3)).double()
        twin = copy.deepcopy(model)
        assert randomize_batchnorm_statistics(model, seed=1) == 2
        randomize_batchnorm_statistics(twin, seed=1)
        for a, b in zip(model.state_dict().values(), twin.state_dict().values()):
            assert torch.equal(a, b)
        bn = model[0]
        assert not torch.equal(bn.running_mean, torch.zeros(5))
        assert not torch.equal(bn.running_var, torch.ones(5))
        assert not torch.equal(bn.weight.data, torch.ones(5))
        assert bool((bn.running_var > 0).all())

    def test_different_seeds_differ_and_linear_is_untouched(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)).double()
        weight = model[0].weight.detach().clone()
        other = copy.deepcopy(model)
        randomize_batchnorm_statistics(model, seed=1)
        randomize_batchnorm_statistics(other, seed=2)
        assert not torch.equal(model[1].running_mean, other[1].running_mean)
        assert torch.equal(model[0].weight, weight)

    def test_affine_free_batchnorm_keeps_no_affine(self):
        model = nn.BatchNorm1d(4, affine=False).double()
        assert randomize_batchnorm_statistics(model, seed=3) == 1
        assert model.weight is None

    def test_probe_inputs_are_seeded_fp64_normal_uniform_zero_and_negative(self):
        x = probe_inputs((2, 3), seed=5)
        assert x.shape == (6, 2, 3) and x.dtype == torch.float64
        assert torch.equal(x, probe_inputs((2, 3), seed=5))
        assert not torch.equal(x, probe_inputs((2, 3), seed=6))
        normal, uniform, zero, negative = x[:2], x[2:4], x[4], x[5]
        assert bool((normal < 0).any()) and bool((normal > 0).any())
        assert bool((uniform >= 0).all()) and bool((uniform < 1).all())
        assert bool((zero == 0).all())
        assert bool((negative < 0).all())

    def test_probe_inputs_keep_both_random_kinds_at_a_larger_batch(self):
        x = probe_inputs((3,), seed=0, batch=8)
        assert x.shape == (8, 3)
        assert bool((x[3:6] >= 0).all()) and bool((x[6] == 0).all()) and bool((x[7] < 0).all())

    def test_probe_inputs_refuse_a_batch_without_two_random_inputs(self):
        with pytest.raises(ValueError, match="batch"):
            probe_inputs((3,), seed=0, batch=3)


class TestZeroParameterRandomization:
    """A zero-initialized parameter (a classifier head, a token) would make the comparison vacuous."""

    def test_all_zero_parameters_are_drawn_deterministically(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2)).double()
        with torch.no_grad():
            model[1].weight.zero_()
            model[1].bias.zero_()
        kept = model[0].weight.detach().clone()
        twin = copy.deepcopy(model)
        assert randomize_zero_parameters(model, seed=1) == ["1.weight", "1.bias"]
        randomize_zero_parameters(twin, seed=1)
        assert torch.equal(model[1].weight, twin[1].weight) and torch.equal(model[1].bias, twin[1].bias)
        assert bool(model[1].weight.abs().gt(0).all()) and bool(model[1].bias.abs().gt(0).all())
        assert torch.equal(model[0].weight, kept)

    def test_buffers_and_non_zero_parameters_are_untouched(self):
        model = nn.BatchNorm1d(3).double()
        with torch.no_grad():
            model.weight.fill_(1.0)
        assert randomize_zero_parameters(model, seed=1) == ["bias"]
        assert torch.equal(model.running_mean, torch.zeros(3, dtype=torch.float64))
        assert torch.equal(model.weight, torch.ones(3, dtype=torch.float64))


# ----------------------------------------------------------------------------- the contract
def _replace_gelu_by_relu(model: nn.Module) -> nn.Module:
    for name, child in list(model.named_children()):
        if isinstance(child, nn.GELU):
            setattr(model, name, nn.ReLU())
        else:
            _replace_gelu_by_relu(child)
    return model


DECLARED_GELU_AS_RELU = FunctionContract(
    name="test_gelu_deployed_as_relu",
    statement="the GELU is deployed as the neuron's ReLU law (declared in the test)",
    reference=_replace_gelu_by_relu,
)


def _gelu_deployed_as_relu(flow) -> None:
    for module in flow.modules():
        if isinstance(getattr(module, "activation", None), nn.GELU):
            module.activation = nn.ReLU()


def _converting_with(mutate):
    """A conversion whose every returned flow is mutated (the probe's witness included)."""
    original = converter.trace_and_convert

    def mutated(*args, **kwargs) -> TracedConversion:
        traced = original(*args, **kwargs)
        mutate(traced.flow)
        return traced

    return converter, mutated


class TestContract:
    def test_identity_contract_is_named_and_declares_no_change(self):
        model = nn.Linear(2, 2)
        assert NATIVE_EVAL_IDENTITY.name == "native_eval_identity"
        assert NATIVE_EVAL_IDENTITY.reference(model) is model
        assert "first" in NATIVE_EVAL_IDENTITY.statement

    def test_tuple_output_is_represented_by_its_first_element(self):
        first = torch.ones(2)
        assert NATIVE_EVAL_IDENTITY.select_output((first, torch.zeros(3))) is first
        assert NATIVE_EVAL_IDENTITY.select_output([first]) is first
        assert NATIVE_EVAL_IDENTITY.select_output(first) is first

    def test_non_tensor_output_is_refused(self):
        with pytest.raises(TypeError, match="tensor"):
            NATIVE_EVAL_IDENTITY.select_output({"logits": torch.ones(1)})

    def test_declared_change_passes_only_under_its_contract(self, monkeypatch):
        module, mutated = _converting_with(_gelu_deployed_as_relu)
        monkeypatch.setattr(module, "trace_and_convert", mutated)
        native = _MLP(nn.GELU())
        convert = _witness_converter(SPIKING_PACKAGING)
        with pytest.raises(ConversionValueParityError, match="native_eval_identity"):
            probe_conversion_values(_fp64(native), convert, (12,))
        result = probe_conversion_values(
            _fp64(native), convert, (12,), contract=DECLARED_GELU_AS_RELU,
        )
        assert result.ok and result.contract == "test_gelu_deployed_as_relu"
        assert result.relative_error is not None and result.relative_error <= 1e-12


# ----------------------------------------------------------------------------- the probe API
class _FailsInFp64(nn.Module):
    """Creates an fp32 operand at run time, so an fp64 copy cannot run."""

    def __init__(self):
        super().__init__()
        self.l = nn.Linear(12, 6)

    def forward(self, x):
        return self.l(x) @ torch.ones(6, 6, dtype=torch.float32)


class _Reshaped(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x).unsqueeze(1)


class _TupleOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(12, 6)

    def forward(self, x):
        y = self.l(x)
        return y, y * 2


class TestProbeApi:
    def test_exact_conversion_passes_with_its_measurements(self):
        result = probe_conversion_values(_fp64(_MLP()), _witness_converter(), (12,))
        assert result.ok and result.stage == STAGE_CONVERTED
        assert result.contract == NATIVE_EVAL_IDENTITY.name
        assert result.relative_error is not None and result.relative_error <= 1e-12
        assert result.max_abs_error is not None and result.reference_max is not None
        assert result.wall_s >= 0.0
        assert "PASS" in result.format()

    def test_native_failure_in_fp64_is_refused_by_name(self):
        def convert(model):
            return WitnessConversion(model, {})

        with pytest.raises(ConversionValueParityError) as excinfo:
            probe_conversion_values(_fp64(_FailsInFp64()), convert, (12,))
        assert excinfo.value.stage == STAGE_NATIVE
        assert "fp64" in str(excinfo.value)

    def test_output_shape_mismatch_is_refused(self):
        convert = _witness_converter()

        def reshaped(model):
            conversion = convert(model)
            return WitnessConversion(_Reshaped(conversion.flow), conversion.absorption_plan)

        with pytest.raises(ConversionValueParityError, match="shape"):
            probe_conversion_values(_fp64(_MLP()), reshaped, (12,))

    def test_tuple_output_native_compares_its_first_element(self):
        result = probe_conversion_values(_fp64(_TupleOut()), _witness_converter(), (12,))
        assert result.ok

    def test_permissive_mode_returns_the_refusal(self):
        def negated(model):
            conversion = _witness_converter()(model)
            with torch.no_grad():
                for p in conversion.flow.parameters():
                    p.neg_()
            return conversion

        result = probe_conversion_values(_fp64(_MLP()), negated, (12,), strict=False)
        assert not result.ok
        assert result.relative_error is not None and result.relative_error > result.bound
        assert "REFUSED" in result.format()

    def test_localization_reads_the_flow_in_eval_mode(self):
        """A BatchNorm upstream of the defect must not read batch statistics during localization."""

        def negated_fc(model):
            conversion = _witness_converter()(model)
            for node in conversion.flow.get_mapper_repr().execution_order():
                if getattr(getattr(node, "perceptron", None), "name", None) == "fc":
                    with torch.no_grad():
                        node.perceptron.layer.weight.neg_()
            return conversion

        result = probe_conversion_values(_fp64(_MLP()), negated_fc, (12,), strict=False)
        assert not result.ok
        assert result.first_diverging_node == "fc"

    def test_toolchain_refusals_pass_through_the_witness_conversion(self):
        from mimarsinan.torch_mapping.representability_analyzer import (
            RepresentabilityError,
            RepresentabilityReport,
        )

        refusal = RepresentabilityError(RepresentabilityReport(is_representable=False))

        def refusing(model):
            raise refusal

        with pytest.raises(RepresentabilityError) as excinfo:
            probe_conversion_values(_fp64(_MLP()), refusing, (12,))
        assert excinfo.value is refusal

    def test_anonymous_witness_conversion_failure_stays_anonymous(self):
        """Restated in review round 2 (E1): the crash reaches the caller as itself, never wrapped as a refusal."""

        def crashing(model):
            raise IndexError("deep inside a mapper")

        with pytest.raises(IndexError, match="deep inside a mapper") as excinfo:
            probe_conversion_values(_fp64(_MLP()), crashing, (12,))
        assert type(excinfo.value) is IndexError and not isinstance(excinfo.value, ConversionProbeError)

    def test_refusal_is_a_conversion_probe_error(self):
        assert issubclass(ConversionValueParityError, ConversionProbeError)

    def test_probe_does_not_touch_the_global_rng(self):
        snapshot = _fp64(_MLP())
        torch.manual_seed(123)
        probe_conversion_values(snapshot, _witness_converter(), (12,))
        after_probe = torch.rand(3)
        torch.manual_seed(123)
        assert torch.equal(after_probe, torch.rand(3))


# ----------------------------------------------------------------------------- converter wiring
class TestConverterWiring:
    def test_trace_and_convert_returns_flow_and_report(self):
        traced = trace_and_convert(_MLP().eval(), (12,), "cpu", packaging=MVM_PACKAGING)
        assert isinstance(traced, TracedConversion)
        assert traced.report.is_representable
        assert "bn" in traced.report.absorption_plan

    def test_convert_does_not_touch_native_statistics_or_rng(self):
        model = _MLP().eval()
        before = copy.deepcopy(model.state_dict())
        torch.manual_seed(5)
        convert_torch_model(model, (12,), 6, packaging=MVM_PACKAGING)
        after_convert = torch.rand(3)
        torch.manual_seed(5)
        trace_and_convert(copy.deepcopy(model), (12,), "cpu", packaging=MVM_PACKAGING)
        assert torch.equal(after_convert, torch.rand(3)), (
            "the value probe consumed the global RNG beyond the conversion itself"
        )
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[key]), key

    def test_returned_flow_equals_an_unprobed_conversion(self):
        model = _MLP().eval()
        flow = convert_torch_model(copy.deepcopy(model), (12,), 6, packaging=SPIKING_PACKAGING)
        twin = trace_and_convert(copy.deepcopy(model), (12,), "cpu", packaging=SPIKING_PACKAGING).flow
        got, want = flow.state_dict(), twin.state_dict()
        assert list(got) == list(want)
        for key in got:
            assert torch.equal(got[key], want[key]), key

    def test_structural_zero_input_stage_still_runs_first(self, monkeypatch):
        from mimarsinan.torch_mapping import conversion_probe

        calls = []
        original = conversion_probe.probe_forward

        def recording(*args, **kwargs):
            calls.append("structural")
            return original(*args, **kwargs)

        monkeypatch.setattr(conversion_probe, "probe_forward", recording)
        convert_torch_model(_MLP().eval(), (12,), 6, packaging=MVM_PACKAGING)
        assert calls == ["structural"]

    def test_value_refusal_in_permissive_conversion_is_printed(self, monkeypatch, capsys):
        def negate_fc(flow):
            for node in flow.get_mapper_repr().execution_order():
                if getattr(getattr(node, "perceptron", None), "name", None) == "fc":
                    with torch.no_grad():
                        node.perceptron.layer.weight.neg_()

        module, mutated = _converting_with(negate_fc)
        monkeypatch.setattr(module, "trace_and_convert", mutated)
        from mimarsinan.torch_mapping.value_parity.certificate import UnverifiedConversionWarning

        with pytest.warns(UnverifiedConversionWarning, match="NOT certified"):
            flow = convert_torch_model(_MLP().eval(), (12,), 6, packaging=MVM_PACKAGING, strict=False)
        assert flow is not None
        assert "REFUSED" in capsys.readouterr().out


# ----------------------------------------------------------------------------- refusal line and degenerate references
class _NullLogit(nn.Module):
    """A learned null logit before a log-sum-exp: dropping it moves the output by about exp(null)."""

    def __init__(self, null: float):
        super().__init__()
        self.fc = nn.Linear(12, 6)
        self.null = nn.Parameter(torch.full((1, 1), null))

    def forward(self, x):
        z = torch.cat([self.fc(x), self.null.expand(x.size(0), -1)], dim=1)
        return torch.logsumexp(z, dim=1, keepdim=True) - z[:, :6]


class _PassFirst(nn.Module):
    def forward(self, *inputs):
        return inputs[0]


def _drop_the_null_logit(flow) -> None:
    """The defect the review measured: the concatenation keeps its activation and drops the parameter."""
    from mimarsinan.torch_mapping.value_parity.localization import mapper_label

    for node in flow.get_mapper_repr().execution_order():
        if mapper_label(node) == "cat":
            node.module = _PassFirst()


class _ZeroHead(nn.Module):
    """A classifier head initialized to zero (ViT style): its native output is all zero."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 16)
        self.r = nn.ReLU()
        self.head = nn.Linear(16, 6)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def forward(self, x):
        return self.head(self.r(self.l1(x)))


class _InputIgnored(nn.Module):
    """Its output never depends on the input (a constant path): the constant is still compared."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(12, 6)

    def forward(self, x):
        return self.head(torch.zeros_like(x))


class _ZeroOutput(nn.Module):
    """Its output is zero on every input whatever the weights: nothing upstream is ever compared."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 6)

    def forward(self, x):
        return self.l1(x) * 0.0


def _negate_head(flow) -> None:
    for node in flow.get_mapper_repr().execution_order():
        perceptron = getattr(node, "perceptron", None)
        if getattr(perceptron, "name", None) == "head":
            with torch.no_grad():
                perceptron.layer.bias.neg_()


def _scale_first_perceptron(flow) -> None:
    for node in flow.get_mapper_repr().execution_order():
        perceptron = getattr(node, "perceptron", None)
        if perceptron is not None:
            with torch.no_grad():
                perceptron.layer.weight.mul_(-3.0)
            return


class TestRefusalLine:
    @pytest.mark.parametrize("null", [-22.0, -25.0])
    def test_a_dropped_null_logit_is_refused(self, monkeypatch, null):
        module, mutated = _converting_with(_drop_the_null_logit)
        monkeypatch.setattr(module, "trace_and_convert", mutated)
        result = probe_conversion_values(_fp64(_NullLogit(null)), _witness_converter(), (12,), strict=False)
        assert not result.ok, result.format()
        assert result.relative_error is not None and 1e-13 < result.relative_error < 1e-9
        assert result.relative_error > result.bound

    def test_exact_conversion_passes_under_tau(self):
        result = probe_conversion_values(_fp64(_NullLogit(-22.0)), _witness_converter(), (12,))
        assert result.ok and result.relative_error is not None
        assert result.relative_error <= result.bound <= EXACTNESS_ACCEPTANCE
        assert result.weight_layer_depth == 1


class TestDegenerateReference:
    def test_zero_initialized_head_is_randomized_and_a_wrong_conversion_is_refused(self, monkeypatch):
        module, mutated = _converting_with(_scale_first_perceptron)
        monkeypatch.setattr(module, "trace_and_convert", mutated)
        with pytest.raises(ConversionValueParityError) as excinfo:
            probe_conversion_values(_fp64(_ZeroHead()), _witness_converter(), (12,))
        assert excinfo.value.reference_max is not None and excinfo.value.reference_max > 0
        assert "randomized 2 all-zero parameters" in str(excinfo.value)

    def test_zero_initialized_head_converts(self):
        result = probe_conversion_values(_fp64(_ZeroHead()), _witness_converter(), (12,))
        assert result.ok and result.reference_max is not None and result.reference_max > 0
        assert result.randomized_parameters == ("head.weight", "head.bias")

    def test_an_all_zero_reference_is_refused_by_name(self):
        with pytest.raises(ConversionValueParityError) as excinfo:
            probe_conversion_values(_fp64(_ZeroOutput()), _witness_converter(), (12,))
        assert excinfo.value.stage == STAGE_NATIVE
        assert "zero on all 6 seeded inputs" in str(excinfo.value)

    def test_a_constant_reference_is_compared_and_flagged(self, monkeypatch):
        result = probe_conversion_values(_fp64(_InputIgnored()), _witness_converter(), (12,))
        assert result.ok and result.reference_constant
        assert "reference constant across the inputs" in result.format()
        module, mutated = _converting_with(_negate_head)
        monkeypatch.setattr(module, "trace_and_convert", mutated)
        with pytest.raises(ConversionValueParityError):
            probe_conversion_values(_fp64(_InputIgnored()), _witness_converter(), (12,))


# ----------------------------------------------------------------------------- review round 2: degeneracy by finite values
def _seeded(factory) -> nn.Module:
    torch.manual_seed(0)
    return _fp64(factory())


class _InfRef(nn.Module):
    """A zero-initialized normalizer buffer: ``x / 0`` makes the native output non-finite on every input."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.zeros(12))
        self.l1 = nn.Linear(12, 16)
        self.r = nn.ReLU()
        self.head = nn.Linear(16, 6)

    def forward(self, x):
        return self.head(self.r(self.l1(x / self.scale)))


class _NanRef(nn.Module):
    """The square root of a signed pre-activation: NaN on every input."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 16)
        self.head = nn.Linear(16, 6)

    def forward(self, x):
        return self.head(torch.sqrt(self.l1(x)))


class _DeadRelu(nn.Module):
    """``head(relu(l1(x) - 10))``: every ReLU is dead on unit-scale inputs (a constant output), alive beyond."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 16)
        self.r = nn.ReLU()
        self.head = nn.Linear(16, 6)

    def forward(self, x):
        return self.head(self.r(self.l1(x) - 10.0))


class _AllMaskedEncoder(nn.Module):
    """A TransformerEncoderLayer whose mask removes every key: the native output is NaN on every input."""

    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True)
        self.register_buffer("mask", torch.full((4, 4), float("-inf")))

    def forward(self, x):
        return self.layer(x, src_mask=self.mask)


class TestNonFiniteReference:
    """The reference's finite values decide degeneracy: no finite non-zero entry is no information."""

    @pytest.mark.parametrize("factory,kind", [(_InfRef, "non-finite"), (_NanRef, "non-finite")],
                             ids=["inf_normalizer", "nan_sqrt"])
    @pytest.mark.parametrize("mutate", [None, _scale_first_perceptron], ids=["exact", "first_perceptron_x-3"])
    def test_a_reference_without_a_finite_non_zero_entry_is_refused_by_name(self, monkeypatch, factory, kind,
                                                                            mutate):
        if mutate is not None:
            module, mutated = _converting_with(mutate)
            monkeypatch.setattr(module, "trace_and_convert", mutated)
        with pytest.raises(ConversionValueParityError) as excinfo:
            probe_conversion_values(_seeded(factory), _witness_converter(), (12,))
        assert excinfo.value.stage == STAGE_NATIVE
        assert f"{kind} on all 6 seeded inputs" in str(excinfo.value)
        assert "escalated" in str(excinfo.value)

    def test_an_all_masked_encoder_is_refused_by_name(self):
        with pytest.raises(ConversionValueParityError) as excinfo:
            probe_conversion_values(
                _seeded(_AllMaskedEncoder), _witness_converter(SPIKING_PACKAGING, (4, 8)), (4, 8),
            )
        assert excinfo.value.stage == STAGE_NATIVE and "non-finite" in str(excinfo.value)

    def test_a_partly_finite_reference_is_compared_on_its_finite_values(self):
        assert relative_error(torch.tensor([float("nan"), 2.0]), torch.tensor([float("nan"), 2.0])).reference_max == 2.0


class TestEscalatedInputs:
    """A constant reference is compared again on escalated-magnitude inputs before it is certified."""

    def test_escalated_inputs_are_seeded_and_scaled(self):
        from mimarsinan.torch_mapping.value_parity.witness import ESCALATED_SCALES, escalated_probe_inputs

        a = escalated_probe_inputs((12,), seed=1, batch=6)
        z = torch.randn(6, 12, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        assert ESCALATED_SCALES == (10.0, 100.0)
        assert a.dtype == torch.float64 and torch.equal(a[:3], 10.0 * z[:3]) and torch.equal(a[3:], 100.0 * z[3:])

    def test_a_dead_relu_conversion_is_compared_where_it_lives(self):
        result = probe_conversion_values(_seeded(_DeadRelu), _witness_converter(), (12,))
        assert result.ok and not result.reference_constant
        assert result.escalated

    def test_a_wrong_dead_relu_conversion_is_refused(self, monkeypatch):
        module, mutated = _converting_with(_scale_first_perceptron)
        monkeypatch.setattr(module, "trace_and_convert", mutated)
        with pytest.raises(ConversionValueParityError, match="escalated"):
            probe_conversion_values(_seeded(_DeadRelu), _witness_converter(), (12,))

    def test_a_reference_constant_everywhere_is_certified_with_the_flag(self):
        result = probe_conversion_values(_fp64(_InputIgnored()), _witness_converter(), (12,))
        assert result.ok and result.reference_constant and result.escalated


# ----------------------------------------------------------------------------- review round 2: the batch-1 run (E2)
class _WrongAtBatchOne(nn.Module):
    """A converted flow whose output is wrong only at batch 1: a batch-dependent constant."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out = self.inner(x)
        return 2.0 * out if x.shape[0] == 1 else out


class _IdentityOfTheBatchSize(nn.Module):
    """``operator.is_(x.size(0), 1)``: tracing folds it to False; natively it is True at batch 1."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(12, 6)

    def forward(self, x):
        import operator

        y = self.fc(x)
        return y * operator.is_(x.size(0), 1) + y


class TestBatchOne:
    """The witness also runs at batch 1 (the first seeded input alone) against the native batch-1 output."""

    def test_a_conversion_wrong_only_at_batch_one_is_refused_by_name(self):
        convert = _witness_converter()

        def wrapped(model):
            conversion = convert(model)
            return WitnessConversion(_WrongAtBatchOne(conversion.flow), conversion.absorption_plan)

        with pytest.raises(ConversionValueParityError, match="batch 1") as excinfo:
            probe_conversion_values(_fp64(_MLP()), wrapped, (12,))
        assert excinfo.value.stage == STAGE_CONVERTED

    @pytest.mark.parametrize("packaging", [MVM_PACKAGING, SPIKING_PACKAGING], ids=["mvm", "spiking"])
    def test_a_traced_batch_constant_is_refused_by_name(self, packaging):
        with pytest.raises(ConversionValueParityError, match="batch 1"):
            convert_torch_model(_seeded(_IdentityOfTheBatchSize), (12,), 6, packaging=packaging)

    def test_an_exact_conversion_passes_at_both_batch_sizes(self):
        result = probe_conversion_values(_fp64(_MLP()), _witness_converter(), (12,))
        assert result.ok and "batch 1" in result.format()


# ----------------------------------------------------------------------------- review round 2: the lowering line
class _ConstantOverActivation(nn.Module):
    """``c / y`` on (B, 5, 6) tokens, y = Linear(x): the host division amplifies the core's reassociation."""

    def __init__(self):
        super().__init__()
        self.pre = nn.Linear(6, 6)
        self.c = nn.Parameter(torch.randn(6))

    def forward(self, x):
        return self.c / self.pre(x)


class _OffsetProgram(nn.Module):
    """A lowering mutation: the value program's output moved by ``rel * max(1, max|out|)``."""

    def __init__(self, inner, rel: float):
        super().__init__()
        self.inner = inner
        self.rel = rel

    def forward(self, x):
        out = self.inner(x)
        return out + self.rel * max(1.0, float(out.abs().max()))

    def run_retaining_state(self, x):
        return self.inner.run_retaining_state(x)


class TestLoweringLine:
    """Stage 2's line: tau(P) plus the first-order reassociation bound of every core, capped at 1e-12."""

    def test_gamma_follows_the_standard_formula(self):
        from mimarsinan.torch_mapping.value_parity.rounding import UNIT_ROUNDOFF, gamma

        assert gamma(7) == 7 * UNIT_ROUNDOFF / (1 - 7 * UNIT_ROUNDOFF)

    def test_one_core_bound_is_twice_gamma_of_its_absolute_dot_products(self):
        from mimarsinan.torch_mapping.value_parity.rounding import gamma, reassociation_bound

        torch.manual_seed(0)
        core = nn.Linear(12, 6).double().eval()
        x = torch.randn(3, 12, dtype=torch.float64)
        with torch.no_grad():
            sums = torch.nn.functional.linear(x.abs(), core.weight.abs(), core.bias.abs())
            reference_max = float(core(x).abs().max())
        expected = float((2 * gamma(13) * sums).max()) / max(1.0, reference_max)
        assert reassociation_bound(core, x, NATIVE_EVAL_IDENTITY.select_output) == pytest.approx(expected, rel=1e-12)

    def test_the_c_over_x_program_passes_under_its_derived_line(self):
        from mimarsinan.torch_mapping.converter import probe_witness_conversion

        model = _seeded(_ConstantOverActivation)
        _, lowering = probe_witness_conversion(model, (5, 6), packaging=MVM_PACKAGING)
        tau = declared_rounding_bound(lowering.weight_layer_depth)
        assert lowering.ok and lowering.relative_error > tau, lowering.format()
        assert lowering.reassociation_bound is not None
        assert lowering.relative_error - tau < lowering.reassociation_bound
        assert lowering.relative_error <= lowering.bound == min(EXACTNESS_ACCEPTANCE, tau + lowering.reassociation_bound)

    def test_a_lowering_mutation_of_5e_13_is_refused(self, monkeypatch):
        from mimarsinan.torch_mapping.converter import probe_witness_conversion
        from mimarsinan.torch_mapping.value_parity import value_program

        original = value_program.fp64_identity_program
        monkeypatch.setattr(value_program, "fp64_identity_program", lambda ir: _OffsetProgram(original(ir), 5e-13))
        with pytest.raises(ConversionValueParityError) as excinfo:
            probe_witness_conversion(_fp64(_MLP()), (12,), packaging=MVM_PACKAGING)
        assert excinfo.value.stage == "identity value program"
        assert excinfo.value.relative_error == pytest.approx(5e-13, rel=1e-2)
        assert excinfo.value.result.bound < 5e-13


# ----------------------------------------------------------------------------- review round 3: escalation for every reference
class _DeadReluEps(nn.Module):
    """``_DeadRelu`` plus a 1e-15-scale live term: near-constant, so exact constancy never escalated it."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 8)
        self.head = nn.Linear(8, 3)
        self.tap = nn.Linear(12, 3)

    def forward(self, x):
        return self.head(torch.relu(self.l1(x) - 10.0)) + 1e-15 * self.tap(x)


class _DeadBranchSkip(nn.Module):
    """A dead ReLU branch beside a live linear skip: the reference varies, the branch is dead on seeded inputs."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 8)
        self.head = nn.Linear(8, 3)
        self.skip = nn.Linear(12, 3)

    def forward(self, x):
        return self.head(torch.relu(self.l1(x) - 10.0)) + self.skip(x)


class _GateNaN(nn.Module):
    """Constant on the seeded inputs (the head bias); NaN on every escalated input (``sqrt(-relu(|x|^2 - 400))``)."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 8)
        self.head = nn.Linear(8, 3)
        self.ident = nn.Linear(12, 12)
        self.sq = nn.Linear(12, 1)
        with torch.no_grad():
            self.ident.weight.copy_(torch.eye(12))
            self.ident.bias.zero_()
            self.sq.weight.fill_(1.0)
            self.sq.bias.fill_(-400.0)

    def forward(self, x):
        h = self.ident(x)
        return self.head(torch.relu(self.l1(x) - 10.0)) + torch.sqrt(-torch.relu(self.sq(h * h)))


class _NearlyConstant(nn.Module):
    """A constant head plus a 1e-17-scale live term: constant under the probe's metric, never under ``==``."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(12, 6)
        self.tap = nn.Linear(12, 6)

    def forward(self, x):
        return self.head(torch.zeros_like(x)) + 1e-17 * self.tap(x)


class TestEscalationForEveryReference:
    """Every reference is compared on the escalated-magnitude inputs, wherever a dead-on-seeded branch sits."""

    @pytest.mark.parametrize("factory", [_DeadReluEps, _DeadBranchSkip], ids=["dead_relu_eps", "dead_branch_skip"])
    def test_a_wrong_dead_branch_is_refused_by_name(self, monkeypatch, factory):
        module, mutated = _converting_with(_scale_first_perceptron)
        monkeypatch.setattr(module, "trace_and_convert", mutated)
        with pytest.raises(ConversionValueParityError, match="escalated-magnitude inputs") as excinfo:
            probe_conversion_values(_seeded(factory), _witness_converter(), (12,))
        assert excinfo.value.stage == STAGE_CONVERTED and excinfo.value.relative_error > 1e-3

    @pytest.mark.parametrize("factory", [_DeadReluEps, _DeadBranchSkip], ids=["dead_relu_eps", "dead_branch_skip"])
    def test_an_exact_dead_branch_is_certified_after_escalation(self, factory):
        result = probe_conversion_values(_seeded(factory), _witness_converter(), (12,))
        assert result.ok and result.escalated and not result.reference_constant

    def test_a_live_reference_is_escalated_too(self):
        result = probe_conversion_values(_fp64(_MLP()), _witness_converter(), (12,))
        assert result.ok and result.escalated and "escalated-magnitude inputs" in result.format()

    @pytest.mark.parametrize("mutate", [None, _scale_first_perceptron], ids=["exact", "first_perceptron_x-3"])
    def test_constant_on_seeded_and_non_finite_on_escalated_inputs_is_refused_by_name(self, monkeypatch, mutate):
        if mutate is not None:
            module, mutated = _converting_with(mutate)
            monkeypatch.setattr(module, "trace_and_convert", mutated)
        with pytest.raises(ConversionValueParityError) as excinfo:
            probe_conversion_values(_seeded(_GateNaN), _witness_converter(), (12,))
        message = str(excinfo.value)
        assert excinfo.value.stage == STAGE_NATIVE
        assert "constant on all 6 seeded inputs" in message and "non-finite on all 6 escalated" in message

    def test_constancy_is_decided_by_the_probe_metric(self):
        from mimarsinan.torch_mapping.value_parity.comparison import constant_across_samples

        base = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float64)
        nudged = base + torch.tensor([[0.0, 0.0], [1e-15, 0.0]], dtype=torch.float64)
        assert constant_across_samples(base) and not constant_across_samples(nudged)
        assert constant_across_samples(nudged, 1e-13) and not constant_across_samples(nudged, 1e-16)
        both_nan = torch.full((2, 2), float("nan"), dtype=torch.float64)
        assert constant_across_samples(both_nan)

    def test_a_reference_constant_under_the_metric_is_flagged(self):
        result = probe_conversion_values(_seeded(_NearlyConstant), _witness_converter(), (12,))
        assert result.ok and result.reference_constant
        assert "reference constant across the inputs" in result.format()


# ----------------------------------------------------------------------------- review round 4: magnitudes between
class _Gate(nn.Module):
    """``s = min(relu(|x|^2 - 400), 1)``: 0 on the seeded and 3 N(0, 1) inputs, 1 from 10 N(0, 1) on."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 8)
        self.head = nn.Linear(8, 3)
        self.ident = nn.Linear(12, 12)
        self.sq = nn.Linear(12, 1)
        with torch.no_grad():
            self.ident.weight.copy_(torch.eye(12))
            self.ident.bias.zero_()
            self.sq.weight.fill_(1.0)
            self.sq.bias.fill_(-400.0)

    def gate(self, x):
        h = self.ident(x)
        return torch.clamp(torch.relu(self.sq(h * h)), max=1.0)


class _GateConst(_Gate):
    """GateNaN with a finite constant for NaN: the head bias on the seeded inputs, 5.0 on the escalated ones."""

    def forward(self, x):
        s = self.gate(x)
        return self.head(torch.relu(self.l1(x) - 10.0)) * (1.0 - s) + 5.0 * s


class _GateConstSkip(_Gate):
    """A live skip beside a ReLU branch dead on the seeded inputs and gated off on the escalated ones."""

    def __init__(self):
        super().__init__()
        self.skip = nn.Linear(12, 3)

    def forward(self, x):
        s = self.gate(x)
        return self.head(torch.relu(self.l1(x) - 10.0) * (1.0 - s)) + self.skip(x)


class _ExpandCat(nn.Module):
    """``test_call_method_fail_loud.ExpandCatModel``: a class token read back, constant on every input."""

    def __init__(self):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, 8))
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        y = torch.cat((self.token.expand(x.shape[0], -1, -1), x), dim=1)
        return self.head(y[:, 0])


class TestIntermediateMagnitudes:
    """Every reference is also compared at 3 N(0, 1) and 30 N(0, 1), between the seeded and escalated magnitudes."""

    def test_intermediate_inputs_are_seeded_and_scaled(self):
        from mimarsinan.torch_mapping.value_parity.witness import INTERMEDIATE_SCALES, intermediate_probe_inputs

        a = intermediate_probe_inputs((12,), seed=1, batch=6)
        z = torch.randn(6, 12, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        assert INTERMEDIATE_SCALES == (3.0, 30.0)
        assert a.dtype == torch.float64 and torch.equal(a[:3], 3.0 * z[:3]) and torch.equal(a[3:], 30.0 * z[3:])

    def test_a_branch_dead_on_seeded_and_gated_off_on_escalated_inputs_is_refused_by_name(self, monkeypatch):
        module, mutated = _converting_with(_scale_first_perceptron)
        monkeypatch.setattr(module, "trace_and_convert", mutated)
        with pytest.raises(ConversionValueParityError, match="intermediate-magnitude inputs") as excinfo:
            probe_conversion_values(_seeded(_GateConstSkip), _witness_converter(), (12,))
        assert excinfo.value.stage == STAGE_CONVERTED and excinfo.value.relative_error > 1e-6

    def test_the_exact_gated_branch_is_certified(self):
        result = probe_conversion_values(_seeded(_GateConstSkip), _witness_converter(), (12,))
        assert result.ok and not result.reference_constant
        assert "6 intermediate-magnitude inputs" in result.format()

    def test_a_reference_constant_on_each_magnitude_is_flagged(self):
        """GateConst: constant on each input set on its own (the union is not): flagged as ExpandCatModel is."""
        result = probe_conversion_values(_seeded(_GateConst), _witness_converter(), (12,))
        assert result.ok and result.reference_constant
        assert "reference constant across the inputs" in result.format()

    def test_a_wrong_conversion_of_a_reference_constant_on_each_magnitude_is_refused_by_name(self, monkeypatch):
        module, mutated = _converting_with(_scale_first_perceptron)
        monkeypatch.setattr(module, "trace_and_convert", mutated)
        with pytest.raises(ConversionValueParityError, match="intermediate-magnitude inputs") as excinfo:
            probe_conversion_values(_seeded(_GateConst), _witness_converter(), (12,))
        assert excinfo.value.stage == STAGE_CONVERTED

    def test_an_expand_cat_reference_stays_certified_and_flagged(self):
        torch.manual_seed(1)
        result = probe_conversion_values(_fp64(_ExpandCat()), _witness_converter(MVM_PACKAGING, (3, 8)), (3, 8))
        assert result.ok and result.reference_constant


# ----------------------------------------------------------------------------- review round 3: open issue, pinned
class _TraceTimeBatchFold(nn.Module):
    """A Python-level evaluation of the batch size's Proxy that FX tracing performs and folds into a constant.

    Identity, hashing, ``repr``, ``str`` and ``format`` of ``x.size(0)`` (review round 4 added the last three),
    ``repr`` of ``x.shape`` (review round 5), and set membership (hashing) and the type checks ``isinstance``,
    ``type``, ``callable`` and ``hasattr`` (review round 6). Natively the value differs only at batch 2, where no probe
    batch (6, 1) looks. The programs behave the same on w0-base, except that w0-base refuses the ``type`` and
    ``hasattr`` forms through its own size-alias defect. Since review round 5 the class is a stated boundary of the
    certificate (``CERTIFICATE_HYPOTHESES``, trace fidelity), not a tolerance; the programs stay pinned as
    known-wrong, and closing the class needs a symbolic-batch trace of the model.
    """

    FORMS = {
        "is": lambda x: operator.is_(x.size(0), 2),
        "is_not": lambda x: operator.is_not(x.size(0), 2),
        "dict_get": lambda x: {2: 3.0}.get(x.size(0), 1.0),
        "repr": lambda x: repr(x.size(0)) == "2",
        "str": lambda x: str(x.size(0)) == "2",
        "format": lambda x: f"{x.size(0)}" == "2",
        "shape_repr": lambda x: repr(x.shape) == "torch.Size([2, 12])",
        "set_membership": lambda x: x.size(0) in {2, 13},
        "isinstance": lambda x: (x.size(0) == 2) if isinstance(x.size(0), int) else False,
        "shape_isinstance": lambda x: (x.size(0) == 2) if isinstance(x.shape, torch.Size) else False,
        "type": lambda x: (x.size(0) == 2) if type(x.size(0)) is int else False,
        "callable": lambda x: False if callable(x.size(0)) else (x.size(0) == 2),
        "hasattr": lambda x: False if hasattr(x.size(0), "node") else (x.size(0) == 2),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(12, 6)

    def forward(self, x):
        y = self.fc(x)
        return y * self.FORMS[self.how](x) + y



@pytest.mark.parametrize("packaging", [MVM_PACKAGING, SPIKING_PACKAGING], ids=["mvm", "spiking"])
@pytest.mark.parametrize("how", sorted(_TraceTimeBatchFold.FORMS))
def test_known_wrong_trace_time_batch_folds_are_pinned(how, packaging):
    """Known wrong (open issue): certified, exact at batches 1, 3 and 6, wrong at batch 2. When this fails, the
    class is closed: restate it as exact or refused by name."""
    model = _seeded(lambda: _TraceTimeBatchFold(how))
    flow = convert_torch_model(copy.deepcopy(model), (12,), 6, packaging=packaging).eval()
    assert flow.conversion_certificate.certified
    errors = {}
    for batch in (1, 2, 3, 6):
        x = torch.randn(batch, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            errors[batch] = relative_error(flow(x), model(x)).relative_error
    assert errors[1] == errors[3] == errors[6] == 0.0, errors
    assert errors[2] >= 0.3, errors
