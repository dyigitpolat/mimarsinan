"""Tests for graph_normalization: MM+ fusion through Identity and BatchNorm.

The MM+ rule: consecutive ops representable as matrix multiplications
(Linear, BatchNorm — a diagonal MM) are fused into a single Linear.
Identity ops between MMs are pure passthrough. BN between two Linears
is folded into the preceding Linear before the two Linears are fused.

Part 2 (integration): verifies that the fused graph correctly flows through
the full conversion pipeline (convert_torch_model) and that the resulting
Perceptrons have the right properties for downstream adaptation steps.
"""

import pytest
import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph


# ---------------------------------------------------------------------------
# Helper: count module types in the FX graph
# ---------------------------------------------------------------------------

def _count_modules(gm: fx.GraphModule):
    """Return a dict mapping module type names to call counts in the FX graph."""
    modules = dict(gm.named_modules())
    counts = {}
    for node in gm.graph.nodes:
        if node.op == "call_module":
            mod = modules.get(node.target)
            if mod is not None:
                name = type(mod).__name__
                counts[name] = counts.get(name, 0) + 1
    return counts


def _trace_and_normalize(model, input_shape):
    gm = trace_model(model, input_shape, device="cpu")
    gm = normalize_fx_graph(gm)
    return gm


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------

class LinearIdentityLinear(nn.Module):
    """Linear -> Identity -> Linear (should fuse to single Linear)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.id = nn.Identity()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.id(self.fc1(x)))


class LinearBNLinear(nn.Module):
    """Linear -> BN -> Linear (BN should be folded, then Linears fused)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.bn(self.fc1(x)))


class LinearBNLinearReLU(nn.Module):
    """Linear -> BN -> Linear -> ReLU (fuse MMs, keep ReLU for perceptron packaging)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 4)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.fc2(self.bn(self.fc1(x))))


class LinearIdentityBNLinear(nn.Module):
    """Linear -> Identity -> BN -> Linear (mixed chain, should still fuse)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.id = nn.Identity()
        self.bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.bn(self.id(self.fc1(x))))


class ThreeLinearChain(nn.Module):
    """Linear -> BN -> Linear -> Linear (iterative fusion: all three become one)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc3(self.fc2(self.bn(self.fc1(x))))


class LinearBNReLU(nn.Module):
    """Linear -> BN -> ReLU — no second Linear, so no fusion should occur."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
        self.bn = nn.BatchNorm1d(4)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.fc(x)))


class BranchingBN(nn.Module):
    """Linear -> BN that fans out to two consumers — BN has two users, no fusion."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.bn = nn.BatchNorm1d(16)
        self.fc2a = nn.Linear(16, 4)
        self.fc2b = nn.Linear(16, 4)

    def forward(self, x):
        h = self.bn(self.fc1(x))
        return self.fc2a(h) + self.fc2b(h)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLinearIdentityLinearFusion:
    """Existing behavior: Linear -> Identity -> Linear fuses to single Linear."""

    def test_fuses_to_single_linear(self):
        model = LinearIdentityLinear()
        gm = _trace_and_normalize(model, (8,))
        counts = _count_modules(gm)
        assert counts.get("Linear", 0) == 1, (
            f"Expected 1 Linear after fusion, got {counts}"
        )

    def test_numerical_equivalence(self):
        model = LinearIdentityLinear().eval()
        x = torch.randn(4, 8)
        expected = model(x)

        gm = _trace_and_normalize(model, (8,))
        gm.eval()
        actual = gm(x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"Max diff: {(expected - actual).abs().max().item():.6f}"
        )


class TestLinearBNLinearFusion:
    """New: Linear -> BN -> Linear fuses to single Linear (BN folded first)."""

    def test_fuses_to_single_linear(self):
        model = LinearBNLinear()
        model.eval()
        # Run a forward pass to initialize BN running stats
        with torch.no_grad():
            model(torch.randn(32, 8))

        gm = _trace_and_normalize(model, (8,))
        counts = _count_modules(gm)
        assert counts.get("Linear", 0) == 1, (
            f"Expected 1 Linear after BN-fold + fusion, got {counts}"
        )
        assert counts.get("BatchNorm1d", 0) == 0, (
            f"BatchNorm1d should be eliminated after folding, got {counts}"
        )

    def test_numerical_equivalence(self):
        model = LinearBNLinear().eval()
        with torch.no_grad():
            model(torch.randn(32, 8))

        x = torch.randn(4, 8)
        expected = model(x)

        gm = _trace_and_normalize(model, (8,))
        gm.eval()
        actual = gm(x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"Max diff: {(expected - actual).abs().max().item():.6f}"
        )


class TestLinearBNLinearReLU:
    """Linear -> BN -> Linear -> ReLU: fuse MMs, keep ReLU."""

    def test_structure(self):
        model = LinearBNLinearReLU().eval()
        with torch.no_grad():
            model(torch.randn(32, 8))

        gm = _trace_and_normalize(model, (8,))
        counts = _count_modules(gm)
        assert counts.get("Linear", 0) == 1
        assert counts.get("ReLU", 0) == 1
        assert counts.get("BatchNorm1d", 0) == 0

    def test_numerical_equivalence(self):
        model = LinearBNLinearReLU().eval()
        with torch.no_grad():
            model(torch.randn(32, 8))

        x = torch.randn(4, 8)
        expected = model(x)

        gm = _trace_and_normalize(model, (8,))
        gm.eval()
        actual = gm(x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"Max diff: {(expected - actual).abs().max().item():.6f}"
        )


class TestLinearIdentityBNLinear:
    """Mixed chain: Linear -> Identity -> BN -> Linear fuses to one."""

    def test_fuses(self):
        model = LinearIdentityBNLinear().eval()
        with torch.no_grad():
            model(torch.randn(32, 8))

        gm = _trace_and_normalize(model, (8,))
        counts = _count_modules(gm)
        assert counts.get("Linear", 0) == 1

    def test_numerical_equivalence(self):
        model = LinearIdentityBNLinear().eval()
        with torch.no_grad():
            model(torch.randn(32, 8))

        x = torch.randn(4, 8)
        expected = model(x)
        gm = _trace_and_normalize(model, (8,))
        gm.eval()
        actual = gm(x)
        assert torch.allclose(expected, actual, atol=1e-5)


class TestThreeLinearChain:
    """Linear -> BN -> Linear -> Linear: iterative fusion to single Linear."""

    def test_fuses(self):
        model = ThreeLinearChain().eval()
        with torch.no_grad():
            model(torch.randn(32, 8))

        gm = _trace_and_normalize(model, (8,))
        counts = _count_modules(gm)
        assert counts.get("Linear", 0) == 1

    def test_numerical_equivalence(self):
        model = ThreeLinearChain().eval()
        with torch.no_grad():
            model(torch.randn(32, 8))

        x = torch.randn(4, 8)
        expected = model(x)
        gm = _trace_and_normalize(model, (8,))
        gm.eval()
        actual = gm(x)
        assert torch.allclose(expected, actual, atol=1e-5)


class TestNoFusionCases:
    """Cases where BN/Identity should NOT trigger fusion."""

    def test_linear_bn_relu_no_fusion(self):
        """Linear -> BN -> ReLU: no second Linear, so no MM fusion."""
        model = LinearBNReLU().eval()
        with torch.no_grad():
            model(torch.randn(32, 8))

        gm = _trace_and_normalize(model, (8,))
        counts = _count_modules(gm)
        assert counts.get("Linear", 0) == 1
        assert counts.get("BatchNorm1d", 0) == 1, (
            "BN should remain when there's no second Linear to fuse with"
        )

    def test_branching_bn_no_fusion(self):
        """Linear -> BN -> (fc2a, fc2b): BN fans out, no fusion possible."""
        model = BranchingBN().eval()
        with torch.no_grad():
            model(torch.randn(32, 8))

        gm = _trace_and_normalize(model, (8,))
        counts = _count_modules(gm)
        assert counts.get("Linear", 0) >= 2, (
            "Branching BN should prevent fusion; need at least 2 Linears"
        )


# ===========================================================================
# Part 2: Integration — full conversion pipeline + downstream compatibility
# ===========================================================================

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.pipelining.pipeline_steps.activation_utils import has_non_relu_activations
from mimarsinan.pipelining.pipeline_steps.activation_analysis_step import scale_from_activations
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.models.layers import TransformedActivation, SavedTensorDecorator


# ---------------------------------------------------------------------------
# Integration test models
# ---------------------------------------------------------------------------

class LinearBNLinearReLUClassifier(nn.Module):
    """Linear -> BN -> Linear -> ReLU -> Linear (output).

    After MM+ normalization the first two Linears + BN fuse into one
    Linear, producing: fused_Linear -> ReLU -> output_Linear.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.bn = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.act = nn.ReLU()
        self.out = nn.Linear(16, 10)

    def forward(self, x):
        x = x.flatten(1)
        return self.out(self.act(self.fc2(self.bn(self.fc1(x)))))


class LinearBNLinearGELUClassifier(nn.Module):
    """Linear -> BN -> Linear -> GELU -> Linear (output).

    Same structure but with GELU — chip-targeted (adaptable) but not
    chip-supported until activation adaptation runs.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.bn = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.act = nn.GELU()
        self.out = nn.Linear(16, 10)

    def forward(self, x):
        x = x.flatten(1)
        return self.out(self.act(self.fc2(self.bn(self.fc1(x)))))


class LinearBNLinearNoActClassifier(nn.Module):
    """Linear -> BN -> Linear -> Linear (output, no activation).

    The first two Linears + BN fuse, but there is no activation between
    fused result and output, so the fused perceptron gets Identity.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.bn = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 10)

    def forward(self, x):
        x = x.flatten(1)
        return self.out(self.fc2(self.bn(self.fc1(x))))


def _warmup_and_convert(model_cls, input_shape=(1, 8, 8), num_classes=10):
    """Instantiate, warmup BN, convert through the full pipeline."""
    torch.manual_seed(42)
    model = model_cls()
    model.eval()
    with torch.no_grad():
        model(torch.randn(4, *input_shape))
    supermodel = convert_torch_model(model, input_shape=input_shape, num_classes=num_classes)
    supermodel.eval()
    return model, supermodel


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestBNFoldFusionPerceptronProperties:
    """After BN-fold + fusion, the resulting Perceptrons must have the
    correct activation, normalization, and chip classification so that
    downstream adaptation steps handle them correctly.
    """

    def test_relu_perceptron_properties(self):
        """Linear -> BN -> Linear -> ReLU produces a chip-supported ReLU perceptron
        with Identity normalization (BN was folded during graph normalization).
        """
        _, supermodel = _warmup_and_convert(LinearBNLinearReLUClassifier)
        perceptrons = supermodel.get_perceptrons()

        relu_perceptrons = [
            p for p in perceptrons if p.base_activation_name == "ReLU"
        ]
        assert len(relu_perceptrons) == 1, (
            f"Expected 1 ReLU perceptron from fused chain, got {len(relu_perceptrons)}. "
            f"All: {[(p.name, p.base_activation_name) for p in perceptrons]}"
        )

        fused_p = relu_perceptrons[0]
        assert isinstance(fused_p.normalization, nn.Identity), (
            f"Fused perceptron should have Identity normalization (BN was folded), "
            f"got {type(fused_p.normalization).__name__}"
        )
        assert not isinstance(fused_p.base_activation, nn.Identity), (
            "ReLU perceptron must be a perceptron activation"
        )

    def test_gelu_perceptron_properties(self):
        """Linear -> BN -> Linear -> GELU produces a chip-targeted but not
        chip-supported perceptron. has_non_relu_activations() must return True.
        """
        _, supermodel = _warmup_and_convert(LinearBNLinearGELUClassifier)
        perceptrons = supermodel.get_perceptrons()

        gelu_perceptrons = [
            p for p in perceptrons if p.base_activation_name == "GELU"
        ]
        assert len(gelu_perceptrons) == 1, (
            f"Expected 1 GELU perceptron from fused chain, got {len(gelu_perceptrons)}. "
            f"All: {[(p.name, p.base_activation_name) for p in perceptrons]}"
        )

        fused_p = gelu_perceptrons[0]
        assert isinstance(fused_p.normalization, nn.Identity)
        assert not isinstance(fused_p.base_activation, nn.Identity), (
            "GELU perceptron must be a perceptron activation"
        )

    def test_gelu_has_non_relu_activations(self):
        """has_non_relu_activations() must be True when a GELU perceptron exists."""
        _, supermodel = _warmup_and_convert(LinearBNLinearGELUClassifier)
        assert has_non_relu_activations(supermodel), (
            "GELU perceptron should be detected as non-ReLU"
        )

    def test_relu_has_no_non_relu_activations(self):
        """has_non_relu_activations() must be False when all perceptrons are ReLU."""
        _, supermodel = _warmup_and_convert(LinearBNLinearReLUClassifier)
        assert not has_non_relu_activations(supermodel), (
            "All-ReLU model should not be detected as having non-ReLU activations"
        )

    def test_no_act_fusion_produces_no_perceptron_mappers(self):
        """Linear -> BN -> Linear -> Linear: all three fuse into one.
        No activation detected → no PerceptronMappers, only ModuleComputeMappers.
        """
        from mimarsinan.mapping.mappers.perceptron import ModuleComputeMapper
        _, supermodel = _warmup_and_convert(LinearBNLinearNoActClassifier)
        mapper_repr = supermodel.get_mapper_repr()
        mapper_repr._ensure_exec_graph()

        perceptron_mappers = [
            n for n in mapper_repr._exec_order if isinstance(n, PerceptronMapper)
        ]
        compute_mappers = [
            n for n in mapper_repr._exec_order if isinstance(n, ModuleComputeMapper)
        ]
        assert len(perceptron_mappers) == 0, (
            f"Expected 0 PerceptronMappers (no activation in chain), "
            f"got {len(perceptron_mappers)}"
        )
        assert len(compute_mappers) >= 1, (
            f"Expected at least 1 ModuleComputeMapper, got {len(compute_mappers)}"
        )

    def test_normalization_fusion_step_skips_fused_perceptrons(self):
        """Fused perceptrons have Identity normalization, so
        NormalizationFusionStep has nothing to fold — it should be a no-op.
        """
        _, supermodel = _warmup_and_convert(LinearBNLinearReLUClassifier)
        for p in supermodel.get_perceptrons():
            assert isinstance(p.normalization, nn.Identity), (
                f"Perceptron '{p.name}' has {type(p.normalization).__name__} "
                f"normalization — BN should have been folded by graph normalization"
            )


class TestBNFoldFusionNumericalEquivalence:
    """Full pipeline numerical equivalence: original model vs converted flow
    for models that exercise BN-fold fusion.
    """

    @pytest.mark.parametrize(
        "model_cls",
        [LinearBNLinearReLUClassifier, LinearBNLinearGELUClassifier, LinearBNLinearNoActClassifier],
        ids=["relu", "gelu", "no_act"],
    )
    def test_forward_match(self, model_cls):
        model, supermodel = _warmup_and_convert(model_cls)
        x = torch.randn(8, 1, 8, 8)
        with torch.no_grad():
            orig = model(x)
            converted = supermodel(x)

        diff = (orig - converted).abs().max().item()
        assert diff < 1e-3, (
            f"{model_cls.__name__}: max diff {diff:.6f} exceeds tolerance"
        )

    @pytest.mark.parametrize(
        "model_cls",
        [LinearBNLinearReLUClassifier, LinearBNLinearGELUClassifier, LinearBNLinearNoActClassifier],
        ids=["relu", "gelu", "no_act"],
    )
    def test_argmax_agreement(self, model_cls):
        model, supermodel = _warmup_and_convert(model_cls)
        x = torch.randn(16, 1, 8, 8)
        with torch.no_grad():
            orig_pred = model(x).argmax(dim=1)
            conv_pred = supermodel(x).argmax(dim=1)

        agreement = (orig_pred == conv_pred).float().mean().item()
        assert agreement == 1.0, (
            f"{model_cls.__name__}: {int((1 - agreement) * 16)}/16 predictions disagree"
        )


# ===========================================================================
# Part 3: Adaptation parameter consistency for BN-folded perceptrons
# ===========================================================================


def _init_adaptation_manager(supermodel, config=None):
    """Wrap perceptron activations in TransformedActivation (mirrors pipeline init).

    The AdaptationManager with zero rates creates TransformedActivation wrappers
    whose decorators are all no-ops. This must happen before ActivationAnalysisStep
    can call ``.decorate()`` on the activation.
    """
    if config is None:
        config = {"target_tq": 64, "spiking_mode": "rate"}
    am = AdaptationManager()
    for p in supermodel.get_perceptrons():
        am.update_activation(config, p)
    return am, config


def _compute_activation_scales(supermodel, x):
    """Simulate ActivationAnalysisStep: init AM → decorate → forward → measure scales."""
    _init_adaptation_manager(supermodel)
    supermodel.eval()
    perceptrons = supermodel.get_perceptrons()
    for p in perceptrons:
        p.activation.decorate(SavedTensorDecorator())

    with torch.no_grad():
        supermodel(x)

    scales = []
    for p in perceptrons:
        saved = p.activation.pop_decorator()
        flat = saved.latest_output.view(-1)
        scales.append(scale_from_activations(flat))
    return scales


class TestAdaptationParameterConsistency:
    """Verify that adaptation parameters (activation_scale, effective weights,
    decorator chain) are correctly set for BN-folded perceptrons and that
    the full adaptation chain preserves numerical correctness.
    """

    @pytest.fixture
    def relu_model_and_supermodel(self):
        return _warmup_and_convert(LinearBNLinearReLUClassifier)

    @pytest.fixture
    def gelu_model_and_supermodel(self):
        return _warmup_and_convert(LinearBNLinearGELUClassifier)

    # -- activation_scale --------------------------------------------------

    def test_activation_scales_are_positive_and_finite(self, relu_model_and_supermodel):
        _, supermodel = relu_model_and_supermodel
        x = torch.randn(32, 1, 8, 8)
        scales = _compute_activation_scales(supermodel, x)
        for i, s in enumerate(scales):
            assert s > 0, f"Perceptron {i}: activation_scale must be > 0, got {s}"
            assert torch.isfinite(torch.tensor(s)), f"Perceptron {i}: scale not finite"

    def test_activation_scales_stable_across_batches(self, relu_model_and_supermodel):
        """Two different random batches should yield similar activation_scales."""
        _, supermodel = relu_model_and_supermodel
        torch.manual_seed(100)
        scales1 = _compute_activation_scales(supermodel, torch.randn(64, 1, 8, 8))
        torch.manual_seed(200)
        scales2 = _compute_activation_scales(supermodel, torch.randn(64, 1, 8, 8))
        for i, (s1, s2) in enumerate(zip(scales1, scales2)):
            ratio = max(s1, s2) / max(min(s1, s2), 1e-9)
            assert ratio < 5.0, (
                f"Perceptron {i}: activation_scale unstable: {s1:.4f} vs {s2:.4f}"
            )

    # -- PerceptronTransformer effective weights ----------------------------

    def test_effective_weight_identity_normalization_formula(self, relu_model_and_supermodel):
        """For fused perceptrons (Identity normalization), effective_W = W / activation_scale."""
        _, supermodel = relu_model_and_supermodel
        pt = PerceptronTransformer()
        x = torch.randn(32, 1, 8, 8)
        scales = _compute_activation_scales(supermodel, x)

        for idx, p in enumerate(supermodel.get_perceptrons()):
            p.set_activation_scale(scales[idx])
            assert isinstance(p.normalization, nn.Identity), (
                f"Perceptron {idx}: expected Identity normalization after BN fold"
            )
            eff_W = pt.get_effective_weight(p)
            expected_W = p.layer.weight.data / p.activation_scale
            assert torch.allclose(eff_W, expected_W, atol=1e-6), (
                f"Perceptron {idx}: effective weight formula mismatch. "
                f"Max diff: {(eff_W - expected_W).abs().max().item():.8f}"
            )

    def test_effective_bias_identity_normalization_formula(self, relu_model_and_supermodel):
        """For fused perceptrons (Identity normalization), effective_b = b / activation_scale."""
        _, supermodel = relu_model_and_supermodel
        pt = PerceptronTransformer()
        x = torch.randn(32, 1, 8, 8)
        scales = _compute_activation_scales(supermodel, x)

        for idx, p in enumerate(supermodel.get_perceptrons()):
            p.set_activation_scale(scales[idx])
            eff_b = pt.get_effective_bias(p)
            b = p.layer.bias.data if p.layer.bias is not None else torch.zeros(p.output_channels)
            expected_b = b / p.activation_scale
            assert torch.allclose(eff_b, expected_b, atol=1e-6), (
                f"Perceptron {idx}: effective bias formula mismatch"
            )

    def test_effective_weights_are_finite(self, relu_model_and_supermodel):
        _, supermodel = relu_model_and_supermodel
        pt = PerceptronTransformer()
        x = torch.randn(32, 1, 8, 8)
        scales = _compute_activation_scales(supermodel, x)
        for idx, p in enumerate(supermodel.get_perceptrons()):
            p.set_activation_scale(scales[idx])
            eff_W = pt.get_effective_weight(p)
            eff_b = pt.get_effective_bias(p)
            assert torch.isfinite(eff_W).all(), f"Perceptron {idx}: non-finite effective weight"
            assert torch.isfinite(eff_b).all(), f"Perceptron {idx}: non-finite effective bias"

    # -- AdaptationManager decorator chain ---------------------------------

    def _apply_adaptation(self, supermodel, x, clamp_rate=1.0):
        """Simulate the clamp adaptation path: init AM → compute scales → re-apply with clamp_rate."""
        scales = _compute_activation_scales(supermodel, x)
        am = AdaptationManager()
        am.clamp_rate = clamp_rate
        config = {"target_tq": 64, "spiking_mode": "rate"}
        for idx, p in enumerate(supermodel.get_perceptrons()):
            p.set_activation_scale(scales[idx])
            am.update_activation(config, p)
        return am, scales

    def test_update_activation_produces_transformed_activation(self, relu_model_and_supermodel):
        _, supermodel = relu_model_and_supermodel
        x = torch.randn(32, 1, 8, 8)
        self._apply_adaptation(supermodel, x)
        for p in supermodel.get_perceptrons():
            assert isinstance(p.activation, TransformedActivation), (
                f"Perceptron '{p.name}': expected TransformedActivation after update_activation"
            )

    def test_clamp_decorator_uses_correct_activation_scale(self, relu_model_and_supermodel):
        _, supermodel = relu_model_and_supermodel
        x = torch.randn(32, 1, 8, 8)
        _, scales = self._apply_adaptation(supermodel, x, clamp_rate=1.0)
        for idx, p in enumerate(supermodel.get_perceptrons()):
            assert abs(p.activation_scale.item() - scales[idx]) < 1e-6, (
                f"Perceptron {idx}: activation_scale {p.activation_scale.item():.6f} "
                f"!= measured scale {scales[idx]:.6f}"
            )

    def test_post_adaptation_forward_pass_succeeds(self, relu_model_and_supermodel):
        """Forward pass must work after adaptation manager updates activations."""
        _, supermodel = relu_model_and_supermodel
        x = torch.randn(8, 1, 8, 8)
        self._apply_adaptation(supermodel, x)
        supermodel.eval()
        with torch.no_grad():
            out = supermodel(x)
        assert out.shape == (8, 10)
        assert torch.isfinite(out).all(), "Non-finite output after adaptation"

    def test_post_adaptation_forward_gelu(self, gelu_model_and_supermodel):
        """GELU model: adaptation replaces GELU with ReLU, forward must still work."""
        _, supermodel = gelu_model_and_supermodel
        x = torch.randn(8, 1, 8, 8)
        self._apply_adaptation(supermodel, x)
        supermodel.eval()
        with torch.no_grad():
            out = supermodel(x)
        assert out.shape == (8, 10)
        assert torch.isfinite(out).all()

    # -- NormalizationFusionStep invariant ---------------------------------

    def test_normalization_fusion_is_noop_after_bn_fold(self, relu_model_and_supermodel):
        """Simulating NormalizationFusionStep: for every chip-targeted perceptron,
        normalization is already Identity, so fusion should change nothing.
        """
        _, supermodel = relu_model_and_supermodel
        pt = PerceptronTransformer()
        x = torch.randn(8, 1, 8, 8)
        scales = _compute_activation_scales(supermodel, x)

        weights_before = {}
        for idx, p in enumerate(supermodel.get_perceptrons()):
            p.set_activation_scale(scales[idx])
            weights_before[idx] = (
                p.layer.weight.data.clone(),
                p.layer.bias.data.clone() if p.layer.bias is not None else None,
            )

        # Simulate NormalizationFusionStep logic
        for p in supermodel.get_perceptrons():
            if isinstance(p.normalization, nn.Identity):
                continue
            # If we reach here, BN was NOT folded — that's a problem.
            u, beta, mean = pt._get_u_beta_mean(p.normalization)
            W = p.layer.weight.data
            b = p.layer.bias.data if p.layer.bias is not None else torch.zeros(W.shape[0])
            p.layer.weight.data = W * u.unsqueeze(-1)
            p.layer.bias.data = (b - mean) * u + beta
            p.normalization = nn.Identity()

        for idx, p in enumerate(supermodel.get_perceptrons()):
            w_before, b_before = weights_before[idx]
            assert torch.equal(p.layer.weight.data, w_before), (
                f"Perceptron {idx}: weights changed by NormalizationFusion — "
                "BN should already be folded"
            )
            if b_before is not None:
                assert torch.equal(p.layer.bias.data, b_before), (
                    f"Perceptron {idx}: biases changed by NormalizationFusion"
                )

    # -- Paired equivalence: fused vs unfused reference --------------------

    def test_effective_params_match_manual_bn_fold(self):
        """Build a model where BN is between two Linears. After graph normalization
        fuses them, verify that get_effective_weight/bias matches a manual computation
        of BN-fold + Linear-fusion.
        """
        torch.manual_seed(42)
        model = LinearBNLinearReLUClassifier()
        model.eval()
        with torch.no_grad():
            model(torch.randn(32, 1, 8, 8))

        # Manually compute the expected fused weight/bias
        with torch.no_grad():
            W1 = model.fc1.weight.data.clone()  # (32, 64)
            b1 = model.fc1.bias.data.clone()    # (32,)
            gamma = model.bn.weight.data.clone()
            beta = model.bn.bias.data.clone()
            mean = model.bn.running_mean.clone()
            var = model.bn.running_var.clone()
            eps = model.bn.eps
            W2 = model.fc2.weight.data.clone()  # (16, 32)
            b2 = model.fc2.bias.data.clone()    # (16,)

            # Fold BN into fc1
            scale = gamma / torch.sqrt(var + eps)
            W1_folded = scale.unsqueeze(1) * W1
            b1_folded = scale * (b1 - mean) + beta

            # Fuse fc1_folded + fc2
            W_fused = W2 @ W1_folded   # (16, 64)
            b_fused = W2 @ b1_folded + b2  # (16,)

        # Convert through the pipeline
        _, supermodel = _warmup_and_convert(LinearBNLinearReLUClassifier)
        perceptrons = supermodel.get_perceptrons()
        relu_perceptrons = [p for p in perceptrons if p.base_activation_name == "ReLU"]
        assert len(relu_perceptrons) == 1
        fused_p = relu_perceptrons[0]

        # The fused perceptron's raw weights (before activation_scale division)
        # should match our manual computation
        assert torch.allclose(fused_p.layer.weight.data, W_fused, atol=1e-5), (
            f"Fused weight mismatch. "
            f"Max diff: {(fused_p.layer.weight.data - W_fused).abs().max().item():.8f}"
        )
        assert torch.allclose(fused_p.layer.bias.data, b_fused, atol=1e-5), (
            f"Fused bias mismatch. "
            f"Max diff: {(fused_p.layer.bias.data - b_fused).abs().max().item():.8f}"
        )
