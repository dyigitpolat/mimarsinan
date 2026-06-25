"""D4 deployment wiring lock: ``prune_sparsity`` structurally prunes BEFORE mapping.

Contract verified here (tests-first), through the REAL instruments the demo cites
(``IRMapping`` → ``estimate_cores_needed`` → ``weight_reuse_plan_from_graph`` →
``phase_cost_band``), never hardcoded:

1. PLAN WIRING — ``DeploymentPlan.resolve`` reads ``prune_sparsity`` (default 0.0).
2. PRUNED DEPLOYMENT — with ``prune_sparsity > 0`` the ``SoftCoreMappingStep`` hook
   shrinks every perceptron's output channels, so the model maps to STRICTLY FEWER
   hard cores, fewer reprogram phases, and a lower weight-reuse cost band.
3. DEFAULT-OFF BYTE-IDENTICAL — unset / 0.0 ⇒ the hook is a no-op: the SAME
   ``nn.Linear`` objects with bit-exact tensors, and an identical mapped core count.

The fixture mirrors the SCM-time model state: perceptrons whose normalization is
already fused to ``Identity`` (the post-``NormalizationFusionStep`` state the hook
runs against), so pruning the linear's output rows is structurally sound.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from conftest import MockPipeline

from mimarsinan.mapping.mapping_utils import (
    EinopsRearrangeMapper,
    Ensure2DMapper,
    InputMapper,
    ModelRepresentation,
    ModuleMapper,
    PerceptronMapper,
)
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.verification.capacity import estimate_cores_needed
from mimarsinan.mapping.weight_reuse import weight_reuse_plan_from_graph
from mimarsinan.chip_simulation.weight_reuse_cost_model import phase_cost_band
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
    SoftCoreMappingStep,
)
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_structured_pruning import (
    apply_structured_pruning_if_enabled,
)


# --------------------------------------------------------------------------- #
# Realistic post-fusion fixture: Identity-norm perceptron MLP (a real model).  #
# --------------------------------------------------------------------------- #
class _FusedMLP(PerceptronFlow):
    """Perceptron-flow MLP with normalization already fused to Identity.

    Wide intermediate widths so the diagonal (axon/neuron-sum) bound — not a
    fixed floor — governs the mapped core count, so structured pruning visibly
    drops it.
    """

    def __init__(self, input_shape, widths):
        super().__init__("cpu")
        self.input_activation = nn.Identity()
        self.input_shape = input_shape
        self.perceptrons = nn.ModuleList(
            Perceptron(
                output_channels=widths[i + 1],
                input_features=widths[i],
                normalization=nn.Identity(),
            )
            for i in range(len(widths) - 1)
        )
        inp = InputMapper(input_shape)
        self._iam = ModuleMapper(inp, self.input_activation)
        out = EinopsRearrangeMapper(self._iam, "... c h w -> ... (c h w)")
        out = Ensure2DMapper(out)
        for p in self.perceptrons:
            out = PerceptronMapper(out, p)
        self._mapper_repr = ModelRepresentation(out)

    def get_perceptrons(self):
        return self._mapper_repr.get_perceptrons()

    def get_mapper_repr(self):
        return self._mapper_repr

    def get_input_activation(self):
        return self.input_activation

    def set_input_activation(self, activation):
        self.input_activation = activation
        self._iam.module = activation

    def forward(self, x):
        return self._mapper_repr(x)


_INPUT_SHAPE = (1, 16, 16)
_WIDTHS = [256, 256, 256, 256, 10]
_PLATFORM = {
    "cores": [{"max_axons": 256, "max_neurons": 64, "count": 4096, "has_bias": True}],
    "allow_coalescing": True,
}
# A tight SCHEDULED budget so oversized segments need multiple reprogram passes;
# structured pruning then provably cuts the phase_count.
_SCHEDULED = {
    "cores": [{"max_axons": 256, "max_neurons": 64, "count": 8, "has_bias": True}],
    "allow_coalescing": True,
    "allow_scheduling": True,
}


def _build_model(seed):
    torch.manual_seed(seed)
    model = _FusedMLP(_INPUT_SHAPE, _WIDTHS)
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, *_INPUT_SHAPE))
    return model


def _ir_of(model):
    mapper_repr = model.get_mapper_repr()
    if hasattr(mapper_repr, "assign_perceptron_indices"):
        mapper_repr.assign_perceptron_indices()
    ir_mapping = IRMapping(
        q_max=127,
        firing_mode="Default",
        max_axons=256,
        max_neurons=64,
        allow_coalescing=True,
        hardware_bias=True,
    )
    return ir_mapping.map(mapper_repr)


def _step(prune_sparsity=None):
    cfg = {}
    if prune_sparsity is not None:
        cfg["prune_sparsity"] = prune_sparsity
    return SoftCoreMappingStep(MockPipeline(config=cfg))


# --------------------------------------------------------------------------- #
# 1. Plan wiring                                                              #
# --------------------------------------------------------------------------- #
class TestPlanWiring:
    def test_default_prune_sparsity_is_zero(self):
        assert DeploymentPlan.resolve({}).prune_sparsity == 0.0

    def test_prune_sparsity_resolved_from_config(self):
        assert DeploymentPlan.resolve({"prune_sparsity": 0.5}).prune_sparsity == 0.5

    def test_prune_sparsity_none_coerces_to_zero(self):
        assert DeploymentPlan.resolve({"prune_sparsity": None}).prune_sparsity == 0.0


# --------------------------------------------------------------------------- #
# 2. Pruned deployment: real measured reduction                              #
# --------------------------------------------------------------------------- #
class TestPrunedDeploymentReducesCost:
    def test_hook_prunes_and_reduces_cores_phases_cost(self):
        dense = _build_model(seed=1)
        dense_cores = estimate_cores_needed(_ir_of(dense), _PLATFORM).cores_needed
        dense_plan = weight_reuse_plan_from_graph(_ir_of(dense))
        dense_band = phase_cost_band(
            reprogram_passes=dense_plan.reprogram_passes,
            reuse_passes=dense_plan.reuse_passes,
            params_reloaded=dense_plan.params_reloaded,
            activation_bytes_moved=0,
        )
        dense_phases = estimate_cores_needed(_ir_of(dense), _SCHEDULED).phase_count

        pruned = _build_model(seed=1)
        result = apply_structured_pruning_if_enabled(
            _step(prune_sparsity=0.5), pruned, "SoftCoreMappingStep"
        )
        assert result is not None and result.pruned is True

        pruned_g = _ir_of(pruned)
        pruned_cores = estimate_cores_needed(pruned_g, _PLATFORM).cores_needed
        pruned_plan = weight_reuse_plan_from_graph(pruned_g)
        pruned_band = phase_cost_band(
            reprogram_passes=pruned_plan.reprogram_passes,
            reuse_passes=pruned_plan.reuse_passes,
            params_reloaded=pruned_plan.params_reloaded,
            activation_bytes_moved=0,
        )
        pruned_phases = estimate_cores_needed(pruned_g, _SCHEDULED).phase_count

        # Real instruments, not hardcoded: every D4 lever drops under pruning.
        assert pruned_cores < dense_cores
        assert pruned_plan.reprogram_passes < dense_plan.reprogram_passes
        assert pruned_plan.params_reloaded < dense_plan.params_reloaded
        assert pruned_phases < dense_phases
        assert pruned_band.nominal_mj < dense_band.nominal_mj
        assert pruned_band.low_mj < dense_band.low_mj
        assert pruned_band.high_mj < dense_band.high_mj

    def test_hook_shrinks_output_channels_in_place(self):
        model = _build_model(seed=2)
        layer_objs_before = [p.layer for p in model.get_perceptrons()]
        apply_structured_pruning_if_enabled(
            _step(prune_sparsity=0.5), model, "SoftCoreMappingStep"
        )
        ps = model.get_perceptrons()
        # intermediate output channels halved (floor), logits exempt
        assert ps[0].layer.out_features == 128
        assert ps[1].layer.out_features == 128
        assert ps[2].layer.out_features == 128
        assert ps[-1].layer.out_features == 10
        # downstream in_features track the upstream pruned outputs
        assert ps[1].layer.in_features == ps[0].layer.out_features
        assert ps[2].layer.in_features == ps[1].layer.out_features
        assert ps[3].layer.in_features == ps[2].layer.out_features
        # the in-place mutation replaced the shrunk layers (not the perceptron objs)
        assert layer_objs_before[0] is not ps[0].layer


# --------------------------------------------------------------------------- #
# 3. Default-off byte-identical                                              #
# --------------------------------------------------------------------------- #
class TestDefaultOffByteIdentical:
    def test_unset_is_noop_same_objects(self):
        model = _build_model(seed=3)
        layers_before = [p.layer for p in model.get_perceptrons()]
        weights_before = [l.weight.clone() for l in layers_before]
        biases_before = [l.bias.clone() for l in layers_before]

        result = apply_structured_pruning_if_enabled(
            _step(prune_sparsity=None), model, "SoftCoreMappingStep"
        )

        assert result is None
        layers_after = [p.layer for p in model.get_perceptrons()]
        for before, after, w0, b0 in zip(
            layers_before, layers_after, weights_before, biases_before
        ):
            assert before is after
            assert torch.equal(after.weight, w0)
            assert torch.equal(after.bias, b0)

    def test_zero_is_noop(self):
        model = _build_model(seed=4)
        layers_before = [p.layer for p in model.get_perceptrons()]
        result = apply_structured_pruning_if_enabled(
            _step(prune_sparsity=0.0), model, "SoftCoreMappingStep"
        )
        assert result is None
        assert [p.layer for p in model.get_perceptrons()] == layers_before

    def test_default_off_maps_to_identical_core_count(self):
        a = _build_model(seed=5)
        b = _build_model(seed=5)
        apply_structured_pruning_if_enabled(
            _step(prune_sparsity=None), b, "SoftCoreMappingStep"
        )
        assert (
            estimate_cores_needed(_ir_of(a), _PLATFORM).cores_needed
            == estimate_cores_needed(_ir_of(b), _PLATFORM).cores_needed
        )
