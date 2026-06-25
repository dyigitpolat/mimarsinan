"""D4 demo lock: structured pruning → fewer cores → fewer phases → lower cost band.

This is the executable companion to
``docs/research/findings/D4_pruning_scheduling_cost.md``: it MEASURES the
dense-vs-pruned deployment cost of a real small perceptron-flow model through the
production instruments and asserts the reduction the doc reports — so the doc's
numbers are reproduced by a real run, never hand-asserted.

Instrument chain (all consumed, none edited here):
  IRMapping.map → estimate_cores_needed (cores, scheduled phase_count)
              → weight_reuse_plan_from_graph (reprogram phases, params reloaded)
              → phase_cost_band (the cited low/nominal/high mJ band).
"""

from __future__ import annotations

import torch
import torch.nn as nn

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
from mimarsinan.transformations.pruning.magnitude import prune_perceptron_chain


# A real perceptron-flow model in its post-normalization-fusion (Identity-norm)
# SCM-time state, with wide intermediate widths so the diagonal core bound — not a
# fixed floor — governs the mapped core count.
class _DemoMLP(PerceptronFlow):
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
_SAMPLE_SPARSITY = 0.5

# Single-pool budget for the static core count.
_PLATFORM = {
    "cores": [{"max_axons": 256, "max_neurons": 64, "count": 4096, "has_bias": True}],
    "allow_coalescing": True,
}
# Tight scheduled budget: oversized segments need multiple reprogram passes.
_SCHEDULED = {
    "cores": [{"max_axons": 256, "max_neurons": 64, "count": 8, "has_bias": True}],
    "allow_coalescing": True,
    "allow_scheduling": True,
}


def _build(seed=1):
    torch.manual_seed(seed)
    m = _DemoMLP(_INPUT_SHAPE, _WIDTHS)
    m.eval()
    with torch.no_grad():
        m(torch.randn(2, *_INPUT_SHAPE))
    return m


def _ir(model):
    mr = model.get_mapper_repr()
    if hasattr(mr, "assign_perceptron_indices"):
        mr.assign_perceptron_indices()
    return IRMapping(
        q_max=127, firing_mode="Default",
        max_axons=256, max_neurons=64,
        allow_coalescing=True, hardware_bias=True,
    ).map(mr)


def _measure(model):
    g = _ir(model)
    cores = estimate_cores_needed(g, _PLATFORM).cores_needed
    phases = estimate_cores_needed(g, _SCHEDULED).phase_count
    plan = weight_reuse_plan_from_graph(g)
    band = phase_cost_band(
        reprogram_passes=plan.reprogram_passes,
        reuse_passes=plan.reuse_passes,
        params_reloaded=plan.params_reloaded,
        activation_bytes_moved=0,
    )
    return {
        "cores": cores,
        "phases": phases,
        "reprogram_passes": plan.reprogram_passes,
        "params_reloaded": plan.params_reloaded,
        "cost_nominal_mj": band.nominal_mj,
        "cost_low_mj": band.low_mj,
        "cost_high_mj": band.high_mj,
    }


def test_demo_measured_reduction_matches_doc():
    dense = _measure(_build())
    pruned_model = _build()
    prune_perceptron_chain(pruned_model.get_perceptrons(), _SAMPLE_SPARSITY)
    pruned = _measure(pruned_model)

    # The exact MEASURED integer numbers the findings doc reports (lock vs drift).
    assert dense["cores"] == 13
    assert dense["phases"] == 2
    assert dense["reprogram_passes"] == 13
    assert dense["params_reloaded"] == 199168
    assert pruned["cores"] == 7
    assert pruned["phases"] == 1
    assert pruned["reprogram_passes"] == 7
    assert pruned["params_reloaded"] == 66816

    # Every D4 lever strictly drops; assert the reduction, not just the points.
    assert pruned["cores"] < dense["cores"]
    assert pruned["phases"] < dense["phases"]
    assert pruned["reprogram_passes"] < dense["reprogram_passes"]
    assert pruned["params_reloaded"] < dense["params_reloaded"]
    assert pruned["cost_nominal_mj"] < dense["cost_nominal_mj"]
    assert pruned["cost_low_mj"] < dense["cost_low_mj"]
    assert pruned["cost_high_mj"] < dense["cost_high_mj"]

    # Headline nominal savings factor reported in the doc (~2.5x).
    savings = dense["cost_nominal_mj"] / pruned["cost_nominal_mj"]
    assert 2.4 < savings < 2.7
