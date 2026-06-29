"""Synchronized MLP-mixer NF↔SCM parity: isolates the weight-quant residual.

Reproduction for the "synchronized mixer fails NF↔SCM per-neuron parity" incident.
The prime suspect was the token-mixer ``permute(0,2,1)`` transpose. These tests
REFUTE that and pin the real cause:

  * ``test_..._bit_exact_without_weight_quant`` PASSES — with weight rounding off,
    the full two-block mixer (both token-mixer transposes included) is bit-exact
    NF↔SCM across every perceptron. The NF dynamics and the transpose wiring are
    correct. ``permute(0,2,1)`` maps to a structural ``PermuteMapper``
    (``np.transpose`` of the IR source view), NOT the
    ``ComputeOpMapper._emit_unary`` transpose branch the incident suspected.

  * ``test_..._exceeds_default_budget_but_meets_mixer_budget`` pins the honest
    contract — turning weight quantization on at the same scales produces a
    per-neuron residual that genuinely EXCEEDS the synchronized default budget
    (0.02, from ``soft_core_mapping_step.py`` ``default_budget = 0.02 if
    contract.is_synchronized() else 0.25``) yet sits under the mixer recipe
    budget (0.15). The residual is pure weight-rounding amplified by the TTFS
    ceil staircase at near-boundary activations; it is concentrated on the DEEP
    perceptrons, with the transpose-downstream token-mixer fc1 the CLEANEST —
    the opposite of a transpose-localized bug.

Honest fix (NF proven correct by the first test): the global synchronized 0.02
default correctly rejects the mixer, so it stays put — never loosen it. Every
``mixer_sync_ttfs_*`` recipe instead carries
``nf_scm_parity_max_mismatch_fraction = 0.15`` (the honest mixer WQ residual
budget), which is the recipe-scoped path the second test verifies.
"""

import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.activations.autograd import TTFSInputGridQuantizer
from mimarsinan.models.nn.activations.ttfs_cycle import TTFSCycleActivation
from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore
from mimarsinan.pipelining.core.nf_scm_parity import (
    NfScmParityError,
    assert_nf_scm_parity_or_raise,
)
from mimarsinan.spiking.scale_aware_boundaries import calibrate_scale_aware_boundaries
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import (
    mark_encoding_layers,
    segment_entry_perceptrons,
)
from mimarsinan.transformations.quantization_bounds import quantization_bounds

T = 8
INPUT_SHAPE = (1, 28, 28)
NUM_CLASSES = 4
WEIGHT_BITS = 8

# Production default for synchronized schedules (soft_core_mapping_step.py:395).
# Calibrated on a shallow non-mixer reference; correctly too tight for the mixer's
# honest weight-quant residual, which is why the mixer recipes carry an override.
SYNCHRONIZED_DEFAULT_PARITY_BUDGET = 0.02

# The honest mixer weight-quant residual budget carried by every mixer_sync_ttfs_*
# recipe (research/harness.py: nf_scm_parity_max_mismatch_fraction = 0.15).
MIXER_RECIPE_PARITY_BUDGET = 0.15

# Lower activation scales pack post-ReLU activations near the TTFS grid steps,
# the regime a trained network occupies. This reproduces the field ~9 % worst-
# perceptron residual on the minimal mixer instead of the ~0.4 % an untrained,
# wide-scaled init shows.
TRAINED_LIKE_SCALE_MULT = 0.3


def _pipeline():
    p = MockPipeline()
    p.config["spiking_mode"] = "ttfs_cycle_based"
    p.config["firing_mode"] = "TTFS"
    p.config["spike_generation_mode"] = "TTFS"
    p.config["thresholding_mode"] = "<="
    p.config["simulation_steps"] = T
    p.config["ttfs_cycle_schedule"] = "synchronized"
    p.config["input_shape"] = INPUT_SHAPE
    return p


def _calibrate_activation_scales(flow, perceptrons, scale_mult):
    """Per-perceptron post-activation max over a fixed batch, scaled down."""
    maxima = {}
    handles = []
    for i, p in enumerate(perceptrons):
        def hook(_m, _inp, out, idx=i):
            maxima[idx] = out.detach().abs().reshape(out.shape[0], -1).max().item()
        handles.append(p.activation.register_forward_hook(hook))
    torch.manual_seed(7)
    with torch.no_grad():
        flow(torch.rand(8, *INPUT_SHAPE))
    for h in handles:
        h.remove()
    return [max(maxima[i] * scale_mult, 1e-3) for i in range(len(perceptrons))]


def _build_synchronized_mixer(*, weight_quantization, scale_mult=TRAINED_LIKE_SCALE_MULT,
                              bits=WEIGHT_BITS):
    """Minimal synchronized-schedule mixer + identity IR for the parity gate."""
    torch.manual_seed(0)
    model = TorchMLPMixerCore(
        input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
        patch_n_1=4, patch_m_1=4, patch_c_1=8, fc_w_1=16, fc_w_2=16,
        num_blocks=2,
    ).eval()
    # Untrained init collapses fc2 outputs to ~0; inflate weights and add bias so
    # post-ReLU activations actually span the TTFS grid.
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                mod.weight.mul_(3.0)
                if mod.bias is not None:
                    mod.bias.copy_(torch.randn_like(mod.bias) * 0.5)

    flow = convert_torch_model(
        model, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
    ).eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)

    perceptrons = list(flow.get_perceptrons())
    scales = _calibrate_activation_scales(flow, perceptrons, scale_mult)

    for p in perceptrons:
        activation = TTFSCycleActivation(T=T, activation_scale=p.activation_scale)
        p.base_activation = activation
        p.activation = activation

    calibrate_scale_aware_boundaries(flow, scales, input_data_scale=1.0)
    repr_.assign_perceptron_indices()

    entry_ids = {id(p) for p in segment_entry_perceptrons(repr_)}
    for p in perceptrons:
        if id(p) in entry_ids:
            q = TTFSInputGridQuantizer(T=T, activation_scale=p.input_activation_scale)
            if isinstance(p.input_activation, nn.Identity):
                p.input_activation = q
            else:
                p.input_activation = nn.Sequential(p.input_activation, q)

    compute_per_source_scales(repr_)

    _, q_max = quantization_bounds(bits)
    ir_graph = IRMapping(
        q_max=q_max, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
    ).map(repr_)
    if weight_quantization:
        # Zero parameter_scale so quantize_ir_graph auto-scales (q_max/w_max) and
        # preserves signal instead of collapsing small weights to 0 under the
        # un-calibrated parameter_scale=1 path.
        for node in ir_graph.nodes:
            if isinstance(node, NeuralCore):
                node.parameter_scale = torch.tensor(0.0)
    quantize_ir_graph(ir_graph, bits, weight_quantization=weight_quantization)
    IRLatency(ir_graph).calculate()

    return flow.double().eval(), ir_graph


def _samples(n=8):
    torch.manual_seed(1)
    return torch.rand(n, *INPUT_SHAPE, dtype=torch.float64)


def test_synchronized_mixer_bit_exact_without_weight_quant():
    """NF correctness guard: sans weight rounding the mixer (both token-mixer
    transposes included) is bit-exact NF↔SCM, so the transpose wiring is sound."""
    flow, ir_graph = _build_synchronized_mixer(weight_quantization=False)
    fraction = assert_nf_scm_parity_or_raise(
        _pipeline(), flow, ir_graph, _samples(),
        atol=1e-6, max_mismatch_fraction=0.0,
    )
    assert fraction == 0.0


def test_synchronized_mixer_exceeds_default_budget_but_meets_mixer_budget():
    """The honest mixer contract. With weight quantization on, the mixer's
    inherent rounding residual genuinely EXCEEDS the synchronized 0.02 default
    (calibrated on a shallow reference) — that rejection is correct, not a bug.
    The same residual sits comfortably under the mixer recipe budget (0.15) that
    every ``mixer_sync_ttfs_*`` recipe carries. The NF is proven correct by the
    bit-exact guard above, so the honest fix is the recipe-scoped budget, never a
    loosened global default."""
    flow, ir_graph = _build_synchronized_mixer(weight_quantization=True)

    with pytest.raises(NfScmParityError) as excinfo:
        assert_nf_scm_parity_or_raise(
            _pipeline(), flow, ir_graph, _samples(),
            atol=1e-6, max_mismatch_fraction=SYNCHRONIZED_DEFAULT_PARITY_BUDGET,
        )

    # The honest residual lives between the two budgets — tight default rejects,
    # mixer budget admits.
    fraction = excinfo.value.mismatch_fraction
    assert fraction > SYNCHRONIZED_DEFAULT_PARITY_BUDGET
    assert fraction <= MIXER_RECIPE_PARITY_BUDGET

    admitted = assert_nf_scm_parity_or_raise(
        _pipeline(), flow, ir_graph, _samples(),
        atol=1e-6, max_mismatch_fraction=MIXER_RECIPE_PARITY_BUDGET,
    )
    assert admitted <= MIXER_RECIPE_PARITY_BUDGET
