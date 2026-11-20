"""[U-PCH2] The per-neuron capability's consumers and projection details, each pinned by one test.

- Every consumer that holds one threshold register per core refuses a per-neuron core by name, the refusal naming the
  consumer and the per-neuron thresholds: the ODIN export (its emission bound included), the ODIN variant generator,
  the ODIN pass and variant pass builds, the ODIN RTL reference, SANA-FE's TTFS soma synthesis, the event-serial guard
  and the event-serial fold.
- IR quantization refuses a trained per-neuron scale off the integer lattice, as it refuses a per-perceptron one.
- The GUI's IR graph snapshot reads every register of a per-neuron program (one reader of a reported threshold).
- The projection's per-neuron formulas, each by equality: the shared grid's ``s_c = max(1, floor(q_max / max(max_j
  |w_cj|, |b_c| / d)))``, the bias rows the largest neuron needs, and the off-floor predicate's coarsest half quantum.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.support.neuron_thresholds import PerNeuronThresholdRefusal
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.lif_kernels import NeuronLattice
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization as NAPQ,
)
from mimarsinan.transformations.perceptron.bias_saturation import clip_off_saturated_effective_bias

from unit.mapping.residency.executed import vehicles as V
from unit.mapping.test_odin_exporter import ODIN_LAW, _export, _two_core_mapping
from unit.mapping.test_per_neuron_weight_scales import BITS, PT, Q_MAX, _uneven_perceptron


def _per_neuron_two_core_mapping():
    """The ODIN exporter tests' two-core mapping with one threshold register per neuron on both cores."""
    mapping = _two_core_mapping()
    mapping.cores[0].threshold = np.array([4.0, 5.0])
    mapping.cores[1].threshold = np.array([6.0, 7.0])
    return mapping


# ---------------------------------------------------------------------------
# One register per core: refused by name
# ---------------------------------------------------------------------------


def test_the_odin_export_refuses_a_per_neuron_core_by_name():
    with pytest.raises(PerNeuronThresholdRefusal, match="the ODIN export"):
        _export(_per_neuron_two_core_mapping())


def test_the_odin_emission_bound_refuses_a_per_neuron_core_by_name():
    from mimarsinan.mapping.export.odin.feasibility import propagate_emission_bounds

    with pytest.raises(PerNeuronThresholdRefusal, match="emission bound"):
        propagate_emission_bounds(_per_neuron_two_core_mapping(), ceiling=255)


def test_the_odin_variant_generator_refuses_a_per_neuron_core_by_name():
    from mimarsinan.mapping.export.odin_gen.feasibility import gate_variant_segment

    from unit.mapping.test_odin_gen_packer import _spec

    with pytest.raises(PerNeuronThresholdRefusal, match="the ODIN variant generator"):
        gate_variant_segment(_per_neuron_two_core_mapping(), spec=_spec(), membrane_init=0, cycles=4)


def test_the_odin_pass_build_refuses_a_per_neuron_core_by_name():
    from mimarsinan.chip_simulation.odin_hacc.pass_build import pass_mapping

    with pytest.raises(PerNeuronThresholdRefusal, match="the ODIN pass build"):
        pass_mapping(_per_neuron_two_core_mapping().cores[0])


def test_the_odin_variant_pass_build_refuses_a_per_neuron_core_by_name():
    from mimarsinan.chip_simulation.odin_hacc.variant_pass_build import VariantPassBuild

    mapping = _per_neuron_two_core_mapping()
    with pytest.raises(PerNeuronThresholdRefusal, match="the ODIN variant pass build"):
        VariantPassBuild(mapping, 0, SimpleNamespace(latencies=[0, 1]), spec=None, membrane_init=0)


def test_the_odin_rtl_reference_refuses_a_per_neuron_core_by_name():
    from mimarsinan.chip_simulation.odin_rtl.reference import simulate_cycles

    with pytest.raises(PerNeuronThresholdRefusal, match="the ODIN RTL reference"):
        simulate_cycles(_per_neuron_two_core_mapping(), soma_law=ODIN_LAW, input_counts=[[0, 0]], simulation_length=2)


@pytest.mark.parametrize("mode", ["ttfs", "ttfs_quantized"])
def test_sanafes_ttfs_soma_synthesis_refuses_a_per_neuron_core_by_name(mode):
    from mimarsinan.chip_simulation.sanafe.net_synth.spike_trains import apply_ttfs_preset_membranes

    mapping = _per_neuron_two_core_mapping()
    with pytest.raises(PerNeuronThresholdRefusal, match="SANA-FE"):
        apply_ttfs_preset_membranes({0: [object(), object()]}, mapping, [torch.zeros(1, 2), torch.zeros(1, 2)],
                                    spiking_mode=mode, simulation_length=4)


def test_the_event_serial_guard_refuses_a_per_neuron_core_by_name():
    from mimarsinan.models.spiking.serial.guards import require_serial_deployment_admissible

    stage = SimpleNamespace(name="s", retimed_level_stages=None, hard_core_mapping=_per_neuron_two_core_mapping())
    with pytest.raises(PerNeuronThresholdRefusal, match="the event-serial fold"):
        require_serial_deployment_admissible(SimpleNamespace(stages=[stage]), ODIN_LAW, membrane_init=0.5)


def test_the_event_serial_fold_refuses_a_per_neuron_lattice_by_name():
    from mimarsinan.models.nn.activations.lif_serial import serial_fold_theta

    lif = LIFActivation(T=4, activation_scale=1.0)
    lif.set_membrane_lattice(NeuronLattice(torch.tensor([3.0, 5.0]), -1))
    with pytest.raises(PerNeuronThresholdRefusal, match="the event-serial fold"):
        serial_fold_theta(lif)


def test_ir_quantization_refuses_a_trained_per_neuron_scale_off_the_integer_lattice():
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=V.MAX_AXONS, max_neurons=V.MAX_NEURONS).map(
        V.random_rep(0))
    for owner in list(ir.get_neural_cores()) + list(ir.weight_banks.values()):
        width = max(int(n.perceptron_output_slice[1]) for n in ir.get_neural_cores()
                    if n.perceptron_index == owner.perceptron_index)
        owner.parameter_scale = torch.tensor([40.5 + c for c in range(width)])
    with pytest.raises(ValueError, match="per-neuron parameter_scale is not on the integer lattice"):
        quantize_ir_graph(ir, 8, weight_quantization=True)


def test_every_arming_reads_a_scales_lattice_through_one_rule():
    from mimarsinan.models.nn.lif_kernels import integral_lattice

    assert integral_lattice(torch.tensor(7.0)) == 7 and integral_lattice(7.0000001) == 7
    assert integral_lattice(torch.tensor(7.5)) is None and integral_lattice(torch.tensor(0.0)) is None
    lattice = integral_lattice(torch.tensor([3.0, 5.0]), 1)
    assert isinstance(lattice, NeuronLattice) and lattice.axis == 1 and lattice.scales.tolist() == [3.0, 5.0]
    assert integral_lattice(torch.tensor([3.0, 5.5])) is None


# ---------------------------------------------------------------------------
# The GUI's IR graph snapshot: every register
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("per_neuron", [False, True])
def test_the_gui_ir_graph_snapshot_reads_every_register(per_neuron):
    from mimarsinan.gui.snapshot.ir_graph import snapshot_ir_graph

    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=V.MAX_AXONS, max_neurons=3).map(V.random_rep(0))
    for owner in list(ir.get_neural_cores()) + list(ir.weight_banks.values()):
        width = max(int(n.perceptron_output_slice[1]) for n in ir.get_neural_cores()
                    if n.perceptron_index == owner.perceptron_index)
        owner.parameter_scale = (torch.tensor([float(40 + 7 * (c % 5)) for c in range(width)]) if per_neuron
                                 else torch.tensor(0.0))
    quantize_ir_graph(ir, 8, weight_quantization=True)
    registers = [float(t) for n in ir.get_neural_cores() for t in np.atleast_1d(np.asarray(n.threshold))]
    assert (len(registers) > len(ir.get_neural_cores())) == per_neuron
    summary, _ = snapshot_ir_graph(ir)
    assert sum(summary["threshold_distribution"]["counts"]) == len(registers)
    ranges = [g["threshold_range"] for g in summary["groups"] if g.get("threshold_range") is not None]
    assert ranges and min(r[0] for r in ranges) == min(registers) and max(r[1] for r in ranges) == max(registers)
    assert all(isinstance(v, float) for r in ranges for v in r)


# ---------------------------------------------------------------------------
# The projection's per-neuron formulas, by equality
# ---------------------------------------------------------------------------


def _effective(p):
    clip_off_saturated_effective_bias(p, transformer=PT)  # idempotent: the projection's own first move
    return PT.get_effective_weight(p).detach().clone(), PT.get_effective_bias(p).detach().clone()


def test_the_shared_grid_sizes_each_neuron_from_its_weight_and_bias_peaks():
    p = _uneven_perceptron()
    w, b = _effective(p)
    peak = torch.maximum(w.abs().amax(dim=1), b.abs())
    want = torch.clamp(torch.floor(Q_MAX / peak), min=1.0)
    NAPQ(BITS, "cpu", rate=1.0, two_scale=False, per_neuron=True).transform(p)
    assert torch.equal(p.parameter_scale.to(want.dtype), want)


def test_the_bias_rows_are_as_many_as_the_largest_neuron_needs():
    from mimarsinan.mapping.support.bias_rows import bias_row_demand

    p = _uneven_perceptron()
    with torch.no_grad():
        p.layer.bias.mul_(8.0)
    w, b = _effective(p)
    scale = torch.clamp(torch.floor(Q_MAX / w.abs().amax(dim=1)), min=1.0)
    demand = torch.clamp(torch.ceil(bias_row_demand(b.abs(), scale, Q_MAX)), min=1.0)
    assert float(demand.min()) < float(demand.max()), "the fixture's neurons need different bias rows"
    NAPQ(BITS, "cpu", rate=1.0, two_scale=True, register_bias=False, per_neuron=True).transform(p)
    rows = (p.parameter_scale / p.bias_scale).to(torch.float64)
    assert torch.allclose(rows, torch.full_like(rows, float(demand.max())), rtol=0, atol=1e-9)


def test_the_off_floor_predicate_reads_the_coarsest_neurons_half_quantum():
    from mimarsinan.models.nn.off_floor import has_off_floor

    lif = LIFActivation(T=4, activation_scale=1.0, membrane_init=0.8)
    lif.set_membrane_lattice(16.0)  # one lattice: the half quantum 1/64 keeps 0.8 below the threshold
    assert has_off_floor(lif)
    lif.set_membrane_lattice(NeuronLattice(torch.tensor([1.0, 8.0]), -1))  # the coarsest: 0.8 + 1/4 reaches it
    assert not has_off_floor(lif)
