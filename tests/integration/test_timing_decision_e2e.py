"""A converter-built timing program decides count for count on nevresim, on SANA-FE and in the Python executor.

A small torch network is converted by the toolchain's converter, mapped to the IR, weight-quantized on its trained
lattice (weights and biases on the grid 1/c, parameter scale c, so the codes are small integers and the threshold is
c), given its latencies and packed; its neural segment (two cores, one reading the other) is run at S 6 and at S 11 on
segment inputs on the window's grid, so every neuron is on the integer chip's lattice and ties are frequent:

- the Python executor (``TtfsAnalyticalExecutor``) is the reference;
- nevresim runs the segment as a quantized timing program (``NevresimDriver``);
- SANA-FE runs it with the quantized timing soma (``ttfs_quantized``) and with the per-cycle timing soma (the
  synchronized ``ttfs_cycle_based`` schedule), each neuron's fire step read from its soma.

Every neuron's fire step must equal the executor's, sample for sample.
"""

from __future__ import annotations

import tempfile

from spiking_points import point_of_mode

import numpy as np
import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.slow, pytest.mark.integration]

INPUTS = 12
SAMPLES = 24
SCALE = 2  # the lattice: weights and biases on 1/SCALE, so the integer threshold is SCALE
CODES = 3


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUTS, INPUTS)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(INPUTS, 10)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 6)
        self.act3 = nn.ReLU()

    def forward(self, x):
        return self.act3(self.fc3(self.act2(self.fc2(self.act1(self.fc1(x))))))


def _on_lattice(net: nn.Module, seed: int) -> nn.Module:
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for layer in (net.fc1, net.fc2, net.fc3):
            layer.weight.copy_(torch.as_tensor(rng.integers(-CODES, CODES + 1, size=tuple(layer.weight.shape)) / SCALE))
            layer.bias.copy_(torch.as_tensor(rng.integers(-CODES, CODES + 1, size=tuple(layer.bias.shape)) / SCALE))
    return net


def _segment(s: int):
    """The converter-built, weight-quantized neural segment at window ``s``: ``(hybrid, its neural stage)``."""
    from mimarsinan.chip_simulation.ttfs.segment_arrays import segment_ttfs_arrays_from_mapping
    from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
    from mimarsinan.mapping.ir import NeuralCore
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.latency.ir import IRLatency
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
    from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
    from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

    torch.manual_seed(5)
    flow = convert_torch_model(_on_lattice(_Net(), 300 + s), (INPUTS,), 6, device="cpu").eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    propagate_boundary_input_scales(flow, input_data_scale=1.0)
    for p in flow.get_perceptrons():
        p.set_activation(TTFSActivation(T=s, activation_scale=p.activation_scale, input_scale=p.input_activation_scale,
                                        bias=p.layer.bias, thresholding_mode="<=",
                                        encoding=getattr(p, "is_encoding_layer", False)))
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)
    ir = IRMapping(q_max=CODES, firing_mode="TTFS", max_axons=64, max_neurons=64).map(repr_)
    for node in ir.nodes:
        if isinstance(node, NeuralCore):
            node.parameter_scale = torch.tensor(float(SCALE))  # the trained lattice, as Weight Quantization leaves it
    quantize_ir_graph(ir, 3, weight_quantization=True)
    IRLatency(ir).calculate()
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 64, "max_neurons": 64, "count": 64}],
        strategy=MappingStrategy.from_permissions(allow_neuron_splitting=False, allow_coalescing=False),
    )
    neural = [st for st in hybrid.stages if st.kind == "neural"]
    assert len(neural) == 1, "the program is one neural segment"
    seg = neural[0].hard_core_mapping
    assert len(seg.cores) >= 2 and len({int(c.latency) for c in seg.cores}) >= 2, "a core reads another core"
    assert segment_ttfs_arrays_from_mapping(seg).integer_lattice(), "every neuron is integral (the class it reads)"
    assert {float(np.asarray(c.threshold).reshape(-1)[0]) for c in seg.cores} == {float(SCALE)}
    return hybrid, neural[0]


def _segment_alone(hybrid, stage):
    """The neural stage as a one-stage program reading its segment input from the raw input buffer."""
    from mimarsinan.mapping.packing.hybrid_types import HybridHardCoreMapping, HybridStage, SegmentIOSlice

    width = sum(int(sl.size) for sl in stage.input_map)
    alone = HybridStage(kind="neural", name=stage.name, hard_core_mapping=stage.hard_core_mapping,
                        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=width)],
                        output_map=list(stage.output_map))
    return HybridHardCoreMapping(stages=[alone], output_sources=hybrid.output_sources), width


def _grid_inputs(s: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + s)
    return rng.integers(0, s + 1, size=(SAMPLES, width)).astype(np.float64) / s


def _fire_steps(levels: np.ndarray, s: int) -> np.ndarray:
    return np.where(levels > 0.0, s - np.rint(levels * s), -1).astype(np.int64)


def _executor(seg, x: np.ndarray, s: int, mode: str):
    """The Python executor on the segment: (per-neuron fire steps of every used neuron in core order, outputs)."""
    from mimarsinan.chip_simulation.ttfs.ttfs_executor import TtfsAnalyticalExecutor
    from mimarsinan.mapping.support.core_geometry import used_neurons

    result = TtfsAnalyticalExecutor().run_segment(seg, x, simulation_length=s, spiking_mode=mode)
    per_neuron = np.concatenate([np.asarray(a, dtype=np.float64)[:, :int(used_neurons(c))]
                                 for a, c in zip(result.per_core_activations, seg.cores) if int(used_neurons(c)) > 0],
                                axis=1)
    return _fire_steps(per_neuron, s), _fire_steps(np.asarray(result.inter_stage, dtype=np.float64), s)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("s", [6, 11])
def test_nevresim_equals_the_executor_count_for_count(s):
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
    from mimarsinan.common.env import nevresim_root
    from mimarsinan.mapping.latency.chip import ChipLatency

    hybrid, stage = _segment(s)
    seg = stage.hard_core_mapping
    _, width = _segment_alone(hybrid, stage)
    x = _grid_inputs(s, width)
    _, want = _executor(seg, x, s, "ttfs_quantized")
    NevresimDriver.nevresim_path = NevresimDriver.nevresim_path or nevresim_root()
    with tempfile.TemporaryDirectory() as tmp:
        driver = NevresimDriver(
            width, seg, tmp, int, spike_generation_mode="TTFS", thresholding_mode="<=",
            firing_mode="TTFS", point=point_of_mode("ttfs_quantized"), threshold_type=float, verbose=False,
        )
        raw = driver.predict_spiking_raw([(x[i], np.zeros(1)) for i in range(x.shape[0])], s,
                                         ChipLatency(seg).calculate())
    got = _fire_steps(np.asarray(raw, dtype=np.float64), s)
    assert got.shape == want.shape, (got.shape, want.shape)
    apart = int(np.count_nonzero(got != want))
    assert apart == 0, f"{apart} of {want.size} output fire steps apart at S {s}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("s", [6, 11])
@pytest.mark.parametrize("schedule", ["quantized", "synchronized"])
def test_sanafe_equals_the_executor_count_for_count(s, schedule):
    pytest.importorskip("sanafe")
    from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

    mode = "ttfs_quantized" if schedule == "quantized" else "ttfs_cycle_based"
    hybrid, stage = _segment(s)
    program, width = _segment_alone(hybrid, stage)
    x = _grid_inputs(s, width)
    want, _ = _executor(stage.hard_core_mapping, x, s, mode)
    behavior = NeuralBehaviorConfig(spiking_mode=mode, firing_mode="TTFS", thresholding_mode="<=",
                                    spike_generation_mode="TTFS")
    runner = SanafeRunner(mapping=program, simulation_length=s, behavior=behavior,
                          ttfs_cycle_schedule="synchronized" if schedule == "synchronized" else "cascaded",
                          log_potential_trace=True)
    rows = []
    for i in range(x.shape[0]):
        rec = runner.run(x[i:i + 1].astype(np.float32), sample_index=i)
        segment = next(iter(rec.segments.values()))
        rows.append(np.asarray(segment.per_neuron_potential_trace, dtype=np.float64)[:, -1])
    got = _fire_steps(np.asarray(rows), s)
    assert got.shape == want.shape, (got.shape, want.shape)
    apart = int(np.count_nonzero(got != want))
    assert apart == 0, f"{apart} of {want.size} neuron fire steps apart at S {s} ({schedule})"
