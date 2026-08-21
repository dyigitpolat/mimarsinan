"""The 2-core ODIN deployment fixture, shared by the transport and E2E gates.

Same family as the R11a cosimulation fixture (``odin_rtl_harness``): integral
weights, a biasless always-on row, a per-event law whose producer emits MORE
than one spike per cycle so a per-cycle executor could not reproduce it. Here it
is wrapped as a HYBRID program, because the physical backend is a pipeline step
and the thing under test is the whole path from a state buffer to device counts.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from integration.odin_rtl_harness import ODIN_LAW, export_of, hard_core, mapping_of

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)

INPUT_LINES = 4
NEURONS = 4
TIMESTEPS = 4
#: theta 3 with weight 3 on four occurrences of one line: several crossings in
#: ONE cycle, which is the multiplicity a per-cycle law cannot produce.
PRODUCER_THETA = 3
CONSUMER_THETA = 12
WEIGHT_BITS = 4


def _producer_weights() -> np.ndarray:
    rng = np.random.default_rng(11)
    weights = rng.integers(1, 8, size=(INPUT_LINES + 1, NEURONS)).astype(np.float64)
    weights[:, 0] = 3.0
    return weights


def _consumer_weights() -> np.ndarray:
    rng = np.random.default_rng(23)
    weights = rng.integers(1, 8, size=(NEURONS, NEURONS)).astype(np.float64)
    weights[0, :] = 7.0
    return weights


def two_core_mapping():
    """The packed segment: a multi-spiking producer feeding one consumer."""
    producer = hard_core(
        _producer_weights(), threshold=float(PRODUCER_THETA),
        sources=[SpikeSource(-2, index, is_input=True) for index in range(INPUT_LINES)]
        + [SpikeSource(-3, 0, is_always_on=True)],
    )
    consumer = hard_core(
        _consumer_weights(), threshold=float(CONSUMER_THETA),
        sources=[SpikeSource(0, index) for index in range(NEURONS)],
    )
    mapping = mapping_of(
        [producer, consumer], [SpikeSource(1, index) for index in range(NEURONS)])
    ChipLatency(mapping).calculate()
    return mapping


def hybrid_program() -> HybridHardCoreMapping:
    """One neural stage carrying the two-core segment — a whole deployable program."""
    stage = HybridStage(
        kind="neural", name="odin_fpga_fixture",
        hard_core_mapping=two_core_mapping(),
        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=INPUT_LINES)],
        output_map=[SegmentIOSlice(node_id=0, offset=0, size=NEURONS)],
    )
    return HybridHardCoreMapping(
        stages=[stage],
        output_sources=np.asarray(
            [IRSource(node_id=0, index=index) for index in range(NEURONS)],
            dtype=object),
    )


def fixture_export():
    """The exported images + program the transports are handed."""
    return export_of(two_core_mapping())


def pipeline_config(**overrides: Any) -> Dict[str, Any]:
    """The declared config of the fixture: the stock-ODIN point, device opt-in."""
    config: Dict[str, Any] = {
        "device": "cpu",
        "input_shape": (1, 1, INPUT_LINES),
        "num_classes": NEURONS,
        "num_workers": 0,
        "simulation_steps": TIMESTEPS,
        "target_tq": TIMESTEPS,
        "weight_bits": WEIGHT_BITS,
        "spiking_family": "lif",
        "spiking_variant": "streamed",
        "spiking_mode": "lif",
        "firing_mode": "Novena",
        "thresholding_mode": "<=",
        "spike_generation_mode": "Deterministic",
        "firing_granularity": "per_event",
        "membrane_bits": 8,
        "membrane_arithmetic": "saturating_unsigned",
        "weight_sign_granularity": "per_axon",
        "enable_odin_fpga_simulation": True,
        "enable_nevresim_simulation": False,
        "enable_sanafe_simulation": False,
        "enable_loihi_simulation": False,
        "odin_fpga_transport": "rtl_cosim",
        "odin_fpga_sample_count": 1,
        "cores": [{"max_axons": 8, "max_neurons": 8, "count": 2}],
    }
    config.update(overrides)
    return config


PLATFORM_RESOLVED = {
    "cores": [{"max_axons": 8, "max_neurons": 8, "count": 2}],
    "weight_bits": WEIGHT_BITS,
    "cores_per_tile": 0, "tile_grid_rows": 0, "tile_grid_cols": 0,
}


def prepare_step(monkeypatch, step_class, *, transport, config_overrides=None):
    """A MockPipeline-driven instance of the REAL step, sharing the fixture.

    Only the SAMPLE SOURCE is stubbed (the sanafe step test's precedent): the
    step, the HCM reference, the runner, the exporter and the transport are all
    the production objects.
    """
    from conftest import MockDataProviderFactory, MockPipeline
    import mimarsinan.pipelining.pipeline_steps.verification.odin_fpga_simulation_step \
        as step_module

    sample = entry_sample()

    def _load(_factory, indices, num_workers=4):
        import torch

        return [
            torch.tensor(sample, dtype=torch.float32).reshape(1, INPUT_LINES)
            for _ in indices
        ]

    monkeypatch.setattr(step_module, "load_test_samples_by_index", _load)
    monkeypatch.setattr(
        step_module, "build_transport", lambda _config: transport)

    pipeline = MockPipeline(
        config=pipeline_config(**(config_overrides or {})),
        data_provider_factory=MockDataProviderFactory(
            input_shape=(1, 1, INPUT_LINES), num_classes=NEURONS, size=4),
    )
    pipeline.reporter = _RecordingReporter()
    pipeline.set_target_metric(0.5)
    pipeline.seed("model", object(), step_name="Model Configuration")
    pipeline.seed("hard_core_mapping", hybrid_program(),
                  step_name="Hard Core Mapping")
    pipeline.seed("platform_constraints_resolved", dict(PLATFORM_RESOLVED),
                  step_name="Model Configuration")
    step = step_class(pipeline)
    pipeline.prepare_step(step)
    return pipeline, step


class _RecordingReporter:
    """Captures the headline metrics the step reports."""

    def __init__(self):
        self.events: list = []

    def report(self, name, value):
        self.events.append((name, value))


def entry_sample() -> np.ndarray:
    """A saturated entry raster: every input line fires in every cycle."""
    return np.ones((1, INPUT_LINES), dtype=np.float64)


__all__ = [
    "CONSUMER_THETA",
    "INPUT_LINES",
    "NEURONS",
    "ODIN_LAW",
    "PRODUCER_THETA",
    "TIMESTEPS",
    "WEIGHT_BITS",
    "PLATFORM_RESOLVED",
    "entry_sample",
    "fixture_export",
    "prepare_step",
    "hybrid_program",
    "pipeline_config",
    "two_core_mapping",
]
