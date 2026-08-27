"""The TINY deployable classifier the HACC export gates run end to end.

Small enough to freeze in a second, but a real classification task rather than
a tautology: two identity-chained ODIN cores turn a one-hot entry raster into a
one-hot readout, so sample ``k``'s TRUE label is ``k`` and a bundle that reports
100% accuracy has actually transported the spikes of the right neuron through
two host-mediated passes. A constant predictor scores 1/3.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from integration.odin_rtl_harness import ODIN_LAW, hard_core, mapping_of

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)

CLASSES = 3
INPUT_LINES = CLASSES
TIMESTEPS = 4
WEIGHT_BITS = 4
#: theta 1 with unit weights: one arriving event is one emitted event, so each
#: core relays its input line's raster exactly and the readout stays one-hot.
THETA = 1


def _identity(rows: int) -> np.ndarray:
    weights = np.zeros((rows, CLASSES), dtype=np.float64)
    for index in range(CLASSES):
        weights[index, index] = 1.0
    return weights


def two_core_mapping():
    """A producer that relays the entry raster into a consumer that relays it on."""
    producer = hard_core(
        _identity(INPUT_LINES + 1), threshold=float(THETA),
        sources=[SpikeSource(-2, index, is_input=True)
                 for index in range(INPUT_LINES)]
        + [SpikeSource(-3, 0, is_always_on=True)],
    )
    consumer = hard_core(
        _identity(CLASSES), threshold=float(THETA),
        sources=[SpikeSource(0, index) for index in range(CLASSES)],
    )
    mapping = mapping_of(
        [producer, consumer], [SpikeSource(1, index) for index in range(CLASSES)])
    ChipLatency(mapping).calculate()
    return mapping


def hybrid_program() -> HybridHardCoreMapping:
    """One neural stage carrying the two-core segment — a whole deployable program."""
    stage = HybridStage(
        kind="neural", name="odin_hacc_fixture",
        hard_core_mapping=two_core_mapping(),
        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=INPUT_LINES)],
        output_map=[SegmentIOSlice(node_id=0, offset=0, size=CLASSES)],
    )
    return HybridHardCoreMapping(
        stages=[stage],
        output_sources=np.asarray(
            [IRSource(node_id=0, index=index) for index in range(CLASSES)],
            dtype=object),
    )


def one_hot(label: int) -> np.ndarray:
    sample = np.zeros((1, INPUT_LINES), dtype=np.float64)
    sample[0, int(label) % INPUT_LINES] = 1.0
    return sample


def pipeline_config(**overrides: Any) -> Dict[str, Any]:
    """The stock-ODIN point with the HACC export opted in."""
    config: Dict[str, Any] = {
        "device": "cpu",
        "input_shape": (1, 1, INPUT_LINES),
        "input_size": INPUT_LINES,
        "num_classes": CLASSES,
        "num_workers": 0,
        "model_type": "odin_hacc_fixture",
        "experiment_name": "odin_hacc_micro",
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
        "enable_odin_hacc_export": True,
        "enable_nevresim_simulation": False,
        "enable_sanafe_simulation": False,
        "enable_loihi_simulation": False,
        "enable_odin_fpga_simulation": False,
        "odin_hacc_bundle_samples": 6,
        "odin_hacc_certification_samples": 3,
        "odin_hacc_bundle_name": "odin_hacc_micro",
        "cores": [{"max_axons": 8, "max_neurons": 8, "count": 2}],
    }
    config.update(overrides)
    return config


PLATFORM_RESOLVED = {
    "cores": [{"max_axons": 8, "max_neurons": 8, "count": 2}],
    "weight_bits": WEIGHT_BITS,
    "cores_per_tile": 0, "tile_grid_rows": 0, "tile_grid_cols": 0,
}


def prepare_step(monkeypatch, step_class, *, working_directory=None,
                 config_overrides=None):
    """A MockPipeline-driven instance of the REAL export step.

    Only the SAMPLE SOURCE is stubbed: the step, the HCM reference, the
    freezer, the exporter and the bundle schema are the production objects.
    """
    import torch
    from conftest import MockDataProviderFactory, MockPipeline
    import mimarsinan.pipelining.pipeline_steps.verification.odin_hacc_deployment_step \
        as step_module

    def _load(_factory, indices, num_workers=4):
        return [
            (torch.tensor(one_hot(index), dtype=torch.float32),
             torch.tensor([int(index) % CLASSES], dtype=torch.int64))
            for index in indices
        ]

    monkeypatch.setattr(step_module, "load_test_pairs_by_index", _load)

    pipeline = MockPipeline(
        config=pipeline_config(**(config_overrides or {})),
        working_directory=working_directory,
        data_provider_factory=MockDataProviderFactory(
            input_shape=(1, 1, INPUT_LINES), num_classes=CLASSES, size=8),
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


__all__ = [
    "CLASSES",
    "INPUT_LINES",
    "ODIN_LAW",
    "PLATFORM_RESOLVED",
    "THETA",
    "TIMESTEPS",
    "WEIGHT_BITS",
    "hybrid_program",
    "one_hot",
    "pipeline_config",
    "prepare_step",
    "two_core_mapping",
]
