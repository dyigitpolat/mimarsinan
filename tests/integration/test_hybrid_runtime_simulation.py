"""Integration: hybrid SimulationRunner precompiled path with runtime connectivity."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn

from integration.parity_harness import build_toy_hybrid_mapping
from mimarsinan.chip_simulation.simulation_runner.hybrid import SimulationHybridMixin
from mimarsinan.common.build_utils import find_cpp20_compiler
from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping, HybridStage
from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver


NEVRESIM_ROOT = Path(__file__).resolve().parents[2] / "nevresim"


def _have_nevresim() -> bool:
    return NEVRESIM_ROOT.is_dir() and find_cpp20_compiler()[0] is not None


def _build_neural_plus_compute_hybrid() -> HybridHardCoreMapping:
    base = build_toy_hybrid_mapping()
    neural_stage = base.stages[0]
    compute_op = ComputeOp(
        id=1,
        name="identity",
        op_type="Identity",
        input_sources=np.array([IRSource(node_id=0, index=0)], dtype=object),
        params={"module": nn.Identity()},
        output_shape=(1, 1),
    )
    compute_stage = HybridStage(
        kind="compute",
        name="identity",
        compute_op=compute_op,
    )
    return HybridHardCoreMapping(
        stages=[neural_stage, compute_stage],
        output_sources=np.array([IRSource(node_id=1, index=0)], dtype=object),
    )


class _HybridSmokeRunner(SimulationHybridMixin):
    """Minimal harness exposing hybrid precompiled segment execution."""

    def __init__(
        self,
        *,
        work_dir: str,
        test_data: list,
        num_classes: int = 2,
    ) -> None:
        self.working_directory = work_dir
        self.test_data = test_data
        self.test_targets = [y for _, y in test_data]
        self.num_classes = num_classes
        self.weight_type = float
        self.threshold_type = float
        self.spike_generation_mode = "Deterministic"
        self.firing_mode = "Default"
        self.thresholding_mode = "<="
        self.spiking_mode = "lif"
        self.simulation_length = 4
        self.nevresim_connectivity_mode = "runtime"
        self.simulation_step_timeout_s = 900.0


@pytest.mark.skipif(not _have_nevresim(), reason="nevresim or C++20 compiler unavailable")
def test_hybrid_precompiled_runtime_multisample() -> None:
    num_samples = 20
    hybrid = _build_neural_plus_compute_hybrid()
    test_data = [
        (np.array([float(v)], dtype=np.float64), np.array([0], dtype=np.int64))
        for v in np.linspace(0.5, 1.0, num_samples)
    ]

    NevresimDriver.nevresim_path = str(NEVRESIM_ROOT)
    with tempfile.TemporaryDirectory() as tmp:
        runner = _HybridSmokeRunner(work_dir=tmp, test_data=test_data)
        prepared = runner._prepare_all_segments(hybrid)
        assert len(prepared) == 1

        seg = prepared[0]
        seg_input = runner._assemble_segment_input_np(
            hybrid.stages[0].input_map,
            {-2: np.stack([d[0] for d in test_data])},
            num_samples,
        )
        seg_data = [(seg_input[i], np.zeros(1)) for i in range(num_samples)]
        raw, membranes = runner._run_neural_segment_precompiled(
            seg, seg_data, num_proc=4,
        )

    # Default-off contract: without the [C2] gate the segment run is counts-only.
    assert membranes is None
    assert raw.shape == (num_samples, seg.output_size)
    assert np.any(raw >= 0)
