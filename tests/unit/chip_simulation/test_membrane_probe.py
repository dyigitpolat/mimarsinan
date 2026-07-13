"""[C2] nevresim-probe membrane-decode helpers: eligibility + correction stash."""

from __future__ import annotations

import numpy as np

from mimarsinan.chip_simulation.simulation_runner.membrane_probe import (
    flat_membrane_slices,
    stash_membrane_corrections,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping


def _single_stage_hybrid(n: int = 3) -> HybridHardCoreMapping:
    core = HardCore(1, n, has_bias_capability=False)
    core.core_matrix = np.full((1, n), 0.5, dtype=np.float64)
    core.axon_sources = [SpikeSource(-2, 0, is_input=True)]
    core.available_axons = 0
    core.available_neurons = 0
    core.threshold = 1.0
    core.latency = 0
    segment = HardCoreMapping([])
    segment.cores = [core]
    segment.output_sources = np.asarray(
        [SpikeSource(0, i) for i in range(n)], dtype=object
    )
    stage = HybridStage(
        kind="neural",
        name="s0",
        hard_core_mapping=segment,
        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=1)],
        output_map=[SegmentIOSlice(node_id=0, offset=0, size=n)],
    )
    return HybridHardCoreMapping(
        stages=[stage],
        output_sources=np.asarray(
            [IRSource(node_id=0, index=i) for i in range(n)], dtype=object
        ),
    )


class TestStashMembraneCorrections:
    def test_stash_removes_half_step_and_keys_by_node(self):
        hybrid = _single_stage_hybrid(3)
        membranes = np.asarray([[0.25, -0.5, 0.75], [0.0, 0.5, 1.0]])
        corrections: dict[int, np.ndarray] = {}
        stash_membrane_corrections(
            hybrid, hybrid.stages[0], membranes, corrections,
            half_step_charge=0.5,
        )
        assert set(corrections) == {0}
        np.testing.assert_allclose(corrections[0], membranes - 0.5)

    def test_zero_half_step_passes_raw_membranes(self):
        hybrid = _single_stage_hybrid(2)
        membranes = np.asarray([[0.25, -0.5]])
        corrections: dict[int, np.ndarray] = {}
        stash_membrane_corrections(
            hybrid, hybrid.stages[0], membranes, corrections,
            half_step_charge=0.0,
        )
        np.testing.assert_allclose(corrections[0], membranes)


class TestPrepareSegmentsArmsExportFlag:
    """The hybrid probe compiles the NEVRESIM_EXPORT_MEMBRANE build exactly for
    eligible stages of a gate-armed runner."""

    def _prepare(self, monkeypatch, tmp_path, *, armed: bool):
        from types import SimpleNamespace

        import mimarsinan.chip_simulation.simulation_runner.hybrid as hybrid_mod
        from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
        from mimarsinan.chip_simulation.simulation_runner.hybrid import (
            SimulationHybridMixin,
        )

        hybrid = _single_stage_hybrid(2)
        monkeypatch.setattr(NevresimDriver, "nevresim_path", "/fake/nevresim")
        recorded: dict[int, tuple] = {}

        def _fake_pool(fn, task_args, *, max_workers, timeout_s, description):
            recorded.update(task_args)
            return {}

        monkeypatch.setattr(hybrid_mod, "run_tasks_in_pool_bounded", _fake_pool)
        fake_self = SimpleNamespace(
            test_data=[(np.zeros(1, dtype=np.float64), np.zeros(1))],
            working_directory=str(tmp_path),
            weight_type=float,
            threshold_type=float,
            spike_generation_mode="Uniform",
            firing_mode="Default",
            thresholding_mode="<=",
            spiking_mode="lif",
            simulation_length=4,
            nevresim_connectivity_mode="runtime",
            simulation_step_timeout_s=900.0,
            membrane_readout=armed,
            membrane_half_step_charge=0.0,
        )
        SimulationHybridMixin._prepare_all_segments(fake_self, hybrid)
        return recorded[0]

    def test_armed_runner_requests_membrane_build(self, monkeypatch, tmp_path):
        args = self._prepare(monkeypatch, tmp_path, armed=True)
        assert args[-1] is True, "export_membrane must reach the compile worker"

    def test_unarmed_runner_keeps_default_build(self, monkeypatch, tmp_path):
        args = self._prepare(monkeypatch, tmp_path, armed=False)
        assert args[-1] is False


class TestFlatMembraneSlices:
    def test_armed_single_stage_yields_eligible_slices(self):
        hybrid = _single_stage_hybrid(3)
        assert flat_membrane_slices(hybrid, armed=True) == [(0, 0, 3)]

    def test_unarmed_is_empty(self):
        assert flat_membrane_slices(_single_stage_hybrid(), armed=False) == []

    def test_bare_hard_core_mapping_is_empty(self):
        """A bare ``HardCoreMapping`` carries no node structure to gate
        eligibility on; the probe stays counts-only."""
        segment = _single_stage_hybrid().stages[0].hard_core_mapping
        assert flat_membrane_slices(segment, armed=True) == []
