"""_prepare_all_segments delegates its emit+compile pool to run_tasks_in_pool_bounded."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.simulation_runner import hybrid as hybrid_mod
from mimarsinan.chip_simulation.simulation_runner.emit import (
    _PreparedSegment,
    _emit_and_compile_segment,
)
from mimarsinan.chip_simulation.simulation_runner.hybrid import SimulationHybridMixin
from mimarsinan.code_generation.cpp_chip_model import SpikeSource


def _fake_hcm():
    core = SimpleNamespace(
        axons_per_core=1, neurons_per_core=1,
        available_axons=0, available_neurons=0,
        threshold=1.0,
        core_matrix=np.zeros((1, 1), dtype=np.float32),
        axon_sources=[SpikeSource(-1, 0, True, False, False)],
        hardware_bias=None,
        latency=0,
    )
    return SimpleNamespace(
        cores=[core],
        output_sources=[SpikeSource(0, 0, is_input=False, is_off=False)],
    )


def _fake_self(tmp_path, timeout_s):
    return SimpleNamespace(
        test_data=[(np.zeros(4, dtype=np.float32), np.zeros(1))],
        working_directory=str(tmp_path),
        weight_type=int,
        threshold_type=int,
        spike_generation_mode="Deterministic",
        firing_mode="Default",
        thresholding_mode="<=",
        spiking_mode="lif",
        simulation_length=4,
        nevresim_connectivity_mode="runtime",
        simulation_step_timeout_s=timeout_s,
    )


def test_prepare_all_segments_passes_the_wall_cap_to_the_bounded_pool(
    tmp_path, monkeypatch,
):
    slice_ = SimpleNamespace(node_id=-2, offset=0, size=4)
    out_slice = SimpleNamespace(node_id=0, offset=0, size=1)
    stage = SimpleNamespace(
        kind="neural", name="s0", hard_core_mapping=_fake_hcm(),
        compute_op=None, input_map=[slice_], output_map=[out_slice],
    )
    hybrid = SimpleNamespace(stages=[stage])

    monkeypatch.setattr(NevresimDriver, "nevresim_path", "/fake/nevresim")

    recorded = {}
    sentinel = _PreparedSegment(
        seg_idx=0, seg_dir="d", binary_path="b", output_size=1, input_size=4,
    )

    def _fake_pool(fn, task_args, *, max_workers, timeout_s, description):
        recorded.update(
            fn=fn, task_args=dict(task_args),
            max_workers=max_workers, timeout_s=timeout_s,
        )
        return {0: sentinel}

    monkeypatch.setattr(hybrid_mod, "run_tasks_in_pool_bounded", _fake_pool)

    prepared = SimulationHybridMixin._prepare_all_segments(
        _fake_self(tmp_path, 432.0), hybrid,
    )

    assert prepared == {0: sentinel}
    assert recorded["fn"] is _emit_and_compile_segment
    assert recorded["timeout_s"] == 432.0
    args = recorded["task_args"][0]
    assert args[0] == 0, "segment index leads the worker args"
    assert 432.0 in args, "the wall cap must reach the in-worker compile"
