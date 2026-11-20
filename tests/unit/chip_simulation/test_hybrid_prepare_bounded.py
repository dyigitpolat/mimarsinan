"""_prepare_all_segments delegates its emit+compile pool to run_tasks_in_pool_bounded."""

from __future__ import annotations

from fake_cores import FakeCore
from types import SimpleNamespace

import numpy as np

from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from spiking_points import point_of_mode
from mimarsinan.chip_simulation.simulation_runner import emit as emit_mod
from mimarsinan.chip_simulation.simulation_runner.emit import (
    _PreparedSegment,
    _emit_and_compile_segment,
)
from mimarsinan.chip_simulation.simulation_runner.hybrid import SimulationHybridMixin
from mimarsinan.code_generation.cpp_chip_model import SpikeSource


def _fake_hcm():
    core = FakeCore(
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
        pass_transfer="collapse",
        simulation_length=4,
        nevresim_connectivity_mode="runtime",
        simulation_step_timeout_s=timeout_s,
        membrane_readout=False,
        membrane_half_step_charge=0.0,
        spiking_point=point_of_mode("lif"),
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

    monkeypatch.setattr(emit_mod, "run_tasks_in_pool_bounded", _fake_pool)

    prepared = SimulationHybridMixin._prepare_all_segments(
        _fake_self(tmp_path, 432.0), hybrid,
    )

    assert prepared == {0: sentinel}
    assert recorded["fn"] is _emit_and_compile_segment
    assert recorded["timeout_s"] == 432.0
    args = recorded["task_args"][0]
    assert args[0] == 0, "segment index leads the worker args"
    assert 432.0 in args, "the wall cap must reach the in-worker compile"


# [U-LIF2 r2] nevresim's code generation takes the runner's resolved point, on both paths.

_DECLARED = (("fire_while_integrating", "integrate_then_emit"), "integrate_then_emit")


def _run_point(window_firing_declared):
    from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point

    window_firing_declared(*_DECLARED)
    return resolve_spiking_point({"spiking_family": "lif", "spiking_variant": "synchronized",
                                  "lif_window_firing": "fire_while_integrating"})


def test_the_emit_pool_carries_the_runner_point(tmp_path, monkeypatch, window_firing_declared):
    """The worker builds its driver from the runner's record (declared against the other default), never a mode."""
    import inspect

    point = _run_point(window_firing_declared)
    recorded = {}
    monkeypatch.setattr(NevresimDriver, "nevresim_path", "/fake/nevresim")
    monkeypatch.setattr(emit_mod, "run_tasks_in_pool_bounded",
                        lambda fn, task_args, **kw: recorded.update(task_args=dict(task_args)) or {})
    slice_ = SimpleNamespace(node_id=-2, offset=0, size=4)
    stage = SimpleNamespace(kind="neural", name="s0", hard_core_mapping=_fake_hcm(), compute_op=None,
                            input_map=[slice_], output_map=[SimpleNamespace(node_id=0, offset=0, size=1)])
    runner = _fake_self(tmp_path, 60.0)
    runner.spiking_point = point
    SimulationHybridMixin._prepare_all_segments(runner, SimpleNamespace(stages=[stage]))
    parameters = list(inspect.signature(_emit_and_compile_segment).parameters)
    args = dict(zip(parameters, recorded["task_args"][0]))
    assert args["point"] is point and "spiking_mode" not in args and "soma_law" not in args


def test_the_flat_path_builds_its_driver_at_the_runner_point(tmp_path, monkeypatch, window_firing_declared):
    from mimarsinan.chip_simulation.simulation_runner import flat as flat_mod
    from mimarsinan.chip_simulation.simulation_runner.flat import SimulationFlatMixin

    point = _run_point(window_firing_declared)
    seen = {}

    class _Built(Exception):
        pass

    def recording_driver(*args, **kwargs):
        seen.update(kwargs)
        raise _Built

    monkeypatch.setattr(flat_mod, "NevresimDriver", recording_driver)
    monkeypatch.setattr(flat_mod, "ChipLatency", lambda hcm: SimpleNamespace(calculate=lambda: 1))
    monkeypatch.setattr(flat_mod, "bare_mapping_output_gauge", lambda mapping: None)  # [composed head, rule 19]
    runner = _fake_self(tmp_path, 60.0)
    runner.spiking_point = point
    runner.input_size = 4
    runner.mapping = None  # [composed head, rule 19] the gauge unit's flat runner reads it
    try:
        SimulationFlatMixin._run_flat_mapping(runner, _fake_hcm())
    except _Built:
        pass
    assert seen["point"] is point and "spiking_mode" not in seen and "soma_law" not in seen
