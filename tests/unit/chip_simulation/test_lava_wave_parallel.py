"""C5 wave-parallel Lava per-core execution: longest-path waves, spawn-pool dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.lava_loihi import segment_runner
from mimarsinan.chip_simulation.lava_loihi.segment_assembly import (
    assemble_core_active_input,
)
from mimarsinan.chip_simulation.lava_loihi.segment_runner import LavaSegmentMixin
from mimarsinan.chip_simulation.lava_loihi.wave_schedule import (
    core_dependency_graph,
    wave_levels,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.common.env import LOIHI_WAVE_WORKERS_VAR
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"


# ---------------------------------------------------------------------------
# Wave levels = longest-path levels of the dependency graph
# ---------------------------------------------------------------------------


class TestWaveLevels:
    def test_levels_are_longest_path_levels(self):
        deps = {0: [], 1: [0], 2: [0], 3: [1, 2], 4: []}
        assert wave_levels(deps) == [[0, 4], [1, 2], [3]]

    def test_chain_yields_one_core_per_wave(self):
        deps = {0: [], 1: [0], 2: [1], 3: [2]}
        assert wave_levels(deps) == [[0], [1], [2], [3]]

    def test_union_is_full_core_set_and_no_intra_wave_edges(self):
        rng = np.random.default_rng(7)
        n = 40
        deps = {
            i: sorted({int(j) for j in rng.integers(0, i, size=int(rng.integers(0, 4)))})
            if i > 0
            else []
            for i in range(n)
        }
        waves = wave_levels(deps)
        flat = [idx for wave in waves for idx in wave]
        assert sorted(flat) == list(range(n))
        assert len(flat) == n

        level = {idx: k for k, wave in enumerate(waves) for idx in wave}
        for idx, idx_deps in deps.items():
            for dep in idx_deps:
                assert level[dep] < level[idx]
            expected = 0 if not idx_deps else 1 + max(level[dep] for dep in idx_deps)
            assert level[idx] == expected

    def test_cycle_fails_loud(self):
        with pytest.raises(RuntimeError, match="Cycle detected"):
            wave_levels({0: [1], 1: [0]})

    def test_self_dependency_fails_loud(self):
        with pytest.raises(RuntimeError, match="Cycle detected"):
            wave_levels({0: [0]})

    def test_empty_graph_has_no_waves(self):
        assert wave_levels({}) == []


# ---------------------------------------------------------------------------
# Shifted-core case: _align_shiftable_cores can invert latency order, so waves
# MUST come from the dependency graph, never from timing.core_latency.
# ---------------------------------------------------------------------------


def _core(axons: int, neurons: int, sources: list, threshold: float = 1.0) -> HardCore:
    core = HardCore(axons_per_core=axons, neurons_per_core=neurons)
    core.core_matrix = np.ones((axons, neurons), dtype=np.float64)
    core.axon_sources = sources
    core.threshold = threshold
    core.available_axons = 0
    core.available_neurons = 0
    return core


def _shifted_source_mapping() -> HardCoreMapping:
    """Source core 0 feeds a deep consumer (4) and a shallow consumer (5)."""
    cores = [
        _core(1, 1, [SpikeSource(-2, 0, is_input=True)]),  # 0: shiftable source
        _core(1, 1, [SpikeSource(-2, 0, is_input=True)]),  # 1: chain root
        _core(1, 1, [SpikeSource(1, 0)]),                  # 2
        _core(1, 1, [SpikeSource(2, 0)]),                  # 3
        _core(2, 1, [SpikeSource(3, 0), SpikeSource(0, 0)]),  # 4: deep consumer
        _core(1, 1, [SpikeSource(0, 0)]),                  # 5: shallow consumer
    ]
    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = cores
    mapping.output_sources = np.array([SpikeSource(4, 0), SpikeSource(5, 0)])
    return mapping


class TestShiftedCoreCase:
    def test_alignment_inverts_latency_order_yet_source_still_precedes_consumer(self):
        mapping = _shifted_source_mapping()
        ChipLatency(mapping).calculate()
        lat = {idx: int(core.latency) for idx, core in enumerate(mapping.cores)}
        # _align_shiftable_cores raised core 0 to max(consumer)-1, above its own consumer 5.
        assert lat[0] > lat[5], f"expected inverted latencies, got {lat}"

        deps = core_dependency_graph(mapping.cores)
        assert deps[5] == [0]
        assert deps[4] == [0, 3]

        waves = wave_levels(deps)
        level = {idx: k for k, wave in enumerate(waves) for idx in wave}
        assert level[0] < level[5]
        assert level[0] < level[4]
        assert level[3] < level[4]


# ---------------------------------------------------------------------------
# Parallel-vs-serial output equality on a diamond segment (mocked core unit)
# ---------------------------------------------------------------------------


def _fake_core_output(weights, threshold, hardware_bias, input_spikes) -> np.ndarray:
    bias = 0.0 if hardware_bias is None else np.asarray(hardware_bias).reshape(-1, 1)
    drive = weights @ input_spikes + bias
    return (drive >= float(threshold)).astype(np.float64)


class _FakeLavaHost(LavaSegmentMixin):
    def __init__(self, T: int):
        self.T = T
        self._behavior = NeuralBehaviorConfig(
            spiking_mode="lif",
            firing_mode="Default",
            thresholding_mode="<=",
            spike_generation_mode="Uniform",
        )
        self._simulation_step_timeout_s = 60.0

    def _run_core_lava(self, *, weights, threshold, hardware_bias, input_spikes):
        return _fake_core_output(weights, threshold, hardware_bias, input_spikes)


def _diamond_segment() -> HardCoreMapping:
    """core0 -> {core1, core2} -> core3; waves must be [[0], [1, 2], [3]]."""
    core0 = _core(2, 2, [
        SpikeSource(-2, 0, is_input=True), SpikeSource(-2, 1, is_input=True),
    ])
    core1 = _core(2, 1, [SpikeSource(0, 0), SpikeSource(0, 1)])
    core2 = _core(2, 2, [SpikeSource(0, 0), SpikeSource(0, 1)], threshold=2.0)
    core3 = _core(3, 2, [
        SpikeSource(1, 0), SpikeSource(2, 0), SpikeSource(2, 1),
    ])
    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = [core0, core1, core2, core3]
    mapping.output_sources = np.array([SpikeSource(3, 0), SpikeSource(3, 1)])
    return mapping


def _run_segment(host: _FakeLavaHost) -> np.ndarray:
    seg = _diamond_segment()
    rates = np.array([[0.5, 1.0], [0.25, 0.75]], dtype=np.float64)
    return host._run_neural_segment_scheduled(seg, rates)


class TestParallelSerialEquality:
    def test_serial_path_never_touches_the_pool(self, monkeypatch, capsys):
        monkeypatch.setenv(LOIHI_WAVE_WORKERS_VAR, "1")

        def _forbidden_pool(*args, **kwargs):
            raise AssertionError("workers=1 must run inline, not through the pool")

        monkeypatch.setattr(segment_runner, "run_tasks_in_pool_bounded", _forbidden_pool)
        out = _run_segment(_FakeLavaHost(T=4))
        assert out.shape == (2, 2)

    def test_parallel_output_equals_serial_output(self, monkeypatch, capsys):
        monkeypatch.setenv(LOIHI_WAVE_WORKERS_VAR, "1")
        serial_out = _run_segment(_FakeLavaHost(T=4))

        monkeypatch.setenv(LOIHI_WAVE_WORKERS_VAR, "4")
        pool_calls: list[dict] = []

        def _fake_pool(fn, task_args, *, max_workers, timeout_s, description):
            pool_calls.append({
                "keys": sorted(task_args),
                "max_workers": max_workers,
                "timeout_s": timeout_s,
            })
            # Reversed completion order: aggregation must be dict-keyed, not positional.
            return {key: fn(*task_args[key]) for key in reversed(sorted(task_args))}

        def _fake_task(T, behavior, weights, threshold, hardware_bias, input_spikes):
            assert T == 4
            assert behavior.thresholding_mode == "<="
            return _fake_core_output(weights, threshold, hardware_bias, input_spikes)

        monkeypatch.setattr(segment_runner, "run_tasks_in_pool_bounded", _fake_pool)
        monkeypatch.setattr(segment_runner, "run_lava_core_task", _fake_task)
        parallel_out = _run_segment(_FakeLavaHost(T=4))

        np.testing.assert_array_equal(serial_out, parallel_out)
        # Only the two-core wave [1, 2] is pool-funded; waves [0] and [3] run inline.
        assert [c["keys"] for c in pool_calls] == [[1, 2]]
        assert pool_calls[0]["max_workers"] == 2
        assert pool_calls[0]["timeout_s"] == 60.0


# ---------------------------------------------------------------------------
# The source-not-scheduled guard stays loud
# ---------------------------------------------------------------------------


class TestSourceNotScheduledGuard:
    def test_missing_upstream_buffer_fails_loud(self):
        core = _core(1, 1, [SpikeSource(7, 0)])
        with pytest.raises(RuntimeError, match="has not been scheduled yet"):
            assemble_core_active_input(
                core,
                core_idx=3,
                N=1,
                T=4,
                latency=1,
                used_ax=1,
                seg_input_logical=np.zeros((0, 1, 8), dtype=np.float64),
                core_buffer_spikes={},
            )


# ---------------------------------------------------------------------------
# Executor seam: bounded spawn-context ProcessPoolExecutor only
# ---------------------------------------------------------------------------


class TestPoolSeam:
    def test_segment_runner_routes_through_the_bounded_pool(self):
        src = (SRC_ROOT / "chip_simulation/lava_loihi/segment_runner.py").read_text()
        assert "run_tasks_in_pool_bounded" in src
        for forbidden in ("ProcessPoolExecutor", "multiprocessing.Pool", "as_completed"):
            assert forbidden not in src

    def test_no_bare_executors_anywhere_in_lava_loihi(self):
        lava_dir = SRC_ROOT / "chip_simulation" / "lava_loihi"
        for path in sorted(lava_dir.glob("*.py")):
            src = path.read_text()
            assert "ProcessPoolExecutor" not in src, path.name
            assert "multiprocessing.Pool" not in src, path.name

    def test_worker_restores_fork_actors_before_any_lava_import(self):
        """Spawn workers arrive with the start method forced to 'spawn'; Lava's
        channel runtime deadlocks under spawn-context actors, so the worker must
        restore 'fork' and apply the set_start_method idempotency patch
        (``_subtractive_lif_cls``) before any lava import."""
        src = (SRC_ROOT / "chip_simulation/lava_loihi/core_worker.py").read_text()
        body = src.split("def run_lava_core_task", 1)[1]
        fork_restore = body.index('multiprocessing.set_start_method("fork", force=True)')
        patch = body.index("_subtractive_lif_cls()")
        run_call = body.index("._run_core_lava(")
        assert fork_restore < patch < run_call

    def test_bounded_pool_is_spawn_context_process_pool(self):
        from mimarsinan.chip_simulation import execution_bounds

        assert execution_bounds._POOL_MP_CONTEXT.get_start_method() == "spawn"
        src = (SRC_ROOT / "chip_simulation/execution_bounds.py").read_text()
        assert "ProcessPoolExecutor(" in src
        assert "mp_context=_POOL_MP_CONTEXT" in src
