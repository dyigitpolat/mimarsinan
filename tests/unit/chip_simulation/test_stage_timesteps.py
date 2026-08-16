"""The executed-window rule has ONE home (E1).

A neural stage does not run ``T`` timesteps. It runs ``T`` plus the latency
its cores sit at plus one cycle for input delivery — and a windowed-LIF
program executes one stage PER DEPTH LEVEL, so the program's step count is a
sum over stages, not ``T x segments``.

The incident: ``quantities/from_candidate`` computed ``timesteps x
neural_segment_count`` while the runner computed ``T + max_latency + 1`` per
stage. Measured on the sealed MLP study run: candidate 4, record 15 — the
candidate under-charged the wall 3.75x, and with it every term that
multiplies latency (static energy, e2e, throughput).
"""

import pytest

from mimarsinan.chip_simulation.stage_timesteps import (
    executed_stage_timesteps,
    program_latency_steps,
)


class TestTheDefaultLifRule:
    def test_a_stage_runs_T_plus_its_latency_plus_the_delivery_cycle(self):
        assert executed_stage_timesteps(
            timesteps=4, max_latency=0, is_cycle=False, is_cascade=False,
        ) == 5

    def test_a_stage_holding_a_later_core_runs_longer(self):
        assert executed_stage_timesteps(
            timesteps=8, max_latency=1, is_cycle=False, is_cascade=False,
        ) == 10


class TestTheTtfsBranches:
    def test_cycle_based_runs_one_window_per_latency_group_plus_one(self):
        assert executed_stage_timesteps(
            timesteps=8, max_latency=3, is_cycle=True, is_cascade=False,
            latency_group_count=3,
        ) == (3 + 1) * 8

    def test_cascade_uses_the_whole_chip_latency(self):
        assert executed_stage_timesteps(
            timesteps=8, max_latency=1, is_cycle=False, is_cascade=True,
            chip_latency=5,
        ) == 8 + 5 + 1

    def test_cycle_without_its_group_count_refuses(self):
        with pytest.raises(ValueError, match="latency_group_count"):
            executed_stage_timesteps(
                timesteps=8, max_latency=0, is_cycle=True, is_cascade=False,
            )

    def test_cascade_without_its_chip_latency_refuses(self):
        with pytest.raises(ValueError, match="chip_latency"):
            executed_stage_timesteps(
                timesteps=8, max_latency=0, is_cycle=False, is_cascade=True,
            )


class TestTheProgramSum:
    def test_the_sealed_study_program_reproduces_its_measured_15(self):
        """The MLP study run: T=4, one segment retimed into 3 depth levels,
        each level a stage at max_latency 0 → 3 x 5 = 15, which is exactly
        the ``compute_steps`` its record sealed."""
        assert program_latency_steps(
            stage_max_latencies=[0, 0, 0], timesteps=4,
            is_cycle=False, is_cascade=False,
        ) == 15

    def test_the_lenet5_shape_reproduces_its_measured_28(self):
        """T=8, three stages whose executed windows were [9, 9, 10]."""
        assert program_latency_steps(
            stage_max_latencies=[0, 0, 1], timesteps=8,
            is_cycle=False, is_cascade=False,
        ) == 28

    def test_no_stages_is_no_steps(self):
        assert program_latency_steps(
            stage_max_latencies=[], timesteps=4,
            is_cycle=False, is_cascade=False,
        ) == 0


class TestTheRunnerDelegates:
    def test_the_sanafe_stage_reads_the_shared_rule(self):
        """The SSOT pin: the runner must not keep its own copy of the rule."""
        import inspect

        import mimarsinan.chip_simulation.sanafe.runner.neural_stage as stage

        source = inspect.getsource(stage)
        assert "executed_stage_timesteps" in source, (
            "neural_stage.py stopped delegating its executed-window rule to "
            "chip_simulation.stage_timesteps"
        )
