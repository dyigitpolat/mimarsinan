"""ArchitectureSearchStep hardware-mode smoke test.

Hardware-only search must run end-to-end on the resolved platform base (the
historical failure: ``fixed_platform_constraints`` stayed ``None`` outside
model mode, every candidate died on ``KeyError: 'cores'``, and the step
reported a misleading "no candidates" error).
"""

import random

import numpy as np
import pytest
import torch
from conftest import MockPipeline, default_config

from mimarsinan.mapping.platform.coalescing import CANONICAL_KEY
from mimarsinan.pipelining.determinism import (
    apply_determinism,
    isolated_rng_stream,
)
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_step import (
    ArchitectureSearchStep,
)


def _hardware_search_config():
    cfg = default_config()
    cfg.update({
        "model_type": "simple_mlp",
        "model_config": {
            "mlp_width_1": 16,
            "mlp_width_2": 16,
            "base_activation": "ReLU",
        },
        "hw_config_mode": "search",
        "weight_bits": 4,
        "allow_scheduling": True,
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
        "arch_search": {
            "optimizer": "nsga2",
            "pop_size": 4,
            "generations": 2,
            "seed": 0,
            "num_core_types": 1,
            "core_axons_bounds": [64, 256],
            "core_neurons_bounds": [64, 256],
            "core_count_bounds": [8, 64],
        },
    })
    return cfg


def _run_step(tmp_path, config=None):
    pipeline = MockPipeline(
        config=config or _hardware_search_config(),
        working_directory=str(tmp_path / "search_step"),
    )
    step = ArchitectureSearchStep(pipeline)
    step.name = "ArchitectureSearch"
    pipeline.prepare_step(step)
    step.run()
    return pipeline


class TestADeploymentRunCanSealAResourceLedger:
    """[TS1] The accountant reaches a real run through the run's DECLARATION.

    A mechanism with no path from a config into a sealed artifact is a
    mechanism only a hand-built campaign problem can use.
    """

    def test_a_declared_budget_seals_a_ledger_in_the_step_artifact(self, tmp_path):
        config = _hardware_search_config()
        config["arch_search"] = {
            **config["arch_search"], "generations": 3, "evaluation_budget": 2,
        }

        pipeline = _run_step(tmp_path, config)

        ledger = pipeline.cache["ArchitectureSearch.architecture_search_result"]["ledger"]
        assert ledger["budget_limit"] == 2
        assert ledger["evaluations_distinct"] >= 2, "the boundary stop overshoots"
        assert ledger["stopped_at_boundary"] is True, "3 generations were declared"
        assert ledger["wall_s"] > 0.0

    def test_an_undeclared_budget_seals_no_ledger(self, tmp_path):
        result = _run_step(tmp_path).cache[
            "ArchitectureSearch.architecture_search_result"
        ]

        assert "ledger" not in result, "an unmetered run makes no claim about spend"

    def test_a_budget_that_is_not_a_positive_count_fails_loud(self, tmp_path):
        config = _hardware_search_config()
        config["arch_search"] = {**config["arch_search"], "evaluation_budget": 0}

        with pytest.raises(ValueError, match="evaluation_budget"):
            _run_step(tmp_path, config)


class TestHardwareModeSmoke:
    def test_step_completes_and_promises_entries(self, tmp_path):
        pipeline = _run_step(tmp_path)
        cache = pipeline.cache
        assert "ArchitectureSearch.model_config" in cache
        assert "ArchitectureSearch.model_builder" in cache
        assert "ArchitectureSearch.platform_constraints_resolved" in cache
        assert "ArchitectureSearch.architecture_search_result" in cache

        result = cache["ArchitectureSearch.architecture_search_result"]
        assert result["search_mode_used"] == "hardware"
        assert result["best"]["configuration"], "search must land a best candidate"
        assert result["discovered_platform_constraints"] is not None

    def test_resolved_platform_is_base_plus_searched_cores(self, tmp_path):
        pipeline = _run_step(tmp_path)
        pcfg = pipeline.cache["ArchitectureSearch.platform_constraints_resolved"]

        assert pcfg["cores"], "searched cores must be present"
        for core in pcfg["cores"]:
            assert 64 <= core["max_axons"] <= 256
            assert 64 <= core["max_neurons"] <= 256
            assert 8 <= core["count"] <= 64
            assert core["has_bias"] is True

        # The deployed base rides along: search and deployment see one chip.
        assert pcfg["weight_bits"] == 4, "config weight_bits, not a hardcoded 8"
        assert pcfg["allow_scheduling"] is True
        assert pcfg[CANONICAL_KEY] is False
        assert "schedule_policy" not in pcfg  # retired [U1]
        assert "max_schedule_passes" in pcfg

    def test_the_promised_platform_is_the_deployment_resolution_of_the_winner(
        self, tmp_path,
    ):
        # W5.1: the step promises the WINNER's chip verbatim — the SAME
        # resolution a fixed-mode deployment of the winner's decision variables
        # would produce. Nothing is merged, re-stamped, or carried afterwards.
        #
        # The expectation is built from the SEARCH RESULT's own record of the
        # winner, never from the promised platform: a golden rebuilt out of the
        # thing under test would pass just as happily on the run's base chip.
        pipeline = _run_step(tmp_path)
        pcfg = pipeline.cache["ArchitectureSearch.platform_constraints_resolved"]
        result = pipeline.cache["ArchitectureSearch.architecture_search_result"]
        winner = result["best"]["configuration"]["platform_constraints"]

        expected = build_platform_constraints_resolved(
            {**_hardware_search_config(),
             "cores": winner["cores"],
             "target_tq": winner["target_tq"]},
        )
        assert pcfg == expected
        assert result["discovered_platform_constraints"] == pcfg

        # ...and the winner is a chip the search FOUND, not the one it started
        # from, so promoting the base instead cannot satisfy the golden above.
        base = build_platform_constraints_resolved(_hardware_search_config())
        assert pcfg["cores"] != base["cores"], (
            "the fixture must let the search move off its declared platform, "
            "or this golden cannot tell the winner from the base"
        )

    def test_fixed_model_config_is_passed_through(self, tmp_path):
        pipeline = _run_step(tmp_path)
        model_config = pipeline.cache["ArchitectureSearch.model_config"]
        assert model_config == _hardware_search_config()["model_config"]

    def test_best_objectives_are_not_penalties(self, tmp_path):
        pipeline = _run_step(tmp_path)
        result = pipeline.cache["ArchitectureSearch.architecture_search_result"]
        best_objectives = result["best"]["objectives"]
        assert best_objectives, "hardware mode has layout-proxy objectives"
        for name, value in best_objectives.items():
            assert abs(float(value)) < 1e17, f"{name} looks like a penalty: {value}"


@pytest.fixture
def _restore_process_rng():
    """These tests seed the process; sibling tests keep their ambient stream."""
    states = (torch.random.get_rng_state(), np.random.get_state(), random.getstate())
    yield
    torch.random.set_rng_state(states[0])
    np.random.set_state(states[1])
    random.setstate(states[2])


@pytest.mark.usefixtures("_restore_process_rng")
class TestTheSearchDoesNotMoveTheRunsRngStream:
    """A search CHOOSES a chip; it must not re-roll the weights the run deploys.

    Candidate scoring seeds the world to its own scoring seed — deliberately,
    since that is what makes a candidate's score reproducible — and it used to
    leave it there. Every later step then drew from a stream that depended on
    how many candidates the search had looked at: the SAME config deployed
    different weights with the search ON than with its winning chip declared by
    hand, so the searched-hardware guard cell was rolling fresh dice against
    every downstream certificate instead of being its fixed anchor's pair.
    """

    @staticmethod
    def _draw():
        return (
            torch.rand(4).tolist(),
            np.random.rand(4).tolist(),
            random.random(),
        )

    def test_the_next_draw_is_the_one_the_run_would_have_had(self, tmp_path):
        apply_determinism(1234)
        expected = self._draw()

        apply_determinism(1234)
        _run_step(tmp_path)
        assert self._draw() == expected

    def test_every_seeded_family_is_put_back(self, tmp_path):
        apply_determinism(4242)
        torch_state = torch.random.get_rng_state()
        np_state = np.random.get_state()
        py_state = random.getstate()

        _run_step(tmp_path)

        assert torch.equal(torch.random.get_rng_state(), torch_state)
        assert np.random.get_state()[1].tolist() == np_state[1].tolist()  # pyright: ignore[reportIndexIssue] — legacy MT19937 state tuple
        assert random.getstate() == py_state

    def test_a_reseed_inside_the_block_does_not_escape(self):
        """The isolation is not a rewind: the search calls ``manual_seed``."""
        apply_determinism(7)
        expected = self._draw()

        apply_determinism(7)
        with isolated_rng_stream():
            apply_determinism(999)
            self._draw()
        assert self._draw() == expected

    def test_no_search_budget_leaves_a_mark_on_the_stream(self, tmp_path):
        # The budget is what used to leak: more candidates, more draws
        # consumed, a different model deployed at the end of it. Every budget
        # must land where a run with NO search at all would have been.
        def stream_after(pop_size, generations):
            apply_determinism(11)
            cfg = _hardware_search_config()
            cfg["arch_search"] = {**cfg["arch_search"],
                                  "pop_size": pop_size, "generations": generations}
            pipeline = MockPipeline(
                config=cfg,
                working_directory=str(tmp_path / f"budget_{pop_size}_{generations}"),
            )
            step = ArchitectureSearchStep(pipeline)
            step.name = "ArchitectureSearch"
            pipeline.prepare_step(step)
            step.run()
            return self._draw()

        apply_determinism(11)
        unsearched = self._draw()
        assert stream_after(4, 2) == unsearched
        assert stream_after(8, 3) == unsearched
