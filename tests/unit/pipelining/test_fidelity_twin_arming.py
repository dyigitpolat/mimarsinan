"""[H3] The fidelity twin is the SEARCH's problem, not a lookalike.

H0 measured the drift this pins closed: `_fidelity_problem` never passed the
firing semantics, so a re-timed run's twin priced the FUSED wall — E1's arming
reached the search step but not the fidelity plane. The step and the twin now
share one resolution (`firing_semantics_kwargs`), so they cannot drift apart
one kwarg at a time again.
"""

from __future__ import annotations

import pytest
from conftest import MockPipeline

from mimarsinan.config_schema.runtime import build_flat_pipeline_config
from mimarsinan.pipelining.pipeline_steps.verification import fidelity_emission


class _Captured(Exception):
    def __init__(self, kwargs):
        super().__init__("captured")
        self.kwargs = kwargs


def _twin_kwargs(monkeypatch, tmp_path, **declared):
    cfg = build_flat_pipeline_config(
        deployment_parameters={
            "model_type": "simple_mlp",
            "model_config": {"mlp_width_1": 16, "mlp_width_2": 16,
                             "base_activation": "ReLU"},
            "hw_config_mode": "search",
            "arch_search": {"objectives": ["param_utilization_pct"]},
            **declared,
        },
        platform_constraints={
            "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
        },
    )
    cfg.update({"device": "cpu", "input_shape": (1, 8, 8), "num_classes": 4})
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path / "run"))

    def _capture(**kwargs):
        raise _Captured(kwargs)

    monkeypatch.setattr(fidelity_emission, "JointArchHwProblem", _capture)
    with pytest.raises(_Captured) as caught:
        fidelity_emission._fidelity_problem(
            pipeline, ["param_utilization_pct"],
            {"cores": cfg["cores"], "weight_bits": 4, "target_tq": 4},
        )
    return caught.value.kwargs


class TestTheTwinIsArmedLikeTheStep:
    def test_a_default_lif_runs_twin_re_times(self, monkeypatch, tmp_path):
        """The drift H0 measured: the exact-QAT recipe arms re-timing during
        resolution, and the twin must price per-level windows like the run."""
        kwargs = _twin_kwargs(monkeypatch, tmp_path, spiking_mode="lif")
        assert kwargs["per_hop_retiming"] is True
        assert kwargs["spiking_mode"] == "lif"
        assert kwargs["pass_transfer"] == "collapse"

    def test_a_streamed_runs_twin_carries_verbatim(self, monkeypatch, tmp_path):
        kwargs = _twin_kwargs(
            monkeypatch, tmp_path,
            spiking_family="lif", spiking_variant="streamed",
        )
        assert kwargs["pass_transfer"] == "verbatim"

    def test_a_cycle_scheduled_runs_twin_knows_its_window_rule(
        self, monkeypatch, tmp_path,
    ):
        kwargs = _twin_kwargs(
            monkeypatch, tmp_path,
            spiking_family="ttfs", spiking_variant="synchronized",
        )
        assert kwargs["spiking_mode"] == "ttfs_cycle_based"
        assert kwargs["ttfs_cycle_schedule"] == "synchronized"

    def test_the_step_and_the_twin_read_one_helper(self):
        """The drift-proof: both call sites source the kwargs from the SAME
        function, so a new semantics kwarg cannot reach one and not the other."""
        import inspect

        from mimarsinan.pipelining.pipeline_steps.config import (
            architecture_search_problem,
        )

        assert "firing_semantics_kwargs" in inspect.getsource(
            architecture_search_problem)
        assert "firing_semantics_kwargs" in inspect.getsource(fidelity_emission)
