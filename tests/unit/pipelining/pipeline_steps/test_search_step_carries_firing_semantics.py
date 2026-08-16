"""[E1] The searched candidate is priced under the run's OWN firing semantics.

The executed wall branches on three of them — LIF vs cycle-based vs cascaded
TTFS, and whether per-hop re-timing splits each depth level into its own
execution stage. A search that assumed the defaults would price a re-timed
program's latency at a third of what the chip runs.

The re-timing arm is the subtle one: no config sets ``lif_per_hop_retiming``
by hand. The exact-QAT recipe pairs it on during resolution, so the step must
read the RESOLVED config — reading the raw JSON reports False on exactly the
runs that re-time. (That misreading is what these pins were written after.)
"""

from __future__ import annotations

import pytest
from conftest import MockPipeline

from mimarsinan.config_schema.runtime import build_flat_pipeline_config
from mimarsinan.pipelining.pipeline_steps.config import architecture_search_step
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_step import (
    ArchitectureSearchStep,
)

SEARCHED_RUN = {
    "model_type": "simple_mlp",
    "model_config": {
        "mlp_width_1": 16, "mlp_width_2": 16, "base_activation": "ReLU",
    },
    "hw_config_mode": "search",
    "arch_search": {"optimizer": "nsga2", "pop_size": 2, "generations": 1},
}


class _Captured(Exception):
    """Stop the step once the problem's declaration is in hand."""

    def __init__(self, kwargs):
        super().__init__("captured")
        self.kwargs = kwargs


def _problem_kwargs(monkeypatch, tmp_path, **declared):
    """The kwargs THIS run's step hands the search problem.

    The config goes through the run's OWN resolution first — recipe pairings
    included — because that is the config a live pipeline carries.
    """
    cfg = build_flat_pipeline_config(
        deployment_parameters={**SEARCHED_RUN, **declared},
        platform_constraints={
            "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
        },
    )
    # Dataset-derived keys the pipeline supplies at runtime, not the schema.
    cfg.update({"device": "cpu", "input_shape": (1, 8, 8), "num_classes": 4})

    def _capture(**kwargs):
        raise _Captured(kwargs)

    monkeypatch.setattr(architecture_search_step, "JointArchHwProblem", _capture)
    step = ArchitectureSearchStep(MockPipeline(
        config=cfg, working_directory=str(tmp_path / "search"),
    ))
    with pytest.raises(_Captured) as caught:
        step.process()
    return caught.value.kwargs


class TestTheFiringSemanticsReachTheProblem:
    def test_a_lif_run_declares_lif_to_the_search(self, monkeypatch, tmp_path):
        kwargs = _problem_kwargs(monkeypatch, tmp_path, spiking_mode="lif")
        assert kwargs["spiking_mode"] == "lif"

    def test_a_cycle_scheduled_ttfs_run_declares_its_schedule(
        self, monkeypatch, tmp_path,
    ):
        """The cycle-based window is ``(latency_groups + 1) x T``, a different
        rule entirely — the search must know which one this run executes."""
        kwargs = _problem_kwargs(
            monkeypatch, tmp_path,
            spiking_family="ttfs", spiking_variant="synchronized",
        )
        assert kwargs["spiking_mode"] == "ttfs_cycle_based"
        assert kwargs["ttfs_cycle_schedule"] == "synchronized"

    def test_a_cascaded_run_declares_its_schedule(self, monkeypatch, tmp_path):
        kwargs = _problem_kwargs(
            monkeypatch, tmp_path,
            spiking_family="ttfs", spiking_variant="cascaded",
        )
        assert kwargs["ttfs_cycle_schedule"] == "cascaded"


class TestTheRetimingArmSurvivesResolution:
    """The arm no config states by hand.

    ``lif_exact_qat`` is a DEFAULT of the LIF recipe, and it pairs re-timing
    on during resolution — so the ordinary LIF run executes one stage per
    depth level. A step reading the raw JSON would hand the search ``False``
    on exactly those runs and price their wall at a fraction of the truth.
    """

    def test_the_default_lif_recipe_reaches_the_search_re_timed(
        self, monkeypatch, tmp_path,
    ):
        kwargs = _problem_kwargs(monkeypatch, tmp_path, spiking_mode="lif")
        assert kwargs["per_hop_retiming"] is True

    def test_opting_out_of_exact_qat_reaches_the_search_fused(
        self, monkeypatch, tmp_path,
    ):
        """The other arm is reachable, and reaches the search as itself —
        otherwise the pin above would pass on a hardcoded True."""
        kwargs = _problem_kwargs(
            monkeypatch, tmp_path, spiking_mode="lif", lif_exact_qat=False,
        )
        assert kwargs["per_hop_retiming"] is False
