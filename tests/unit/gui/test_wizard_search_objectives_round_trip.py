"""Honest chips, end to end: what the wizard can offer is what the step accepts.

The chip list in ``wizard/structured.js`` filters ``nas.objective_options`` by
the ``available_in_modes`` rows of ``nas.objective_catalog``. The registry now
ABORTS a run on an axis the mode cannot measure, so that filter is the only
thing standing between a hardware-only draft and a dead run. These tests walk
the whole trip the wizard's own payload takes — schema payload -> JS filter
(reproduced here from the served data) -> emitted config -> ``derive_search_mode``
-> ``ArchitectureSearchStep`` -> ``resolve_active`` — and pin that the unfiltered
list would in fact abort, so the filter can never quietly stop being load-bearing.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from conftest import MockPipeline, default_config

from mimarsinan.deployment_record.objectives import OBJECTIVES, SEARCH_MODES
from mimarsinan.gui.wizard.config_builder import build_deployment_config_from_state
from mimarsinan.gui.wizard.schema import get_wizard_nas_schema
from mimarsinan.pipelining.core.search_mode import derive_search_mode
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_step import (
    ArchitectureSearchStep,
)
from mimarsinan.search.results import objectives_for_mode, resolve_active_objectives


def offered_objective_ids(search_mode: str) -> List[str]:
    """The chips ``structured.js`` renders for *search_mode*, from the served payload.

    Mirrors the JS one-for-one, escape hatch included: an option with no
    catalog row (``!modes``) is offered in EVERY mode. Reproducing the hole is
    the point — a missing row is how an unavailable axis would reach the step.
    """
    nas = get_wizard_nas_schema()
    availability = {
        row["id"]: row["available_in_modes"] for row in nas["objective_catalog"]
    }
    return [
        option["id"] for option in nas["objective_options"]
        if option["id"] not in availability
        or search_mode in availability[option["id"]]
    ]


ARCH_SEARCH_BUDGET: Dict[str, Any] = {
    "optimizer": "nsga2",
    "pop_size": 4,
    "generations": 2,
    "seed": 0,
    "num_core_types": 1,
    "core_axons_bounds": [64, 256],
    "core_neurons_bounds": [64, 256],
    "core_count_bounds": [8, 64],
}


def hardware_only_draft(objectives: List[str]) -> Dict[str, Any]:
    """The wizard state a hardware-only co-search draft produces."""
    return {
        "pipeline_mode": "vanilla",
        "experiment_name": "w53_round_trip",
        "deployment_parameters": {
            "model_config_mode": "user",
            "hw_config_mode": "search",
            "model_type": "simple_mlp",
            "model_config": {
                "mlp_width_1": 16, "mlp_width_2": 16, "base_activation": "ReLU",
            },
            "spiking_family": "lif",
            "arch_search": {**ARCH_SEARCH_BUDGET, "objectives": objectives},
        },
        "platform_constraints": {
            "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
            "weight_bits": 4,
        },
    }


def run_search_step(tmp_path, objectives: List[str]):
    emitted = build_deployment_config_from_state(hardware_only_draft(objectives))
    flat = {
        **default_config(),
        **emitted["deployment_parameters"],
        **emitted["platform_constraints"],
    }
    assert derive_search_mode(flat) == "hardware"

    pipeline = MockPipeline(
        config=flat, working_directory=str(tmp_path / "round_trip"),
    )
    step = ArchitectureSearchStep(pipeline)
    step.name = "ArchitectureSearch"
    pipeline.prepare_step(step)
    step.run()
    return pipeline.cache["ArchitectureSearch.architecture_search_result"]


class TestTheServedPayloadCannotOfferAnUnavailableAxis:
    def test_every_offerable_option_has_an_availability_row(self):
        # The JS treats a MISSING row as "available everywhere"; an option
        # without one would put an unavailable chip in front of the user.
        nas = get_wizard_nas_schema()
        rows = {row["id"] for row in nas["objective_catalog"]}
        missing = [o["id"] for o in nas["objective_options"] if o["id"] not in rows]
        assert not missing, f"options with no availability row: {missing}"

    @pytest.mark.parametrize("search_mode", sorted(SEARCH_MODES))
    def test_catalog_availability_is_the_registrys_own_answer(self, search_mode):
        available = {spec.name for spec in objectives_for_mode(search_mode)}
        served = {
            row["id"] for row in get_wizard_nas_schema()["objective_catalog"]
            if search_mode in row["available_in_modes"]
        }
        assert served == available

    @pytest.mark.parametrize("search_mode", sorted(SEARCH_MODES))
    def test_the_offered_set_resolves_without_a_word(self, search_mode):
        offered = offered_objective_ids(search_mode)
        assert offered, f"{search_mode} must have something to offer"
        resolved = resolve_active_objectives(search_mode, offered)
        assert [spec.name for spec in resolved] == offered

    def test_hardware_only_search_never_offers_the_training_proxy(self):
        assert "estimated_accuracy" not in offered_objective_ids("hardware")
        assert "estimated_accuracy" in offered_objective_ids("model")


class TestTheFilterIsLoadBearing:
    def test_the_unfiltered_option_list_would_abort_a_hardware_search(self):
        # If this ever stops raising, the chip filter has stopped mattering and
        # the tests above are checking nothing.
        unfiltered = [o["id"] for o in get_wizard_nas_schema()["objective_options"]]
        assert "estimated_accuracy" in unfiltered
        with pytest.raises(ValueError) as excinfo:
            resolve_active_objectives("hardware", unfiltered)
        message = str(excinfo.value)
        assert "estimated_accuracy" in message
        assert OBJECTIVES.get("estimated_accuracy").requires in message

    def test_an_unavailable_axis_cannot_reach_the_step(self, tmp_path):
        with pytest.raises(ValueError, match="estimated_accuracy"):
            run_search_step(tmp_path, ["fragmentation_pct", "estimated_accuracy"])


class TestTheOfferedChipsSurviveTheWholeTrip:
    def test_the_step_optimizes_exactly_what_the_wizard_offered(self, tmp_path):
        offered = offered_objective_ids("hardware")
        result = run_search_step(tmp_path, offered)
        assert result["search_mode_used"] == "hardware"
        assert result["active_objectives"] == offered
        assert set(result["best"]["objectives"]) == set(offered)
        assert result["discovered_platform_constraints"] is not None

    def test_a_single_offered_chip_also_survives(self, tmp_path):
        result = run_search_step(tmp_path, ["fragmentation_pct"])
        assert result["active_objectives"] == ["fragmentation_pct"]
        assert set(result["best"]["objectives"]) == {"fragmentation_pct"}
