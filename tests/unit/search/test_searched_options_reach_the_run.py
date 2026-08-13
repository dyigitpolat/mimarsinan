"""A searched deployment option must become the run's — end to end, through the step."""

import pytest

from conftest import MockPipeline, default_config

from mimarsinan.pipelining.pipeline_steps.config.architecture_search_step import (
    ArchitectureSearchStep,
)

_ARCH_SEARCH = {
    "optimizer": "nsga2",
    "pop_size": 4,
    "generations": 2,
    "seed": 0,
    "num_core_types": 1,
    "core_axons_bounds": [64, 256],
    "core_neurons_bounds": [64, 256],
    "core_count_bounds": [8, 64],
    "objectives": ["total_param_capacity", "fragmentation_pct"],
}


def _run(tmp_path, option_axes, **over):
    config = {
        **default_config(),
        "model_config_mode": "user",
        "hw_config_mode": "search",
        "model_type": "simple_mlp",
        "model_config": {
            "mlp_width_1": 16, "mlp_width_2": 16, "base_activation": "ReLU",
        },
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
        "encoding_layer_placement": "subsume",
        "arch_search": {**_ARCH_SEARCH, "option_axes": option_axes},
    }
    config.update(over)
    pipeline = MockPipeline(config=config, working_directory=str(tmp_path / "opt"))
    step = ArchitectureSearchStep(pipeline)
    step.name = "ArchitectureSearch"
    pipeline.prepare_step(step)
    step.run()
    return pipeline, pipeline.cache["ArchitectureSearch.architecture_search_result"]


def test_a_search_without_option_axes_declares_none(tmp_path):
    pipeline, result = _run(tmp_path, None)
    assert result["discovered_deployment_options"] == {}
    assert pipeline.config["encoding_layer_placement"] == "subsume"


def test_a_searched_option_is_reported_as_discovered(tmp_path):
    _pipeline, result = _run(tmp_path, ["encoding_layer_placement"])
    discovered = result["discovered_deployment_options"]
    assert set(discovered) == {"encoding_layer_placement"}
    assert discovered["encoding_layer_placement"] in ("subsume", "offload")


def test_the_winners_option_becomes_the_runs_config(tmp_path):
    """The deployed run must execute under the options the winner was SCORED with,
    or the deployed thing is not the thing the search chose.

    The axis is narrowed to the value the run does NOT declare, so the stamp has
    to actually write something — a search that quietly kept the declaration would
    otherwise pass whenever the winner happened to agree with it.
    """
    pipeline, result = _run(
        tmp_path, {"encoding_layer_placement": ["offload"]},
        encoding_layer_placement="subsume",
    )
    assert result["discovered_deployment_options"] == {
        "encoding_layer_placement": "offload"
    }
    assert pipeline.config["encoding_layer_placement"] == "offload"


def test_a_searched_platform_option_reaches_the_resolved_chip(tmp_path):
    _pipeline, result = _run(tmp_path, {"weight_bits": {"bounds": [4, 8]}})
    discovered = result["discovered_deployment_options"]
    assert discovered["weight_bits"] in (4, 5, 6, 7, 8)
    assert (
        result["discovered_platform_constraints"]["weight_bits"]
        == discovered["weight_bits"]
    )


def test_the_constraint_census_is_reported(tmp_path):
    """A search that rejected candidates on a declared floor must say so, rather
    than leaving the rejections buried in undifferentiated penalties."""
    _pipeline, result = _run(tmp_path, ["encoding_layer_placement"])
    assert "constraint_census" in result
    assert isinstance(result["constraint_census"], dict)


def test_an_undeclarable_axis_fails_loud_at_launch(tmp_path):
    with pytest.raises((KeyError, ValueError), match="not_a_config_key"):
        _run(tmp_path, ["not_a_config_key"])
