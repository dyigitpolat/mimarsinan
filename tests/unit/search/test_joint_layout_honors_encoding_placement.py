"""A searched candidate is laid out with the placement the run will DEPLOY.

The joint search builds and converts its own models, so it is a second flow-
birth site. It carried the deployment's ``encoding_layer_placement`` nowhere:
the native build hook and the torch conversion hook both resolved the default,
so an ``offload`` run scored every candidate on the SUBSUMED mapping — a
different core count, a different host-segment count, a different packing — and
then deployed the other one.

Both hooks are pinned here with a candidate whose layout genuinely differs
between the placements: hardcoding either back to ``'subsume'`` fails these.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.models.builders.lenet5_builder import LeNet5Builder
from mimarsinan.models.builders.simple_mlp_builder import SimpleMLPBuilder
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    make_platform_resolver,
)
from mimarsinan.search.problems.joint import JointArchHwProblem
from mimarsinan.torch_mapping.encoding_layers import resolved_encoding_placement

#: Wide enough for the 784-fan-in encoder the OFFLOAD candidate must map on chip
#: (a narrower chip refuses it outright instead of laying it out differently).
PIPELINE_CONFIG = {
    "device": "cpu",
    "input_shape": (1, 28, 28),
    "num_classes": 10,
    "target_tq": 4,
    "weight_bits": 4,
    "lr": 0.001,
    "allow_scheduling": True,
    "allow_coalescing": True,
    "cores": [{"max_axons": 784, "max_neurons": 256, "count": 256}],
}

NATIVE_CONFIG = {"mlp_width_1": 64, "mlp_width_2": 32, "base_activation": "ReLU"}
TORCH_CONFIG = {"variant": "lenet5", "base_activation": "ReLU"}


def _problem(builder_factory, model_config, placement: str) -> JointArchHwProblem:
    return JointArchHwProblem(
        data_provider_factory=None,
        device=torch.device("cpu"),
        input_shape=tuple(PIPELINE_CONFIG["input_shape"]),
        num_classes=PIPELINE_CONFIG["num_classes"],
        target_tq=PIPELINE_CONFIG["target_tq"],
        lr=PIPELINE_CONFIG["lr"],
        search_mode="hardware",
        builder_factory=builder_factory,
        arch_options=(),
        model_config_assembler=lambda raw: dict(raw),
        fixed_model_config=dict(model_config),
        platform_resolver=make_platform_resolver(PIPELINE_CONFIG),
        active_objective_names=["total_param_capacity", "param_utilization_pct"],
        num_core_types=1,
        core_axons_bounds=(64, 784),
        core_neurons_bounds=(64, 256),
        core_count_bounds=(8, 256),
        accuracy_seed=0,
        encoding_placement=placement,
    )


def _candidate(builder_factory, model_config, placement: str):
    """The candidate exactly as the search scores it: build, convert, lay out, pack."""
    problem = _problem(builder_factory, model_config, placement)
    pcfg = dict(problem.fixed_platform_constraints or {})
    model, total_params = problem._build_model(dict(model_config), pcfg, problem.encoding_placement)
    softcores, host_segments, _census = problem._collect_softcores(model, pcfg)
    stats, error = problem._pack_candidate(softcores, pcfg)
    return {
        "stamp": resolved_encoding_placement(model.get_mapper_repr()),
        "encoders": sum(
            bool(getattr(p, "is_encoding_layer", False)) for p in model.get_perceptrons()
        ),
        "softcores": len(softcores),
        "host_segments": host_segments,
        "total_params": total_params,
        "stats": stats,
        "error": error,
    }


class TestNativeBuildHookCarriesThePlacement:
    """``_build_raw_model`` -> ``build_model(..., encoding_placement=...)``."""

    @pytest.mark.parametrize("placement", ["subsume", "offload"])
    def test_the_candidate_flow_is_stamped_with_the_runs_placement(self, placement):
        assert _candidate(SimpleMLPBuilder, NATIVE_CONFIG, placement)["stamp"] == placement

    def test_the_two_placements_lay_the_same_model_out_differently(self):
        subsumed = _candidate(SimpleMLPBuilder, NATIVE_CONFIG, "subsume")
        offloaded = _candidate(SimpleMLPBuilder, NATIVE_CONFIG, "offload")

        # Same model, same parameter census — only WHERE the encoder runs moves.
        assert subsumed["total_params"] == offloaded["total_params"]
        assert (subsumed["encoders"], offloaded["encoders"]) == (1, 0)
        assert offloaded["softcores"] > subsumed["softcores"]
        assert offloaded["host_segments"] < subsumed["host_segments"]

    def test_the_packed_candidate_the_search_scores_differs(self):
        subsumed = _candidate(SimpleMLPBuilder, NATIVE_CONFIG, "subsume")
        offloaded = _candidate(SimpleMLPBuilder, NATIVE_CONFIG, "offload")
        assert subsumed["error"] is None and offloaded["error"] is None
        assert offloaded["stats"].total_softcores > subsumed["stats"].total_softcores
        assert offloaded["stats"].total_cores > subsumed["stats"].total_cores


class TestTorchConversionHookCarriesThePlacement:
    """``_convert_to_mapper_repr`` -> ``convert_torch_model(..., encoding_layer_placement=...)``."""

    @pytest.mark.parametrize("placement", ["subsume", "offload"])
    def test_the_converted_candidate_is_stamped_with_the_runs_placement(self, placement):
        assert _candidate(LeNet5Builder, TORCH_CONFIG, placement)["stamp"] == placement

    def test_offloading_the_encoder_conv_grows_the_on_chip_layout(self):
        subsumed = _candidate(LeNet5Builder, TORCH_CONFIG, "subsume")
        offloaded = _candidate(LeNet5Builder, TORCH_CONFIG, "offload")

        assert subsumed["total_params"] == offloaded["total_params"]
        assert (subsumed["encoders"], offloaded["encoders"]) == (1, 0)
        # The encoder conv's weight-bank replication is the whole difference.
        assert offloaded["softcores"] > 2 * subsumed["softcores"]

    def test_the_packed_candidate_the_search_scores_differs(self):
        subsumed = _candidate(LeNet5Builder, TORCH_CONFIG, "subsume")
        offloaded = _candidate(LeNet5Builder, TORCH_CONFIG, "offload")
        assert subsumed["error"] is None and offloaded["error"] is None
        assert offloaded["stats"].total_softcores > subsumed["stats"].total_softcores


class TestTheProblemDefaultIsNotTheDeploymentDefault:
    """A placement the run never configured must not reach the layout hooks."""

    def test_the_search_step_passes_the_configured_placement(self):
        import inspect

        from mimarsinan.pipelining.pipeline_steps.config import architecture_search_step

        source = inspect.getsource(architecture_search_step)
        assert "encoding_placement=str(" in source
        assert 'self.pipeline.config.get("encoding_layer_placement"' in source
