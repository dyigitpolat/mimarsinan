"""A run directory written before the placement stamp existed stays resumable.

The stamp lives on the ``ModelRepresentation``, so every flow cached before it
existed deserializes with no ``encoding_placement`` attribute at all — which is
exactly what the placement guard refuses. Resume (``start_step``/``stop_step``)
is a first-class feature, so a legacy artifact must RESOLVE, not die; and it
must resolve by having the configured placement APPLIED, not by being waved
through (a legacy native flow really never had one applied — that was the
defect this unit exists to fix).

The one thing it must not do is erase a host placement another writer owns: the
negative-boundary subsume-forward policy marks perceptrons that placement does
not own, and those survive.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.mapping.platform.packaging_contract import (
    MVM_PACKAGING,
    SPIKING_PACKAGING,
)
from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model
from mimarsinan.pipelining.cache.legacy_placement import (
    resolve_cached_flow_placements,
)
from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache
from mimarsinan.torch_mapping.encoding_layers import (
    PLACEMENT_NOT_APPLICABLE,
    UnresolvedEncodingPlacementError,
    require_resolved_encoding_placement,
    resolved_encoding_placement,
)

INPUT_SHAPE = (1, 28, 28)
NUM_CLASSES = 10
MLP_CONFIG = {"mlp_width_1": 64, "mlp_width_2": 32}


def _flow(placement: str = "subsume"):
    builder = BUILDERS_REGISTRY["simple_mlp"]("cpu", INPUT_SHAPE, NUM_CLASSES, {})
    model = build_model(builder, MLP_CONFIG, encoding_placement=placement)
    with torch.no_grad():
        model.train()
        model(torch.randn(2, *INPUT_SHAPE))
    return model


def _make_legacy(model):
    """Exactly what a pre-stamp pickle restores: the attribute never existed."""
    mapper_repr = model.get_mapper_repr()
    del mapper_repr.encoding_placement
    assert resolved_encoding_placement(mapper_repr) is None
    return model


def _encoders(model) -> list[bool]:
    return [bool(getattr(p, "is_encoding_layer", False)) for p in model.get_perceptrons()]


def _cache_with(model) -> PipelineCache:
    cache = PipelineCache()
    cache.add("Model Building/model", model, "torch_model")
    cache.add("Model Building/model_config", dict(MLP_CONFIG), "basic")
    return cache


class TestTheLegacyArtifactWasBroken:
    def test_an_unstamped_flow_is_refused_by_the_guard(self):
        legacy = _make_legacy(_flow("subsume"))
        with pytest.raises(UnresolvedEncodingPlacementError, match="never resolved"):
            require_resolved_encoding_placement(
                legacy.get_mapper_repr(), "subsume", context="the on-chip fraction gate"
            )


class TestResumeResolvesTheLegacyArtifact:
    @pytest.mark.parametrize("placement", ["subsume", "offload"])
    def test_the_configured_placement_is_applied_and_stamped(self, placement):
        legacy = _make_legacy(_flow("subsume"))
        keys = resolve_cached_flow_placements(
            _cache_with(legacy), placement=placement, packaging=SPIKING_PACKAGING,
        )
        assert keys == ["Model Building/model"]
        assert resolved_encoding_placement(legacy.get_mapper_repr()) == placement
        require_resolved_encoding_placement(
            legacy.get_mapper_repr(), placement, context="the on-chip fraction gate"
        )

    def test_offload_actually_moves_the_baked_encoder_on_chip(self):
        """Not a rubber stamp: the legacy marking said host, offload says chip."""
        legacy = _make_legacy(_flow("subsume"))
        assert any(_encoders(legacy)), "fixture must carry a host-marked encoder"
        resolve_cached_flow_placements(
            _cache_with(legacy), placement="offload", packaging=SPIKING_PACKAGING,
        )
        assert not any(_encoders(legacy))

    def test_subsume_puts_the_encoder_host_side_on_an_unmarked_legacy_flow(self):
        legacy = _make_legacy(_flow("offload"))
        assert not any(_encoders(legacy))
        resolve_cached_flow_placements(
            _cache_with(legacy), placement="subsume", packaging=SPIKING_PACKAGING,
        )
        assert any(_encoders(legacy))

    def test_a_value_domain_resume_is_stamped_not_applicable(self):
        legacy = _make_legacy(_flow("subsume"))
        resolve_cached_flow_placements(
            _cache_with(legacy), placement="subsume", packaging=MVM_PACKAGING,
        )
        assert (
            resolved_encoding_placement(legacy.get_mapper_repr())
            == PLACEMENT_NOT_APPLICABLE
        )


class TestItTouchesNothingElse:
    def test_an_already_stamped_flow_is_left_alone(self):
        model = _flow("subsume")
        before = _encoders(model)
        keys = resolve_cached_flow_placements(
            _cache_with(model), placement="offload", packaging=SPIKING_PACKAGING,
        )
        assert keys == []
        assert resolved_encoding_placement(model.get_mapper_repr()) == "subsume"
        assert _encoders(model) == before

    def test_a_cache_without_flows_resolves_nothing(self):
        cache = PipelineCache()
        cache.add("Model Building/model_config", dict(MLP_CONFIG), "basic")
        cache.add("Pretraining/model", torch.nn.Linear(4, 4), "torch_model")
        assert resolve_cached_flow_placements(
            cache, placement="offload", packaging=SPIKING_PACKAGING,
        ) == []

    def test_a_subsume_forward_host_placement_survives_an_offload_resume(self):
        """The negative-boundary policy owns perceptrons placement does not.

        A blanket re-mark would clear them and silently put a negative host
        boundary back on chip; the resume path only writes the encoding-segment
        starts.
        """
        legacy = _make_legacy(_flow("subsume"))
        perceptrons = legacy.get_perceptrons()
        assert len(perceptrons) >= 2
        deep = perceptrons[-1]  # not a segment start: another perceptron feeds it
        deep.is_encoding_layer = True

        resolve_cached_flow_placements(
            _cache_with(legacy), placement="offload", packaging=SPIKING_PACKAGING,
        )
        assert deep.is_encoding_layer, (
            "an offload resume erased a negative-boundary subsume-forward host "
            "placement it does not own"
        )
        assert not perceptrons[0].is_encoding_layer


class TestTheDeploymentPipelineIsWiredToIt:
    """The recovery must run for real, at pipeline construction.

    ``Pipeline.__init__`` loads the cache before the config exists, and every
    later seam is after a step could have added a host placement — so this is
    the one point the resolution can happen, and it has to be wired there.
    """

    @staticmethod
    def _pipeline(tmp_path, placement: str):
        from conftest import MockDataProviderFactory

        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import (
            DeploymentPipeline,
        )

        reporter = type(
            "R", (), {
                "report": lambda *a, **kw: None,
                "console_log": lambda *a, **kw: None,
                "finish": lambda *a, **kw: None,
            },
        )()
        return DeploymentPipeline(
            data_provider_factory=MockDataProviderFactory(),
            deployment_parameters={"encoding_layer_placement": placement},
            platform_constraints={
                "cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}],
                "target_tq": 4,
                "weight_bits": 8,
            },
            reporter=reporter,
            working_directory=str(tmp_path),
        )

    @pytest.mark.parametrize("placement", ["subsume", "offload"])
    def test_constructing_the_pipeline_resolves_a_legacy_cached_flow(
        self, tmp_path, placement
    ):
        cache = _cache_with(_make_legacy(_flow("subsume")))
        cache.store(str(tmp_path))

        pipeline = self._pipeline(tmp_path, placement)

        model = pipeline.cache.get("Model Building/model")
        assert model is not None
        assert resolved_encoding_placement(model.get_mapper_repr()) == placement
        assert any(_encoders(model)) == (placement == "subsume")


class TestRoundTripThroughTheRealCache:
    def test_a_stored_and_reloaded_legacy_flow_resolves(self, tmp_path):
        """End to end over the on-disk artifact, not just the in-memory object."""
        stored = _make_legacy(_flow("subsume"))
        cache = _cache_with(stored)
        cache.store(str(tmp_path))

        reloaded = PipelineCache()
        reloaded.load(str(tmp_path))
        model = reloaded.get("Model Building/model")
        assert resolved_encoding_placement(model.get_mapper_repr()) is None

        keys = resolve_cached_flow_placements(
            reloaded, placement="offload", packaging=SPIKING_PACKAGING,
        )
        assert keys == ["Model Building/model"]
        assert resolved_encoding_placement(model.get_mapper_repr()) == "offload"
        assert not any(_encoders(model))
