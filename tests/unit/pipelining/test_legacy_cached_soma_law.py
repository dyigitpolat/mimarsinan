"""[U-H targeted pass, review M3] A LIF cache written before the soma axes existed stays resumable at the default point.

LIF Adaptation pickles its ``ChipAlignedNFForward`` into the step cache as the model's instance ``forward`` (for every
LIF run: ``cycle_accurate_lif_forward`` defaults on). The port made that forward's point default-FREE, so an instance
unpickled with no ``soma_law`` refuses at call time instead of silently running the default point. That is right for
the forward alone, but it made every pre-port LIF cache unresumable, although such a pickle was written by a tree that
knew only the default point.

The point is therefore resolved at cache load, beside the legacy placement (``DeploymentPipeline._initialize_config``:
the cache is loaded, the run's point is resolved, and no step has run). A forward without a point is bound to the run's
resolved law when that law is a default point, and refused by name when the run declares any other point, whose
physics the pickle does not carry.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw
from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal
from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache
from mimarsinan.tuning.forward_install import ChipAlignedNFForward, SomaLawMissingError

from unit.models.spiking.serial.test_serial_executors import PER_EVENT_LAW, T, build_chain
from unit.models.spiking.serial.test_serial_nf_twin import _nf_model

KEY = "LIF Adaptation.model"

#: A default point that is not the bare default: the comparator and reset are config keys every pre-axes tree read.
DEFAULT_POINT_RUN = SomaLaw.resolve({"thresholding_mode": "<=", "firing_mode": "Novena"})


def _installed(soma_law: SomaLaw = DEFAULT_SOMA_LAW) -> nn.Module:
    """The serial chain's NF model with the LIF install's forward at ``soma_law`` (what LIF Adaptation caches)."""
    model = _nf_model(build_chain()[0])
    model.forward = ChipAlignedNFForward(model, T, soma_law=soma_law)
    return model


def _pre_axes(model: nn.Module) -> nn.Module:
    """Exactly what a pre-axes pickle restores: the forward's ``soma_law`` attribute never existed."""
    del model.__dict__["forward"].__dict__["soma_law"]
    return model


def _sample() -> torch.Tensor:
    torch.manual_seed(3)
    return torch.rand(4, 8) * 0.9


def resolve_cached_forward_soma_laws(cache, *, soma_law):
    """The resolver under test, imported per call so each test reads red or green on its own."""
    from mimarsinan.pipelining.cache.legacy_soma_law import resolve_cached_forward_soma_laws as resolve

    return resolve(cache, soma_law=soma_law)


def _cache_with(model: nn.Module) -> PipelineCache:
    cache = PipelineCache()
    cache.add(KEY, model, "torch_model")
    cache.add("LIF Adaptation.adaptation_manager", {"lif_active": True}, "basic")
    return cache


class TestThePreAxesArtifactWasUnresumable:
    def test_the_forward_alone_still_refuses_without_a_point(self):
        """The forward never re-defaults on read; only the cache load, which knows the run's point, may bind it."""
        model = _pre_axes(_installed())
        with pytest.raises(SomaLawMissingError, match="no soma_law"):
            model(_sample())

    def test_the_refusal_is_a_refusal_by_name(self):
        assert issubclass(SomaLawMissingError, ToolchainRefusal)
        assert issubclass(SomaLawMissingError, RuntimeError)


class TestTheCacheLoadResolvesIt:
    def test_a_default_point_run_binds_the_run_s_law_and_runs_the_default_walk(self):
        cache = _cache_with(_pre_axes(_installed()))
        assert resolve_cached_forward_soma_laws(cache, soma_law=DEFAULT_SOMA_LAW) == [KEY]
        model = cache.get(KEY)
        assert model.forward.soma_law == DEFAULT_SOMA_LAW
        with torch.no_grad():
            resumed = model(_sample())
            fresh = _installed(DEFAULT_SOMA_LAW)(_sample())
        assert torch.equal(resumed, fresh)

    def test_the_bound_law_is_the_one_the_lif_install_would_record_for_this_run(self):
        """A default point carries the run's comparator and reset (keys that predate the axes), so the run's own law
        is bound, not the bare default: exactly what LIF Adaptation installs (``SomaLaw.resolve(pipeline.config)``)."""
        assert DEFAULT_POINT_RUN.is_default_point and DEFAULT_POINT_RUN != DEFAULT_SOMA_LAW
        cache = _cache_with(_pre_axes(_installed()))
        assert resolve_cached_forward_soma_laws(cache, soma_law=DEFAULT_POINT_RUN) == [KEY]
        assert cache.get(KEY).forward.soma_law == DEFAULT_POINT_RUN

    def test_a_non_default_point_run_refuses_by_name(self):
        cache = _cache_with(_pre_axes(_installed()))
        with pytest.raises(SomaLawMissingError) as refused:
            resolve_cached_forward_soma_laws(cache, soma_law=PER_EVENT_LAW)
        message = str(refused.value)
        assert KEY in message and "per_event" in message and "quarantine_entry" in message
        assert isinstance(refused.value, ToolchainRefusal)
        assert "soma_law" not in cache.get(KEY).forward.__dict__, "a refused forward is never bound"

    @pytest.mark.parametrize("run_law", [DEFAULT_SOMA_LAW, PER_EVENT_LAW])
    def test_a_forward_that_carries_its_point_is_never_rebound(self, run_law):
        cache = _cache_with(_installed(PER_EVENT_LAW))
        assert resolve_cached_forward_soma_laws(cache, soma_law=run_law) == []
        assert cache.get(KEY).forward.soma_law == PER_EVENT_LAW

    def test_a_forward_installed_on_a_submodule_is_found_too(self):
        outer = nn.Sequential(_pre_axes(_installed()))
        cache = _cache_with(outer)
        assert resolve_cached_forward_soma_laws(cache, soma_law=DEFAULT_SOMA_LAW) == [KEY]
        assert outer[0].forward.soma_law == DEFAULT_SOMA_LAW

    def test_a_cache_without_a_pre_axes_forward_is_a_no_op(self):
        assert resolve_cached_forward_soma_laws(PipelineCache(), soma_law=PER_EVENT_LAW) == []
        plain = PipelineCache()
        plain.add("Pretraining.model", nn.Linear(2, 2), "torch_model")
        plain.add("Model Configuration.model_config", {"a": 1}, "basic")
        assert resolve_cached_forward_soma_laws(plain, soma_law=PER_EVENT_LAW) == []


class TestRoundTripThroughTheRealCache:
    def test_a_stored_and_reloaded_pre_axes_forward_resolves_and_runs(self, tmp_path):
        """End to end over the on-disk artifact: the pickle carries no point, the reload refuses, the resolve binds."""
        _cache_with(_pre_axes(_installed())).store(str(tmp_path))
        reloaded = PipelineCache()
        reloaded.load(str(tmp_path))
        model = reloaded.get(KEY)
        assert "soma_law" not in model.forward.__dict__
        with pytest.raises(SomaLawMissingError):
            model(_sample())
        assert resolve_cached_forward_soma_laws(reloaded, soma_law=DEFAULT_SOMA_LAW) == [KEY]
        with torch.no_grad():
            assert torch.equal(model(_sample()), _installed(DEFAULT_SOMA_LAW)(_sample()))


class TestTheDeploymentPipelineIsWiredToIt:
    """The resolution must run for real at pipeline construction, with the run's own resolved point."""

    @staticmethod
    def _pipeline(tmp_path, deployment_parameters: dict):
        from conftest import MockDataProviderFactory

        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline

        reporter = type("R", (), {"report": lambda *a, **kw: None, "console_log": lambda *a, **kw: None,
                                  "finish": lambda *a, **kw: None})()
        return DeploymentPipeline(
            data_provider_factory=MockDataProviderFactory(),
            deployment_parameters=dict(deployment_parameters),
            platform_constraints={"cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}],
                                  "target_tq": 4, "weight_bits": 8},
            reporter=reporter,
            working_directory=str(tmp_path),
        )

    def test_constructing_the_pipeline_binds_a_pre_axes_cached_forward(self, tmp_path):
        _cache_with(_pre_axes(_installed())).store(str(tmp_path))
        pipeline = self._pipeline(tmp_path, {})
        law = pipeline.cache.get(KEY).forward.soma_law
        assert law == SomaLaw.resolve(pipeline.config) and law.is_default_point

    def test_the_pipeline_passes_its_own_resolved_point(self, tmp_path, monkeypatch):
        import mimarsinan.pipelining.core.pipelines.deployment_pipeline as dp

        seen = []
        monkeypatch.setattr(dp, "resolve_cached_forward_soma_laws",
                            lambda cache, *, soma_law: seen.append(soma_law) or [])
        pipeline = self._pipeline(tmp_path, {"thresholding_mode": "<="})
        assert seen == [SomaLaw.resolve(pipeline.config)]
        assert seen[0].thresholding_mode == "<="
