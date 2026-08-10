"""W3-S1 zero-draw pin: adaptation instrumentation adds NO validation draws.

The retention-ledger exit read must reuse what the run already measured
(``TunerBase.exit_metric_estimate``: cached validate metric, else the last
single-batch validation read, else the ``_after_run`` final metric) — NEVER a
fresh ``validate()`` at commit time. A fresh read would draw one extra
validation batch for every non-caching family, advancing the loader round-robin
and breaking the "artifacts only, byte-identical training path" contract.

The pin: two identically-seeded stepped runs of a REAL tuner family — one with
instrumentation off (no working directory: the incumbent path) and one with it
on — must consume EXACTLY the same number of validation-loader draws (spied at
``BasicTrainer.next_validation_batch``). Restoring ``float(self.validate())``
in ``_persist_adaptation_instrumentation`` fails the non-caching-family pin.
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import (
    MockPipeline,
    default_config,
    make_activation_scale_stats,
    make_tiny_supermodel,
)

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.pipelining.pipeline_steps.adaptation.clamp_adaptation_step import (
    ClampAdaptationStep,
)
from mimarsinan.pipelining.pipeline_steps.adaptation.noise_adaptation_step import (
    NoiseAdaptationStep,
)
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.tuner_base import TunerBase


@pytest.fixture
def draw_counter(monkeypatch):
    """Count every validation-loader draw made through any BasicTrainer."""
    counts = {"n": 0}
    original = BasicTrainer.next_validation_batch

    def spy(self):
        counts["n"] += 1
        return original(self)

    monkeypatch.setattr(BasicTrainer, "next_validation_batch", spy)
    return counts


def _reseed():
    """Identical RNG entry for each compared run: same data, same init, same
    trajectory — so draw counts differ ONLY if instrumentation itself draws."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def _pipeline(working_directory):
    pipeline = MockPipeline(config=default_config())
    # None = the incumbent path: the persistence seam no-ops entirely.
    pipeline.working_directory = working_directory
    pipeline._target_metric = 0.25
    return pipeline


def _run_noise_step(working_directory):
    """A REAL non-caching family: NoiseTuner.validate() is TunerBase's fresh
    single-batch read (no cached-metric override)."""
    _reseed()
    pipeline = _pipeline(working_directory)
    model = make_tiny_supermodel()
    pipeline.seed("model", model, step_name="Prev")
    pipeline.seed("adaptation_manager", AdaptationManager(), step_name="Prev")
    step = NoiseAdaptationStep(pipeline)
    step.name = "Noise Adaptation"
    pipeline.prepare_step(step)
    step.pipeline_previous_metric = 0.25
    step.run()
    step.cleanup()
    return step


def _run_clamp_step(working_directory):
    """A REAL caching family: ClampTuner serves its cached ``_final_metric``
    through ``cached_validate_metric``."""
    _reseed()
    pipeline = _pipeline(working_directory)
    pipeline.config["activation_quantization"] = True
    model = make_tiny_supermodel()
    scales = [1.0] * len(model.get_perceptrons())
    stats = make_activation_scale_stats(model, scales, num_batches=2)
    pipeline.seed("model", model, step_name="Activation Adaptation")
    pipeline.seed("adaptation_manager", AdaptationManager(), step_name="Activation Adaptation")
    pipeline.seed("activation_scales", scales, step_name="Activation Analysis")
    pipeline.seed("activation_scale_stats", stats, step_name="Activation Analysis")
    step = ClampAdaptationStep(pipeline)
    step.name = "Clamp Adaptation"
    pipeline.prepare_step(step)
    step.pipeline_previous_metric = 0.25
    step.run()
    step.cleanup()
    return step


def _ledger_entries(tmp_path):
    return json.loads((tmp_path / "retention_ledger.json").read_text())["entries"]


class TestDrawParity:
    def test_non_caching_family_zero_extra_draws(
        self, deterministic_rng, draw_counter, tmp_path
    ):
        _run_noise_step(None)
        incumbent = draw_counter["n"]
        assert incumbent > 0, (
            "the non-caching family must read validation batches during its "
            "run, or the parity assertion is vacuous"
        )

        draw_counter["n"] = 0
        step = _run_noise_step(str(tmp_path))
        instrumented = draw_counter["n"]

        assert instrumented == incumbent, (
            "instrumentation must not draw a single extra validation batch "
            f"(incumbent={incumbent}, instrumented={instrumented}); a fresh "
            "validate() at commit time is the exact defect this test pins"
        )
        # Non-vacuous: the instrumented run DID persist the retention entry,
        # and its exit metric is the run's own last single-batch read.
        entries = _ledger_entries(tmp_path)
        assert [e["step"] for e in entries] == ["Noise Adaptation"]
        assert entries[0]["exit_metric"] == pytest.approx(
            step.tuner.trainer.last_validation_accuracy
        )

    def test_caching_family_zero_extra_draws(
        self, deterministic_rng, draw_counter, tmp_path
    ):
        _run_clamp_step(None)
        incumbent = draw_counter["n"]

        draw_counter["n"] = 0
        step = _run_clamp_step(str(tmp_path))
        instrumented = draw_counter["n"]

        assert instrumented == incumbent
        entries = _ledger_entries(tmp_path)
        assert [e["step"] for e in entries] == ["Clamp Adaptation"]
        # The caching family's exit metric is the cached full-val metric.
        assert step.tuner._final_metric is not None
        assert entries[0]["exit_metric"] == pytest.approx(step.tuner._final_metric)


class TestExitMetricEstimateResolution:
    """TunerBase.exit_metric_estimate: zero-draw by construction."""

    @staticmethod
    def _bare(last_read=None, final=None, cached=None):
        tuner = TunerBase.__new__(TunerBase)
        tuner.trainer = SimpleNamespace(
            last_validation_accuracy=last_read,
            validate=TestExitMetricEstimateResolution._must_not_read,
        )
        if final is not None:
            tuner._final_metric = final
        if cached is not None:
            tuner.cached_validate_metric = lambda: cached
        return tuner

    @staticmethod
    def _must_not_read():
        raise AssertionError("exit_metric_estimate must never read the loader")

    def test_cached_metric_wins(self):
        tuner = self._bare(last_read=0.5, final=0.6, cached=0.7)
        assert tuner.exit_metric_estimate() == pytest.approx(0.7)

    def test_last_single_batch_read_when_no_cache(self):
        tuner = self._bare(last_read=0.5, final=0.6)
        assert tuner.exit_metric_estimate() == pytest.approx(0.5)

    def test_final_metric_when_never_single_batch_validated(self):
        tuner = self._bare(final=0.6)
        assert tuner.exit_metric_estimate() == pytest.approx(0.6)

    def test_none_when_nothing_was_measured(self):
        assert self._bare().exit_metric_estimate() is None

    def test_base_validate_still_reads_fresh(self):
        # The generic cached-else-fresh dispatch must NOT cache for base
        # families: validate() stays a fresh read (only the estimate is free).
        tuner = TunerBase.__new__(TunerBase)
        tuner.trainer = SimpleNamespace(validate=lambda: 0.42)
        assert tuner.validate() == pytest.approx(0.42)


class TestValidationReadStash:
    def test_validate_measured_records_the_read(self, deterministic_rng):
        _reseed()
        pipeline = _pipeline(None)
        tuner = TunerBase(pipeline, make_tiny_supermodel(), 0.9, 0.001)
        try:
            assert tuner.trainer.last_validation_accuracy is None
            acc = tuner.trainer.validate()
            assert tuner.trainer.last_validation_accuracy == pytest.approx(acc)
        finally:
            tuner.close()
