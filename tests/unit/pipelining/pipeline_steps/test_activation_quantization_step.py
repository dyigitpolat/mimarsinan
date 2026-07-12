"""ActivationQuantizationStep: the ``aq_reference_read`` promise (R2b).

The step stashes its post-AQ VALIDATION read as a plain float entry — the
premise teacher for the LIF affine fold (lossless_refinement_ledger.md §2D).
Validation domain on purpose: the fold premise differences it against a
validation-batch calibration read, and test reads are never differenced
against eval-subset reads (ledger conventions).
"""

from __future__ import annotations

import json

from conftest import MockPipeline, default_config

from mimarsinan.pipelining.pipeline_steps.quantization.activation_quantization_step import (
    ActivationQuantizationStep,
)


class _StubTuner:
    def __init__(self, value):
        self._value = value

    def validate(self):
        return self._value


def _run_stubbed_step(monkeypatch, validation_read):
    pipeline = MockPipeline(config=default_config())
    pipeline.seed("model", object(), step_name="Torch Mapping")
    pipeline.seed("adaptation_manager", object(), step_name="Torch Mapping")

    def fake_run_tuner(self, tuner_cls, model, adaptation_manager, **kwargs):
        self.tuner = _StubTuner(validation_read)
        self._commit_tuner_entries(model, adaptation_manager)

    monkeypatch.setattr(ActivationQuantizationStep, "run_tuner", fake_run_tuner)
    step = ActivationQuantizationStep(pipeline)
    step.name = "Activation Quantization"
    pipeline.prepare_step(step)
    step.run()
    return pipeline


class TestAqReferencePromise:
    def test_promises_the_reference_entry(self):
        assert "aq_reference_read" in ActivationQuantizationStep.PROMISES

    def test_stashes_the_tuner_validation_read_as_a_plain_float(self, monkeypatch):
        pipeline = _run_stubbed_step(monkeypatch, validation_read=0.9812)
        value = pipeline.cache["Activation Quantization.aq_reference_read"]
        assert isinstance(value, float)
        assert value == 0.9812
        # A plain scalar entry (basic/JSON strategy), never a model copy.
        assert json.dumps(value)

    def test_coerces_tensor_like_reads_to_float(self, monkeypatch):
        import torch

        pipeline = _run_stubbed_step(
            monkeypatch, validation_read=torch.tensor(0.5),
        )
        value = pipeline.cache["Activation Quantization.aq_reference_read"]
        assert isinstance(value, float)
        assert value == 0.5
