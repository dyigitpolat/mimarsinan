"""Regression tests for Phase A1: no test-set data leak inside tuner decisions.

The pipeline enforces its hard floor via ``trainer.test()`` called from
``PipelineStep.pipeline_metric()`` AFTER a step completes. Tuners must never
call ``trainer.test()`` from inside their decision logic (rate commits,
rollbacks, LR probes, per-cycle gates). Doing so leaks test labels into the
training loop, invalidating both the tuner's self-measurement and the
pipeline's step-level integrity check.

These tests lock the invariant by inspecting source (for files we don't want
to import without heavy deps) and by instrumenting the trainer during a
short tuner run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from conftest import MockPipeline, make_tiny_supermodel
from mimarsinan.tuning.adaptation_manager import AdaptationManager


SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"


def _strip_comments_and_docstrings(text: str) -> str:
    """Remove Python comments and triple-quoted string literals.

    Not a full Python parser -- sufficient for our files (no f-string
    embedded triple quotes etc.).
    """
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    text = re.sub(r"'''(?:.|\n)*?'''", "", text)
    out_lines = []
    for line in text.splitlines():
        idx = line.find("#")
        if idx >= 0:
            line = line[:idx]
        out_lines.append(line)
    return "\n".join(out_lines)


_TUNER_FILES_WITHOUT_TEST_CALLS = [
    "tuning/unified_tuner.py",
    "tuning/tuners/clamp_tuner.py",
    "tuning/tuners/pruning_tuner.py",
    "tuning/tuners/activation_shift_tuner.py",
    "tuning/tuners/activation_quantization_tuner.py",
    "tuning/tuners/activation_adaptation_tuner.py",
    "tuning/tuners/normalization_aware_perceptron_quantization_tuner.py",
    "tuning/tuners/perceptron_transform_tuner.py",
]


@pytest.mark.parametrize("rel", _TUNER_FILES_WITHOUT_TEST_CALLS)
def test_tuner_source_has_no_trainer_test_call(rel):
    """No tuner source may call ``self.trainer.test(...)`` or ``_trainer.test(...)``.

    The pipeline calls ``trainer.test()`` exactly once per step from
    ``PipelineStep.pipeline_metric``. Tuners must use ``validate()`` or
    ``validate_n_batches()`` for any decision that could feed back into
    training.
    """
    path = SRC / rel
    assert path.exists(), f"missing file: {path}"
    src = _strip_comments_and_docstrings(path.read_text())
    # Match any attribute access ending with `.test(` where the base is
    # (self|something)trainer, capturing the tuner-owned variants.
    bad = re.findall(r"\btrainer\.test\s*\(", src)
    bad += re.findall(r"_trainer\.test\s*\(", src)
    assert not bad, (
        f"{rel}: forbidden trainer.test() call(s) inside tuner code -- "
        f"test-set leaks into tuner decision logic. Use validate_n_batches()."
    )


def test_unified_tuner_has_no_test_baseline_attribute():
    """``SmoothAdaptationTuner`` must not cache a test-set baseline internally.

    The previous implementation stored ``self._test_baseline = trainer.test()``
    at run start and compared against it during the rate=1.0 gate. This was
    a test-set data leak. Validation baselines are fine.
    """
    src = (SRC / "tuning/unified_tuner.py").read_text()
    stripped = _strip_comments_and_docstrings(src)
    assert "_test_baseline" not in stripped, (
        "SmoothAdaptationTuner must not use a _test_baseline field derived from "
        "trainer.test(); use _validation_baseline instead."
    )


def test_live_tuner_does_not_call_trainer_test():
    """Run a smoothly-adapted tuner end-to-end and assert trainer.test is untouched.

    Instruments a ``ClampTuner`` run with a ``trainer.test`` that raises on
    call. The tuner must complete without ever hitting it.
    """
    from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner

    mock = MockPipeline()
    ce = nn.CrossEntropyLoss()
    mock.loss = lambda model, x, y: ce(model(x), y)
    model = make_tiny_supermodel()

    am = AdaptationManager()
    scales = [1.0 for _ in model.get_perceptrons()]

    tuner = ClampTuner(
        pipeline=mock,
        model=model,
        target_accuracy=0.0,
        lr=1e-3,
        adaptation_manager=am,
        activation_scales=scales,
        activation_scale_stats=None,
    )

    def _boom(*a, **k):
        raise AssertionError(
            "trainer.test() called from inside tuner decision logic -- leak!"
        )

    with patch.object(tuner.trainer, "test", side_effect=_boom):
        # Run a tiny adaptation step (not full run) -- enough to exercise
        # the decision paths including the rate=1.0 internal gate.
        tuner._committed_rate = 0.0
        tuner._natural_rate = 0.0
        tuner._missed_target_streak = 0
        tuner._pre_relaxation_target = None
        tuner._cycle_log = []
        tuner._cached_lr = None
        tuner._pipeline_tolerance = 0.05
        tuner._pipeline_hard_floor = None
        tuner._validation_baseline = 0.0
        tuner._rollback_tolerance = 0.05
        # _adaptation at rate=1.0 formerly triggered the test() gate.
        tuner._adaptation(1.0)
