"""Tests that tuners tag exploratory validations with ``probe`` context.

The Accuracy panel in the GUI renders committed tuning progress and
exploratory probes (LR search, rate proposals) as distinct traces on the
same chart. Tuners mark exploratory validations via
``BasicTrainer.validation_context("probe")`` so the emitted metric names
carry a ``(probe)`` suffix; committed progress keeps the untagged name.

These tests assert that contract at the tuner layer.
"""

from __future__ import annotations

import pytest

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


class _RecordingTuner(SmoothAdaptationTuner):
    """Concrete tuner that patches the trainer to record emitted metric names."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.emitted: list[tuple[str, float]] = []
        self.trainer.report_function = lambda n, v: self.emitted.append((n, v))

        # Make validate_n_batches cheap and deterministic: it must still
        # call ``_report`` with the context-suffixed metric name so we can
        # assert on the emissions.
        trainer = self.trainer

        def fake_validate_n_batches(n):
            acc = 0.75
            trainer._report(trainer._validation_metric_name("Validation accuracy"), acc)
            return acc

        trainer.validate_n_batches = fake_validate_n_batches  # type: ignore[method-assign]
        trainer.train_steps_until_target = lambda *a, **kw: None  # type: ignore[method-assign]
        trainer.train_n_steps = lambda *a, **kw: None  # type: ignore[method-assign]

    def _update_and_evaluate(self, rate):
        return self.trainer.validate_n_batches(1)


@pytest.fixture
def tuner(tmp_path):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()
    t = _RecordingTuner(pipeline, model, target_accuracy=0.9, lr=1e-3)
    t._rollback_tolerance = 0.05
    return t


class TestValidationContextTagging:
    def test_find_lr_validations_are_tagged_probe(self, tuner):
        """Every validation call inside ``_find_lr`` is tagged as a probe."""
        tuner.emitted.clear()
        lr = tuner._find_lr()

        probe_emits = [n for (n, _) in tuner.emitted if n.endswith("(probe)")]
        plain_emits = [n for (n, _) in tuner.emitted if n == "Validation accuracy"]

        assert probe_emits, "expected at least one probe-tagged validation"
        assert not plain_emits, (
            f"no untagged validations should be emitted during _find_lr, got {plain_emits}"
        )
        assert isinstance(lr, float) and lr > 0

    def test_update_and_evaluate_is_tagged_probe(self, tuner):
        """``_adaptation`` wraps ``_update_and_evaluate`` in probe context."""
        tuner.emitted.clear()
        tuner._adaptation(0.5)

        tagged_after_pre = False
        seen_pre_cycle = False
        for name, _ in tuner.emitted:
            if name == "Validation accuracy":
                if not seen_pre_cycle:
                    seen_pre_cycle = True
                else:
                    continue
            if seen_pre_cycle and name == "Validation accuracy (probe)":
                tagged_after_pre = True
                break

        assert seen_pre_cycle, "pre-cycle validation should be emitted untagged"
        assert tagged_after_pre, (
            "instant-acc validation inside _update_and_evaluate must be "
            "tagged (probe), got emissions: "
            f"{[n for (n, _) in tuner.emitted]}"
        )

    def test_committed_progress_validations_are_untagged(self, tuner):
        """pre-cycle baseline and post-recovery decision validations are progress."""
        tuner.emitted.clear()
        tuner._adaptation(0.5)

        untagged = [n for (n, _) in tuner.emitted if n == "Validation accuracy"]
        assert len(untagged) >= 2, (
            "expected at least pre-cycle and post-recovery untagged validations; "
            f"got {[n for (n, _) in tuner.emitted]}"
        )

    def test_context_is_restored_after_find_lr(self, tuner):
        tuner._find_lr()
        assert getattr(tuner.trainer, "_validation_context", None) is None

    def test_context_is_restored_after_adaptation(self, tuner):
        tuner._adaptation(0.5)
        assert getattr(tuner.trainer, "_validation_context", None) is None
