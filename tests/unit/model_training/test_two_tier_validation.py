"""Phase D2: Two-tier validation API on BasicTrainer.

Contract:
  * ``BasicTrainer`` exposes ``validate_fast()`` and ``validate_full()``
    as first-class, explicit entry points.
  * ``validate_fast`` evaluates on a small subset (≤ ``_fast_n_batches``)
    and is intended for LR probes / per-cycle progress checks.
  * ``validate_full`` evaluates on the full validation set (≤
    ``_full_n_batches``, defaults to the total number of validation
    batches) and is intended for rollback / safety-net decisions.
  * Both APIs share the trainer's existing validation loader and
    iterator wraparound -- no new DataLoader is created.
  * The batch counts are configurable via
    ``set_fast_validation_batches`` /
    ``set_full_validation_batches`` so that tuners can align them with
    their ``TuningBudget`` (``progress_eval_batches`` for fast,
    ``eval_n_batches`` for full).
  * ``validate_fast`` must consume strictly fewer (or equal) batches
    than ``validate_full`` on a trainer where the full batch budget
    exceeds the fast one.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from conftest import MockDataProviderFactory


class _WrapperLoss:
    def __call__(self, model, x, y):
        return nn.CrossEntropyLoss()(model(x), y)


def _make_trainer(num_classes=4, input_shape=(1, 8, 8)):
    dp_factory = MockDataProviderFactory(
        input_shape=input_shape, num_classes=num_classes
    )
    dlf = DataLoaderFactory(dp_factory, num_workers=0)
    in_features = 1
    for d in input_shape:
        in_features *= d
    model = nn.Sequential(nn.Flatten(), nn.Linear(in_features, num_classes))
    return BasicTrainer(model, "cpu", dlf, _WrapperLoss())


class TestTwoTierValidationAPI:
    def test_validate_fast_exists_and_returns_float(self):
        trainer = _make_trainer()
        assert hasattr(trainer, "validate_fast"), (
            "BasicTrainer must expose an explicit validate_fast() API "
            "for per-cycle / LR-probe evaluations (Phase D2)."
        )
        acc = trainer.validate_fast()
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_validate_full_exists_and_returns_float(self):
        trainer = _make_trainer()
        assert hasattr(trainer, "validate_full"), (
            "BasicTrainer must expose an explicit validate_full() API "
            "for rollback / safety-net evaluations (Phase D2)."
        )
        acc = trainer.validate_full()
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_fast_full_configurable(self):
        trainer = _make_trainer()
        trainer.set_fast_validation_batches(2)
        trainer.set_full_validation_batches(4)
        assert trainer._fast_n_batches == 2
        assert trainer._full_n_batches == 4

    def test_fast_uses_fewer_batches_than_full(self):
        """validate_fast must iterate over strictly fewer batches when
        the full budget exceeds the fast one."""
        trainer = _make_trainer()
        trainer.set_fast_validation_batches(1)
        trainer.set_full_validation_batches(4)

        call_counts = {"forward": 0}
        orig_forward = trainer.model.forward

        def counted_forward(x):
            call_counts["forward"] += 1
            return orig_forward(x)

        trainer.model.forward = counted_forward  # type: ignore[method-assign]

        call_counts["forward"] = 0
        trainer.validate_fast()
        fast_calls = call_counts["forward"]

        call_counts["forward"] = 0
        trainer.validate_full()
        full_calls = call_counts["forward"]

        assert fast_calls < full_calls, (
            f"validate_fast used {fast_calls} forwards, "
            f"validate_full used {full_calls}; fast must be a strict "
            f"subset of full."
        )
        assert fast_calls == 1
        assert full_calls == 4

    def test_fast_and_full_do_not_hit_test_loader(self):
        """Two-tier validation must never touch the test set."""
        trainer = _make_trainer()

        def boom(*_a, **_k):
            raise AssertionError(
                "validate_fast / validate_full must not invoke test()"
            )

        trainer.test = boom
        trainer.validate_fast()
        trainer.validate_full()


class TestTunerWiresBudgetsIntoTrainer:
    """The tuner must configure the trainer's fast/full batch counts
    from its :class:`TuningBudget` so ``validate_fast`` aligns with
    ``progress_eval_batches`` and ``validate_full`` aligns with
    ``eval_n_batches``."""

    def test_tuner_base_configures_fast_full_from_budget(self):
        """``TunerBase.__init__`` must push its budget's
        ``progress_eval_batches`` / ``eval_n_batches`` into the
        trainer's fast / full batch counts."""
        from mimarsinan.tuning.unified_tuner import TunerBase

        src = inspect.getsource(TunerBase.__init__)
        # The init must configure fast+full batch counts from the
        # budget.  We accept either explicit setter calls or direct
        # attribute assignments -- both are equally valid wiring.
        assert (
            "set_fast_validation_batches" in src
            or "_fast_n_batches" in src
        ), (
            "TunerBase.__init__ must wire progress_eval_batches into the "
            "trainer's fast-validation batch count."
        )
        assert (
            "set_full_validation_batches" in src
            or "_full_n_batches" in src
        ), (
            "TunerBase.__init__ must wire eval_n_batches into the "
            "trainer's full-validation batch count."
        )


class TestRoutingInTuners:
    """Tuner call sites must route through the two-tier API, not the
    legacy ``validate_n_batches(<budget>)`` spelling:

      * ``progress_eval_batches`` -> ``validate_fast``
      * ``eval_n_batches``       -> ``validate_full``

    The legacy ``validate_n_batches`` helper stays on the trainer as an
    implementation detail, but no tuner should reference it with a
    budget field any more -- budget-awareness belongs in the trainer
    itself, not repeated at every call site.
    """

    TUNER_FILES = [
        "src/mimarsinan/tuning/unified_tuner.py",
        "src/mimarsinan/tuning/tuners/pruning_tuner.py",
        "src/mimarsinan/tuning/tuners/clamp_tuner.py",
        "src/mimarsinan/tuning/tuners/activation_adaptation_tuner.py",
        "src/mimarsinan/tuning/tuners/activation_quantization_tuner.py",
        "src/mimarsinan/tuning/tuners/activation_shift_tuner.py",
        "src/mimarsinan/tuning/tuners/normalization_aware_perceptron_quantization_tuner.py",
    ]

    def _read(self, rel_path: str) -> str:
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[3]
        with open(root / rel_path, "r") as f:
            return f.read()

    def test_no_tuner_calls_validate_n_batches_with_progress_eval(self):
        for rel in self.TUNER_FILES:
            src = self._read(rel)
            assert "validate_n_batches(self._budget.progress_eval_batches" not in src, (
                f"{rel} still routes a fast/per-cycle probe through "
                f"validate_n_batches(progress_eval_batches).  Use "
                f"trainer.validate_fast() instead (Phase D2)."
            )

    def test_no_tuner_calls_validate_n_batches_with_eval_n_batches(self):
        for rel in self.TUNER_FILES:
            src = self._read(rel)
            assert "validate_n_batches(self._budget.eval_n_batches" not in src, (
                f"{rel} still routes a rollback/baseline probe through "
                f"validate_n_batches(eval_n_batches).  Use "
                f"trainer.validate_full() instead (Phase D2)."
            )
