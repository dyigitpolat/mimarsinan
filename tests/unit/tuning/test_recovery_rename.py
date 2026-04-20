"""Phase D1: Rename + tighten the safety-net recovery method.

Per the refactor plan:

  > _ensure_pipeline_threshold redesign
  > Replace the current behaviour:
  >   Currently: calls trainer.test(), if below hard_floor tries 2 recovery
  >   passes, returns best test.
  >   New: called only once at the very end of _after_run. Uses
  >   trainer.validate() as the signal for whether to attempt safety-net
  >   recovery (no test set involved during recovery decisions). Recovery
  >   training reuses the cached LR. The pipeline-level assertion (line 230
  >   of pipeline.py) remains the single test-based check.
  >   Rename to _attempt_recovery_if_below_floor so the intent is visible;
  >   failure path logs a clear warning instead of silently returning a
  >   below-floor value.

These tests pin the new name and the failure-path warning behaviour.
"""

from __future__ import annotations

import inspect
import logging

import pytest

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.activation_adaptation_tuner import ActivationAdaptationTuner
from mimarsinan.tuning.tuners.activation_quantization_tuner import ActivationQuantizationTuner
from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
    NormalizationAwarePerceptronQuantizationTuner,
)
from mimarsinan.tuning.tuners.perceptron_transform_tuner import PerceptronTransformTuner
from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner
from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


class TestRenamed:
    """Phase D1: ``_ensure_validation_threshold`` becomes
    ``_attempt_recovery_if_below_floor``.  The new name is the
    authoritative one; the legacy names live on as deprecated aliases
    that route to the same implementation so external code that still
    references them keeps working."""

    def test_new_method_exists(self):
        assert hasattr(SmoothAdaptationTuner, "_attempt_recovery_if_below_floor")

    def test_legacy_aliases_route_to_new_name(self):
        new = SmoothAdaptationTuner._attempt_recovery_if_below_floor
        assert SmoothAdaptationTuner._ensure_validation_threshold is new
        assert SmoothAdaptationTuner._ensure_pipeline_threshold is new


class TestNoTestSetInRecovery:
    """Sentinel against re-introducing a test-set probe inside the
    safety net."""

    def test_no_trainer_test_call_in_recovery(self):
        """Strip docstrings + comments before searching, then confirm
        no live ``trainer.test()`` (or other ``.test()``) call is made
        from the recovery body."""
        import ast
        import io
        import textwrap
        import tokenize

        src = textwrap.dedent(
            inspect.getsource(
                SmoothAdaptationTuner._attempt_recovery_if_below_floor
            )
        )
        # Drop function docstring.
        tree = ast.parse(src)
        for fn in ast.walk(tree):
            if (
                isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                and fn.body
                and isinstance(fn.body[0], ast.Expr)
                and isinstance(fn.body[0].value, ast.Constant)
                and isinstance(fn.body[0].value.value, str)
            ):
                fn.body = fn.body[1:]
        body = ast.unparse(tree)
        # Drop comments.
        tokens = tokenize.generate_tokens(io.StringIO(body).readline)
        kept = [t for t in tokens if t.type != tokenize.COMMENT]
        clean = tokenize.untokenize(kept)
        # The helper must not call ``trainer.test()`` -- the pipeline
        # assertion is the only test-set gate.
        assert "trainer.test(" not in clean
        # AST-level check: any direct call to ``.test`` is forbidden.
        for node in ast.walk(ast.parse(clean)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "test"
            ):
                pytest.fail(
                    f"unexpected .test() call inside recovery body: "
                    f"{ast.dump(node)}"
                )


class TestFailurePathWarns:
    """When recovery fails to reach the floor, the helper must log a
    clear warning instead of silently returning a below-floor value."""

    @pytest.fixture
    def tuner(self, tmp_path):
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        model = make_tiny_supermodel()
        am = AdaptationManager()
        # ClampTuner is the simplest non-PerceptronTransform tuner; it
        # exercises the same shared safety-net code path.
        scales = [1.0] * len(list(model.get_perceptrons()))
        tuner = ClampTuner(
            pipeline,
            model=model,
            target_accuracy=0.9,
            lr=0.001,
            adaptation_manager=am,
            activation_scales=scales,
            activation_scale_stats=None,
        )
        # Force the threshold high so the floor is unreachable.
        tuner.target_adjuster.original_metric = 1.0
        tuner._pipeline_tolerance = 0.0
        # Pretend recovery never improves accuracy beyond a sub-floor value.
        tuner.trainer.validate_n_batches = lambda _n: 0.10
        tuner.trainer.train_steps_until_target = lambda *a, **kw: None
        return tuner

    def test_warns_when_recovery_fails(self, tuner, caplog):
        with caplog.at_level(logging.WARNING):
            result = tuner._attempt_recovery_if_below_floor()
        assert result == pytest.approx(0.10)
        assert any(
            "safety-net recovery could not" in rec.message.lower()
            or "could not bring validation above" in rec.message.lower()
            or "below the floor" in rec.message.lower()
            for rec in caplog.records
        ), (
            f"expected a clear warning when recovery fails; got "
            f"{[rec.message for rec in caplog.records]!r}"
        )
        # And it must be at WARNING level or higher.
        assert any(
            rec.levelno >= logging.WARNING for rec in caplog.records
        )


class TestCallSitesUpdated:
    """``_after_run`` of every tuner that previously called the old
    name must now call the new ``_attempt_recovery_if_below_floor``."""

    @pytest.mark.parametrize(
        "tuner_cls",
        [
            ClampTuner,
            ActivationAdaptationTuner,
            ActivationQuantizationTuner,
            PruningTuner,
            PerceptronTransformTuner,
        ],
    )
    def test_after_run_uses_new_name(self, tuner_cls):
        source = inspect.getsource(tuner_cls._after_run)
        assert "_attempt_recovery_if_below_floor" in source, (
            f"{tuner_cls.__name__}._after_run still uses the legacy name"
        )


class TestCalledOnceAtEndOfAfterRun:
    """The plan is explicit: 'called only once at the very end of
    _after_run'.  No tuner should call the helper twice in one
    ``_after_run`` and no tuner should call it from anywhere else."""

    @pytest.mark.parametrize(
        "tuner_cls",
        [
            ClampTuner,
            ActivationAdaptationTuner,
            ActivationQuantizationTuner,
            PruningTuner,
            PerceptronTransformTuner,
        ],
    )
    def test_after_run_calls_helper_exactly_once(self, tuner_cls):
        """Count actual ``Call`` AST nodes -- not docstring or comment
        mentions -- so the test fails only when the helper is invoked
        more than once in the runtime body."""
        import ast
        import textwrap

        source = textwrap.dedent(inspect.getsource(tuner_cls._after_run))
        tree = ast.parse(source)
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "_attempt_recovery_if_below_floor":
                    count += 1
        assert count == 1, (
            f"{tuner_cls.__name__}._after_run calls the safety net "
            f"{count} times; should call it exactly once at the end"
        )
