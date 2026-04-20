"""Post-DSQ contract for ``ActivationQuantizationTuner``.

The tuner no longer drives a smooth ``quantization_rate`` scalar
through DSQ.  Each cycle's ``rate`` argument is a *binary-discrete*
schedule knob: at rate ``t`` the first ``round(t * N)`` perceptrons
(in ascending-sensitivity order, determined by an empirical probe
at tuner start) receive the hard ``QuantizeDecorator`` via
``AdaptationManager.set_per_perceptron_rate``; the others stay
un-quantised.  ``rate == 1.0`` means "every perceptron quantised";
``rate == 0.0`` means "none quantised" (the baseline).

These tests pin:

1. ``_measure_layer_sensitivities`` probes one perceptron at a time
   (hard-quantise, validate_fast, drop, restore) and returns the
   perceptrons sorted by accuracy drop ascending (least-sensitive
   first).
2. ``_update_and_evaluate(rate)`` enables exactly ``round(rate * N)``
   perceptrons in that order via per-perceptron overrides.
3. Cycle rollback (via ``_clone_state`` / ``_restore_state``)
   correctly resets ``_per_perceptron_rates`` to the pre-cycle
   snapshot.
4. ``_after_run`` forces every perceptron to the hard quantiser
   before calling the shared recovery helper.
"""

from __future__ import annotations

import copy

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.models.decorators import NestedDecoration, QuantizeDecorator
from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.activation_quantization_tuner import (
    ActivationQuantizationTuner,
)


def _flatten_decorators(activation) -> list:
    decorators = list(activation.decorators)
    out: list = []
    while decorators:
        d = decorators.pop(0)
        if isinstance(d, NestedDecoration):
            decorators = list(d.decorators) + decorators
        else:
            out.append(d)
    return out


def _has_hard_quantizer(perceptron) -> bool:
    return any(
        isinstance(d, QuantizeDecorator)
        for d in _flatten_decorators(perceptron.activation)
    )


def _make_pipeline(tmp_path):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["target_tq"] = 4
    return MockPipeline(config=cfg, working_directory=str(tmp_path))


def _make_named_model():
    """Tiny supermodel with distinct, stable perceptron names so
    per-perceptron overrides (keyed by ``perceptron.name``) don't
    collide.  The real pipeline gives perceptrons unique names via
    the torch->mapper converter; the tiny test model does not."""
    model = make_tiny_supermodel()
    for i, p in enumerate(model.get_perceptrons()):
        p.name = f"layer_{i}"
    return model


def _make_tuner(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    model = _make_named_model()
    am = AdaptationManager()
    for p in model.get_perceptrons():
        am.update_activation(pipeline.config, p)
    return ActivationQuantizationTuner(
        pipeline, model, target_tq=4, target_accuracy=0.9, lr=0.001,
        adaptation_manager=am,
    )


class TestMeasureLayerSensitivities:
    """The tuner must have a helper that probes every perceptron once
    at the start of ``run``, then caches an ascending-sensitivity
    order of perceptron names (least-sensitive first)."""

    def test_returns_names_of_all_perceptrons(self, tmp_path):
        tuner = _make_tuner(tmp_path)
        names = [p.name for p in tuner.model.get_perceptrons()]
        order = tuner._measure_layer_sensitivities()
        assert sorted(order) == sorted(names), (
            "sensitivity order must cover every perceptron exactly once"
        )

    def test_is_deterministic_for_fixed_trainer(self, tmp_path):
        """With a mocked deterministic ``validate_fast`` the resulting
        order must be a pure function of the recorded drops."""
        tuner = _make_tuner(tmp_path)

        baseline = 0.90
        call_log: list[tuple[str, float]] = []

        def fake_validate_fast():
            # Detect which perceptron has the hard quantiser installed
            # (there must be exactly one at probe time, or zero when
            # the helper re-measures the baseline).
            quantised = [
                p.name
                for p in tuner.model.get_perceptrons()
                if _has_hard_quantizer(p)
            ]
            assert len(quantised) <= 1, (
                f"sensitivity probe must quantise one perceptron at "
                f"a time; saw {quantised}"
            )
            if not quantised:
                call_log.append(("baseline", baseline))
                return baseline
            name = quantised[0]
            # Deterministic "drops": layer_1 hurts more than layer_0.
            drop_by_name = {"layer_0": 0.05, "layer_1": 0.20}
            acc = baseline - drop_by_name.get(name, 0.10)
            call_log.append((name, acc))
            return acc

        tuner.trainer.validate_fast = fake_validate_fast  # type: ignore[assignment]

        order = tuner._measure_layer_sensitivities()
        assert order == ["layer_0", "layer_1"], (
            f"expected ascending-sensitivity order layer_0 (drop=0.05) "
            f"before layer_1 (drop=0.20); got {order}"
        )

    def test_restores_state_after_probes(self, tmp_path):
        """After ``_measure_layer_sensitivities`` returns, the model
        must be back in its pre-measurement state (no lingering
        per-perceptron overrides, no leftover QuantizeDecorators).
        Otherwise the tuner's subsequent cycles would start from a
        partially-quantised model."""
        tuner = _make_tuner(tmp_path)
        am = tuner.adaptation_manager
        tuner.trainer.validate_fast = lambda: 0.9  # type: ignore[assignment]

        pre_overrides = copy.deepcopy(am._per_perceptron_rates)
        tuner._measure_layer_sensitivities()
        post_overrides = am._per_perceptron_rates

        assert post_overrides == pre_overrides, (
            "sensitivity probe must restore the per-perceptron override "
            "dict to its pre-probe state"
        )
        for p in tuner.model.get_perceptrons():
            assert not _has_hard_quantizer(p), (
                f"perceptron {p.name} still has a hard quantiser after "
                f"the sensitivity probe finished"
            )


class TestUpdateAndEvaluateBinaryDiscreteRollout:
    """``_update_and_evaluate(rate)`` must enable the first
    ``round(rate * N)`` perceptrons in ``_sensitivity_order`` and
    leave the rest un-quantised."""

    def _prime(self, tuner, order):
        """Seed the tuner's sensitivity cache so the rollout helper
        can run without an actual probing pass (tested separately)."""
        tuner._sensitivity_order = list(order)

    @pytest.mark.parametrize(
        "rate,expected_k",
        [
            (0.0, 0),
            (0.4, 1),  # round(0.4 * 2) == 1
            (0.5, 1),  # round(0.5 * 2) == 1  (banker's rounding even)
            (0.6, 1),  # round(0.6 * 2) == 1
            (0.75, 2),
            (1.0, 2),
        ],
    )
    def test_enables_first_k_perceptrons(self, tmp_path, rate, expected_k):
        tuner = _make_tuner(tmp_path)
        self._prime(tuner, ["layer_0", "layer_1"])
        tuner.trainer.validate_fast = lambda: 0.9  # type: ignore[assignment]

        tuner._update_and_evaluate(rate)

        perceptrons_by_name = {p.name: p for p in tuner.model.get_perceptrons()}
        expected_enabled = set(tuner._sensitivity_order[:expected_k])
        for name, p in perceptrons_by_name.items():
            has_q = _has_hard_quantizer(p)
            if name in expected_enabled:
                assert has_q, (
                    f"rate={rate}: perceptron {name} is in the first "
                    f"{expected_k} sensitivity slots but did not receive "
                    f"the hard quantiser"
                )
            else:
                assert not has_q, (
                    f"rate={rate}: perceptron {name} is outside the first "
                    f"{expected_k} sensitivity slots but was quantised"
                )

    def test_rate_one_enables_all(self, tmp_path):
        tuner = _make_tuner(tmp_path)
        self._prime(tuner, ["layer_0", "layer_1"])
        tuner.trainer.validate_fast = lambda: 0.9  # type: ignore[assignment]

        tuner._update_and_evaluate(1.0)
        for p in tuner.model.get_perceptrons():
            assert _has_hard_quantizer(p), (
                f"rate=1.0: every perceptron must be quantised "
                f"(failed for {p.name})"
            )

    def test_rate_zero_disables_all(self, tmp_path):
        """rate=0 must restore the un-quantised baseline regardless
        of any per-perceptron overrides left by a previous cycle."""
        tuner = _make_tuner(tmp_path)
        self._prime(tuner, ["layer_0", "layer_1"])
        tuner.trainer.validate_fast = lambda: 0.9  # type: ignore[assignment]

        tuner._update_and_evaluate(1.0)
        tuner._update_and_evaluate(0.0)
        for p in tuner.model.get_perceptrons():
            assert not _has_hard_quantizer(p), (
                f"rate=0 after rate=1 must drop the hard quantiser "
                f"from every perceptron (failed for {p.name})"
            )


class TestRollbackRestoresPerPerceptronOverrides:
    """Cycle rollback uses ``_clone_state`` / ``_restore_state`` to
    return to the pre-cycle snapshot.  The snapshot must include the
    per-perceptron quantisation-rate overrides so a failed cycle does
    not leak the "quantised" flag into the next attempt."""

    def test_clone_captures_overrides(self, tmp_path):
        tuner = _make_tuner(tmp_path)
        self._prime = lambda order: tuner.__setattr__(
            "_sensitivity_order", list(order)
        )
        tuner._sensitivity_order = ["layer_0", "layer_1"]
        tuner.trainer.validate_fast = lambda: 0.9  # type: ignore[assignment]

        tuner._update_and_evaluate(1.0)
        state = tuner._clone_state()
        _, extra = state
        overrides = extra["per_perceptron_quant_overrides"]
        assert overrides.get("layer_0") == pytest.approx(1.0)
        assert overrides.get("layer_1") == pytest.approx(1.0)

    def test_restore_undoes_mid_cycle_overrides(self, tmp_path):
        tuner = _make_tuner(tmp_path)
        tuner._sensitivity_order = ["layer_0", "layer_1"]
        tuner.trainer.validate_fast = lambda: 0.9  # type: ignore[assignment]

        baseline_state = tuner._clone_state()

        tuner._update_and_evaluate(1.0)
        for p in tuner.model.get_perceptrons():
            assert _has_hard_quantizer(p)

        tuner._restore_state(baseline_state)
        for p in tuner.model.get_perceptrons():
            assert not _has_hard_quantizer(p), (
                f"rollback must undo the quantiser on {p.name}"
            )

    def test_restore_reinstates_partial_override_set(self, tmp_path):
        """Snapshot at k=1, advance to k=2, roll back: must end at k=1
        with only the first perceptron quantised."""
        tuner = _make_tuner(tmp_path)
        tuner._sensitivity_order = ["layer_0", "layer_1"]
        tuner.trainer.validate_fast = lambda: 0.9  # type: ignore[assignment]

        tuner._update_and_evaluate(0.5)  # k = round(0.5*2) = 1
        mid_state = tuner._clone_state()

        tuner._update_and_evaluate(1.0)  # k = 2
        tuner._restore_state(mid_state)

        perceptrons_by_name = {p.name: p for p in tuner.model.get_perceptrons()}
        assert _has_hard_quantizer(perceptrons_by_name["layer_0"])
        assert not _has_hard_quantizer(perceptrons_by_name["layer_1"])


class TestAfterRunForcesFullQuantisationAndCallsRecovery:
    """``_after_run`` is contractually required to:
      * install the hard quantiser on every perceptron (rate=1 world),
      * call the shared ``_attempt_recovery_if_below_floor`` safety
        net exactly once,
      * return the recovered validation metric.
    """

    def test_after_run_enables_all_perceptrons(self, tmp_path):
        tuner = _make_tuner(tmp_path)
        tuner._sensitivity_order = ["layer_0", "layer_1"]

        # Neutralise the recovery helper so we can focus on the
        # quantisation state it leaves behind.
        tuner._attempt_recovery_if_below_floor = lambda: 0.9  # type: ignore[assignment]
        tuner._flush_enforcement_hooks = lambda: None  # type: ignore[assignment]
        tuner._continue_to_full_rate = lambda: None  # type: ignore[assignment]
        tuner._validation_floor_for_commit = lambda: 0.0  # type: ignore[assignment]

        tuner._after_run()

        for p in tuner.model.get_perceptrons():
            assert _has_hard_quantizer(p), (
                f"after_run must leave every perceptron hard-quantised "
                f"(failed for {p.name})"
            )

    def test_after_run_calls_recovery_helper_once(self, tmp_path):
        """Sentinel for the Phase D1 'called exactly once in _after_run'
        contract -- the per-layer rewrite must not regress it."""
        import ast
        import inspect
        import textwrap

        src = textwrap.dedent(
            inspect.getsource(ActivationQuantizationTuner._after_run)
        )
        tree = ast.parse(src)
        count = 0
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_attempt_recovery_if_below_floor"
            ):
                count += 1
        assert count == 1, (
            f"_after_run must call _attempt_recovery_if_below_floor exactly "
            f"once; got {count} call(s)"
        )
