"""Tests for PruningTuner and AdaptationManager pruning_rate integration."""

import pytest
import torch
import torch.nn as nn
import copy
from unittest.mock import patch

from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.layers import LeakyGradReLU
from conftest import default_config, MockPipeline, make_tiny_supermodel


class TestAdaptationManagerPruningRate:
    def test_initial_pruning_rate_is_zero(self):
        am = AdaptationManager()
        assert am.pruning_rate == 0.0

    def test_pruning_rate_can_be_set(self):
        am = AdaptationManager()
        am.pruning_rate = 0.5
        assert am.pruning_rate == 0.5

    def test_update_activation_still_works_with_pruning_rate(self):
        """Setting pruning_rate should not break update_activation."""
        am = AdaptationManager()
        am.pruning_rate = 0.7
        p = Perceptron(8, 16)
        p.base_activation = LeakyGradReLU()
        p.activation = LeakyGradReLU()
        p.set_activation_scale(2.0)
        cfg = default_config()
        am.update_activation(cfg, p)
        x = torch.randn(2, 16)
        out = p(x)
        assert not torch.isnan(out).any()


class TestPruningTuner:
    def test_constructs(self):
        """PruningTuner should be constructable with mock pipeline."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner

        mock = MockPipeline()
        model = make_tiny_supermodel()
        am = AdaptationManager()

        tuner = PruningTuner(
            pipeline=mock,
            model=model,
            target_accuracy=0.0,
            lr=0.001,
            adaptation_manager=am,
            pruning_fraction=0.25,
        )
        assert tuner is not None
        assert tuner.pruning_fraction == 0.25

    def test_apply_pruning_at_rate_one_zeros_weights(self):
        """At rate=1.0, pruned weights should be zeroed."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner
        from mimarsinan.transformations.pruning import apply_pruning_masks

        mock = MockPipeline()
        ce = nn.CrossEntropyLoss()
        mock.loss = lambda model, x, y: ce(model(x), y)
        model = make_tiny_supermodel()
        am = AdaptationManager()

        tuner = PruningTuner(
            pipeline=mock,
            model=model,
            target_accuracy=0.0,
            lr=0.001,
            adaptation_manager=am,
            pruning_fraction=0.5,
        )

        perceptrons = model.get_perceptrons()
        # Manually set up importance (normally done in run())
        for p in perceptrons:
            w = p.layer.weight.data
            tuner.base_row_imp.append(w.abs().sum(dim=1))
            tuner.base_col_imp.append(w.abs().sum(dim=0))

        original_weights = [p.layer.weight.data.clone() for p in perceptrons]
        row_masks, col_masks = tuner._get_masks(1.0)

        for i, p in enumerate(perceptrons):
            apply_pruning_masks(p, row_masks[i], col_masks[i], 1.0,
                                original_weights[i], None)

        # At rate=1.0, pruned weights should be exactly zero
        any_zeroed = False
        for p in perceptrons:
            if (p.layer.weight.data == 0.0).any():
                any_zeroed = True
        assert any_zeroed, "At least some weights should be zeroed at rate=1.0"

    def test_apply_pruning_at_rate_zero_preserves_weights(self):
        """At rate=0.0, weights should be unchanged."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner
        from mimarsinan.transformations.pruning import apply_pruning_masks

        mock = MockPipeline()
        ce = nn.CrossEntropyLoss()
        mock.loss = lambda model, x, y: ce(model(x), y)
        model = make_tiny_supermodel()
        am = AdaptationManager()

        tuner = PruningTuner(
            pipeline=mock,
            model=model,
            target_accuracy=0.0,
            lr=0.001,
            adaptation_manager=am,
            pruning_fraction=0.5,
        )

        perceptrons = model.get_perceptrons()
        for p in perceptrons:
            w = p.layer.weight.data
            tuner.base_row_imp.append(w.abs().sum(dim=1))
            tuner.base_col_imp.append(w.abs().sum(dim=0))

        original_weights = [p.layer.weight.data.clone() for p in perceptrons]
        row_masks, col_masks = tuner._get_masks(1.0)

        for i, p in enumerate(perceptrons):
            apply_pruning_masks(p, row_masks[i], col_masks[i], 0.0,
                                original_weights[i], None)

        for orig, p in zip(original_weights, perceptrons):
            assert torch.allclose(orig, p.layer.weight.data), \
                "Weights should be unchanged at rate=0.0"

    def test_first_layer_columns_and_last_layer_rows_exempt_at_rate_one(self):
        """At rate=1.0, first layer column mask and last layer row mask must be all True
        (input-buffer and output-buffer dimensions are never pruned)."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner

        mock = MockPipeline()
        model = make_tiny_supermodel()
        am = AdaptationManager()

        tuner = PruningTuner(
            pipeline=mock,
            model=model,
            target_accuracy=0.0,
            lr=0.001,
            adaptation_manager=am,
            pruning_fraction=1.0,
        )

        perceptrons = model.get_perceptrons()
        for p in perceptrons:
            w = p.layer.weight.data
            tuner.base_row_imp.append(w.abs().sum(dim=1))
            tuner.base_col_imp.append(w.abs().sum(dim=0))

        row_masks, col_masks = tuner._get_masks(1.0)

        assert col_masks[0].all(), "First layer column mask must be all True (input-buffer exempt)"
        assert row_masks[-1].all(), "Last layer row mask must be all True (output-buffer exempt)"

    def test_refresh_pruning_importance_called_per_cycle(self):
        """When run(max_cycles=N) is used, activation stats (importance) should be collected at least N times."""
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner
        from mimarsinan.transformations.pruning import _collect_activation_stats

        mock = MockPipeline()
        mock.config["training_epochs"] = 10
        model = make_tiny_supermodel()
        am = AdaptationManager()
        tuner = PruningTuner(
            pipeline=mock,
            model=model,
            target_accuracy=0.99,
            lr=0.001,
            adaptation_manager=am,
            pruning_fraction=0.25,
        )
        # Baseline calibration returns 0.99 (high target).
        # After pruning, validate returns 0.5 — well below target — so the
        # one-shot at rate 1.0 triggers catastrophic fast-fail and the adapter
        # falls through to gradual cycles.
        validate_call_count = [0]
        def _mock_validate_n_batches(self_unused, n):
            validate_call_count[0] += 1
            if validate_call_count[0] <= 1:
                return 0.99
            return 0.5

        tuner.trainer = type(
            "T",
            (),
            {
                "validate": lambda self: 0.99,
                "validate_n_batches": _mock_validate_n_batches,
                "train_one_step": lambda self, lr: None,
                "train_until_target_accuracy": lambda self, *a: None,
                "train_steps_until_target": lambda self, *a, **k: None,
                "test": lambda self: 0.5,
                "train_n_steps": lambda self, lr, n, **kw: None,
            },
        )()
        tuner.trainer.model = model
        tuner.trainer.report_function = lambda *a: None
        tuner.trainer.validation_loader = [(torch.randn(2, 1, 8, 8), torch.zeros(2, dtype=torch.long))]
        collect_calls = []

        def counting_collect(*args, **kwargs):
            collect_calls.append(1)
            return _collect_activation_stats(*args, **kwargs)

        with patch("mimarsinan.tuning.tuners.pruning_tuner._collect_activation_stats", side_effect=counting_collect):
            tuner.run(max_cycles=3)
        assert len(collect_calls) >= 3, "Activation stats should be collected at least once per cycle (3 cycles)"

    def test_pruned_set_grows_monotonically_with_rate(self):
        """Regression: pruned set at rate r₂ must be a superset of the set at r₁ < r₂.

        Pre-fix, per-cycle importance refresh could flip which row was the
        "least important" near the k_r boundary, producing a different pruned
        subset at rates 0.8550 and 0.8551 — the mask churn that caused
        catastrophic fast-fails in Pruning Adaptation. With the monotonic
        fix, once an index is pruned it stays pruned at all higher rates.
        """
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner

        mock = MockPipeline()
        model = make_tiny_supermodel()
        am = AdaptationManager()
        tuner = PruningTuner(
            pipeline=mock, model=model,
            target_accuracy=0.0, lr=0.001,
            adaptation_manager=am, pruning_fraction=0.5,
        )
        tuner._persistent_pruned_rows = [set() for _ in range(len(model.get_perceptrons()))]
        tuner._persistent_pruned_cols = [set() for _ in range(len(model.get_perceptrons()))]

        # Populate importance with values designed to have near-ties at the
        # boundary, so unstable sort would flip ranks across refreshes.
        for i, p in enumerate(model.get_perceptrons()):
            out_f, in_f = p.layer.weight.data.shape
            tuner.base_row_imp.append(torch.linspace(0.0, 1.0, out_f))
            tuner.base_col_imp.append(torch.linspace(0.0, 1.0, in_f))

        # Simulate a sequence of rate proposals that exercise the bisection
        # pattern (bounce around a committed rate near a k_r boundary).
        pruned_sets_by_rate = []
        for rate in [0.10, 0.30, 0.50, 0.49, 0.51, 0.70, 0.65, 0.90, 1.0]:
            tuner._get_masks(rate)
            pruned_sets_by_rate.append(
                [set(s) for s in tuner._persistent_pruned_rows]
            )

        # Monotonic invariant: every subsequent snapshot is a superset of all prior ones.
        for k in range(1, len(pruned_sets_by_rate)):
            for i, (prev, cur) in enumerate(
                zip(pruned_sets_by_rate[k - 1], pruned_sets_by_rate[k])
            ):
                assert prev.issubset(cur), (
                    f"perceptron {i}: set at step {k} lost indices from step {k-1}: "
                    f"{prev - cur}"
                )

    def test_enforcement_hooks_are_pickle_safe(self):
        """Regression: the persistent pruning hooks must survive ``torch.save``.

        The pipeline pickles the adaptation_manager and the model between
        steps. Closures captured inside ``_enforce_pruning_persistently``
        would fail with ``Can't pickle local object`` — module-level hook
        functions backed by buffers survive instead.
        """
        import io
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner

        mock = MockPipeline()
        model = make_tiny_supermodel()
        am = AdaptationManager()
        tuner = PruningTuner(
            pipeline=mock, model=model,
            target_accuracy=0.0, lr=0.001,
            adaptation_manager=am, pruning_fraction=0.5,
        )
        perceptrons = model.get_perceptrons()
        for p in perceptrons:
            w = p.layer.weight.data
            tuner.base_row_imp.append(w.abs().sum(dim=1))
            tuner.base_col_imp.append(w.abs().sum(dim=0))
        tuner._persistent_pruned_rows = [set() for _ in perceptrons]
        tuner._persistent_pruned_cols = [set() for _ in perceptrons]

        row_masks, col_masks = tuner._get_masks(1.0)
        for i, p in enumerate(perceptrons):
            rm, cm = row_masks[i], col_masks[i]
            p.layer.register_buffer("prune_row_mask", (~rm).clone())
            p.layer.register_buffer("prune_col_mask", (~cm).clone())
            p.layer.register_buffer(
                "prune_mask",
                ((~rm).unsqueeze(1) | (~cm).unsqueeze(0)).clone(),
            )
            if p.layer.bias is not None:
                p.layer.register_buffer("prune_bias_mask", (~rm).clone())
        tuner._enforce_pruning_persistently(perceptrons, row_masks, col_masks)

        buf = io.BytesIO()
        torch.save(model, buf)
        buf.seek(0)
        reloaded = torch.load(buf, weights_only=False)

        x = torch.randn(1, 1, 8, 8)
        out_orig = model(x)
        out_reloaded = reloaded(x)
        assert torch.allclose(out_orig, out_reloaded, atol=1e-5), (
            "Model forward diverged after pickle round-trip — hooks lost or buffers corrupted."
        )

    def test_enforcement_runs_before_final_metric_measurement(self):
        """Regression: ``_enforce_pruning_persistently`` must run *before*
        ``_ensure_validation_threshold`` probes validation accuracy.

        Pre-fix: enforcement (which zeros BN ``running_mean`` and ``beta``
        at pruned rows) ran at the end of ``run()``, *after* the tuner's
        ``_final_metric`` was captured. The cached metric was measured
        on a pre-enforcement model; the next pipeline step then tested a
        post-enforcement (different) model and saw a spurious accuracy
        drop — surfacing at "Activation Analysis" (a step that doesn't
        conceptually change the model) as a tolerance failure.

        Phase A1 moved the recovery probe from ``trainer.test()`` to
        ``trainer.validate_n_batches()`` (no test-set leak), so the
        ordering check now inspects the validation probe.
        """
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner

        mock = MockPipeline()
        model = make_tiny_supermodel()
        am = AdaptationManager()
        tuner = PruningTuner(
            pipeline=mock, model=model,
            target_accuracy=0.0, lr=0.001,
            adaptation_manager=am, pruning_fraction=0.5,
        )
        tuner._init_original_weights()
        perceptrons = model.get_perceptrons()
        tuner._persistent_pruned_rows = [set() for _ in perceptrons]
        tuner._persistent_pruned_cols = [set() for _ in perceptrons]
        for p in perceptrons:
            w = p.layer.weight.data
            tuner.base_row_imp.append(w.abs().sum(dim=1))
            tuner.base_col_imp.append(w.abs().sum(dim=0))

        # Prime BN running_mean with a non-zero value on every row so we
        # can detect whether enforcement's row-zeroing ran by the time
        # the validation probe is called.
        bn = perceptrons[0].normalization
        with torch.no_grad():
            bn.running_mean.fill_(0.5)

        captured = {}

        def _capture_validate_n_batches(_self, _n):
            captured["running_mean_at_probe_time"] = bn.running_mean.clone()
            return 1.0  # satisfy validation-threshold safety net

        def _fail_test(_self):
            raise AssertionError(
                "trainer.test() must not be called inside tuner code (A1)."
            )

        tuner.trainer = type(
            "T", (),
            {
                "test": _fail_test,
                "validate": lambda self: 1.0,
                "validate_n_batches": _capture_validate_n_batches,
                "train_one_step": lambda self, lr: None,
                "train_steps_until_target": lambda self, *a, **k: None,
                "train_n_steps": lambda self, *a, **k: None,
            },
        )()
        tuner.trainer.model = model
        tuner.trainer.report_function = lambda *a: None
        tuner.trainer.validation_loader = [
            (torch.randn(2, 1, 8, 8), torch.zeros(2, dtype=torch.long))
        ]

        tuner._committed_rate = 1.0
        tuner._after_run()

        rm_at_probe = captured["running_mean_at_probe_time"]
        row_masks, _ = tuner._get_masks(1.0)
        pruned_rows_mask = ~row_masks[0]
        assert pruned_rows_mask.any(), (
            "test precondition: expected some pruned rows at rate=1.0 "
            "so we can check enforcement ordering"
        )
        assert torch.all(rm_at_probe[pruned_rows_mask] == 0.0), (
            "BN running_mean at pruned rows must be zero when the validation "
            "probe is called from _ensure_validation_threshold — otherwise "
            "the tuner's _final_metric reflects a different model than the "
            "pipeline consumes."
        )

    def test_rollback_restores_persistent_pruned_sets(self):
        """Rollback must discard any pruned-set expansion from the failed cycle.

        The base-class rollback path calls ``_restore_state`` with the
        ``(model_state, extra_state)`` tuple captured before the cycle.
        PruningTuner's ``_get_extra_state`` / ``_set_extra_state`` must
        snapshot and restore ``_persistent_pruned_rows`` / ``_persistent_pruned_cols``.
        """
        from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner

        mock = MockPipeline()
        model = make_tiny_supermodel()
        am = AdaptationManager()
        tuner = PruningTuner(
            pipeline=mock, model=model,
            target_accuracy=0.0, lr=0.001,
            adaptation_manager=am, pruning_fraction=0.5,
        )
        n = len(model.get_perceptrons())
        tuner._persistent_pruned_rows = [{0} for _ in range(n)]
        tuner._persistent_pruned_cols = [{1} for _ in range(n)]

        snapshot = tuner._get_extra_state()

        # Simulate a cycle that expands the persistent sets (as _get_masks does).
        for s in tuner._persistent_pruned_rows:
            s.add(99)
        for s in tuner._persistent_pruned_cols:
            s.add(99)

        tuner._set_extra_state(snapshot)

        for s in tuner._persistent_pruned_rows:
            assert s == {0}, f"rollback should restore rows to {{0}}, got {s}"
        for s in tuner._persistent_pruned_cols:
            assert s == {1}, f"rollback should restore cols to {{1}}, got {s}"
