"""Genuine full-transformation probe: dual-drop (value vs genuine) trajectory.

After each commit, ``tuning_full_transform_probe`` records BOTH the value-domain
rate-1.0 accuracy and the GENUINE full-transform accuracy. The trend verdict is
keyed on ``genuine_drop`` (the deployed cliff), not the value-domain proxy: when
the value proxy converges but the genuine forward stays flat, the OLD value-keyed
logic was *fooled* (verdict CONVERGING) while the genuine cliff stays large. The
new logic keys on ``genuine_drop`` and warns "probe was fooled" — making it
explicit when ``value_drop`` shrank while ``genuine_drop`` did not."""

import pytest

from conftest import MockPipeline, make_tiny_supermodel, default_config
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner


class _GenuineProbeTuner(SmoothAdaptationTuner):
    """Scripts BOTH the value-domain and genuine rate-1.0 probe accuracies.

    ``value_full_acc_fn(committed)`` drives ``_value_full_transform_eval`` (via
    ``_update_and_evaluate(1.0)``); ``genuine_full_acc_fn(committed)`` overrides
    ``_full_transform_eval`` directly (the genuine forward). When
    ``genuine_full_acc_fn`` is None the tuner uses the BASE default
    (genuine == value), exercising the no-separate-forward parity path."""

    def __init__(self, pipeline, model, target, lr, *,
                 value_full_acc_fn, genuine_full_acc_fn=None):
        super().__init__(pipeline, model, target, lr)
        self._committed_rate = 0.0
        self._validation_baseline = 0.9
        self._rollback_tolerance = 0.05
        self._value_full_acc_fn = value_full_acc_fn
        self._genuine_full_acc_fn = genuine_full_acc_fn
        self._value_eval_calls = 0
        self.trainer.train_steps_until_target = lambda *a, **k: None
        self.trainer.validate_n_batches = lambda n: 0.9  # committed-rate acc

    def _find_lr(self):
        return 0.001

    def _update_and_evaluate(self, rate):
        # rate == 1.0 is the value-domain probe; sub-1.0 is the cycle commit eval.
        if rate >= 1.0 - 1e-9:
            self._value_eval_calls += 1
            return self._value_full_acc_fn(self._committed_rate)
        return 0.9

    def _full_transform_eval(self):
        if self._genuine_full_acc_fn is None:
            return super()._full_transform_eval()
        return float(self._genuine_full_acc_fn(self._committed_rate))


def _make(tmp_path, *, probe=True, value_full_acc_fn, genuine_full_acc_fn=None):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["tuning_full_transform_probe"] = probe
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    return _GenuineProbeTuner(
        pipeline, make_tiny_supermodel(), 0.9, 0.001,
        value_full_acc_fn=value_full_acc_fn,
        genuine_full_acc_fn=genuine_full_acc_fn,
    )


def test_log_record_carries_all_dual_fields_and_legacy_aliases(tmp_path, deterministic_rng):
    # value flat @0.85, genuine climbs 0.6->0.9 → proxy_gap large, genuine_drop shrinks.
    tuner = _make(
        tmp_path, probe=True,
        value_full_acc_fn=lambda a: 0.85,
        genuine_full_acc_fn=lambda a: 0.6 + 0.3 * a,
    )
    tuner._adaptation(0.3)
    tuner._adaptation(0.9)
    log = tuner._full_transform_log
    assert len(log) == 2
    rec = log[0]
    for field in ("committed", "committed_acc", "value_full_acc", "genuine_full_acc",
                  "value_drop", "genuine_drop", "proxy_gap"):
        assert field in rec, f"missing field {field}"
    # Legacy aliases preserved (test_full_transform_probe.py stays green).
    assert rec["full_acc"] == rec["value_full_acc"]
    assert rec["drop"] == rec["value_drop"]
    # Field arithmetic.
    assert rec["value_drop"] == pytest.approx(rec["committed_acc"] - rec["value_full_acc"])
    assert rec["genuine_drop"] == pytest.approx(rec["committed_acc"] - rec["genuine_full_acc"])
    assert rec["proxy_gap"] == pytest.approx(rec["value_full_acc"] - rec["genuine_full_acc"])
    # value flat at 0.85, genuine 0.6@0.3 → proxy_gap is the value-genuine gap.
    assert rec["value_full_acc"] == pytest.approx(0.85)
    assert rec["genuine_full_acc"] == pytest.approx(0.6 + 0.3 * 0.3)
    assert rec["proxy_gap"] > 0.1
    tuner.close()


def test_converging_genuine_surface_verdict_keys_on_genuine(tmp_path, deterministic_rng, capsys):
    # value flat (would be FLAT under value-keyed logic if it were the signal),
    # genuine climbs → genuine_drop shrinks → CONVERGING (keyed on genuine).
    tuner = _make(
        tmp_path, probe=True,
        value_full_acc_fn=lambda a: 0.85,
        genuine_full_acc_fn=lambda a: 0.55 + 0.35 * a,
    )
    for r in (0.3, 0.6, 0.9):
        tuner._adaptation(r)
    log = tuner._full_transform_log
    assert log[0]["genuine_drop"] > log[-1]["genuine_drop"]  # genuine cliff shrank
    tuner._log_full_transform_trend()
    out = capsys.readouterr().out
    assert "CONVERGING" in out
    # proxy_gap large (value 0.85 vs genuine ~0.55-0.86).
    assert log[0]["proxy_gap"] > 0.1
    tuner.close()


def test_fooled_value_converges_genuine_flat_warns(tmp_path, deterministic_rng, capsys):
    # THE regression: value_full_acc climbs (OLD value-keyed verdict = CONVERGING)
    # but genuine_full_acc is low/flat → NEW verdict FLAT/DIVERGING + "fooled" warn.
    tuner = _make(
        tmp_path, probe=True,
        value_full_acc_fn=lambda a: 0.6 + 0.3 * a,   # value drop shrinks
        genuine_full_acc_fn=lambda a: 0.55,          # genuine flat
    )
    for r in (0.3, 0.6, 0.9):
        tuner._adaptation(r)
    log = tuner._full_transform_log
    # value drop DID shrink (the fooling condition); genuine drop did NOT.
    assert log[0]["value_drop"] > log[-1]["value_drop"] + 1e-9
    assert log[-1]["genuine_drop"] >= log[0]["genuine_drop"] - 1e-9
    with pytest.warns(UserWarning, match="(?i)fooled"):
        tuner._log_full_transform_trend()
    out = capsys.readouterr().out
    assert "FLAT/DIVERGING" in out
    tuner.close()


def test_base_default_genuine_equals_value(tmp_path, deterministic_rng):
    # No genuine override → BASE DEFAULT _full_transform_eval == value path.
    tuner = _make(
        tmp_path, probe=True,
        value_full_acc_fn=lambda a: 0.6 + 0.3 * a,
        genuine_full_acc_fn=None,
    )
    for r in (0.3, 0.9):
        tuner._adaptation(r)
    for rec in tuner._full_transform_log:
        assert rec["genuine_full_acc"] == pytest.approx(rec["value_full_acc"])
        assert rec["genuine_drop"] == pytest.approx(rec["value_drop"])
        assert rec["proxy_gap"] == pytest.approx(0.0)
    tuner.close()


def test_value_full_transform_eval_is_non_destructive(tmp_path, deterministic_rng):
    # The base value path clones state, evaluates at 1.0, and restores —
    # committed rate must be unchanged after the probe.
    tuner = _make(
        tmp_path, probe=True,
        value_full_acc_fn=lambda a: 0.5,
        genuine_full_acc_fn=lambda a: 0.5,
    )
    tuner._adaptation(0.4)
    assert tuner._committed_rate == pytest.approx(0.4)
    # Direct call also restores (it clones->applies 1.0->restores).
    before = tuner._committed_rate
    acc = tuner._value_full_transform_eval()
    assert acc == pytest.approx(0.5)
    assert tuner._committed_rate == pytest.approx(before)
    tuner.close()


def test_trend_reports_first_last_proxy_gap_and_value_drop(tmp_path, deterministic_rng):
    tuner = _make(
        tmp_path, probe=True,
        value_full_acc_fn=lambda a: 0.85,
        genuine_full_acc_fn=lambda a: 0.55 + 0.35 * a,
    )
    for r in (0.3, 0.6, 0.9):
        tuner._adaptation(r)
    reported = {}
    tuner.pipeline.reporter.report = lambda key, val: reported.__setitem__(key, val)
    tuner._log_full_transform_trend()
    trend = reported[f"{tuner.name} full_transform_trend"]
    for field in ("first_genuine_drop", "last_genuine_drop",
                  "first_value_drop", "last_value_drop",
                  "first_proxy_gap", "last_proxy_gap", "shrinking"):
        assert field in trend, f"missing trend field {field}"
    assert trend["shrinking"] is True  # genuine drop shrank
    tuner.close()
