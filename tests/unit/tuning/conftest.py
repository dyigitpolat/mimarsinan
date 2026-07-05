"""Tuning-suite fixtures: opt-in accepting gate for fast-ladder mechanics tests."""

import pytest

from mimarsinan.tuning.orchestration import mbh_ledger


@pytest.fixture
def accepting_gate(monkeypatch):
    """Never-regressing injected D-hat: the default fast-ladder gate rides the
    accept path, so tests of the ladder MECHANICS (rung walk, budgets, traces)
    stay deterministic on tiny random fixtures. Gate SEMANTICS (reject/bisect/
    stall) are covered by test_mbh_gate.py with explicit injections."""
    monkeypatch.setattr(
        mbh_ledger, "rung_measurements",
        lambda tuner: {
            "blended_fp32": 0.5, "full_acc": 0.5,
            "rho": 1.0, "grad_norm_t": 0.0,
        },
    )
    monkeypatch.setattr(
        mbh_ledger, "full_transform_measurement", lambda tuner: 0.5,
    )
