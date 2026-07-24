"""[cert-plan W4] certification conformance over the mode matrix: every
(spiking_mode × schedule) cell yields exactly one typed outcome — a counts/
events observable or a non-empty skip reason — and the gate honors it."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.chip_simulation.spiking_semantics import ALL_SPIKING_MODES

_SCHEDULES = (None, "cascaded", "synchronized")


def _cells():
    for mode, schedule in itertools.product(sorted(ALL_SPIKING_MODES), _SCHEDULES):
        yield mode, schedule


@pytest.mark.parametrize("mode,schedule", list(_cells()))
def test_every_mode_cell_has_a_typed_certification_outcome(mode, schedule):
    policy = policy_for_spiking_mode(mode, schedule)
    observable, reason = policy.certification_observable()
    assert observable in ("counts", "events", "skip"), (
        f"{mode}/{schedule}: unknown observable {observable!r}"
    )
    if observable == "skip":
        assert isinstance(reason, str) and reason.strip(), (
            f"{mode}/{schedule}: skip must carry a non-empty typed reason"
        )
    else:
        assert reason is None


def test_lif_family_certifies_counts():
    observable, reason = policy_for_spiking_mode("lif").certification_observable()
    assert (observable, reason) == ("counts", None)


def test_analytic_ttfs_is_a_typed_skip_never_certified():
    """[§17.5 law] nothing analytic is certified as deployed physics."""
    observable, reason = policy_for_spiking_mode("ttfs").certification_observable()
    assert observable == "skip"
    assert "analytic" in (reason or "")


@pytest.mark.parametrize("mode,schedule", list(_cells()))
def test_gate_certifies_or_skips_typed_for_every_cell(mode, schedule, monkeypatch, capsys):
    """The pipeline gate resolves every cell to certify-or-typed-skip; no
    unhandled exceptions, no silent skips."""
    import mimarsinan.pipelining.core.spike_count_gate as gate_mod

    config = {
        "spiking_mode": mode,
        "spike_count_parity_samples": 2,
        "device": "cpu",
    }
    if schedule is not None:
        config["ttfs_cycle_schedule"] = schedule
    pipeline = SimpleNamespace(config=config)
    # Stop the counts path right after the predicate: sample fetch returns
    # None, which the gate treats as a typed no-op.
    monkeypatch.setattr(
        gate_mod, "_certificate_samples", lambda pipeline, model, n: None,
    )
    result = gate_mod.run_spike_count_certificate_gate(
        pipeline, model=None, ir_graph=None, hybrid_mapping=None,
    )
    assert result is None
    observable, _ = policy_for_spiking_mode(mode, schedule).certification_observable()
    out = capsys.readouterr().out
    if observable != "counts":
        assert "SKIP" in out, f"{mode}/{schedule}: skip must be printed, not silent"
