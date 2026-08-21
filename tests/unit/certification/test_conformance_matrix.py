"""[cert-plan W4] certification conformance over the mode matrix: every
(spiking_mode × schedule) cell yields exactly one typed outcome — a counts/
events observable or a non-empty skip reason — and the gate honors it."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw
from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.chip_simulation.spiking_semantics import ALL_SPIKING_MODES

_SCHEDULES = (None, "cascaded", "synchronized")

# The soma points the matrix enumerates. A new point must appear HERE or it
# silently inherits another point's certification outcome.
_ODIN_POINT = SomaLaw.resolve({
    "spiking_family": "lif", "spiking_variant": "streamed",
    "firing_mode": "Novena", "firing_granularity": "per_event",
    "membrane_bits": 8,
})
_SOMA_POINTS = ((None, "lawless"), (DEFAULT_SOMA_LAW, "per_cycle"),
                (_ODIN_POINT, "per_event-sat8"))


def _cells():
    for mode, schedule in itertools.product(sorted(ALL_SPIKING_MODES), _SCHEDULES):
        yield mode, schedule


def _point_cells():
    for mode, schedule in _cells():
        for law, point_id in _SOMA_POINTS:
            yield mode, schedule, law, point_id


@pytest.mark.parametrize(
    "mode,schedule,law,point_id",
    [(m, s, law, pid) for m, s, law, pid in _point_cells()],
    ids=[f"{m}-{s}-{pid}" for m, s, _law, pid in _point_cells()],
)
def test_every_mode_cell_has_a_typed_certification_outcome(
    mode, schedule, law, point_id
):
    del point_id
    policy = policy_for_spiking_mode(mode, schedule, soma_law=law)
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


@pytest.mark.parametrize("law,point_id", _SOMA_POINTS,
                         ids=[pid for _law, pid in _SOMA_POINTS])
def test_counts_stay_the_observable_at_every_soma_point(law, point_id):
    """A per-event law changes how many spikes a window holds, not the
    currency that crosses a boundary — the POINT separates the cells instead."""
    del point_id
    policy = policy_for_spiking_mode("lif", soma_law=law)
    assert policy.certification_observable() == ("counts", None)


def test_the_new_point_does_not_inherit_the_default_points_cell():
    from mimarsinan.chip_simulation.certification import CertificationCell

    default = CertificationCell.from_mode_policy(
        policy_for_spiking_mode("lif", soma_law=DEFAULT_SOMA_LAW), backend="hcm")
    odin = CertificationCell.from_mode_policy(
        policy_for_spiking_mode("lif", soma_law=_ODIN_POINT), backend="hcm")
    assert default.cell_key != odin.cell_key


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
