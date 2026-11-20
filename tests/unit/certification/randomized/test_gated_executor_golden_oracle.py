"""[U-LIF2 W14, t09's second oracle] The randomized certification's synchronized LIF semantics declares integrate then
emit and runs it on the executor of record (the gated cycle-level executor, never the closed form), and the toolchain's
golden machine (scripts/x3a_golden, the reference model whose synchronized LIF integrates the window and then emits)
agrees with it on every neuron of seeded generated programs.

Red on the executor side's tree: there the semantics' flow sets its two-window flag by a literal and its config resolves
the declared default, so the run's point is not the one its executor runs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def golden(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.syspath_prepend(str(ROOT / "scripts" / "x3a_crosscheck"))
    import golden_vs_code

    return golden_vs_code


@pytest.mark.parametrize("seed", range(1000, 1010))
def test_t09_the_golden_machine_agrees_with_the_gated_executor_of_synchronized_lif(golden, monkeypatch, seed):
    from mimarsinan.certification.randomized.instance import SYNCHRONIZED_LIF
    from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point
    from mimarsinan.certification.randomized.executors import RUNS
    from mimarsinan.models.spiking.hybrid import lif_step

    assert resolve_spiking_point(RUNS[SYNCHRONIZED_LIF].config).window_firing == "integrate_then_emit"
    real = lif_step.run_integrate_then_emit
    flows = []

    def spy(flow, *args, **kwargs):
        flows.append((bool(getattr(flow, "window_count_closed_form", False)), flow.spiking_point().window_firing))
        return real(flow, *args, **kwargs)

    monkeypatch.setattr(lif_step, "run_integrate_then_emit", spy)
    row = golden.check(seed, SYNCHRONIZED_LIF)
    assert row.get("outcome") != "code_error", row
    assert row["neurons"] > 0 and row["code_vs_golden_identity"] == 0, row
    assert flows and set(flows) == {(False, "integrate_then_emit")}
    sys.modules.pop("golden_vs_code", None)
