"""``scripts/residency_plan.py``: the provenance a planned row's label cites (closing round 2). Only members certified
against the identity run are "certified on"; a member certified against the U1 run only (``/u1``, t0_08) is listed
apart, so the label never overstates its certification."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "residency_plan.py"


def _script():
    spec = importlib.util.spec_from_file_location("residency_plan_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_u1_only_member_is_listed_apart_from_the_certified_ones():
    executed = {"t0_03@1:floor": "", "t0_03@1:stateless": "segments", "t0_08@1:floor": "u1",
                "t0_22@0.7:floor": "segments", "t0_26@1.2:floor": "", "t0_53@1.34:floor": "",
                "t0_44@4:floor": "segments"}
    assert _script().provenance(executed) == (
        "t0_03, t0_22, t0_26, t0_44, t0_53; executed, certified against U1 only on t0_08")


def test_a_member_certified_against_identity_on_another_row_is_not_listed_as_u1_only():
    assert _script().provenance({"t0_08@1:floor": "u1", "t0_08@2:floor": ""}) == "t0_08"
    assert _script().provenance({}) == "none yet"


def test_each_row_carries_its_period_census_under_its_label():
    """Acceptance 6: a row's ``period_census`` is S0's census of the row's step, labelled with the row's label."""
    from mimarsinan.mapping.support.period_census import PassPeriodCensus

    script = _script()
    fixture = SCRIPT.parents[1] / "tests" / "fixtures" / "deployment_configs" / "lif_simplemlp.json"
    rows = [r for r in script.member_plan(fixture, 0.5, {}, "none yet", False)["rows"] if "period_census" in r]
    assert rows
    for row in rows:
        census = PassPeriodCensus.from_dict(row["period_census"])
        assert census.label.startswith(row["label"])
        assert len(census.passes) == row["Q"]
        assert (census.carry_out_bytes + census.carry_in_bytes) == row["carry_elements"] * \
            script.CARRY_BYTES_PER_ELEMENT
