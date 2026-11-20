"""[U-LIF2 f-1] The coverage coordinate and the ledger row carry the window firing a run executed.

A run that integrates then emits occupies its own coverage cell, so its ledger rows never count as coverage of the
cells that fired while integrating. A run that fires while integrating keeps every historical string and key. A stored
row that states no window firing states the historical firing (``HISTORICAL_WINDOW_FIRING``) through the named holder
``recorded_window_firing``, never the declared default, so under option A's default every earlier row keeps its cell.
The coordinate's suffix composes as the certification cell's variant does (``SpikingPoint.point_tag``): the soma law's
tag, then a window firing other than firing while integrating.

Red on the widened tree before f-1: the coordinate reads the soma law alone, and the holder does not exist."""

from __future__ import annotations

import importlib

import pytest

from mimarsinan.config_schema.runtime import derive_run_config

FWI, ITE = "fire_while_integrating", "integrate_then_emit"
_CORES = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 8}]}


def _new(module: str, name: str):
    return getattr(importlib.import_module(module), name)


def _run(**dp):
    return derive_run_config({"model_type": "simple_mlp", "spiking_family": "lif", "spiking_variant": "synchronized",
                              **dp}, dict(_CORES))


def _row(**fields):
    """A science ledger row at the (lif, synchronized) point, as a campaign writes it."""
    return {"deployment_validity": "VALID", "model": "simple_mlp", "dataset": "mnist", "spiking_mode": "lif", **fields}


@pytest.mark.parametrize("declared,expected", [(None, f"lif+{ITE}"), (FWI, "lif"), (ITE, f"lif+{ITE}")])
def test_f1_the_coordinate_of_a_resolved_run_carries_its_window_firing(declared, expected):
    from mimarsinan.chip_simulation.hypervolume_axis_encoder import AxisCoordinates, firing_axis
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

    cfg = _run(**({} if declared is None else {"lif_window_firing": declared}))
    assert firing_axis(cfg["spiking_mode"], cfg) == expected
    assert AxisCoordinates.from_plan(DeploymentPlan.resolve(cfg), cfg).firing == expected


@pytest.mark.parametrize("declared", [None, FWI])
def test_f1_a_stored_row_keeps_its_cell_under_the_default_that_moved(declared):
    from mimarsinan.chip_simulation.coverage_rows import row_to_cell
    from mimarsinan.chip_simulation.ledger_schema import normalize_ledger_record

    assert _new("mimarsinan.chip_simulation.window_firing_axis", "DEFAULT_WINDOW_FIRING") == ITE
    historical = _new("mimarsinan.chip_simulation.spiking_point", "HISTORICAL_WINDOW_FIRING")
    row = _row(**({} if declared is None else {"lif_window_firing": declared}))
    out = normalize_ledger_record(row)
    assert row_to_cell(row).firing == out["spiking_mode"] == "lif"
    assert out["window_firing"] == (declared or historical) == FWI
    assert out["hypervolume_cell_key"].startswith("lif@")


def test_f1_an_integrate_then_emit_row_occupies_its_own_cell():
    from mimarsinan.chip_simulation.coverage_rows import row_to_cell
    from mimarsinan.chip_simulation.ledger_schema import normalize_ledger_record

    ite, fwi = _row(lif_window_firing=ITE), _row(lif_window_firing=FWI)
    assert row_to_cell(ite).firing == f"lif+{ITE}" and row_to_cell(fwi).firing == "lif"
    assert row_to_cell(ite).cell_key != row_to_cell(fwi).cell_key
    out_ite, out_fwi = normalize_ledger_record(ite), normalize_ledger_record(fwi)
    assert (out_ite["window_firing"], out_ite["spiking_mode"]) == (ITE, f"lif+{ITE}")
    assert out_ite["hypervolume_cell_key"].startswith(f"lif+{ITE}@")
    assert out_ite["hypervolume_cell_key"] != out_fwi["hypervolume_cell_key"]


@pytest.mark.parametrize("declared,expected", [(None, "lif+ssat12"), (FWI, "lif+ssat12"), (ITE, f"lif+ssat12-{ITE}")])
def test_f1_the_suffix_composes_as_the_certification_cell_variant(declared, expected):
    from mimarsinan.chip_simulation.coverage_rows import row_to_cell

    row = _row(membrane_bits=12, membrane_signed=True, **({} if declared is None else {"lif_window_firing": declared}))
    assert row_to_cell(row).firing == expected


@pytest.mark.parametrize("row", [
    _row(spiking_family="lif", spiking_variant="streamed", lif_window_firing=ITE),
    {"deployment_validity": "VALID", "model": "simple_mlp", "dataset": "mnist", "spiking_mode": "ttfs"},
])
def test_f1_away_from_the_point_no_window_firing_is_carried(row):
    from mimarsinan.chip_simulation.coverage_rows import row_to_cell
    from mimarsinan.chip_simulation.ledger_schema import normalize_ledger_record

    recorded = _new("mimarsinan.chip_simulation.spiking_point", "recorded_window_firing")
    assert recorded(row) is None
    assert normalize_ledger_record(row)["window_firing"] is None
    assert "+" not in row_to_cell(row).firing


def test_f1_the_holder_reads_a_stored_thing_as_written_and_resolves_nothing(monkeypatch):
    recorded = _new("mimarsinan.chip_simulation.spiking_point", "recorded_window_firing")
    historical = _new("mimarsinan.chip_simulation.spiking_point", "HISTORICAL_WINDOW_FIRING")
    import mimarsinan.chip_simulation.spiking_point as sp

    monkeypatch.setattr(sp, "resolve_window_firing", lambda *_a, **_k: pytest.fail("a stored row was resolved"))
    assert recorded({"spiking_mode": "lif"}) == historical == FWI
    assert recorded({"spiking_mode": "lif", "lif_window_firing": ITE}) == ITE
    assert recorded({"spiking_mode": "lif", "lif_window_firing": FWI}) == FWI


@pytest.mark.parametrize("declared,stamped,contract", [(None, FWI, "BIT_PARITY"), (FWI, FWI, "BIT_PARITY"),
                                                       (ITE, ITE, "INAPPLICABLE")])
def test_r6_m1_a_planned_rows_parity_contracts_are_resolved_at_the_rows_own_window_firing(declared, stamped, contract):
    """[U-LIF2 r6, m-1] A planned ledger row's backend contract is read at the window firing the row states (its own,
    else the historical one), the firing the science row of the same run is stamped with: a silent synchronized LIF row
    reads SANA-FE's bit parity with firing while integrating stamped, a row that integrates then emits reads SANA-FE as
    inapplicable. Red on the final tree of r5: the silent row was resolved at the declared default (integrate then
    emit) and read inapplicable while its science row was stamped firing while integrating."""
    from mimarsinan.chip_simulation.ledger_schema import normalize_ledger_record, normalize_planned_ledger_row
    from mimarsinan.chip_simulation.parity_contract import ParityContractKind

    row = _row(backend="sanafe", spiking_family="lif", spiking_variant="synchronized", firing_mode="Default",
               simulation_steps=6, **({} if declared is None else {"lif_window_firing": declared}))
    assert normalize_ledger_record(dict(row))["window_firing"] == stamped
    planned = normalize_planned_ledger_row({"axes": dict(row)})
    assert planned["parity_contracts"]["backend_contract"] == ParityContractKind[contract].value
