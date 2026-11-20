"""[U-LIF2 r3] The stored run's own point: a reader of a stored run, record, ledger row or cached entry takes the point
that was resolved when it was written, and never resolves a saved document again under today's entry data.

A run persists its resolved point beside its document (``_RUN_CONFIG/point.json``, written by the session once the
run's plan is resolved). A stored run that persisted none states the historical point: its document's own fields, the
window firing it declared, else ``HISTORICAL_WINDOW_FIRING``. The readers of a run directory: the cost record's reader
(``extract_cost_record_from_run``, ``CostScatter.from_runs``) and the display of a run directory (the live overview's
view, a past run's pipeline view, a managed run's detail), each at the run's stored point.

Red on revision 2: the cost reader resolves a plan from the saved document under today's entry data, the display
derives the saved document under it, and no run persists its point. Every test runs under the four declared sets through
the tests' own widening fixture (``window_firing_declared``). The new symbols are imported inside each test."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

from mimarsinan.mapping.platform.target_expansion import door_read
from mimarsinan.config_schema.runtime import derive_run_config

FWI, ITE = "fire_while_integrating", "integrate_then_emit"
KEY = "lif_window_firing"
_CORES = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 8}]}
#: The four declared sets of the entry's data (its values, its default): today's, the two of option O1, option O2.
_SETTINGS = (((FWI,), FWI), ((FWI, ITE), FWI), ((FWI, ITE), ITE), ((ITE,), ITE))
_SET_IDS = ("fwi~fwi", "fwi+ite~fwi", "fwi+ite~ite", "ite~ite")
#: Stored documents as runs made before the point was persisted saved them: silent on the key (the legacy mode, the
#: family and variant axes, a zero reset whose point a later default refuses) or declaring the value they ran.
_STORED = {
    "legacy lif": {"spiking_mode": "lif"},
    "axes": {"spiking_family": "lif", "spiking_variant": "synchronized", "firing_mode": "Default"},
    "zero reset": {"spiking_family": "lif", "spiking_variant": "synchronized", "firing_mode": "Novena"},
    "declares fwi": {"spiking_family": "lif", "spiking_variant": "synchronized", KEY: FWI},
}


def _new(module: str, name: str):
    return getattr(importlib.import_module(module), name)


@pytest.fixture
def declared(window_firing_declared):
    return window_firing_declared


def _snapshot() -> dict:
    """A SANA-FE snapshot in the shape the step persists: one segment of 120 cores of 256 neurons, 13 steps, T = 4."""
    segment = {"stage_index": 0, "stage_name": "neural_segment_until:mean", "timesteps_executed": 13,
               "spikes": 607726, "per_core": [{"n_neurons": 256}] * 120}
    return {"arch_preset": "loihi", "sample_indices": [0],
            "aggregate": {"sample_count": 1, "total_energy_mj": 0.04733, "total_spikes": 607726, "total_packets": 0},
            "per_sample": [{"sample_index": 0, "T": 4, "segments": [segment]}]}


def _stored_run(run_dir: Path, deployment_parameters: dict) -> str:
    """A generated run directory of an external simulator: its SANA-FE step, its metric and its saved document."""
    (run_dir / "_GUI_STATE").mkdir(parents=True)
    (run_dir / "_RUN_CONFIG").mkdir()
    (run_dir / "__target_metric.json").write_text("0.97")
    steps = {"steps": {"SANA-FE Simulation": {"snapshot": {"sanafe_simulation": _snapshot()}}}}
    (run_dir / "_GUI_STATE" / "steps.json").write_text(json.dumps(steps))
    (run_dir / "_RUN_CONFIG" / "config.json").write_text(json.dumps({"deployment_parameters": deployment_parameters}))
    return str(run_dir)


def _run_point(declared, values, default, deployment_parameters, platform_constraints=None):
    """The point a run resolved at its own time: its document derived under the declared set of that time."""
    from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point

    declared(values, default)
    return resolve_spiking_point(derive_run_config(dict(deployment_parameters), platform_constraints or _CORES))


# -- the cost record's reader ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("stored", sorted(_STORED))
@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r3_a_stored_run_of_an_external_simulator_keeps_its_earlier_key_under_every_declared_set(
        tmp_path, declared, values, default, stored):
    from mimarsinan.chip_simulation.cost_extraction import CostScatter, extract_cost_record_from_run

    run_dir = _stored_run(tmp_path / "run", _STORED[stored])
    declared(values, default)
    for backend in ("sanafe", "loihi"):
        assert extract_cost_record_from_run(run_dir, backend=backend, door=door_read).cell_key == f"lif@{backend}", backend
    assert CostScatter.from_runs([run_dir], door=door_read).cell_keys() == ["lif@sanafe"]


@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r3_a_stored_declaration_is_the_point_its_run_resolved_under_every_declared_set(tmp_path, declared, values,
                                                                                        default):
    """A run that persisted no point and declared integrate then emit ran it (a declaration resolves to itself, or the
    run is refused): its key holds that value under every declared set, today's included."""
    from mimarsinan.chip_simulation.cost_extraction import extract_cost_record_from_run

    run_dir = _stored_run(tmp_path / "run", {**_STORED["axes"], KEY: ITE})
    declared(values, default)
    assert extract_cost_record_from_run(run_dir, door=door_read).cell_key == "lif@sanafe#integrate_then_emit"


@pytest.mark.parametrize("persisted", (FWI, ITE))
@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r3_a_run_is_keyed_by_the_point_it_persisted_under_every_declared_set(tmp_path, declared, values, default,
                                                                               persisted):
    persist_run_point = _new("mimarsinan.chip_simulation.stored_point", "persist_run_point")
    from mimarsinan.chip_simulation.cost_extraction import CostScatter, extract_cost_record_from_run

    silent = _STORED["axes"]
    point = _run_point(declared, (FWI, ITE), persisted, silent)
    run_dir = _stored_run(tmp_path / "run", silent)
    persist_run_point(run_dir, point)
    declared(values, default)
    expected = "lif@sanafe" if persisted == FWI else "lif@sanafe#integrate_then_emit"
    assert extract_cost_record_from_run(run_dir, door=door_read).cell_key == expected
    assert CostScatter.from_runs([run_dir], door=door_read).cell_keys() == [expected]


@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r3_a_value_domain_run_has_no_point_persisted_or_historical(tmp_path, declared, values, default):
    persist_run_point = _new("mimarsinan.chip_simulation.stored_point", "persist_run_point")
    stored_run_point = _new("mimarsinan.chip_simulation.stored_point", "stored_run_point")
    from mimarsinan.chip_simulation.cost_extraction import extract_cost_record_from_run

    document = {"core_semantics": "mvm", "model_type": "simple_mlp"}
    historical = _stored_run(tmp_path / "historical", document)
    persisted = _stored_run(tmp_path / "persisted", document)
    persist_run_point(persisted, None)
    declared(values, default)
    for run_dir in (historical, persisted):
        assert stored_run_point(run_dir, {"deployment_parameters": document}, door=door_read) is None
        assert extract_cost_record_from_run(run_dir, door=door_read).cell_key == "mvm@sanafe"
    os.remove(os.path.join(historical, "_RUN_CONFIG", "config.json"))  # a run that saved no document: skipped
    assert extract_cost_record_from_run(historical, door=door_read) is None


def test_r3_a_persisted_point_reads_back_whole_and_refuses_a_drifted_format(tmp_path, declared):
    persist_run_point = _new("mimarsinan.chip_simulation.stored_point", "persist_run_point")
    stored_run_point = _new("mimarsinan.chip_simulation.stored_point", "stored_run_point")
    point = _run_point(declared, (FWI, ITE), ITE, {**_STORED["axes"], "spike_phase_dither": True},
                       {**_CORES, "membrane_bits": 12, "membrane_signed": True})
    path = persist_run_point(str(tmp_path), point)
    assert point.execution_extensions == ("spike_phase_dither",) and point.soma_law.membrane_bits == 12
    declared((FWI,), FWI)
    assert stored_run_point(str(tmp_path), None, door=door_read) == point
    stored = json.loads(Path(path).read_text())
    Path(path).write_text(json.dumps({**stored, "format_version": 0}))
    with pytest.raises(ValueError, match="format_version"):
        stored_run_point(str(tmp_path), None, door=door_read)
    Path(path).write_text(json.dumps({**stored, "spiking_point": {**stored["spiking_point"], "extra": 1}}))
    with pytest.raises(ValueError, match="extra"):
        stored_run_point(str(tmp_path), None, door=door_read)
    law = {**stored["spiking_point"]["soma_law"], "bias_slot": "head"}  # a law its one constructor does not read back
    Path(path).write_text(json.dumps({**stored, "spiking_point": {**stored["spiking_point"], "soma_law": law}}))
    with pytest.raises(ValueError, match="not as written"):
        stored_run_point(str(tmp_path), None, door=door_read)


# -- the run persists its point where its reader finds it ------------------------------------------------------------

def _session_document(tmp_path, **dp) -> dict:
    return {"experiment_name": "stored_point", "data_provider_name": "MNIST_DataProvider",
            "generated_files_path": str(tmp_path),
            "deployment_parameters": {"model_type": "mlp_mixer", "spiking_family": "lif",
                                      "spiking_variant": "synchronized", "model_config_mode": "user",
                                      "hw_config_mode": "fixed", "model_config": {}, **dp},
            "platform_constraints": {}}


@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r3_the_session_persists_the_point_its_plan_resolved(tmp_path, declared, values, default):
    """[U-LIF2 r4, m-2] Under each of the four declared sets the silent document's session, in a new run directory,
    persists that set's default. Under the default of integrate then emit the session assembles: the entry's data
    alone moves the default, and the backends whose rows do not execute it are derived off."""
    stored_run_point = _new("mimarsinan.chip_simulation.stored_point", "stored_run_point")
    read_run_document = _new("mimarsinan.chip_simulation.stored_point", "read_run_document")
    from conftest import MockDataProviderFactory
    from mimarsinan.pipelining.session import PipelineSession

    declared(values, default)
    session = PipelineSession.from_config(_session_document(tmp_path), data_provider_factory=MockDataProviderFactory())
    run_dir = session.parsed.working_directory
    point = session.pipeline.plan.spiking_point
    assert point.window_firing == default
    persisted = json.loads((Path(run_dir) / "_RUN_CONFIG" / "point.json").read_text())["spiking_point"]
    assert persisted["window_firing"] == default
    for later_values, later_default in _SETTINGS:
        declared(later_values, later_default)
        assert stored_run_point(run_dir, read_run_document(run_dir), door=door_read) == point, later_default


# -- the display of a run directory ----------------------------------------------------------------------------------

def _saved_document(**dp) -> dict:
    return {"seed": 0, "experiment_name": "stored_view", "generated_files_path": "./generated",
            "data_provider_name": "MNIST_DataProvider", "platform_constraints": dict(_CORES),
            "deployment_parameters": {"model_type": "simple_mlp", **dp}}


#: (the saved document's parameters, the point its run resolved at its time: a declared set and default, or None for a
#: run that persisted no point and states the historical one).
_VIEWS = {
    "historical": (_STORED["axes"], None),
    "historical legacy lif": (_STORED["legacy lif"], None),
    "historical zero reset": (_STORED["zero reset"], None),
    "persisted fwi": (_STORED["axes"], ((FWI,), FWI)),
    "persisted ite": (_STORED["axes"], ((FWI, ITE), ITE)),
}


def _view_run(declared, run_dir: Path, case: str):
    """A run directory as its run left it: the saved document, its persisted point (or none), one persisted step; and
    the window firing its run resolved."""
    from mimarsinan.gui.runtime.persistence import save_step_to_persisted
    from mimarsinan.gui.runtime.persistence.store import save_run_info

    dp, at = _VIEWS[case]
    document = _saved_document(**dp)
    (run_dir / "_RUN_CONFIG").mkdir(parents=True)
    (run_dir / "_RUN_CONFIG" / "config.json").write_text(json.dumps(document))
    firing = FWI
    if at is not None:
        point = _run_point(declared, *at, document["deployment_parameters"], document["platform_constraints"])
        _new("mimarsinan.chip_simulation.stored_point", "persist_run_point")(str(run_dir), point)
        firing = point.window_firing
    save_step_to_persisted(str(run_dir), step_name="Pretraining", start_time=0.0, end_time=1.0, target_metric=None,
                           metrics=[], snapshot=None, snapshot_key_kinds=None)
    save_run_info(str(run_dir), pid=os.getpid(), step_names=["Pretraining"])
    return document, firing


def _doc_point() -> dict:
    return {"spiking_family": "lif", "spiking_variant": "synchronized"}


def _shown(view) -> str:
    """The window firing a run directory's view shows, once its point's variant is the one the run resolved."""
    assert view is not None, "the run's view was not built"
    assert view["raw_resolved"]["spiking_variant"] == "synchronized", view["raw_resolved"]["spiking_variant"]
    return view["raw_resolved"][KEY]


@pytest.mark.parametrize("case", sorted(_VIEWS))
@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r3_the_live_overview_derives_the_saved_document_at_the_run_point(tmp_path, declared, values, default, case):
    from mimarsinan.config_schema.display_view import build_pipeline_config_view

    run_dir = tmp_path / "live"
    document, firing = _view_run(declared, run_dir, case)
    live = {**document["deployment_parameters"], KEY: firing}  # the live run's own resolved keys
    declared(values, default)
    assert _shown(build_pipeline_config_view(live, working_dir=str(run_dir))) == firing
    from mimarsinan.chip_simulation.window_firing import declared_window_firings, derived_window_firing

    assert declared_window_firings() == values and derived_window_firing(_doc_point()) == default  # the scope ended


@pytest.mark.parametrize("case", sorted(_VIEWS))
@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r3_a_past_run_view_shows_the_point_its_run_resolved(tmp_path, monkeypatch, declared, values, default, case):
    from mimarsinan.gui.runs import get_run_pipeline

    monkeypatch.setenv("MIMARSINAN_RUNS_ROOT", str(tmp_path))
    _document, firing = _view_run(declared, tmp_path / "past_run", case)
    declared(values, default)
    assert _shown(get_run_pipeline("past_run").get("config_view")) == firing


@pytest.mark.parametrize("case", sorted(_VIEWS))
@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r3_a_managed_run_detail_shows_the_point_its_run_resolved(tmp_path, declared, values, default, case):
    from mimarsinan.gui.runtime.process_monitor import get_run_detail
    from mimarsinan.gui.runtime.process_spawn import ManagedRun

    run_dir = tmp_path / "managed_run"
    _document, firing = _view_run(declared, run_dir, case)
    managed = ManagedRun(run_id="managed_run", working_dir=str(run_dir), pid=os.getpid(), started_at=0.0)
    declared(values, default)
    assert _shown(get_run_detail({"managed_run": managed}, "managed_run").get("config_view")) == firing
