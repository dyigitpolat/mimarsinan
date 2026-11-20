"""[U-LIF2 r4] A run directory holds one point, at the writer as at the readers.

The point a run directory holds is the point it persisted (``_RUN_CONFIG/point.json``), else, where it already holds a
saved document and any step or cache entry, the historical point that document states, else none (an unused directory).
A run in a directory that holds a point takes it before its pipeline is built (a resume with a start step, a rerun in
place from its first step, a run continued from a seeded cache alike): a document silent on ``lif_window_firing``
derives at the stored point, and a document that declares another value is refused by name with both values and the
remedy. The cache seed writes the source run's point into the run it seeds.

Red on revision 3: the session resolves the handed document under today's entry data and writes that point over or beside
the stored run's steps, and the seed copies the entries without their point (the short check's M-1). Every test runs
under the four declared sets through the tests' own widening fixture (``window_firing_declared``). The new symbols are
imported inside each test.

[U-LIF2 r5] A run directory's states are a finite list with one outcome each: unused (no entry, or a document alone),
persisted, historical, or refused by name (a point that cannot be read, or entries without a saved document). Red on
revision 4: an unreadable point raised the statement's own error, the seeds copied entries before they read the
source's point, and entries without a document read as an unused directory. The writer test compares the whole point
record (n-4)."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import itertools
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mimarsinan.mapping.platform.target_expansion import door_read
from mimarsinan.config_schema.runtime import derive_run_config

FWI, ITE = "fire_while_integrating", "integrate_then_emit"
KEY = "lif_window_firing"
_CORES = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 8}]}
#: The four declared sets of the entry's data (its values, its default): today's, the two of option O1, option O2.
_SETTINGS = (((FWI,), FWI), ((FWI, ITE), FWI), ((FWI, ITE), ITE), ((ITE,), ITE))
_SET_IDS = ("fwi~fwi", "fwi+ite~fwi", "fwi+ite~ite", "ite~ite")
#: The torch executors' rows the check's probe widened beside the entry's data (its column form).
_TORCH_ROWS = ("hcm", "unified", "hybrid", "training", "scm")


def _new(module: str, name: str):
    return getattr(importlib.import_module(module), name)


@pytest.fixture
def declared(window_firing_declared):
    return window_firing_declared


def _document(root, **dp) -> dict:
    return {"experiment_name": "stored", "data_provider_name": "MNIST_DataProvider", "generated_files_path": str(root),
            "deployment_parameters": {"model_type": "mlp_mixer", "spiking_family": "lif",
                                      "spiking_variant": "synchronized", "model_config_mode": "user",
                                      "hw_config_mode": "fixed", "model_config": {}, **dp},
            "platform_constraints": {}}


def _session(document: dict):
    """The headless runner's call (``run.py --headless``) with the tests' mock provider: no data is read."""
    from conftest import MockDataProviderFactory
    from mimarsinan.pipelining.session import PipelineSession

    return PipelineSession.from_config(document, data_provider_factory=MockDataProviderFactory())


def _snapshot() -> dict:
    segment = {"stage_index": 0, "stage_name": "neural_segment_until:mean", "timesteps_executed": 13,
               "spikes": 607726, "per_core": [{"n_neurons": 256}] * 120}
    return {"arch_preset": "loihi", "sample_indices": [0],
            "aggregate": {"sample_count": 1, "total_energy_mj": 0.04733, "total_spikes": 607726, "total_packets": 0},
            "per_sample": [{"sample_index": 0, "T": 4, "segments": [segment]}]}


def _finish(run_dir: str) -> None:
    """A finished run's SANA-FE step, its metric and one cache entry, as the pipeline and its monitor store them."""
    from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache

    state = Path(run_dir) / "_GUI_STATE"
    state.mkdir(exist_ok=True)
    steps = {"steps": {"SANA-FE Simulation": {"snapshot": {"sanafe_simulation": _snapshot()}}}}
    (state / "steps.json").write_text(json.dumps(steps))
    (Path(run_dir) / "__target_metric.json").write_text("0.97")
    cache = PipelineCache()
    cache.add("model_config", {"stored": True})
    cache.store(run_dir)


def _made_twin(declared, root, values, default, stored="persisted", **dp):
    """A run made at its own time (under ``values, default``), its directory named after its document as a CLI run's
    is: 'persisted' as the session wrote it, 'historical' as a run made before runs persisted their point; with its
    steps and the whole point record the session persisted (a historical run's persisted twin)."""
    declared(values, default)
    session = _session(_document(root, **dp))
    run_dir = session.parsed.working_directory
    _finish(run_dir)
    twin = _reading(run_dir)[0]
    if stored == "historical":
        os.remove(os.path.join(run_dir, "_RUN_CONFIG", "point.json"))
    return run_dir, [name for name, _ in session.pipeline.steps], twin


def _made(declared, root, values, default, stored="persisted", **dp):
    run_dir, steps, _twin = _made_twin(declared, root, values, default, stored, **dp)
    return run_dir, steps


def _reading(run_dir: str):
    """(the whole point record the directory persisted, else None; the cost record's key of its SANA-FE step, None for
    a directory that holds no saved document, which the reader skips)."""
    from mimarsinan.chip_simulation.cost_extraction import extract_cost_record_from_run

    path = Path(run_dir) / "_RUN_CONFIG" / "point.json"
    persisted = json.loads(path.read_text())["spiking_point"] if path.is_file() else None
    record = extract_cost_record_from_run(run_dir, door=door_read)
    return persisted, None if record is None else record.cell_key


def _firing(run_dir: str):
    """(the window firing the directory persisted, else None; the key)."""
    persisted, key = _reading(run_dir)
    return (persisted or {}).get("window_firing"), key


def _saved(run_dir: str) -> bytes:
    return (Path(run_dir) / "_RUN_CONFIG" / "config.json").read_bytes()


def _handed(root, run_dir: str, way: str, start: str, **dp):
    """The document a later run is handed and the directory it writes: the run's own document handed back with a start
    step in its directory (resume), or from its first step under its CLI name (rerun); or the earlier run continued
    into a new directory by the GUI's spawn (its three seeds), by the steps file's seed alone (continue) or by the cache
    seed alone (cache)."""
    document = _document(root, **dp)
    if way == "resume":
        return {**document, "_working_directory": run_dir, "start_step": start}, run_dir
    if way == "rerun":
        return document, run_dir
    if way == "spawn":
        from mimarsinan.gui.runtime import process_spawn

        with patch.object(process_spawn.subprocess, "Popen", return_value=MagicMock(pid=os.getpid())), \
                patch.object(process_spawn, "start_console_reader"), \
                patch.object(process_spawn, "read_proc_starttime", return_value=None):
            run_id = process_spawn.spawn_run({}, str(root), {**document, "start_step": start,
                                                            "_continue_from_run_id": os.path.basename(run_dir)})
        continued = os.path.join(str(root), run_id)
        return json.loads(_saved(continued)), continued
    from mimarsinan.gui.runtime.run_cache_seed import (copy_pipeline_cache_from_previous_run,
                                                       copy_steps_json_from_previous_run)

    seed = {"continue": copy_steps_json_from_previous_run, "cache": copy_pipeline_cache_from_previous_run}[way]
    continued = os.path.join(str(root), "continued")
    os.makedirs(os.path.join(continued, "_GUI_STATE"))
    seed(str(root), os.path.basename(run_dir), continued)
    return {**document, "_working_directory": continued, "start_step": start}, continued


def _seeded_key(way: str, ran: bool):
    """The key a continued directory's SANA-FE step reads: none where it holds no steps file (the cache seed alone) or
    no saved document (the steps file's seed alone, refused before the session wrote one)."""
    return None if way == "cache" or (way == "continue" and not ran) else "lif@sanafe"


def _refused(document: dict, *parts: str) -> str:
    with pytest.raises(ValueError) as refused:
        _session(document)
    message = str(refused.value)
    assert all(part in message for part in parts), (parts, message)
    return message


# -- a stored run that fired while integrating keeps its point and its key ----------------------------------------------

@pytest.mark.parametrize("way", ("resume", "rerun", "spawn", "continue", "cache"))
@pytest.mark.parametrize("stored", ("persisted", "historical"))
@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r4_a_stored_fwi_run_resumed_rerun_or_continued_keeps_its_point_and_its_key(tmp_path, declared, values,
                                                                                    default, stored, way):
    """Resumed in place, rerun in place from its first step, and continued through the cache seed (the GUI's spawn, the
    steps file's seed alone, the cache seed alone): the later run takes fire while integrating and the earlier run's
    SANA-FE step keeps the key ``lif@sanafe``. Under a declared set without that value the later run is refused by name, and nothing
    of the stored run moves. [U-LIF2 r5, n-4] The whole point record is kept: the later run persists the stored record
    (persisted), or its persisted twin's (historical), field for field."""
    run_dir, steps, twin = _made_twin(declared, tmp_path, (FWI,), FWI, stored)
    before, saved = _reading(run_dir), _saved(run_dir)
    assert twin["window_firing"] == FWI
    assert before == (twin if stored == "persisted" else None, "lif@sanafe")
    declared(values, default)
    document, written = _handed(tmp_path, run_dir, way, steps[-1])
    if FWI in values:
        session = _session(document)
        assert session.parsed.working_directory == written
        assert session.pipeline.plan.spiking_point.window_firing == FWI
        assert _reading(written) == (twin, _seeded_key(way, ran=True))
    else:
        _refused(document, KEY, repr(FWI), "new run directory")
        if written == run_dir:
            assert _reading(run_dir) == before and _saved(run_dir) == saved
        else:  # the seeded point, beside the spawn's document (a seed alone wrote none)
            assert _reading(written) == (twin, _seeded_key(way, ran=False))
    if written != run_dir:
        assert _reading(run_dir) == before and _saved(run_dir) == saved


@pytest.mark.parametrize("way", ("resume", "rerun", "spawn", "continue", "cache"))
@pytest.mark.parametrize("stored", ("persisted", "historical"))
@pytest.mark.parametrize("values,default", _SETTINGS[1:], ids=_SET_IDS[1:])
def test_r4_a_declaration_of_another_value_on_a_used_directory_is_refused_by_name(tmp_path, declared, values, default,
                                                                                  stored, way):
    """A document that declares integrate then emit for a directory that holds a run at fire while integrating is
    refused by name with both values and the remedy (a new run directory, or declare the stored value), under every
    declared set that offers integrate then emit; the directory keeps its point, its key and its saved document."""
    run_dir, steps, twin = _made_twin(declared, tmp_path, (FWI,), FWI, stored)
    before, saved = _reading(run_dir), _saved(run_dir)
    declared(values, default)
    document, written = _handed(tmp_path, run_dir, way, steps[-1], **{KEY: ITE})
    _refused(document, KEY, repr(ITE), repr(FWI), "new run directory", f"declare {KEY}={FWI!r}")
    assert _reading(run_dir) == before and _saved(run_dir) == saved
    if written != run_dir:  # the seeded point, beside the spawn's document (a seed alone wrote none)
        assert _reading(written) == (twin, _seeded_key(way, ran=False))


@pytest.mark.parametrize("made", _SETTINGS[2:], ids=_SET_IDS[2:])
@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r4_a_new_directory_under_the_ite_default_persists_it_and_keeps_it(tmp_path, declared, made, values, default):
    """A run made in a new directory under the default of integrate then emit persists it (its SANA-FE step keyed
    ``lif@sanafe#integrate_then_emit``); resumed in place under each declared set it keeps it, or is refused by name
    where the set does not offer it."""
    run_dir, steps = _made(declared, tmp_path, *made)
    expected = (ITE, "lif@sanafe#integrate_then_emit")
    assert _firing(run_dir) == expected
    declared(values, default)
    document, _written = _handed(tmp_path, run_dir, "resume", steps[-1])
    if ITE in values:
        assert _session(document).pipeline.plan.spiking_point.window_firing == ITE
    else:
        _refused(document, KEY, repr(ITE), "new run directory")
    assert _firing(run_dir) == expected


@pytest.mark.parametrize("stored", ("persisted", "historical"))
@pytest.mark.parametrize("values,default", _SETTINGS[2:3], ids=_SET_IDS[2:3])
def test_r4_a_zero_reset_run_resumed_under_the_ite_default_keeps_its_point_and_its_cli_rerun_is_refused(
        tmp_path, declared, values, default, stored):
    """A stored zero-reset run (fire while integrating, the one value its point admits) resumed in its declared
    directory under the default of integrate then emit derives at its stored point. A CLI rerun's directory is named
    after the pipeline mode the run derives, so the doors judge its silent document at today's default before the
    directory is known: it is refused by name (declare the value), and nothing of the directory moves (a stated
    limit of the rule's form)."""
    reset = {"firing_mode": "Novena"}
    run_dir, steps = _made(declared, tmp_path, (FWI,), FWI, stored, **reset)
    before, saved = _reading(run_dir), _saved(run_dir)
    declared(values, default)
    document, _written = _handed(tmp_path, run_dir, "rerun", steps[-1], **reset)
    _refused(document, KEY, repr(ITE), f"declare {KEY}")
    assert _reading(run_dir) == before and _saved(run_dir) == saved
    document, _written = _handed(tmp_path, run_dir, "resume", steps[-1], **reset)
    assert _session(document).pipeline.plan.spiking_point.window_firing == FWI
    assert _firing(run_dir) == (FWI, "lif@sanafe")


@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r4_a_document_away_from_the_window_firings_point_is_not_judged_on_it(tmp_path, declared, values, default):
    """The rule judges the window firing where the handed document is at its point: a document at another family in a
    directory that held a run at fire while integrating is the pipeline's own case, not the key's, under every set."""
    run_dir, _steps = _made(declared, tmp_path, (FWI,), FWI)
    declared(values, default)
    document = _document(tmp_path, spiking_family="ttfs", spiking_variant="cascaded")
    session = _session(document)
    assert session.parsed.working_directory == run_dir and session.pipeline.plan.spiking_point.window_firing is None


# -- the check's probe k_resume as a test of record ---------------------------------------------------------------------

@pytest.mark.parametrize("tree", ("today", "entry data", "entry data and the torch rows"))
def test_r4_k_resume_a_finished_run_resumed_and_continued_on_a_later_tree_keeps_its_point(tmp_path, declared,
                                                                                          monkeypatch, tree):
    """The short check's probe k_resume (check_r3_evidence/probes/k_resume.py) on its three trees: a run made today and
    finished with a SANA-FE step; then, on the later tree, a resume in place (a copy of the run, its saved document
    handed back with the run's last step as the start) and an edit and continue (the steps file seeded by the GUI's own
    copy into a new directory, the earlier run's saved document with no edit). The later trees: today's; the entry's
    data at option A (both values, integrate then emit by default); and that with the five torch executor rows
    declaring both values (the probe's column form)."""
    import shutil

    from mimarsinan.chip_simulation import spiking_semantics
    from mimarsinan.chip_simulation.stored_point import read_run_document
    from mimarsinan.gui.runtime.run_cache_seed import copy_steps_json_from_previous_run

    declared((FWI,), FWI)
    orig = str(tmp_path / "orig")
    first = _session({**_document(tmp_path), "experiment_name": "orig", "_working_directory": orig})
    start = [name for name, _ in first.pipeline.steps][-1]
    _finish(orig)
    assert _firing(orig) == (FWI, "lif@sanafe")
    if tree != "today":
        declared((FWI, ITE), ITE)
    if tree == "entry data and the torch rows":
        caps = dict(spiking_semantics._BACKEND_CAPS)
        for row in _TORCH_ROWS:
            caps[row] = dataclasses.replace(caps[row], window_firings=frozenset({FWI, ITE}))
        monkeypatch.setattr(spiking_semantics, "_BACKEND_CAPS", caps)
    resumed = str(tmp_path / "resumed")
    shutil.copytree(orig, resumed)
    session = _session({**read_run_document(resumed), "_working_directory": resumed, "start_step": start})
    assert session.pipeline.plan.spiking_point.window_firing == FWI and _firing(resumed) == (FWI, "lif@sanafe")
    continued = str(tmp_path / "continued")
    os.makedirs(os.path.join(continued, "_GUI_STATE"))
    copy_steps_json_from_previous_run(str(tmp_path), "orig", continued)
    session = _session({**read_run_document(orig), "_working_directory": continued, "start_step": start})
    assert session.pipeline.plan.spiking_point.window_firing == FWI and _firing(continued) == (FWI, "lif@sanafe")
    assert _firing(orig) == (FWI, "lif@sanafe")


# -- (a): the directory's own point and the files its entries live in ---------------------------------------------------

def test_r4_a_run_directory_holds_a_point_once_it_holds_a_document_and_an_entry(tmp_path, declared):
    """The directory's own point: the persisted point, else the historical statement where the directory holds a saved
    document and any step or cache entry (the files their owners write), else none. [U-LIF2 r5, n-2] Entries without a
    saved document are refused by name."""
    run_directory_point = _new("mimarsinan.chip_simulation.stored_point", "run_directory_point")
    entry_files = _new("mimarsinan.chip_simulation.stored_point", "RUN_ENTRY_FILES")
    refusal = _new("mimarsinan.chip_simulation.stored_point", "DirectoryPointRefusal")
    from mimarsinan.gui.runtime.persistence.paths import steps_path
    from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache

    owners = tmp_path / "owners"
    PipelineCache().store(str(owners))
    assert {str(path.relative_to(owners)) for path in owners.rglob("*") if path.is_file()} == {entry_files[0]}
    assert steps_path(str(owners)) == owners / entry_files[1]
    run_dir, _steps = _made(declared, tmp_path, (FWI,), FWI, "historical")
    assert run_directory_point(str(tmp_path / "unused"), door=door_read) is None
    held = run_directory_point(run_dir, door=door_read)
    assert held.point.window_firing == FWI and not held.persisted
    document = Path(run_dir) / "_RUN_CONFIG" / "config.json"
    saved = document.read_bytes()
    document.unlink()
    with pytest.raises(refusal) as refused:  # entries without a saved document: refused by name
        run_directory_point(run_dir, door=door_read)
    assert run_dir in str(refused.value) and "restore its _RUN_CONFIG" in str(refused.value)
    document.write_bytes(saved)
    for entry in entry_files:
        os.remove(os.path.join(run_dir, entry))
    assert run_directory_point(run_dir, door=door_read) is None  # a document alone: a run refused before any step (f-4)
    declared((FWI, ITE), ITE)
    _session(_document(tmp_path))
    held = run_directory_point(run_dir, door=door_read)
    assert held.persisted and held.point.window_firing == ITE


# -- m-1: the historical statement reads both sections through the one constructor --------------------------------------

@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r4_m1_a_stored_run_with_a_signed_register_reads_back_its_own_tag_under_every_declared_set(tmp_path, declared,
                                                                                                   values, default):
    """A stored run that persisted no point, silent on the key, its 12-bit signed register declared in the platform
    section: its stored point is the point its own door resolved at its time, tag ``ssat12``, under every set."""
    from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point
    from mimarsinan.chip_simulation.stored_point import read_run_document, stored_run_point

    dp = {"model_type": "simple_mlp", "spiking_family": "lif", "spiking_variant": "synchronized"}
    platform = {**_CORES, "membrane_bits": 12, "membrane_signed": True}
    declared((FWI,), FWI)
    own = resolve_spiking_point(derive_run_config(dict(dp), platform))
    assert own.soma_law.point_tag() == "ssat12"
    (tmp_path / "_RUN_CONFIG").mkdir()
    (tmp_path / "_RUN_CONFIG" / "config.json").write_text(json.dumps({"deployment_parameters": dp,
                                                                      "platform_constraints": platform}))
    declared(values, default)
    stated = stored_run_point(str(tmp_path), read_run_document(str(tmp_path)), door=door_read)
    assert stated.soma_law == own.soma_law and stated.soma_law.point_tag() == "ssat12"
    assert stated == own


# -- f-2: nevresim's compile key covers every field its row admits more than one value of ------------------------------

#: The point basis a document declares, over the vocabulary of each axis the capability rows judge: the mode (family
#: and variant), the soma law (reset, granularity, register), the window firing and an execution extension.
_LIF_AXES = (("lif", "streamed"), ("lif", "synchronized"))
_TTFS_AXES = (("ttfs", "analytical"), ("ttfs", "quantized"), ("ttfs", "synchronized"), ("ttfs", "cascaded"))
_LAWS = [dict(firing_mode=mode, firing_granularity=granularity)
         for mode, granularity in itertools.product(("Default", "Novena"), ("per_cycle", "per_event"))]
_REGISTERS = ({}, {"membrane_bits": 8}, {"membrane_bits": 8, "membrane_signed": True})
_DECLARATIONS = ({}, {KEY: FWI}, {KEY: ITE}, {"spike_phase_dither": True})


def _judged(point) -> tuple:
    """The fields of a point the capability rows judge (``require_backend_executes``): the mode, the soma law, the
    window firing and the execution extensions."""
    return point.spiking_mode, point.soma_law, point.window_firing, point.execution_extensions


def _nevresim_key(point, tmp_path, monkeypatch) -> str:
    """The compile key nevresim's own driver builds for ``point`` (its chip emission stubbed: no compile, no file of
    record), at one mapping, one window and one latency."""
    from mimarsinan.chip_simulation.nevresim import nevresim_driver as driver

    monkeypatch.setattr(driver, "hard_cores_to_chip", lambda *a, **k: SimpleNamespace(get_chip_json=lambda: "{}"))
    monkeypatch.setattr(driver, "save_weights_and_chip_code", lambda *a, **k: None)
    monkeypatch.setattr(driver, "mapping_connectivity_hash", lambda _mapping: "m" * 64)
    monkeypatch.setattr(driver.NevresimDriver, "nevresim_path", str(tmp_path))
    law = point.soma_law
    mapping = SimpleNamespace(axons_per_core=1, neurons_per_core=1, cores=[])  # [composed head, rule 12] U-CSOMA
    chip = driver.NevresimDriver(1, mapping, str(tmp_path), float, spike_generation_mode="Uniform",
                                 thresholding_mode=law.thresholding_mode, firing_mode=law.firing_mode, point=point,
                                 connectivity_mode="runtime", verbose=False)
    return chip._compile_cache_key(8, 2)


def test_r4_f2_nevresims_compile_key_covers_every_field_its_row_admits_more_than_one_value_of(tmp_path, declared,
                                                                                              monkeypatch):
    """Every point nevresim's row executes, over the vocabulary of the judged axes at the widest declared set: a judged
    field of which these points take more than one value (where it applies) is a field the row admits more than one
    value of, and two points that differ in such a field never share a compile key (a binary is served on the key
    alone). Green while the key holds every such field; red the moment the row admits a second value of a field the
    key does not hold (a window firing, an execution extension)."""
    from mimarsinan.chip_simulation.point_capability import require_backend_executes
    from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point

    declared((FWI, ITE), FWI)
    documents = [dict(zip(("spiking_family", "spiking_variant"), axes)) for axes in _TTFS_AXES]
    documents += [{"spiking_family": family, "spiking_variant": variant, **law, **register, **declaration}
                  for (family, variant), law, register, declaration
                  in itertools.product(_LIF_AXES, _LAWS, _REGISTERS, _DECLARATIONS)]
    executed = {}
    for document in documents:
        platform = {**_CORES, **{k: document.pop(k) for k in ("membrane_bits", "membrane_signed") if k in document}}
        try:
            point = resolve_spiking_point(derive_run_config({"model_type": "simple_mlp", **document}, platform))
            require_backend_executes(point, backend="nevresim", context="the compile key's test")
        except ValueError:
            continue
        executed[_judged(point)] = point
    fields = ("spiking_mode", "soma_law", "window_firing", "execution_extensions")
    admitted = {name: {judged[index] for judged in executed} - {None} for index, name in enumerate(fields)}
    checked = [index for index, name in enumerate(fields) if len(admitted[name]) > 1]
    assert {fields[index] for index in checked} >= {"spiking_mode", "soma_law"}, admitted
    keys = {judged: _nevresim_key(point, tmp_path, monkeypatch) for judged, point in executed.items()}
    shared = [(a, b) for a, b in itertools.combinations(executed, 2)
              if keys[a] == keys[b] and any(a[index] != b[index] for index in checked)]
    assert shared == [], shared[:3]


# -- [U-LIF2 r5] a run directory's states: a finite list with one outcome each ------------------------------------------

_SP = "mimarsinan.chip_simulation.stored_point"
#: The words every refusal of a directory's point carries: the rule and the remedy.
_RULE, _REMEDY = "a run directory holds one point", "Run in a new run directory that reuses none of its entries"
#: A run directory's states (ruling (g) of 2026-11-13), each built from a stored run that fired while integrating, with
#: the one outcome of every run that meets it: 'default' (the declared set's default, persisted), 'stored' (the stored
#: point, its whole record, where the declared set offers its value, else refused by name) or 'refused' (by name: the
#: directory, the rule and the remedy, with the stored directory byte-equal and no entry copied).
_STATES = {
    "unused, no entry": "default",
    "unused, a document alone": "default",
    "persisted": "stored",
    "historical": "stored",
    "a point of another format": "refused",
    "a point that is no JSON": "refused",
    "a document whose point does not resolve": "refused",
    "entries without a document": "refused",
}
#: n-1: the states of a point that cannot be read.
_UNREADABLE = ("a point of another format", "a point that is no JSON", "a document whose point does not resolve")
#: Every way a run meets a run directory: in place (a resume in its declared directory; a CLI rerun from a later step or
#: from its first step; its first step in its declared directory), or in a new directory seeded with its entries (the
#: cache seed, the steps file's seed, the GUI's spawn with its three seeds).
_WAYS = ("resume", "rerun from a later step", "rerun from the first step", "from the first step in its directory",
         "cache seed", "steps seed", "spawn")
_IN_PLACE = _WAYS[:4]


def _files(directory) -> dict:
    """Every file under ``directory`` with its bytes (none for a directory that does not exist)."""
    root = Path(directory)
    if not root.is_dir():
        return {}
    held = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                held[str(path.relative_to(root))] = path.read_bytes()
            except PermissionError:
                held[str(path.relative_to(root))] = "unreadable"
    return held


def _entries_and_point(directory) -> list:
    """The entry files and the point file ``directory`` holds."""
    names = (*_new(_SP, "RUN_ENTRY_FILES"), os.path.join("_RUN_CONFIG", "point.json"))
    return [name for name in names if os.path.isfile(os.path.join(str(directory), name))]


def _rewrite(path: Path, edit) -> None:
    path.write_text(json.dumps(edit(json.loads(path.read_text()))))


def _in_state(run_dir: str, state: str) -> None:
    """Put a stored run that fired while integrating into ``state``."""
    run = Path(run_dir)
    config = run / "_RUN_CONFIG"
    point = config / "point.json"
    if state == "unused, no entry":
        shutil.rmtree(run)
        run.mkdir()
    elif state == "unused, a document alone":
        for path in [path for path in run.rglob("*") if path.is_file() and path != config / "config.json"]:
            path.unlink()
    elif state == "historical":
        point.unlink()
    elif state == "a point of another format":
        _rewrite(point, lambda stored: {**stored, "format_version": 2})
    elif state == "a point that is no JSON":
        point.write_text("{")
    elif state == "a document whose point does not resolve":
        point.unlink()
        _rewrite(config / "config.json", lambda document: {**document, "deployment_parameters": {
            **document["deployment_parameters"], "spiking_family": "no_such_family"}})
    elif state == "entries without a document":
        shutil.rmtree(config)
    else:
        assert state == "persisted", state


def _met(root: Path, run_dir: str, way: str, start: str):
    """The document a later run is handed and the directory it writes; a seed's or the spawn's refusal propagates."""
    document = _document(root)
    if way in _IN_PLACE:
        return {"resume": {**document, "_working_directory": run_dir, "start_step": start},
                "rerun from a later step": {**document, "start_step": start},
                "rerun from the first step": document,
                "from the first step in its directory": {**document, "_working_directory": run_dir}}[way], run_dir
    if way == "spawn":
        return _handed(root, run_dir, "spawn", start)
    from mimarsinan.gui.runtime.run_cache_seed import (copy_pipeline_cache_from_previous_run,
                                                       copy_steps_json_from_previous_run)

    seed = {"cache seed": copy_pipeline_cache_from_previous_run, "steps seed": copy_steps_json_from_previous_run}[way]
    continued = os.path.join(str(root), "continued")
    os.makedirs(continued)
    seed(str(root), os.path.basename(run_dir), continued)
    return {**document, "_working_directory": continued, "start_step": start}, continued


def _other_directory(root: Path, run_dir: str):
    """The directory a seed or the spawn made beside the stored run (None for a way in place)."""
    others = [str(path) for path in Path(root).iterdir() if path.is_dir() and str(path) != run_dir]
    assert len(others) <= 1, others
    return others[0] if others else None


@pytest.mark.parametrize("way", _WAYS)
@pytest.mark.parametrize("state", tuple(_STATES))
@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r5_a_run_directory_is_in_one_state_of_a_finite_list_and_every_way_a_run_meets_it_has_one_outcome(
        tmp_path, declared, values, default, state, way):
    """[U-LIF2 r5] The finite list: a run directory is unused (no entry, or a document alone), persisted, historical,
    or refused by name (a point that cannot be read, entries without a saved document). Under the four declared sets
    every way a run meets it takes the state's one outcome: an unused directory gives the set's default; a persisted or
    historical one its stored point, whole, or a refusal by name where the set does not offer its value; a refused one
    a refusal by name with the rule and the remedy, before a seed copies any entry. After a refusal the stored
    directory is byte-equal."""
    from mimarsinan.chip_simulation.spiking_point import point_record

    refusal = _new(_SP, "DirectoryPointRefusal")
    run_dir, steps, twin = _made_twin(declared, tmp_path, (FWI,), FWI)
    _in_state(run_dir, state)
    before = _files(run_dir)
    declared(values, default)
    outcome = _STATES[state]
    if outcome == "stored" and FWI not in values:
        outcome = "refused"
    document = refused = session = None
    try:
        document, _written = _met(tmp_path, run_dir, way, steps[-1])
        session = _session(document)
    except refusal as raised:
        refused = str(raised)
    written = run_dir if way in _IN_PLACE else _other_directory(tmp_path, run_dir)
    if outcome == "refused":
        assert refused is not None, (state, way, values, default)
        assert _RULE in refused and _REMEDY in refused and (run_dir in refused or written in refused), refused
        assert _files(run_dir) == before
        if _STATES[state] == "refused" and way not in _IN_PLACE:  # the seed refused the source before its first copy
            assert document is None and _entries_and_point(written) == [], (state, way)
        return
    assert refused is None and session is not None, (state, way, refused)
    plan = session.pipeline.plan.spiking_point
    assert session.parsed.working_directory == written
    if outcome == "default":
        assert plan.window_firing == default and _reading(written)[0] == point_record(plan)
    else:
        assert plan.window_firing == FWI and _reading(written) == (twin, None if way == "cache seed" else "lif@sanafe")
    if written != run_dir:
        assert _files(run_dir) == before


@pytest.mark.parametrize("way", ("resume", "rerun from a later step", "cache seed", "steps seed"))
@pytest.mark.parametrize("state", _UNREADABLE)
@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r5_n1_a_point_that_cannot_be_read_is_refused_by_name_and_a_seed_reads_it_before_its_first_copy(
        tmp_path, declared, values, default, state, way):
    """[U-LIF2 r5, n-1] A stored point of another format, a point file that is no JSON, and a saved document whose point
    does not resolve: the resume, the CLI rerun, the cache seed and the steps seed each raise ``DirectoryPointRefusal``
    naming the directory, the cause, the rule and the remedy; the stored directory's files are byte-equal after, and
    the seeded directory holds no entry and no point."""
    refusal = _new(_SP, "DirectoryPointRefusal")
    run_dir, steps, _twin = _made_twin(declared, tmp_path, (FWI,), FWI)
    _in_state(run_dir, state)
    before = _files(run_dir)
    declared(values, default)
    with pytest.raises(refusal) as refused:
        _session(_met(tmp_path, run_dir, way, steps[-1])[0])
    message = str(refused.value)
    assert run_dir in message and "holds a point that cannot be read" in message, message
    assert type(refused.value.__cause__).__name__ in message and _RULE in message and _REMEDY in message, message
    assert "repair its _RUN_CONFIG/point.json or config.json" in message, message
    assert _files(run_dir) == before
    if way not in _IN_PLACE:
        assert _entries_and_point(_other_directory(tmp_path, run_dir)) == []


@pytest.mark.parametrize("way", ("resume", "rerun from a later step", "cache seed", "steps seed"))
@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r5_n2_entries_without_a_saved_document_are_refused_by_name(tmp_path, declared, values, default, way):
    """[U-LIF2 r5, n-2] A stored run that fired while integrating with its ``_RUN_CONFIG`` removed (entries, no point,
    no saved document): the resume, the CLI rerun and both seeds are refused by name, naming the directory, its entries,
    the rule and the remedy; nothing is written (no point, no document, no seeded entry)."""
    refusal = _new(_SP, "DirectoryPointRefusal")
    entry_files = _new(_SP, "RUN_ENTRY_FILES")
    run_dir, steps, _twin = _made_twin(declared, tmp_path, (FWI,), FWI)
    _in_state(run_dir, "entries without a document")
    before = _files(run_dir)
    declared(values, default)
    with pytest.raises(refusal) as refused:
        _session(_met(tmp_path, run_dir, way, steps[-1])[0])
    message = str(refused.value)
    assert run_dir in message and f"holds entries ({', '.join(entry_files)})" in message, message
    assert "no point and no saved document" in message and _RULE in message and _REMEDY in message, message
    assert "restore its _RUN_CONFIG" in message, message
    assert "cannot be read" not in message and refused.value.__cause__ is None, message  # its own refusal, not wrapped
    assert _files(run_dir) == before
    if way not in _IN_PLACE:
        assert _entries_and_point(_other_directory(tmp_path, run_dir)) == []


@pytest.mark.parametrize("seed", ("cache seed", "steps seed"))
@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r5_n5_a_seed_from_a_document_only_source_copies_no_entry_and_writes_no_point(tmp_path, declared, values,
                                                                                     default, seed):
    """[U-LIF2 r5, n-5] A source run that holds its saved document alone (a run refused before any step): each seed
    copies no entry and writes no point, under every declared set, and the source is unchanged."""
    run_dir, steps, _twin = _made_twin(declared, tmp_path, (FWI,), FWI)
    _in_state(run_dir, "unused, a document alone")
    before = _files(run_dir)
    declared(values, default)
    _document_handed, continued = _met(tmp_path, run_dir, seed, steps[-1])
    assert _files(continued) == {} and _files(run_dir) == before


def _point_shapes(stored: dict) -> dict:
    """point.json: absent, its readable form (the stored run's own file), and shapes outside that form (a pair
    ('unreadable', text): the text written, then its read permission removed)."""
    point = stored["spiking_point"]
    return {"absent": None, "readable": json.dumps(stored),
            "another format version": json.dumps({**stored, "format_version": 2}),
            "no format version": json.dumps({"spiking_point": point}),
            "no point field": json.dumps({"format_version": 1}), "no JSON": "{", "empty": "", "a list": "[]",
            "null": "null", "a string": '"x"', "a point that is a string": json.dumps({**stored, "spiking_point": "x"}),
            "a point that is a list": json.dumps({**stored, "spiking_point": []}),
            "an empty point": json.dumps({**stored, "spiking_point": {}}),
            "a point without its soma law": json.dumps({**stored, "spiking_point": {
                key: value for key, value in point.items() if key != "soma_law"}}),
            "bytes that are no text": b"\xff\xfe\x00", "no read permission": ("unreadable", json.dumps(stored))}


def _document_shapes(saved: dict) -> dict:
    """config.json: absent, its readable form (the stored run's own document), and shapes outside that form."""
    params = saved["deployment_parameters"]
    return {"absent": None, "readable": json.dumps(saved), "no JSON": "{", "empty": "", "a list": "[]", "null": "null",
            "a string": '"x"', "parameters that are a list": json.dumps({**saved, "deployment_parameters": []}),
            "parameters that are a string": json.dumps({**saved, "deployment_parameters": "x"}),
            "no parameters": json.dumps({k: v for k, v in saved.items() if k != "deployment_parameters"}),
            "a platform that is a list": json.dumps({**saved, "platform_constraints": []}),
            "an unknown family": json.dumps({**saved, "deployment_parameters": {**params, "spiking_family": "zz"}}),
            "an unknown variant": json.dumps({**saved, "deployment_parameters": {**params, "spiking_variant": "zz"}}),
            "bytes that are no text": b"\xff\xfe\x00", "no read permission": ("unreadable", json.dumps(saved)),
            **_target_shapes(saved)}


def _write_shape(path: Path, shape) -> None:
    if shape is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(shape, bytes):
        path.write_bytes(shape)
    elif isinstance(shape, tuple):
        path.write_text(shape[1])
        path.chmod(0)
    else:
        path.write_text(shape)


def _shape_state(point: str, document: str, entries: tuple) -> str:
    """The state the files name: the point file decides where it exists; else no entry is an unused directory; else a
    readable document states the historical point (a target-carried one the point its door resolves); any other
    document is refused (a target that no longer expands among them)."""
    if point == "readable":
        return "persisted"
    if point != "absent":
        return "refused"
    if not entries:
        return "unused"
    return "historical" if document == "readable" or document in _STATABLE_TARGETS else "refused"


@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r5_the_list_is_closed_every_generated_directory_is_in_the_one_state_its_files_name(tmp_path, declared, values,
                                                                                          default, soma_targets):
    """[U-LIF2 r5] The list is closed by construction: over every combination of a point file, a saved document (each
    absent, in its readable form, or in a shape outside that form) and the entry files, the directory's point is the
    state the files name: persisted or historical (the stored run's own record, whole), unused, or a refusal by name
    (the directory, the rule and the remedy), never another error and never a point the files do not state; the
    directory is never written. [U-LIF2 r6] The documents include target-carried ones (the core semantics from one
    target, the soma keys from two others, a target that no longer expands): a historical directory holds the point its
    document's own door resolves."""
    from mimarsinan.chip_simulation.spiking_point import point_record

    run_directory_point = _new(_SP, "run_directory_point")
    refusal = _new(_SP, "DirectoryPointRefusal")
    entry_files = _new(_SP, "RUN_ENTRY_FILES")
    made, _steps, twin = _made_twin(declared, tmp_path / "made", (FWI,), FWI)
    stored = json.loads((Path(made) / "_RUN_CONFIG" / "point.json").read_text())
    saved = json.loads((Path(made) / "_RUN_CONFIG" / "config.json").read_text())
    entry_bytes = {entry: (Path(made) / entry).read_bytes() for entry in entry_files}
    declared(values, default)
    points, documents = _point_shapes(stored), _document_shapes(saved)
    stated = {name: _door_point(json.loads(documents[name])) for name in _TARGET_SHAPES if name in _STATABLE_TARGETS}
    assert stated[_TARGET_SHAPES[0]] is None and len({json.dumps(point) for point in stated.values()}) == 3, stated
    if os.geteuid() == 0:  # a read permission binds no root process
        points.pop("no read permission")
        documents.pop("no read permission")
    entry_sets = [combo for size in range(len(entry_files) + 1) for combo in itertools.combinations(entry_files, size)]
    wrong = []
    for index, (point, document, entries) in enumerate(itertools.product(points, documents, entry_sets)):
        directory = tmp_path / f"d{index}"
        directory.mkdir()
        _write_shape(directory / "_RUN_CONFIG" / "point.json", points[point])
        _write_shape(directory / "_RUN_CONFIG" / "config.json", documents[document])
        for entry in entries:
            _write_shape(directory / entry, entry_bytes[entry])
        before = _files(directory)
        try:
            held = run_directory_point(str(directory), door=door_read)
            state = "unused" if held is None else "persisted" if held.persisted else "historical"
            expected = twin if held is None or held.persisted or document not in stated else stated[document]
            if held is not None and (None if held.point is None else point_record(held.point)) != expected:
                state = f"{state} at another point"
        except refusal as refused:
            message = str(refused)
            state = "refused" if str(directory) in message and _RULE in message and _REMEDY in message else message
        except Exception as other:  # noqa: BLE001  recorded: any other error is a state outside the list
            state = f"raised {type(other).__name__}"
        if state != _shape_state(point, document, entries) or _files(directory) != before:
            wrong.append((point, document, entries, state))
        for path in directory.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
    assert wrong == [], (len(wrong), wrong[:6])


# -- [U-LIF2 r6] a stored document is read as its run's door read it ---------------------------------------------------

#: Two targets of the tests' own that carry soma keys (a composed head's targets carry a register and a comparison),
#: each its cores, the spiking core semantics and its soma keys alone: laid beside the shipped bundles by
#: ``soma_targets``.
_SOMA_TARGETS = {"a_target_with_a_signed_register": {"membrane_bits": 20, "membrane_signed": True},
                 "a_target_with_a_strict_comparison": {"thresholding_mode": "<"}}
#: The target-carried documents of the generator: the core semantics from a shipped target (the value domain), the soma
#: keys from the two targets above, and a target no bundle holds (a target that no longer expands).
_TARGET_SHAPES = ("a target that carries the core semantics", "a target that carries a signed register",
                  "a target that carries a strict comparison", "a target that no longer expands")
_STATABLE_TARGETS = _TARGET_SHAPES[:3]
_VALUE_TARGET, _GONE_TARGET = "generic_estimated_22nm", "a_target_no_bundle_holds"


@pytest.fixture
def soma_targets(monkeypatch):
    """The shipped bundles with ``_SOMA_TARGETS`` beside them, for one test."""
    import types

    import mimarsinan.mapping.platform.target_bundles as bundles

    shipped = bundles.target_bundles()
    added = {name: dataclasses.replace(shipped[_VALUE_TARGET], name=name, values_json=json.dumps(
        {**_CORES, "core_semantics": "spiking", **keys})) for name, keys in _SOMA_TARGETS.items()}
    every = types.MappingProxyType({**shipped, **added})
    monkeypatch.setattr(bundles, "target_bundles", lambda: every)
    return tuple(added)


def _targeted(saved: dict, target: str, *, silent_axes: bool = False) -> dict:
    """``saved`` with ``target`` declared (its declared axes dropped where the target is the value domain's)."""
    params = {key: value for key, value in saved["deployment_parameters"].items()
              if not (silent_axes and key in ("spiking_family", "spiking_variant"))}
    return {**saved, "deployment_parameters": params, "platform_constraints": {"target": target}}


def _target_shapes(saved: dict) -> dict:
    targets = (_VALUE_TARGET, *_SOMA_TARGETS, _GONE_TARGET)
    return {name: json.dumps(_targeted(saved, target, silent_axes=target == _VALUE_TARGET))
            for name, target in zip(_TARGET_SHAPES, targets)}


def _door_point(document: dict):
    """The point the run of ``document`` resolves at its own door (the target expanded, the run's sections, the run's
    config, the one resolution), with the window firing a run that persisted no point states (its declaration, else
    the historical one), as a record; None for the value domain."""
    from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics
    from mimarsinan.chip_simulation.spiking_point import HISTORICAL_WINDOW_FIRING, point_record, resolve_spiking_point
    from mimarsinan.config_schema.runtime import run_config, run_sections

    dp, pc, _layout = run_sections(door_read(document))
    config = run_config(dict(dp), dict(pc))
    if is_mvm_core_semantics(config.get("core_semantics")):
        return None
    record = point_record(resolve_spiking_point(config))
    if record["window_firing"] is not None:
        record["window_firing"] = document["deployment_parameters"].get(KEY, HISTORICAL_WINDOW_FIRING)
    return record


def _stored(root: Path, document: dict, *, finished: bool = True) -> str:
    """A run directory made before runs persisted their point: ``document`` saved, with a finished run's entries."""
    run_dir = root / "stored"
    (run_dir / "_RUN_CONFIG").mkdir(parents=True)
    (run_dir / "_RUN_CONFIG" / "config.json").write_text(json.dumps(document))
    if finished:
        _finish(str(run_dir))
    return str(run_dir)


@pytest.mark.parametrize("shape", _STATABLE_TARGETS)
@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r6_M1_a_stored_target_carried_document_is_stated_at_the_point_its_door_resolves(tmp_path, declared, values,
                                                                                        default, soma_targets, shape):
    """[U-LIF2 r6, M-1] A stored run whose document took its core semantics or its soma keys from its platform target
    is stated, by every reader of its point (the directory's point, the stored point, the cost record's cell), at the
    point its own door resolves: the value domain (no spiking point) for the first, the target's register and
    comparison for the others. Red on the final tree of r5: the statement read the sections without the target."""
    from mimarsinan.chip_simulation.spiking_point import point_record
    from mimarsinan.chip_simulation.stored_point import read_run_document, run_directory_point, stored_run_point

    saved = _document(tmp_path)
    document = json.loads(_target_shapes(saved)[shape])
    expected = _door_point(document)
    run_dir = _stored(tmp_path, document)
    declared(values, default)
    held = run_directory_point(run_dir, door=door_read)
    assert not held.persisted and (None if held.point is None else point_record(held.point)) == expected
    stated = stored_run_point(run_dir, read_run_document(run_dir), door=door_read)
    assert (None if stated is None else point_record(stated)) == expected
    key = _reading(run_dir)[1]
    assert key == ("mvm@sanafe" if expected is None else key) and (expected is None or key.startswith("lif")), key


@pytest.mark.parametrize("seed", ("cache seed", "steps seed", "spawn"))
@pytest.mark.parametrize("values,default", _SETTINGS, ids=_SET_IDS)
def test_r6_M1_a_seed_from_a_value_domain_source_carries_no_spiking_point(tmp_path, declared, values, default, seed):
    """[U-LIF2 r6, M-1] A stored run whose target made it a value-domain run (its spiking axes silent): each seed and
    the GUI's spawn writes the source's point, the value domain's (no spiking point), beside the entries it copies,
    never a spiking point its run did not resolve; the source is unchanged. Red on the final tree of r5: the seeds
    wrote a streamed LIF point."""
    saved = _document(tmp_path)
    run_dir = _stored(tmp_path, _targeted(saved, _VALUE_TARGET, silent_axes=True))
    before = _files(run_dir)
    declared(values, default)
    _document_handed, continued = _met(tmp_path, run_dir, seed, "Soft Core Mapping") if seed != "spawn" else \
        _handed(tmp_path, run_dir, "spawn", "Soft Core Mapping")
    point = Path(continued) / "_RUN_CONFIG" / "point.json"
    assert json.loads(point.read_text())["spiking_point"] is None
    assert _files(run_dir) == before


@pytest.mark.parametrize("reader", ("directory", "stored point", "session", "cache seed", "steps seed", "cost record",
                                    "display"))
def test_r6_M1_a_target_that_no_longer_expands_is_refused_by_name_at_every_reader(tmp_path, declared, reader):
    """[U-LIF2 r6, M-1] A stored run whose document names a target today's bundles do not hold is a document that cannot
    be read: every reader of its point is refused by ``DirectoryPointRefusal`` naming the directory, the cause, the rule
    and the remedy, and the stored directory is unchanged. Red on the final tree of r5: the target was not read."""
    from mimarsinan.chip_simulation.cost_extraction import extract_cost_record_from_run
    from mimarsinan.chip_simulation.stored_point import read_run_document, run_directory_point, stored_run_point
    from mimarsinan.config_schema.display_view import build_pipeline_config_view

    refusal = _new(_SP, "DirectoryPointRefusal")
    run_dir = _stored(tmp_path, _targeted(_document(tmp_path), _GONE_TARGET))
    before = _files(run_dir)
    declared((FWI, ITE), ITE)
    calls = {
        "directory": lambda: run_directory_point(run_dir, door=door_read),
        "stored point": lambda: stored_run_point(run_dir, read_run_document(run_dir), door=door_read),
        "session": lambda: _session({**_document(tmp_path), "_working_directory": run_dir,
                                     "start_step": "Soft Core Mapping"}),
        "cache seed": lambda: _met(tmp_path, run_dir, "cache seed", "Soft Core Mapping"),
        "steps seed": lambda: _met(tmp_path, run_dir, "steps seed", "Soft Core Mapping"),
        "cost record": lambda: extract_cost_record_from_run(run_dir, door=door_read),
        "display": lambda: build_pipeline_config_view(read_run_document(run_dir), working_dir=run_dir),
    }
    with pytest.raises(refusal) as refused:
        calls[reader]()
    message = str(refused.value)
    assert run_dir in message and _GONE_TARGET in message and _RULE in message and _REMEDY in message, message
    assert type(refused.value.__cause__).__name__ in message
    assert _files(run_dir) == before


def _toolchain_documents(root: Path) -> dict:
    """Every deployment document of the tree: its templates and its fixtures' deployment configs."""
    out = {}
    for base in ("templates", "tests/fixtures/deployment_configs"):
        for path in sorted((root / base).rglob("*.json")):
            try:
                document = json.loads(path.read_text())
            except ValueError:
                continue
            if isinstance(document, dict) and isinstance(document.get("deployment_parameters"), dict):
                out[str(path.relative_to(root))] = document
    return out


def test_r6_M1_the_historical_statement_of_every_toolchain_document_is_its_own_doors_point(tmp_path):
    """[U-LIF2 r6, M-1] For every template and fixture document the door resolves, the point a stored run of it that
    persisted none states equals the point its own door resolves (the window firing the historical one). Red on a
    copy of the final tree of r5 composed with U-CHIPKEYS7 (three templates take their soma keys from the target)."""
    from mimarsinan.chip_simulation.spiking_point import point_record
    from mimarsinan.chip_simulation.stored_point import stored_run_point

    root = Path(__file__).resolve().parents[3]
    documents = _toolchain_documents(root)
    differ, compared = {}, 0
    for name, document in documents.items():
        try:
            expected = _door_point(document)
        except ValueError:
            continue  # the door refuses the document today: no run of it exists to state
        compared += 1
        stated = stored_run_point(str(tmp_path), document, door=door_read)
        if (None if stated is None else point_record(stated)) != expected:
            differ[name] = expected
    assert compared >= 80 and differ == {}, (compared, sorted(differ))


#: [U-LIF2 r6, M-1] The class swept: every src module that reads a stored run's document or point (a name of the one
#: reader or of the run directory's files, or a path piece of them), with what it reads. The search over the syntax
#: trees fails on a module the list does not hold.
_DOCUMENT_READERS = {
    "chip_simulation/stored_point.py": "door", "chip_simulation/cost_extraction.py": "door",
    "config_schema/display_view.py": "door", "gui/runtime/run_cache_seed.py": "door", "pipelining/session.py": "door",
    "gui/runtime/process_monitor.py": "the display view", "gui/runs.py": "top-level keys",
    "gui/server/routes_pipeline.py": "the document as data", "gui/runtime/process_spawn.py": "writes, seeds through door",
    "search/optimizers/compilagent/backend/backend_layout.py": "another file of that name",
    "pipelining/core/engine/pipeline.py": "a message naming the file",
}
_READER_NAMES = frozenset({"read_run_document", "stored_run_point", "run_directory_point", "door_document",
                           "RUN_CONFIG_DIRECTORY", "RUN_DOCUMENT_FILENAME", "RUN_POINT_FILENAME", "run_point_path",
                           "load_saved_config_from_run_dir", "get_run_config"})
_PATH_PIECES = ("_RUN_CONFIG", "config.json", "point.json")
_POINT_READS = frozenset({"stored_run_point", "run_directory_point", "extract_cost_record_from_run", "from_runs"})


def _source_trees():
    src = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
    return {path.relative_to(src).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
            for path in sorted(src.rglob("*.py"))}


def _mentions(tree) -> set:
    prose = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Expr)}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _READER_NAMES or \
                isinstance(node, ast.Attribute) and node.attr in _READER_NAMES:
            found.add(getattr(node, "id", None) or node.attr)
        elif isinstance(node, ast.alias) and node.name in _READER_NAMES:
            found.add(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose and \
                any(piece in node.value for piece in _PATH_PIECES):
            found.add(node.value)
    return found


def _named(node) -> str:
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""


def _without_door(tree) -> list:
    """The lines of a module's references to a reader of a stored run's point that do not hand in the door: every load
    of a name bound to a reader (its own name, an import alias, a simple binding of one, followed to a fixpoint) or of
    an attribute named like one, other than a call with a ``door`` keyword (a reader passed as an object included)."""
    bound, size = set(_POINT_READS), -1
    while size != len(bound):
        size = len(bound)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                bound.update(alias.asname for alias in node.names if alias.name in bound and alias.asname)
            elif isinstance(node, ast.Assign) and _named(node.value) in bound:
                bound.update(target.id for target in node.targets if isinstance(target, ast.Name))
    handed = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call) and
              any(keyword.arg == "door" for keyword in node.keywords)}
    return sorted(node.lineno for node in ast.walk(tree) if isinstance(node, (ast.Name, ast.Attribute)) and
                  isinstance(node.ctx, ast.Load) and _named(node) in bound and id(node) not in handed)


#: The closure test's self-check: shapes of a reference to a reader without its door, each found (the last is no
#: reference), as the search must find them in src.
_DOOR_SHAPES = {
    "a call without the door": ("from x import run_directory_point\nrun_directory_point(d)\n", [2]),
    "an aliased import": ("from x import stored_run_point as read\nread(d, doc)\n", [2]),
    "a bound alias": ("from x import run_directory_point\nread = run_directory_point\nread(d, door=g)\n", [2]),
    "a reader passed as an object": ("from x import run_directory_point\nlist(map(run_directory_point, ds))\n", [2]),
    "a module attribute": ("import x\nx.extract_cost_record_from_run(d)\n", [2]),
    "a call that hands in the door": ("from x import run_directory_point\nrun_directory_point(d, door=g)\n", []),
}


@pytest.mark.parametrize("shape", tuple(_DOOR_SHAPES))
def test_r6_M1_self_check_the_search_finds_each_reference_without_the_door(shape):
    source, lines = _DOOR_SHAPES[shape]
    assert _without_door(ast.parse(source)) == lines, shape


def test_r6_M1_every_reader_of_a_stored_runs_document_or_point_is_listed_and_reads_it_through_the_door():
    """[U-LIF2 r6, M-1] The class swept by search: every src module that reads a stored run's document or point is in
    the finite list; every call of a reader of a stored run's point outside the one reader hands in the door; the
    reader of the run list reads top-level keys only, which no target sets (a target writes into the deployment
    parameters and the platform constraints alone)."""
    from mimarsinan.config_schema.registry import REGISTRY
    from mimarsinan.mapping.platform.target_expansion import DP, PC

    trees = _source_trees()
    found = {name for name, tree in trees.items() if _mentions(tree)}
    assert found == set(_DOCUMENT_READERS), sorted(found ^ set(_DOCUMENT_READERS))
    without_door = [f"{name}:{line}" for name, tree in trees.items() for line in _without_door(tree)]
    assert without_door == []
    list_keys = {node.args[0].value for node in ast.walk(trees["gui/runs.py"]) if isinstance(node, ast.Call) and
                 getattr(node.func, "attr", None) == "get" and isinstance(getattr(node.func, "value", None), ast.Name)
                 and node.func.value.id == "config" and node.args and isinstance(node.args[0], ast.Constant)}
    assert list_keys == {"experiment_name", "pipeline_mode"}
    assert all(REGISTRY.get(key) is None or REGISTRY[key].section not in (DP, PC) for key in list_keys)
