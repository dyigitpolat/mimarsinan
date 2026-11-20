"""[U-LIF2 W4 to W6, W16] Every reader of the backend capability judges one resolved point: the five callers of the
recipe give one answer, the configuration surface (the resolve payload, the wizard's rows, the display's derived keys)
says what the run derives and why, each enable derives from the keys of the point, the documents name the window
firing, and the toolchain's own windowed LIF documents resolve as today but for the new key (a document declaring an
execution extension now runs with the backends that lack it derived off by name).

Red on the set export: the recipe's callers read the mode and the schedule only (four of the five without the soma
law), the surface says SANA-FE and Loihi run a document whose extensions they refuse at assembly, the HACC row reads a
key that does not exist, and no document names the window firing."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest
from spiking_points import settled_window_firing

from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.runtime import derive_run_config

FWI, ITE = "fire_while_integrating", "integrate_then_emit"
KEY = "lif_window_firing"
ROOT = Path(__file__).resolve().parents[3]
_CORES = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 8}]}
_ENABLES = ("enable_nevresim_simulation", "enable_sanafe_simulation", "enable_loihi_simulation",
            "enable_odin_fpga_simulation", "enable_odin_hacc_export")
_OPT_IN = ("enable_odin_fpga_simulation", "enable_odin_hacc_export")
_TEMPLATE = ROOT / "templates" / "tier_0" / "t0_05_lifsync_simplemlp_wq_s4.json"
_POINTS = {"silent": {}, "fused": {"lif_exact_qat": False}, "zero reset": {"firing_mode": "Novena"},
           "dither": {"spike_phase_dither": True}, "guard": {"lif_membrane_init": -0.25},
           "streamed": {"spiking_variant": "streamed"},
           "ttfs synchronized": {"spiking_family": "ttfs", "spiking_variant": "synchronized"}}


def _new(module: str, name: str):
    return getattr(importlib.import_module(module), name)


def _doc(**dp):
    return {"model_type": "simple_mlp", "spiking_family": "lif", "spiking_variant": "synchronized", **dp}


@pytest.fixture
def declared(window_firing_declared):
    """Set the entry's data (the declared set and the default) for one test (the shared fixture)."""
    return window_firing_declared


def _template(**dp):
    document = json.loads(_TEMPLATE.read_text())
    document["deployment_parameters"].update(dp)
    # [U-LIF2 W16] the template declares firing during the window; a point away from the axis declares none
    document["deployment_parameters"] = settled_window_firing(document["deployment_parameters"])
    return document


# -- W4: the five callers of the recipe give one answer --------------------------------------------------------------

@pytest.mark.parametrize("value", [FWI, ITE])
@pytest.mark.parametrize("point", sorted(_POINTS))
def test_t05a_the_five_callers_give_equal_enables(declared, point, value):
    resolve = _new("mimarsinan.chip_simulation.spiking_point", "resolve_spiking_point")
    from mimarsinan.config_schema.registry.entries_residency import _run_value
    from mimarsinan.gui.wizard.vehicle_rows import vehicle_rows
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
    from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

    declared((FWI, ITE), FWI)
    away = point in ("streamed", "ttfs synchronized")
    if value == ITE and (away or point == "zero reset"):
        pytest.skip("not a legal point (t01, t19)")
    dp = dict(_POINTS[point], **({} if away else {KEY: value}))
    cfg = derive_run_config(_doc(**dp), dict(_CORES))
    recipe = ConversionPolicy.derive(resolve(cfg))
    for key in _ENABLES:
        assert cfg[key] == (recipe.sim_enables[key] and key not in _OPT_IN), key
        assert REGISTRY[key].meta(cfg)["supported"] == recipe.sim_enables[key], key
    assert DeploymentPlan.resolve(cfg).conversion_recipe.sim_enables == recipe.sim_enables
    rows = {row["key"]: row["supported"] for row in vehicle_rows(_template(**dp))}
    assert rows == dict(recipe.sim_enables)
    assert _run_value({k: v for k, v in cfg.items() if k != "lif_depth_balancing_relays"},
                      "lif_depth_balancing_relays") == recipe.knobs.get("lif_depth_balancing_relays")


def test_t05c_every_enable_derives_from_the_keys_of_the_point():
    basis = _new("mimarsinan.chip_simulation.spiking_point", "POINT_BASIS_KEYS")
    for key in _ENABLES:
        assert tuple(REGISTRY[key].derived_from) == tuple(basis), key
    for key in ("spiking_family", "spiking_variant", "firing_mode", "firing_granularity", "membrane_arithmetic",
                "membrane_bits", KEY, "spike_phase_dither", "lif_membrane_init"):
        assert key in basis, key


# -- W5: the configuration surface reads the record ------------------------------------------------------------------

def _payload(document):
    from mimarsinan.config_schema.resolve import resolve_draft

    resolution = resolve_draft(copy.deepcopy(document))
    assert resolution.ok, resolution.errors
    return resolution


@pytest.mark.parametrize("declares", [False, True])
def test_t05b_the_payload_and_the_wizard_rows_equal_the_resolution(declares):
    from mimarsinan.gui.wizard.vehicle_rows import vehicle_rows

    document = _template(**({KEY: FWI} if declares else {}))
    resolution = _payload(document)
    rows = {row["key"]: row for row in vehicle_rows(copy.deepcopy(document))}
    assert resolution.resolved[KEY] == FWI
    for key in _ENABLES:
        assert resolution.derived[key]["meta"]["supported"] is True, key
        assert rows[key]["supported"] is True and rows[key]["on"] == resolution.resolved[key], key
    for key in _OPT_IN:
        assert resolution.derived[key]["why"].startswith("off (opt-in)"), key
        assert "no executor" not in resolution.derived[key]["why"], key


@pytest.mark.parametrize("extension,value", [("spike_phase_dither", True), ("lif_membrane_init", -0.25)])
def test_t05b_the_surface_names_the_extension_a_backend_does_not_execute(extension, value):
    from mimarsinan.gui.wizard.vehicle_rows import vehicle_rows

    document = _template(**{extension: value})
    resolution = _payload(document)
    rows = {row["key"]: row for row in vehicle_rows(copy.deepcopy(document))}
    for key in _ENABLES:
        assert resolution.resolved[key] is False, key
        assert resolution.derived[key]["meta"]["supported"] is False, key
        assert extension in resolution.derived[key]["why"], key
        assert rows[key]["supported"] is False and extension in rows[key]["why"], key


def test_t05b_the_display_marks_what_the_sixth_recipe_entry_derives():
    """The review's p-4(b): the display's derived keys run the recipe fold on a defaults-merged draft."""
    from mimarsinan.config_schema.display_view_meta import _collect_derived_keys

    silent = _collect_derived_keys(_template())
    declaring = _collect_derived_keys(_template(spike_phase_dither=True))
    assert {"enable_sanafe_simulation", "enable_loihi_simulation"} <= silent & declaring
    assert KEY not in declaring


# -- W6: the documents name the window firing ------------------------------------------------------------------------

def test_t14_the_variant_the_advisory_and_the_nevresim_reason_name_the_window_firing():
    from mimarsinan.advisories import rules_config

    assert KEY in REGISTRY["spiking_variant"].doc
    assert KEY in Path(rules_config.__file__).read_text()
    why = REGISTRY["enable_nevresim_simulation"].why(
        derive_run_config({"model_type": "simple_mlp", "spiking_family": "ttfs", "spiking_variant": "synchronized"},
                          dict(_CORES)))
    assert "synchronized-TTFS" in why


def test_t14_pipelined_passes_stay_refused_for_the_variant_with_both_reasons():
    from mimarsinan.config_schema.registry import entries_residency

    assert REGISTRY["pipelined_multiplexing"].legal_values(_doc()) == (False,)
    doc = entries_residency._pipelined_legal.__doc__ or ""
    assert KEY in doc and "2T" in doc


def test_t14_the_calculus_states_throughput_on_the_declared_cores():
    text = (ROOT / "docs" / "research" / "findings" / "spiking_deployment_calculus.md").read_text()
    section = " ".join(text[text.index("### 16.3"):text.index("### 16.4")].split())
    assert "pipelined THROUGHPUT is unchanged" not in section
    assert KEY in section and "one over the latency" in section


def test_t14_the_value_run_keeps_the_executor_flag_inert():
    from mimarsinan.chip_simulation.value_run.value_flow import ValueHybridCoreFlow

    assert "lif_execution_synchronized" in (ValueHybridCoreFlow.__doc__ or "")
    assert KEY not in Path(importlib.import_module(ValueHybridCoreFlow.__module__).__file__).read_text()


# -- W16: the toolchain's own windowed LIF documents ------------------------------------------------------------------

def _documents():
    from mimarsinan.chip_simulation.window_firing import window_firing_applies

    paths = sorted((ROOT / "templates").glob("tier_*/t*.json")) + sorted(
        (ROOT / "tests" / "fixtures" / "deployment_configs").glob("*.json"))
    out = []
    for path in paths:
        document = json.loads(path.read_text())
        dp = document.get("deployment_parameters") or {}
        if window_firing_applies(dp):
            out.append(pytest.param(path, id=path.stem))
    return out


def test_w16_the_census_of_windowed_lif_documents():
    assert len(_documents()) == 23


@pytest.mark.parametrize("path", _documents())
def test_w16_every_windowed_lif_document_resolves_the_default_and_runs(path):
    from mimarsinan.chip_simulation.backend import BACKEND_REGISTRY
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

    from mimarsinan.config_schema.runtime import run_config

    document = json.loads(path.read_text())
    dp = dict(document["deployment_parameters"])
    cfg = run_config(dp, dict(document.get("platform_constraints") or {}))
    assert cfg[KEY] == FWI
    extensions = [key for key in ("spike_phase_dither", "lif_membrane_init") if dp.get(key) not in (None, False, 0.0)]
    for key in ("enable_nevresim_simulation", "enable_sanafe_simulation", "enable_loihi_simulation"):
        if extensions:
            assert cfg[key] is False and all(ext in REGISTRY[key].why(cfg) for ext in extensions), (path.stem, key)
        elif key not in dp:
            assert cfg[key] is True, (path.stem, key)
    BACKEND_REGISTRY.selected_step_specs(DeploymentPlan.resolve(cfg))
