"""[U-CHIPKEYS3] M-2 as a class, "a statement in a description that no run door reads": a stated pricing absence is
read at the run door. A pricing a run declares on a target whose description or realization states pricing absent
refuses by name at each site (the declared key, an override, a bundle), the pricer's entry refuses by the statement
whatever physics a platform carries, no label is derived for a pricing the statement excludes, and a realization that
states its bundle's profile as its own refuses at load. Then the sweep: every statement kind a description or
realization may carry, listed with the door that reads it (``STATEMENT_DOORS``), whose keys equal the loader's closed
field sets, so a statement kind added without its door fails here.

[U-CHIPKEYS4] 3(c): the table covers every statement kind (the weight grid's composition and its readings' forms and
fields, a derivation entry's fields, an assumed field's entry, the realization record's fields) and says which fields a
script prints for people (``PRINTED``, swept over every script that reads a description's own file). 3(f): the label is
derived from the pricing the run resolves, one source (``departures.run_pricing``, the resolver's own reading), with one
test per label site: the run door, the absolute-pricing refusal, the scaling ladder's target spec and the design point
instance's chip block.

[U-CHIPKEYS5] m-6: every set the sweep compares is read from the loader: the statement and reading forms, the assumed
field's entry and the composition's fields from each judge's own refusal of a form or field it does not know, so a form
the loader gains turns the comparison red; the weight grid's composition is listed among its statement forms' fields.
The texts a script prints beside the label (a display name, the class's note) and the typed clauses (named, given) are
judged at load by the label grammar (test_chip_label_clauses.py)."""

import copy
import json
import re
import types
from pathlib import Path

import pytest

import mimarsinan

TARGETS_DIR = Path(mimarsinan.__file__).parent / "mapping" / "platform" / "targets"
WIDE = "odin@odin_wide_1024x256_mb16"
ABSENT = "the hand-made chip is an FPGA build no shipped profile describes"


def _expand(pc, bundles=None):
    from mimarsinan.mapping.platform.target_expansion import expand_target
    return expand_target({}, pc, write_derived=True, bundles=bundles)


def _odin_core(**extra):
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    core, = get_target_bundle(WIDE).values["cores"]
    return [{**core, **extra}]


# --- the declared key, beside the target -------------------------------------------------------------------------------

@pytest.mark.parametrize("key, value", [
    ("platform_physics_profile", "odin"),  # the ASIC's profile (the verify's case b)
    ("platform_physics_profile", "generic_estimated_22nm"),  # another profile (case c)
    ("platform_physics_overrides", {"e_mac": {"value": 1.0, "unit": "pJ", "note": "a test constant"}}),  # a custom one
])
def test_a_pricing_declared_beside_a_target_stating_pricing_absent_refuses_by_name(key, value):
    from mimarsinan.mapping.platform.target_bundles import PricingAbsentError
    with pytest.raises(PricingAbsentError, match=rf"target='{re.escape(WIDE)}' states pricing absent \(No physics "
                                                 rf"profile describes the realization.*the run declares {key}="):
        _expand({"target": WIDE, key: value})


def test_an_empty_pricing_declares_nothing():
    """A blank profile and an empty constant block resolve no physics (the resolver's reading): no pricing is declared."""
    _, pc = _expand({"target": WIDE, "platform_physics_profile": "", "platform_physics_overrides": {}})
    assert pc["target"] == WIDE


# --- the override --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key, value", [
    ("platform_physics_profile", "odin"),
    ("platform_physics_overrides", {"e_mac": {"value": 1.0, "unit": "pJ", "note": "a test constant"}}),
])
def test_a_pricing_declared_in_target_overrides_refuses_by_the_stated_absence(key, value):
    """Before the override is judged (it would otherwise refuse as a key the realization does not set)."""
    from mimarsinan.mapping.platform.target_bundles import PricingAbsentError
    with pytest.raises(PricingAbsentError, match=rf"states pricing absent .*the run declares target_overrides\.{key}="):
        _expand({"target": WIDE, "target_overrides": {key: {"value": value, "note": "a test override"}}})


@pytest.mark.parametrize("field", ["compute_physics", "write_physics"])
def test_a_core_technology_in_the_overridden_cores_refuses_by_the_stated_absence(field):
    from mimarsinan.mapping.platform.target_bundles import PricingAbsentError
    with pytest.raises(PricingAbsentError, match=rf"the run declares target_overrides\.cores\[0\]\.{field}='odin'"):
        _expand({"target": WIDE, "target_overrides": {"cores": {"value": _odin_core(**{field: "odin"}),
                                                                 "note": "a test override"}}})


def test_the_realization_s_cores_overridden_without_a_technology_are_no_pricing():
    from mimarsinan.mapping.platform.target_bundles import PricingAbsentError
    try:
        _expand({"target": WIDE, "target_overrides": {"cores": {"value": _odin_core(count=2), "note": "two cores"}}})
    except PricingAbsentError as refusal:  # pragma: no cover - the failure this test exists to show
        pytest.fail(f"cores without a technology are no pricing: {refusal}")
    except Exception:  # noqa: BLE001 - another door may judge the cores; this test reads the pricing door only
        pass


# --- a bundle: a description (not only a realization) may state its pricing absent ----------------------------------

def _bundle(**extra):
    data = {"name": "absent_chip", "geometry": "recom_64x64_256", "keys": {"core_semantics": "mvm"},
            "faithfulness": {"class": "exact"}, "pricing_absent": ABSENT, **extra}
    from unit.mapping.platform.hand_descriptions import described
    return described(data)


def _load_dir(tmp_path, *datas):
    from mimarsinan.mapping.platform.target_bundles import load_bundles
    for data in datas:
        (tmp_path / f"{data['name']}.json").write_text(json.dumps(data), encoding="utf-8")
    return load_bundles(tmp_path)


def test_a_bundle_states_its_pricing_absent_and_a_run_on_it_declaring_a_profile_refuses(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import PricingAbsentError
    bundles = _load_dir(tmp_path, _bundle())
    assert bundles["absent_chip"].pricing_absent == ABSENT
    with pytest.raises(PricingAbsentError, match=r"target='absent_chip' states pricing absent \(the hand-made chip"):
        _expand({"target": "absent_chip", "platform_physics_profile": "generic_estimated_22nm"}, bundles)
    _, pc = _expand({"target": "absent_chip"}, bundles)
    assert pc["target_faithfulness_class"] == "exact"


@pytest.mark.parametrize("extra, match", [
    ({"keys": {"core_semantics": "mvm", "platform_physics_profile": "generic_estimated_22nm"}},
     r"pricing_absent is stated and the description declares platform_physics_profile='generic_estimated_22nm'"),
    ({"core_fields": {"compute_physics": "generic_estimated_22nm"}},
     r"pricing_absent is stated and the description declares cores\[0\]\.compute_physics="),
    ({"core_fields": {"write_physics": "generic_estimated_22nm"}},
     r"pricing_absent is stated and the description declares cores\[0\]\.write_physics="),
    ({"pricing_absent": " "}, r"pricing_absent ' ': a stated pricing absence says why"),
    ({"pricing_absent": 5}, r"pricing_absent 5: a stated pricing absence says why"),
])
def test_a_bundle_stating_its_pricing_twice_refuses_at_load(extra, match, tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError
    with pytest.raises(TargetBundleError, match=match):
        _load_dir(tmp_path, _bundle(**extra))


def test_a_bundle_stating_no_pricing_absence_takes_a_declared_profile(tmp_path):
    """A description that states no absence leaves its pricing to the run (R5's rule, unchanged)."""
    data = _bundle()
    data.pop("pricing_absent")
    bundles = _load_dir(tmp_path, data)
    assert bundles["absent_chip"].pricing_absent == ""
    _, pc = _expand({"target": "absent_chip", "platform_physics_profile": "generic_estimated_22nm"}, bundles)
    assert pc["platform_physics_profile"] == "generic_estimated_22nm"


# --- the realization stating its bundle's profile as its own (refused at load) -------------------------------------

def test_a_realization_stating_its_bundle_s_profile_as_its_own_refuses_at_load(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError, load_bundle
    raw = json.loads((TARGETS_DIR / "odin.json").read_text(encoding="utf-8"))
    spec = raw["realizations"]["odin_wide_1024x256_mb16"]
    spec.pop("pricing_absent")
    spec["keys"]["platform_physics_profile"] = raw["keys"]["platform_physics_profile"]
    (tmp_path / "odin.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(TargetBundleError, match=r"keys\.platform_physics_profile 'odin' is its bundle's profile"):
        load_bundle(tmp_path / "odin.json")


def test_a_realization_stating_pricing_absent_beside_an_inherited_core_technology_refuses_at_load(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError, load_bundle
    raw = json.loads((TARGETS_DIR / "odin.json").read_text(encoding="utf-8"))
    raw["core_fields"] = {"compute_physics": "odin"}
    (tmp_path / "odin.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(TargetBundleError, match=r"odin_wide_1024x256_mb16.*pricing_absent is stated and the "
                                                r"description declares cores\[0\]\.compute_physics='odin'"):
        load_bundle(tmp_path / "odin.json")


# --- the lookup carries the statement; the pricer's entry reads it ----------------------------------------------------

def test_the_lookup_carries_the_realization_s_stated_absence():
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    real = json.loads(get_target_bundle("odin").realizations_json)["odin_wide_1024x256_mb16"]
    assert get_target_bundle(WIDE).pricing_absent == real["pricing_absent"] and real["pricing_absent"]
    assert get_target_bundle("odin").pricing_absent == ""


def _odin_physics():
    from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics
    return resolve_platform_physics("odin", {}).to_dict()


def test_the_pricing_refusal_reads_the_statement_whatever_physics_the_platform_carries():
    """A platform built past the expansion (a sealed record, a hand-made platform) that carries the ASIC's physics on
    the realization refuses by the statement, and its reason never prints the label's 'exact'."""
    from mimarsinan.mapping.platform.target_bundles import absolute_pricing_refusal
    name, reason = absolute_pricing_refusal({"target": WIDE, "platform_physics_resolved": _odin_physics()})
    assert name == "absolute_pricing" and f"target='{WIDE}' states pricing absent" in reason
    assert "exact" not in reason


def test_a_bundle_with_physics_and_no_stated_absence_has_no_pricing_refusal():
    from mimarsinan.mapping.platform.target_bundles import absolute_pricing_refusal
    assert absolute_pricing_refusal({"target": "odin", "platform_physics_resolved": _odin_physics()}) is None
    assert absolute_pricing_refusal({"platform_physics_resolved": _odin_physics()}) is None


def test_a_target_no_description_selects_keeps_r5_s_reading():
    """An unknown name priced by its own physics stays priced (a record of another tree), and without physics refuses
    by the unknown name, as R5 reads it."""
    from mimarsinan.mapping.platform.target_bundles import UnknownTargetError, absolute_pricing_refusal
    assert absolute_pricing_refusal({"target": "no_such_chip", "platform_physics_resolved": _odin_physics()}) is None
    with pytest.raises(UnknownTargetError):
        absolute_pricing_refusal({"target": "no_such_chip"})


def test_the_record_pricer_s_entry_refuses_by_the_statement():
    from mimarsinan.deployment_record.cost.absolute.report import absolute_pricing_for_record
    record = types.SimpleNamespace(identity=types.SimpleNamespace(
        platform={"target": WIDE, "platform_physics_resolved": _odin_physics()}))
    pricing = absolute_pricing_for_record(record)
    assert pricing.terms == () and [r.name for r in pricing.refusals] == ["absolute_pricing"]
    assert "states pricing absent" in pricing.refusals[0].reason


def test_the_record_pricer_s_entry_prices_nothing_for_a_record_of_no_target_and_no_physics():
    """The entry's refusal test: a record of no target refuses nothing, and with no physics prices nothing (None)."""
    from mimarsinan.deployment_record.cost.absolute.report import absolute_pricing_for_record
    assert absolute_pricing_for_record(types.SimpleNamespace(identity=types.SimpleNamespace(platform={}))) is None


def test_the_candidate_pricer_s_entry_refuses_by_the_statement():
    from mimarsinan.deployment_record.cost.absolute.report import candidate_cost_report
    from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics
    from mimarsinan.deployment_record.quantities.spec import Quantities
    from mimarsinan.mapping.platform.target_bundles import absolute_pricing_refusal
    refusal = absolute_pricing_refusal({"target": WIDE, "platform_physics_resolved": _odin_physics()})
    report = candidate_cost_report(Quantities({}), resolve_platform_physics("odin", {}), refusal=refusal)
    assert report.energy == () and report.area == () and report.latency == ()
    assert any("states pricing absent" in note for note in report.notes)


# --- the label: never derived for a pricing the statement excludes --------------------------------------------------

def test_no_label_is_derived_for_a_pricing_the_statement_excludes():
    from mimarsinan.mapping.platform.target_bundles import PricingAbsentError
    from mimarsinan.mapping.platform.target_expansion import target_assignments
    with pytest.raises(PricingAbsentError):
        target_assignments({}, {"target": WIDE, "platform_physics_profile": "odin"}, write_derived=True)


def test_a_run_on_the_realization_keeps_its_class_and_its_pricing_refusal_names_the_statement():
    from mimarsinan.mapping.platform.target_bundles import absolute_pricing_refusal, get_target_bundle
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    dp, pc = _expand({"target": WIDE})
    assert pc["target_faithfulness_class"] == get_target_bundle(WIDE).faithfulness.label()
    resolved = build_platform_constraints_resolved(merge_pipeline_config(dp, pc))
    assert resolved.get("platform_physics_resolved") is None
    _, reason = absolute_pricing_refusal(resolved)
    assert "states pricing absent" in reason and "exact" not in reason


# --- every door a run enters by -------------------------------------------------------------------------------------

def test_the_document_door_refuses_a_pricing_the_statement_excludes():
    from mimarsinan.mapping.platform.target_bundles import PricingAbsentError
    from mimarsinan.mapping.platform.target_expansion import expand_target_document
    with pytest.raises(PricingAbsentError):
        expand_target_document({"deployment_parameters": {}, "platform_constraints": {
            "target": WIDE, "platform_physics_profile": "odin"}}, write_derived=True)


def test_the_run_door_refuses_a_pricing_the_statement_excludes():
    """The verify's p02 case (b) through the merge every run takes."""
    from mimarsinan.mapping.platform.target_bundles import PricingAbsentError
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
    with pytest.raises(PricingAbsentError, match=r"platform_physics_profile='odin'"):
        merge_pipeline_config({}, {"target": WIDE, "platform_physics_profile": "odin"})


# --- the sweep: every statement kind a description or realization may carry, with the door that reads it ----------

#: (field set, field) -> (the door that reads it, how). A door is a function of the toolchain (module:function); a
#: field read by people only names no door ("people").
STATEMENT_DOORS = {
    ("bundle", "name"): ("mapping.platform.target_bundles:load_bundle", "equals the file name; the target's name"),
    ("bundle", "display_name"): ("people", "a printed name; no run reads it; judged at load: no derived clause"),
    ("bundle", "geometry"): ("mapping.platform.target_bundles:bundle_values", "the cores and weight grid written"),
    ("bundle", "keys"): ("mapping.platform.target_expansion:target_assignments", "each key written at the run door"),
    ("bundle", "core_fields"): ("mapping.platform.target_bundles:bundle_values", "on every core, written"),
    ("bundle", "faithfulness"): ("mapping.platform.target_expansion:target_assignments", "the label derived"),
    ("bundle", "notes"): ("people", "the description's notes; no run reads them"),
    ("bundle", "computation"): ("mapping.platform.target_bundles:judged_statements", "the loader rule at load"),
    ("bundle", "realizations"): ("mapping.platform.descriptions.realizations:lookup", "each a target name"),
    ("bundle", "pricing_absent"): ("mapping.platform.descriptions.departures:declared_pricing_refusal",
                                   "a declared pricing refused at the expansion; absolute_pricing_refusal at the "
                                   "pricer's entry"),
    ("geometry", "name"): ("mapping.platform.target_bundles:load_bundles", "a geometry bundled twice refuses"),
    ("geometry", "cores"): ("mapping.platform.target_expansion:target_assignments", "the cores written"),
    ("geometry", "weight_bits"): ("mapping.platform.target_expansion:target_assignments", "the grid written"),
    ("geometry", "provenance"): ("people", "the geometry's citation; the registry holds it"),
    ("geometry", "claim_class"): ("mapping.platform.target_expansion:target_assignments", "target_claim_class"),
    ("faithfulness", "class"): ("mapping.platform.target_expansion:target_assignments", "the label's kind"),
    ("faithfulness", "named"): ("mapping.platform.target_expansion:target_assignments", "the label's departures, each "
                                "a quoted field; judged at load: no separator, no derived clause in its words"),
    ("faithfulness", "given"): ("mapping.platform.target_expansion:target_assignments", "the label's given, each a "
                                "quoted field; judged at load: no separator, no derived clause in its words"),
    ("faithfulness", "note"): ("people", "the class's explanation; no run reads it; judged at load: no derived clause"),
    ("realization", "display_name"): ("people", "a printed name; no run reads it; judged at load: no derived clause"),
    ("realization", "geometry"): ("mapping.platform.descriptions.realizations:lookup", "the realization's values"),
    ("realization", "keys"): ("mapping.platform.descriptions.realizations:lookup", "the realization's values"),
    ("realization", "computation"): ("mapping.platform.target_bundles:judged_statements", "the loader rule at load"),
    ("realization", "record"): ("mapping.platform.descriptions.realizations:judged", "judged at load (the record "
                                                                                      "that produced it)"),
    ("realization", "notes"): ("people", "the realization's notes; no run reads them"),
    ("realization", "pricing_absent"): ("mapping.platform.descriptions.departures:declared_pricing_refusal",
                                        "carried by the lookup to the expansion and the pricer's entry"),
    ("statement", "stated"): ("mapping.platform.descriptions.computation_keys:statement_errors",
                              "its value equals the key the bundle writes"),
    ("statement", "unstated"): ("mapping.platform.descriptions.computation_keys:statement_errors",
                                "its assumed value is the key written; without it the deployment declares the key"),
    ("statement", "departs"): ("mapping.platform.descriptions.computation_keys:departures",
                               "its assumed value is the key written; the label prints its departure"),
    # [U-CHIPKEYS4] 3(c): the weight grid's composition and its readings, a derivation entry, an assumed field's entry
    # and the realization record, each judged at load by the function named; none is read by a run.
    ("composition", "slices_per_weight"): ("mapping.platform.descriptions.readings:composition_errors",
                                           "a reading judged at load and tied to the cores' cells_per_weight"),
    ("composition", "cell_bits"): ("mapping.platform.descriptions.readings:composition_errors",
                                   "a reading judged at load"),
    ("composition", "law"): ("mapping.platform.descriptions.readings:composition_errors",
                             "composes the two stated readings into the grid the bundle writes (weight_bits)"),
    ("reading", "stated"): ("mapping.platform.descriptions.readings:judged_reading",
                            "a composition reading stated from its source, read in its quote at load"),
    ("reading", "unstated"): ("mapping.platform.descriptions.readings:judged_reading",
                              "a composition reading the source does not state, with its note"),
    ("derivation", "field"): ("mapping.platform.descriptions.readings:value_reading_errors",
                              "the field of the value the entry derives, one entry per field"),
    ("derivation", "rule"): ("mapping.platform.descriptions.readings:value_reading_errors",
                             "a named derivation of computation_keys.DERIVATIONS"),
    ("derivation", "reads"): ("mapping.platform.descriptions.readings:value_reading_errors",
                              "the integer the rule derives the value from, read in the span"),
    ("assumed_fields", "field"): ("mapping.platform.descriptions.readings:assumed_field_errors",
                                  "a field of the stated object value, never every field: not read in the span"),
    ("assumed_fields", "why"): ("mapping.platform.descriptions.readings:assumed_field_errors",
                                "non-empty at load; for readers"),
    ("record", "report"): ("mapping.platform.descriptions.realizations:judged",
                           "its form judged at load (a relative path); carried, not judged: that the report is there"),
    ("record", "sha256"): ("mapping.platform.descriptions.realizations:judged",
                           "its form judged at load (64 lower hex digits); carried, not judged: the report's digest"),
    ("record", "design"): ("mapping.platform.descriptions.realizations:judged",
                           "its form judged at load (a non-empty text); carried, not judged: what it says"),
    ("record", "session"): ("mapping.platform.descriptions.realizations:judged",
                            "its form judged at load (a non-empty text); carried, not judged: what it says"),
    ("record", "result"): ("mapping.platform.descriptions.realizations:judged",
                           "its form judged at load (a non-empty text); carried, not judged: what it says"),
}
#: The statements no run door reads (documentation for people): the sweep's findings, reported by the unit.
PEOPLE_ONLY = sorted(key for key, (door, _) in STATEMENT_DOORS.items() if door == "people")


def _listed_in(message, head):
    """The list a loader's refusal names after ``head``: read from the loader, never written here."""
    import ast
    found = re.search(re.escape(head) + r" (\[.*?\])(?:, got| \(each)", message)
    assert found, message
    return ast.literal_eval(found.group(1))


def _statement_judge(path):
    """The loader's judge of one statement at ``path`` (its errors)."""
    from mimarsinan.mapping.platform.descriptions import computation_keys
    return lambda entry: computation_keys._judge_one(path, entry, {})[1]


def _reading_judge(entry):
    from mimarsinan.mapping.platform.descriptions import readings
    return readings.judged_reading("a reading", entry)[2]


def _forms(judge):
    """[U-CHIPKEYS5] m-6: the forms a judge takes, from its refusal of a form it does not know."""
    return frozenset(next(iter(form)) for form in _listed_in(judge({"no_such_form": {}})[0], "a statement is one of"))


def _form_fields(judge, form):
    """[U-CHIPKEYS5] m-6: the fields a judge takes in ``form``, from its refusal of a field it does not know."""
    return frozenset(_listed_in(judge({form: {"no_such_field": 1}})[0], f"{form} takes the fields"))


def _field_sets():
    from mimarsinan.mapping.platform import target_bundles
    from mimarsinan.mapping.platform.descriptions import readings, realizations
    composition = readings.composition_errors("a grid", {"no_such_reading": 1}, None,
                                              readings.COMPOSITION_READINGS["weight_bits"], {})[0]
    assumed = re.search(r"non-empty object \{(\w+): (\w+)\}", readings.assumed_field_errors(
        "a value", {"assumed_fields": [], "value": {"a": 1}})[0])
    return {"bundle": target_bundles._BUNDLE_FIELDS, "geometry": target_bundles._GEOMETRY_FIELDS,
            "faithfulness": target_bundles._FAITHFULNESS_FIELDS, "realization": realizations._FIELDS,
            # [U-CHIPKEYS5] m-6: the forms, the entry and the composition, each read from its judge's refusal
            "statement": _forms(_statement_judge("weight_bits")),
            "composition": frozenset({*_listed_in(composition, "composition takes"),
                                      re.search(r"and a (\w+), got", composition).group(1)}),
            "reading": _forms(_reading_judge), "derivation": frozenset(readings.DERIVATION_FIELDS),
            "assumed_fields": frozenset(assumed.groups()), "record": realizations._RECORD_FIELDS}


def test_the_sweep_lists_every_statement_kind_a_description_may_carry():
    listed = {}
    for kind, field in STATEMENT_DOORS:
        listed.setdefault(kind, set()).add(field)
    assert {kind: set(fields) for kind, fields in _field_sets().items()} == listed


@pytest.mark.parametrize("site", sorted(STATEMENT_DOORS), ids=lambda site: ".".join(site))
def test_each_listed_door_is_a_function_of_the_toolchain(site):
    import importlib
    door, how = STATEMENT_DOORS[site]
    assert how.strip()
    if door == "people":
        return
    module, _, function = door.partition(":")
    assert callable(getattr(importlib.import_module(f"mimarsinan.{module}"), function))


def test_the_statements_no_run_door_reads_are_documentation_only():
    """The sweep's finding, pinned: names, notes, provenance and the class's note are read by people only."""
    assert PEOPLE_ONLY == [("bundle", "display_name"), ("bundle", "notes"), ("faithfulness", "note"),
                           ("geometry", "provenance"), ("realization", "display_name"), ("realization", "notes")]


def test_a_statement_form_the_loader_does_not_know_refuses(tmp_path):
    """The statement forms are closed: a fourth refuses by name (the sweep's statement kinds are all of them)."""
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError, load_bundle
    from unit.mapping.platform.hand_descriptions import described
    data = described({"name": "hand", "geometry": "recom_64x64_256", "keys": {}, "faithfulness": {"class": "exact"}})
    data["computation"]["readout_law"] = {"guessed": {"note": "n"}}
    (tmp_path / "hand.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TargetBundleError, match=r"readout_law: a statement is one of \[\{'stated': \{\}\}, "
                                                r"\{'unstated': \{\}\}, \{'departs': \{\}\}\]"):
        load_bundle(tmp_path / "hand.json")


def test_every_shipped_statement_is_of_a_listed_kind():
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle, target_names
    kinds = {s["kind"] for name in target_names() for s in get_target_bundle(name).statements.values()}
    assert kinds <= {field for kind, field in STATEMENT_DOORS if kind == "statement"}
    assert copy.deepcopy(kinds) == {"stated", "unstated", "departs"}


# --- [U-CHIPKEYS3] the departures a run's declarations print, and the pricing door's own functions -------------------

def test_target_departures_print_each_override_beside_the_bundle_s_value():
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle, target_departures
    assert target_departures({}) == ()
    bits = get_target_bundle("neurram").values["activation_bits"]
    rows = target_departures({"target": "neurram", "target_overrides": {
        "activation_bits": {"value": 8, "note": "a test override"}}})
    assert rows == ({"key": "activation_bits", "bundle": bits, "declared": 8, "note": "a test override"},)


def test_declared_departures_name_each_key_and_core_field_set_otherwise():
    from mimarsinan.mapping.platform.target_bundles import declared_departures, get_target_bundle
    values = get_target_bundle("neurram").values
    core = values["cores"][0]
    assert declared_departures("neurram") == ()
    assert declared_departures("neurram", {"activation_bits": values["activation_bits"]}) == ()  # the same value
    assert declared_departures("neurram", {"no_key_the_bundle_sets": 1}) == ()  # completes the description
    assert declared_departures("neurram", {"activation_bits": 2}) == (
        {"key": "activation_bits", "bundle": values["activation_bits"], "declared": 2},)
    assert declared_departures("neurram", None, {"max_axons": core["max_axons"]}) == ()
    assert declared_departures("neurram", None, {"max_axons": 3}) == (
        {"key": "cores.max_axons", "bundle": core["max_axons"], "declared": 3},)
    assert declared_departures("neurram", None, {"compute_physics": "x"}) == ()  # a field the cores do not state


def test_pricing_declarations_name_each_pricing_a_value_set_declares():
    from mimarsinan.mapping.platform.descriptions.departures import pricing_declarations
    assert pricing_declarations({}) == ()
    assert pricing_declarations({"platform_physics_profile": "odin", "platform_physics_overrides": {"a": 1}}) == (
        "platform_physics_profile='odin'", "platform_physics_overrides={'a': 1}")
    assert pricing_declarations({"cores": [{"write_physics": "a"}, {"compute_physics": "b", "write_physics": ""}]}, "p.") == (
        "p.cores[0].write_physics='a'", "p.cores[1].compute_physics='b'")
    assert pricing_declarations({"cores": {"write_physics": "a"}}) == ()  # cores that are no list
    assert pricing_declarations({"cores": ["write_physics"]}) == ()  # a core that is no object


def _ladder():
    import importlib.util
    path = Path(mimarsinan.__file__).resolve().parents[2] / "scripts" / "scaling_ladder.py"
    spec = importlib.util.spec_from_file_location("scaling_ladder_script_doors", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("declared, core_fields, overrides, named", [
    ({"platform_physics_profile": "odin"}, None, None, "platform_physics_profile='odin'"),
    ({}, None, {"e_mac": {"value": 1.0, "unit": "pJ", "note": "a test constant"}}, "platform_physics_overrides="),
    ({}, {"compute_physics": "odin"}, None, "cores[0].compute_physics='odin'"),
])
def test_the_ladder_s_platform_builder_refuses_a_pricing_the_statement_excludes(declared, core_fields, overrides, named):
    """The scaling ladder's and the design point instance's platform builder for a target row: one site of the door."""
    from mimarsinan.mapping.platform.target_bundles import PricingAbsentError
    with pytest.raises(PricingAbsentError, match=re.escape(named)):
        _ladder().resolved_platform(WIDE, "declared", None, declared, core_fields, overrides)


def test_the_ladder_s_platform_builder_takes_the_realization_unpriced_and_a_bundle_priced():
    ladder = _ladder()
    assert ladder.resolved_platform(WIDE, "declared", None, {}, None, None)["platform_physics_resolved"] is None
    assert ladder.resolved_platform("odin", "declared", None, {}, None, None)["platform_physics_resolved"] is not None
    assert ladder.target_spec(WIDE)["pricing_absent"] and ladder.target_spec("odin")["pricing_absent"] == ""


def test_a_platform_naming_no_target_has_no_pricing_refusal():
    from mimarsinan.mapping.platform.target_bundles import absolute_pricing_refusal
    assert absolute_pricing_refusal({}) is None


def test_the_pricing_door_reads_only_override_entries_that_are_objects():
    from mimarsinan.mapping.platform.target_bundles import declared_pricing_refusal
    assert declared_pricing_refusal("x", "", {"platform_physics_profile": "odin"}, {}) is None  # no stated absence
    assert declared_pricing_refusal("x", "why", {}, {}) is None  # nothing declared
    assert declared_pricing_refusal("x", "why", {}, {"platform_physics_profile": "odin"}) is None  # no object entry
    assert "target_overrides.platform_physics_profile='odin'" in declared_pricing_refusal(
        "x", "why", {}, {"platform_physics_profile": {"value": "odin"}})


# --- [U-CHIPKEYS3] the sweep's statement fields: each field of each statement form, with the door that reads it --------

#: (form, field) -> (door, how): the fields a statement form carries.
FIELD_DOORS = {
    ("stated", "value"): ("mapping.platform.descriptions.computation_keys:statement_errors", "equals the key written"),
    ("stated", "source"): ("mapping.platform.descriptions.readings:judged_reading", "non-empty at load"),
    ("stated", "location"): ("mapping.platform.descriptions.readings:judged_reading", "non-empty at load; pages and code "
                                                                                     "lines checked by the unit's scripts"),
    ("stated", "quote"): ("mapping.platform.descriptions.readings:read_spans", "holds every span"),
    ("stated", "reads"): ("mapping.platform.descriptions.readings:value_reading_errors", "the value read there"),
    ("stated", "derivation"): ("mapping.platform.descriptions.readings:value_reading_errors", "its input read"),
    ("stated", "assumed_fields"): ("mapping.platform.descriptions.readings:assumed_field_errors", "judged"),
    ("stated", "note"): ("people", "why, for readers"),
    ("unstated", "note"): ("mapping.platform.descriptions.readings:judged_reading", "non-empty at load"),
    ("unstated", "assumed"): ("mapping.platform.descriptions.computation_keys:statement_errors", "the key written"),
    ("departs", "computation"): ("mapping.platform.descriptions.computation_keys:departures", "the label prints it "
                                 "as a quoted field; judged at load: no derived clause in its words"),
    ("departs", "source"): ("mapping.platform.descriptions.readings:judged_reading", "non-empty at load"),
    ("departs", "location"): ("mapping.platform.descriptions.readings:judged_reading", "non-empty at load"),
    ("departs", "quote"): ("mapping.platform.descriptions.readings:read_spans", "holds the span"),
    ("departs", "reads"): ("mapping.platform.descriptions.readings:read_spans", "a span at token boundaries"),
    ("departs", "assumed"): ("mapping.platform.descriptions.computation_keys:statement_errors", "the key written; "
                             "the label prints its JSON text as a quoted field, judged by its words"),
    ("departs", "reason"): ("mapping.platform.descriptions.readings:judged_reading", "non-empty at load; for readers"),
    ("departs", "note"): ("people", "why, for readers"),
    # [U-CHIPKEYS5] m-6: the weight grid's composition, a field of each of its statement forms
    ("stated", "composition"): ("mapping.platform.descriptions.readings:composition_errors",
                                "beside weight_bits: its slices, cell bits and law, judged at load"),
    ("unstated", "composition"): ("mapping.platform.descriptions.readings:composition_errors",
                                  "beside weight_bits: each reading stated or unstated, judged at load"),
    ("departs", "composition"): ("mapping.platform.descriptions.readings:composition_errors",
                                 "beside weight_bits: each reading stated or unstated, judged at load"),
    # [U-CHIPKEYS4] 3(c): a composition reading's two forms and their fields
    ("reading.stated", "value"): ("mapping.platform.descriptions.readings:composition_errors",
                                  "a positive integer read in its span; the law composes it"),
    ("reading.stated", "source"): ("mapping.platform.descriptions.readings:judged_reading", "non-empty at load"),
    ("reading.stated", "location"): ("mapping.platform.descriptions.readings:judged_reading", "non-empty at load"),
    ("reading.stated", "quote"): ("mapping.platform.descriptions.readings:read_spans", "holds the span"),
    ("reading.stated", "reads"): ("mapping.platform.descriptions.readings:value_reading_errors", "the value read there"),
    ("reading.stated", "derivation"): ("mapping.platform.descriptions.readings:value_reading_errors", "its input read"),
    ("reading.stated", "note"): ("people", "why, for readers"),
    ("reading.unstated", "note"): ("mapping.platform.descriptions.readings:judged_reading", "non-empty at load"),
}


def test_the_sweep_lists_every_field_of_every_statement_form():
    """[U-CHIPKEYS5] m-6: each form's fields read from the loader's judge (a key with a composition beside it and one
    without), so a field or a form the loader gains turns the comparison red."""
    listed = {}
    for form, field in FIELD_DOORS:
        listed.setdefault(form, set()).add(field)
    loader = {form: set(_form_fields(_statement_judge("weight_bits"), form) | _form_fields(
        _statement_judge("activation_bits"), form)) for form in _forms(_statement_judge("weight_bits"))}
    loader.update({f"reading.{form}": set(_form_fields(_reading_judge, form)) for form in _forms(_reading_judge)})
    assert listed == loader
    assert "composition" in loader["stated"] and "composition" not in _form_fields(_statement_judge("activation_bits"),
                                                                                    "stated")


@pytest.mark.parametrize("site", sorted(FIELD_DOORS), ids=lambda site: ".".join(site))
def test_each_statement_field_s_door_is_a_function_of_the_toolchain(site):
    import importlib
    door, _ = FIELD_DOORS[site]
    if door != "people":
        module, _, function = door.partition(":")
        assert callable(getattr(importlib.import_module(f"mimarsinan.{module}"), function))


# --- [U-CHIPKEYS4] 3(f): the label derived from the pricing the run resolves, one source, one test per label site -------

NO_PHYSICS = "no physics profile"
GENERIC = "generic_estimated_22nm"
CONSTANTS = {"e_mac": {"nominal": 1.0, "unit": "pJ", "note": "a test constant"}}
#: The declarations a run may price a description stating no pricing by, and the physics the resolver names.
PRICINGS = [({}, None), ({"platform_physics_profile": GENERIC}, GENERIC),
            ({"platform_physics_overrides": CONSTANTS}, "custom"),
            ({"platform_physics_profile": GENERIC, "platform_physics_overrides": CONSTANTS}, GENERIC),
            ({"platform_physics_profile": "", "platform_physics_overrides": {}}, None)]


def _clause(priced):
    """[U-CHIPKEYS6] The pricing clause, the profile's name a quoted field."""
    return NO_PHYSICS if priced is None else f"physics profile {json.dumps(priced)} declared"


def _unpriced_bundle(name="unpriced_chip", faith=None):
    """A hand-made description that states no pricing (neither a profile nor its absence)."""
    from unit.mapping.platform.hand_descriptions import described
    return described({"name": name, "geometry": "recom_64x64_256", "keys": {"core_semantics": "mvm"},
                      "faithfulness": faith or {"class": "refused"}})


@pytest.fixture
def unpriced(tmp_path, monkeypatch):
    """The hand-made description beside the shipped ones, read by every reader of a target name."""
    from mimarsinan.mapping.platform import target_bundles as bundle_module
    known = {**bundle_module.target_bundles(), **_load_dir(tmp_path, _unpriced_bundle())}
    monkeypatch.setattr(bundle_module, "target_bundles", lambda: known)
    return "unpriced_chip"


def test_the_pricing_a_run_resolves_is_read_once_for_the_resolver_and_the_label():
    """The one source: the resolver prices by priced_under over physics_declarations, and run_pricing reads the same
    (a resolved platform's physics where it is resolved)."""
    from mimarsinan.deployment_record.platform_physics.resolve import physics_declarations, priced_under
    from mimarsinan.mapping.platform.descriptions.departures import run_pricing
    for declared, priced in PRICINGS:
        assert priced_under(*physics_declarations(declared)) == priced == run_pricing(declared)
    assert physics_declarations({}) == ("", {})
    assert run_pricing({"platform_physics_resolved": None, "platform_physics_profile": GENERIC}) is None
    assert run_pricing({"platform_physics_resolved": {"name": GENERIC}}) == GENERIC


@pytest.mark.parametrize("declared, priced", PRICINGS)
def test_the_run_door_s_label_prints_the_pricing_its_run_resolves(declared, priced, unpriced):
    """Label site 1, the run door (target_assignments): the label and the resolved platform name one physics."""
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    dp, pc = _expand({"target": unpriced, **declared})
    assert pc["target_faithfulness_class"] == f"refused: {_clause(priced)}"
    resolved = build_platform_constraints_resolved(merge_pipeline_config(dp, pc))
    assert (resolved["platform_physics_resolved"] or {}).get("name") == priced
    assert resolved["target_faithfulness_class"] == pc["target_faithfulness_class"]


def test_the_run_door_s_label_names_the_pricing_before_the_declared_departures(unpriced):
    _, pc = _expand({"target": unpriced, "platform_physics_profile": GENERIC,
                     "target_overrides": {"weight_bits": {"value": 4, "note": "a test override"}}})
    assert pc["target_faithfulness_class"] == f"refused: {_clause(GENERIC)}; weight_bits declared"


def test_a_description_stating_its_pricing_prints_no_pricing_at_the_run_door(tmp_path):
    """Its own profile, or its absence, is its pricing: a run's constants beside its own profile name nothing."""
    from mimarsinan.mapping.platform.target_bundles import load_bundles
    priced = {**_unpriced_bundle("priced_chip", {"class": "exact"})}
    priced["keys"] = {**priced["keys"], "platform_physics_profile": GENERIC}
    (tmp_path / "priced_chip.json").write_text(json.dumps(priced), encoding="utf-8")
    bundles = load_bundles(tmp_path)
    assert _expand({"target": "priced_chip", "platform_physics_overrides": CONSTANTS}, bundles)[1][
        "target_faithfulness_class"] == "exact"


def test_the_absolute_pricing_refusal_prints_the_run_s_label(unpriced):
    """Label site 2, the pricer's entries (absolute_pricing_refusal): an unpriced run prints the pricing it resolves."""
    from mimarsinan.mapping.platform.target_bundles import absolute_pricing_refusal
    name, reason = absolute_pricing_refusal({"target": unpriced, "platform_physics_resolved": None})
    assert name == "absolute_pricing" and reason == (f"target={unpriced!r} declares no physics profile (refused: "
                                                     f"{NO_PHYSICS}): no area, energy or latency is priced")
    assert absolute_pricing_refusal({"target": unpriced, "platform_physics_resolved": _odin_physics()}) is None


@pytest.mark.parametrize("declared, priced", PRICINGS[:3])
def test_the_ladder_s_target_spec_prints_the_pricing_its_declarations_resolve(declared, priced, unpriced,
                                                                              monkeypatch):
    """Label site 3, the scaling ladder's target spec (the rows' bundle label)."""
    ladder = _ladder()
    monkeypatch.setitem(ladder.TARGETS, unpriced, unpriced)
    assert ladder.target_spec(unpriced, declared)["bundle"]["faithfulness"] == f"refused: {_clause(priced)}"
    assert ladder.target_spec(unpriced)["bundle"]["faithfulness"] == f"refused: {NO_PHYSICS}"


def _design_point():
    import importlib.util
    path = Path(mimarsinan.__file__).resolve().parents[2] / "scripts" / "design_point_instance.py"
    spec = importlib.util.spec_from_file_location("design_point_instance_doors", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("declared, priced", PRICINGS[:3])
def test_the_design_point_s_chip_block_prints_the_pricing_its_platform_resolves(declared, priced, unpriced,
                                                                                monkeypatch):
    """Label site 4, the design point instance's chip block of a bundle row: its resolved platform's pricing."""
    driver = _design_point()
    monkeypatch.setattr(driver, "bundle_document", lambda name: _unpriced_bundle())
    ladder = driver.ladder
    monkeypatch.setitem(ladder.TARGETS, unpriced, unpriced)
    platform = ladder.resolved_platform(unpriced, "declared", None, declared, None, None)
    block = driver.chip_block({}, {"target": unpriced, "arithmetic_class": "a test class"}, 1, resolved=platform)
    assert block["faithfulness"]["class"] == "refused" and block["faithfulness"]["named"][0] == _clause(priced)
    unresolved = driver.chip_block({}, {"target": unpriced, "arithmetic_class": "a test class"}, None)
    assert unresolved["faithfulness"]["named"][0] == NO_PHYSICS


@pytest.mark.parametrize("name", ["puma_node_like", "prime_mat_256x256", "dynap_se_64x256x4"])
def test_a_shipped_description_stating_no_pricing_prints_the_profile_a_run_declares(name):
    """The verify's probe (m-5): PUMA, PRIME and DYNAP-SE with the generic profile declared are priced, and their label
    prints that profile, never "refused: no physics profile"."""
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    dp, pc = _expand({"target": name, "platform_physics_profile": GENERIC})
    resolved = build_platform_constraints_resolved(merge_pipeline_config(dp, pc))
    assert resolved["platform_physics_resolved"]["name"] == GENERIC
    assert pc["target_faithfulness_class"].startswith(f"refused: {_clause(GENERIC)}")
    assert NO_PHYSICS not in pc["target_faithfulness_class"]
    assert _expand({"target": name})[1]["target_faithfulness_class"].startswith(f"refused: {NO_PHYSICS}")


# --- [U-CHIPKEYS4] 3(c): the sweep's fields a script prints for people -----------------------------------------------

#: (field set, field) -> the script function that prints it for people (a description's own file read by name).
PRINTED = {("bundle", "display_name"): "scripts/design_point_instance.py:chip_block",
           ("faithfulness", "note"): "scripts/design_point_instance.py:chip_block"}


def _root(node):
    """The name an expression reads from: ``x.get(..)``, ``(x.get(..) or {})``, ``x[..]`` all read ``x``."""
    import ast
    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
        elif isinstance(node, ast.BoolOp):
            node = node.values[0]
        elif isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        else:
            return node.id if isinstance(node, ast.Name) else None


def _document_reads(fn):
    """The paths (field, or object field then its field) a function reads from a description's own file: every .get or
    subscript by a text whose receiver is the name it bound to bundle_document's result."""
    import ast
    names = {t.id for n in ast.walk(fn) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Name) and n.value.func.id == "bundle_document"
             for t in n.targets if isinstance(t, ast.Name)}
    reads = set()
    for sub in ast.walk(fn):
        key = sub.args[0] if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and \
            sub.func.attr == "get" and sub.args else sub.slice if isinstance(sub, ast.Subscript) else None
        if isinstance(key, ast.Constant) and isinstance(key.value, str) and _root(sub) in names:
            inner = sub.func.value if isinstance(sub, ast.Call) else sub.value
            inner = inner.values[0] if isinstance(inner, ast.BoolOp) else inner
            outer = inner.args[0].value if isinstance(inner, ast.Call) and inner.args and isinstance(
                inner.args[0], ast.Constant) else None
            reads.add((outer, key.value) if outer else key.value)
    return reads


def test_the_scripts_that_read_a_description_s_file_print_only_the_listed_fields():
    """The sweep, over every script: a function that reads a description's own file (bundle_document) reads its
    display name and its faithfulness's note, each listed as printed for people, and nothing else of the file."""
    import ast
    repo = Path(mimarsinan.__file__).resolve().parents[2]
    reads = {}
    for path in sorted((repo / "scripts").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            if found := _document_reads(fn):
                reads[f"{path.relative_to(repo)}:{fn.name}"] = found
    assert reads == {"scripts/design_point_instance.py:chip_block": {"display_name", "faithfulness",
                                                                      ("faithfulness", "note")}}
    assert set(PRINTED.values()) == set(reads)
    for (kind, field), _ in PRINTED.items():
        assert STATEMENT_DOORS[(kind, field)][0] == "people", (kind, field)


def _direct_faithfulness_reads(tree):
    """(function, attribute) of each read of a stored faithfulness (``<x>.faithfulness.<label|with_*|named|kind>``),
    with the innermost function holding it ("<module>" for none)."""
    import ast
    out = []

    def visit(node, function):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else function
            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Attribute) \
                    and child.value.attr == "faithfulness" and child.attr in (
                        "label", "with_declared", "with_departures", "with_pricing", "named", "kind"):
                out.append((function, child.attr))
            visit(child, inner)
    visit(tree, "<module>")
    return out


def test_every_label_site_reads_the_run_s_faithfulness():
    """The class swept: no reader of a description's faithfulness in the toolchain or its scripts prints its stored
    class itself; each takes run_faithfulness (the pricing the run resolves), whose own body is the one direct read."""
    import ast
    repo = Path(mimarsinan.__file__).resolve().parents[2]
    direct = set()
    for path in sorted([*(repo / "src" / "mimarsinan").rglob("*.py"), *(repo / "scripts").rglob("*.py")]):
        for function, attribute in _direct_faithfulness_reads(ast.parse(path.read_text(encoding="utf-8"))):
            direct.add((str(path.relative_to(repo)), function, attribute))
    assert direct == {("src/mimarsinan/mapping/platform/descriptions/departures.py", "run_faithfulness", "with_pricing")}
