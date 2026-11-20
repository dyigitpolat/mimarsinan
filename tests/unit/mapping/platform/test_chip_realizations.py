"""[U-CHIPKEYS, ruling D-5] A chip description declares its produced realizations: ODIN's 1,024 x 256 card realization
is a declared geometry of the ``odin`` bundle with the card's record, addressed as ``odin@<realization>``, and every
reader of a target name reads it through the one lookup (``get_target_bundle``).

[U-CHIPKEYS2] One hand-made refusal case per clause of the realization rule and the lookup, each red on the mutant that
removes its clause; a realization states its pricing (its own profile, or pricing absent and refused by name), never its
bundle's profile by inheritance; technology_blocks reads every target name through the one lookup.

[U-CHIPKEYS3] Every condition of the realization rule and every element of its closed sets has its own case (MC62: the
record's required report, sha256 and result, each); a realization's faithfulness is its bundle's class with the
departures of its own statements (test_chip_statement_doors.py holds the pricing door).

[U-CHIPKEYS4] 3(e): the realization record is judged by form at load, each field's type and form (the report's path in
its repository, the 64-hex digest, the texts), one case per form; what cannot be judged at load (that the report at that
path holds that digest, what the texts say) is carried, not judged."""

import copy
import json
import re
from pathlib import Path

import pytest

import mimarsinan
from mimarsinan.chip_simulation.soma_law import SomaLaw

TARGETS_DIR = Path(mimarsinan.__file__).parent / "mapping" / "platform" / "targets"
WIDE = "odin@odin_wide_1024x256_mb16"
SPEC_FIELDS = ("max_axons", "max_neurons", "count", "has_bias")


def _core_of(spec):
    return {**{f: getattr(spec, f) for f in SPEC_FIELDS}, "weight_sign_granularity": spec.weight_sign_granularity}


def test_odin_declares_its_produced_wide_realization():
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    wide = get_target_bundle(WIDE)
    values = wide.values
    assert values["cores"] == [{"max_axons": 1024, "max_neurons": 256, "count": 1, "has_bias": False,
                                "weight_sign_granularity": "per_synapse"}]
    assert (values["weight_bits"], values["membrane_bits"]) == (8, 16)
    stock = get_target_bundle("odin").values
    # [U-CHIPKEYS2] the realization's pricing is its own: it inherits no physics profile (v-m8)
    own = ("cores", "weight_bits", "membrane_bits", "platform_physics_profile")
    assert {k: v for k, v in values.items() if k not in own} == {k: v for k, v in stock.items() if k not in own}
    assert "platform_physics_profile" not in values and stock["platform_physics_profile"] == "odin"
    assert wide.claim_class == get_target_bundle("odin").claim_class
    for path in ("weight_bits", "membrane_bits", "cores.has_bias"):
        assert wide.statements[path]["kind"] == "stated", path


def test_the_wide_realization_is_the_fabric_the_card_ran():
    """One quantity, two sources: the realization's core, weight grid and soma law equal the wide fabric's spec
    (``chip_configs``), the projection the card's bitstream was built from."""
    from mimarsinan.chip_simulation.odin_fpga.chip_configs import WIDE_CHIP, chip_config_named
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    assert WIDE.split("@")[1] == WIDE_CHIP
    spec = chip_config_named(WIDE_CHIP).core_spec
    values = get_target_bundle(WIDE).values
    assert values["cores"] == [_core_of(spec)]
    assert values["weight_bits"] == spec.weight_bits
    assert SomaLaw.resolve(values) == spec.soma_law


def test_the_stock_description_is_the_stock_fabric():
    """The stock fabric's declaration counts logical slots (a row pair per signed input under the per-axon sign) and
    the description its physical rows: the law, the weight grid, the neurons, the count and the bias agree."""
    from mimarsinan.mapping.export.odin_gen.passthrough import stock_core_spec
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle
    spec = stock_core_spec()
    values = get_target_bundle("odin").values
    assert values["weight_bits"] == spec.weight_bits
    assert SomaLaw.resolve(values) == spec.soma_law
    core, = values["cores"]
    same = ("max_neurons", "count", "has_bias")
    assert {f: core[f] for f in same} == {f: getattr(spec, f) for f in same}
    assert core["weight_sign_granularity"] == spec.weight_sign_granularity == "per_axon"
    assert core["max_axons"] == 2 * spec.max_axons


def test_the_realization_carries_the_card_s_record():
    raw = json.loads((TARGETS_DIR / "odin.json").read_text(encoding="utf-8"))
    record = raw["realizations"]["odin_wide_1024x256_mb16"]["record"]
    assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
    assert record["report"].endswith("deployment_report.json") and record["design"] and record["result"]


def test_the_target_names_list_the_realization_and_an_unknown_one_refuses_listing_them():
    from mimarsinan.mapping.platform.target_bundles import UnknownTargetError, get_target_bundle, target_names
    assert WIDE in target_names() and "odin" in target_names()
    with pytest.raises(UnknownTargetError, match=re.escape(WIDE)):
        get_target_bundle("odin@no_such_realization")


def test_a_run_on_the_realization_resolves_its_keys_with_no_departure():
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle, target_departures
    from mimarsinan.mapping.platform.target_expansion import expand_target
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    dp, pc = expand_target({}, {"target": WIDE}, write_derived=True)
    assert pc["cores"] == get_target_bundle(WIDE).values["cores"] and pc["membrane_bits"] == 16
    assert pc["target_claim_class"] == get_target_bundle("odin").claim_class
    assert pc["target_faithfulness_class"] == get_target_bundle("odin").faithfulness.label()
    resolved = build_platform_constraints_resolved(merge_pipeline_config(dp, pc))
    assert resolved["weight_bits"] == 8 and target_departures(resolved) == ()


def test_a_realization_that_changes_a_stated_key_restates_it(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError, load_bundle
    raw = json.loads((TARGETS_DIR / "odin.json").read_text(encoding="utf-8"))
    changed = copy.deepcopy(raw)
    changed["realizations"]["odin_wide_1024x256_mb16"]["computation"].pop("weight_bits")
    path = tmp_path / "odin.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(TargetBundleError, match=r"odin_wide_1024x256_mb16.*weight_bits: stated 4, but the "
                                                r"description sets 8"):
        load_bundle(path)


def test_a_realization_names_a_declared_geometry_of_its_own(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError, load_bundle
    raw = json.loads((TARGETS_DIR / "odin.json").read_text(encoding="utf-8"))
    changed = copy.deepcopy(raw)
    changed["realizations"]["odin_wide_1024x256_mb16"]["geometry"]["name"] = raw["geometry"]["name"]
    path = tmp_path / "odin.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(TargetBundleError, match=r"geometry"):
        load_bundle(path)


_REPO = Path(mimarsinan.__file__).resolve().parents[2]
#: Every script and template generator that reads the bundle mapping (the readers of a target name outside the
#: package, whose own readers go through get_target_bundle).
_READERS = sorted(str(p.relative_to(_REPO)) for d in ("scripts", "templates") for p in (_REPO / d).rglob("*.py")
                  if "target_bundles" in p.read_text(encoding="utf-8"))


def test_the_readers_of_a_target_name_are_swept():
    assert {"scripts/scaling_ladder.py", "scripts/design_point_instance.py", "templates/generate.py"} <= set(_READERS)


@pytest.mark.parametrize("reader", _READERS)
def test_a_reader_of_a_target_name_takes_the_realization(reader):
    """The class swept, one test per site: a reader of a target name reads it through the one lookup, never by its
    own index into the bundle mapping (which holds no realization)."""
    text = (_REPO / reader).read_text(encoding="utf-8")
    assert not re.search(r"target_bundles\(\)\[", text), f"{reader} indexes the bundle mapping by name"


# --- [U-CHIPKEYS2] one hand-made refusal case per clause of the realization rule ------------------------------------

def _odin_with(change):
    raw = json.loads((TARGETS_DIR / "odin.json").read_text(encoding="utf-8"))
    changed = copy.deepcopy(raw)
    change(changed["realizations"]["odin_wide_1024x256_mb16"], changed)
    return changed


def _set(key, value):
    def change(spec, _):
        spec[key] = value
    return change


def _drop(key):
    def change(spec, _):
        spec.pop(key)
    return change


def _record(key, value):
    def change(spec, _):
        spec["record"][key] = value
    return change


def _keys(**keys):
    def change(spec, _):
        spec.setdefault("keys", {}).update(keys)
    return change


def _drop_record(key):
    def change(spec, _):
        spec["record"].pop(key)
    return change


def _both(absent):
    def change(spec, _):
        spec["pricing_absent"] = absent
        spec["keys"]["platform_physics_profile"] = "generic_estimated_22nm"
    return change


def _realizations(value):
    def change(_, data):
        data["realizations"] = value
    return change


def _rename(new):
    def change(_, data):
        data["realizations"] = {new: data["realizations"]["odin_wide_1024x256_mb16"]}
    return change


#: [U-CHIPKEYS4] 3(e): the refusal of a report path of no form.
_REPORT = r"record.report is the report's path in its repository"
#: clause id -> (what it refuses, the change to ODIN's file, the refusal's message); read off realizations.judged.
REALIZATION_CLAUSES = {
    "R01": ("realizations that are no object", _realizations(["odin_wide_1024x256_mb16"]),
            r"realizations must be an object"),
    "R02": ("a realization name of other characters", _rename("odin-wide"),
            r"a realization name is letters, digits and underscores"),
    "R03": ("a realization field of no kind (closed fields)", _set("extra", 1),
            r"the realization has unknown fields \['extra'\]"),
    "R04": ("a realization without its record (required fields)", _drop("record"), r"lacks \['record'\]"),
    "R05": ("a realization on a registered geometry", _set("geometry", "neurram_like"),
            r"a realization declares its geometry .* never a registered one"),
    "R06": ("a record field of no kind", _record("extra", 1), r"record has unknown fields \['extra'\]"),
    "R06b": ("a record without its report (MC62)", _drop_record("report"), r"record has .* lacks \['report'\]"),
    "R06c": ("a record without its sha256", _drop_record("sha256"), r"record has .* lacks \['sha256'\]"),
    "R06d": ("a record without its result (MC62)", _drop_record("result"), r"record has .* lacks \['result'\]"),
    "R06e": ("a record that is no object", _set("record", ["a.json"]), r"record must be an object"),
    "R07": ("a record whose sha256 is not 64 hex digits", _record("sha256", "74c659107020dd47"),
            r"record.sha256 is the report's 64-hex sha256"),
    "R08": ("a realization that states no pricing (it would inherit its bundle's profile)", _drop("pricing_absent"),
            r"a realization states its pricing"),
    "R09": ("a realization that states its own profile and its pricing absent",
            _keys(platform_physics_profile="odin"), r"a realization states its pricing"),
    "R10": ("a pricing absent with no reason", _set("pricing_absent", " "), r"a realization states its pricing"),
    "R10b": ("a pricing absent that is no text", _set("pricing_absent", 5), r"a realization states its pricing"),
    "R10c": ("an own profile beside a pricing absent of no text", _both(5), r"a realization states its pricing"),
    "R11": ("a realization key the geometry sets", _keys(weight_bits=4), r"weight_bits is set by the geometry"),
    "R12": ("a realization key no bundle may set", _keys(no_such_key=1), r"no_such_key is not a registry key"),
    "R12b": ("a realization key the geometry sets (its cores)", _keys(cores=[]), r"cores is set by the geometry"),
    "R12c": ("a realization key of an illegal value", _keys(membrane_bits="wide"), r"membrane_bits"),
    "R14b": ("a realization geometry of no field set", _set("geometry", {"name": "odin_x"}), r"geometry has unknown"),
    # [U-CHIPKEYS4] 3(e): the record judged by form at load, each field's type and form
    "R15": ("a report at an absolute path", _record("report", "/abs/deployment_report.json"), _REPORT),
    "R15b": ("a report above the repository", _record("report", "../deployment_report.json"), _REPORT),
    "R15c": ("a report path with an empty name", _record("report", "a//deployment_report.json"), _REPORT),
    "R15d": ("a report path with a '.' name", _record("report", "./deployment_report.json"), _REPORT),
    "R15e": ("an empty report path", _record("report", ""), _REPORT),
    "R15f": ("a report path that is no text", _record("report", 5), _REPORT),
    "R15g": ("a report path holding a space", _record("report", "a b/deployment_report.json"), _REPORT),
    "R15h": ("a report path ending at its separator", _record("report", "reports/"), _REPORT),
    "R15i": ("a report path with a '..' name inside", _record("report", "a/../deployment_report.json"), _REPORT),
    "R16": ("a sha256 that is a number of 64 digits", _record("sha256", int("7" * 64)),
            r"record.sha256 is the report's 64-hex sha256"),
    "R17": ("a blank design", _record("design", " "), r"record.design is a non-empty text, got ' '"),
    "R17b": ("a design that is no text", _record("design", 7), r"record.design is a non-empty text, got 7"),
    "R17c": ("a session that is no text", _record("session", {}), r"record.session is a non-empty text, got \{\}"),
    "R17d": ("an empty session", _record("session", ""), r"record.session is a non-empty text, got ''"),
    "R17e": ("a blank result", _record("result", " "), r"record.result is a non-empty text, got ' '"),
    "R17f": ("a result that is no text", _record("result", ["x"]), r"record.result is a non-empty text, got \['x'\]"),
}


@pytest.mark.parametrize("clause", sorted(REALIZATION_CLAUSES))
def test_the_loader_refuses_each_clause_of_the_realization_rule(clause, tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError, load_bundle
    _, change, match = REALIZATION_CLAUSES[clause]
    path = tmp_path / "odin.json"
    path.write_text(json.dumps(_odin_with(change)), encoding="utf-8")
    with pytest.raises(TargetBundleError, match=match):
        load_bundle(path)


@pytest.mark.parametrize("name", ["odin@odin_wide_1024x256_mb16@odin_wide_1024x256_mb16", "no_bundle@" + WIDE[5:],
                                  "odin@no_such_realization", 3])
def test_a_name_that_selects_no_bundle_or_realization_refuses_by_name(name):
    """R13 to R16: two separators, an unknown bundle, an unknown realization, a name that is no text."""
    from mimarsinan.mapping.platform.target_bundles import UnknownTargetError, get_target_bundle
    with pytest.raises(UnknownTargetError, match=r"names no target bundle"):
        get_target_bundle(name)


# --- [U-CHIPKEYS2] a realization states its pricing (v-m8) ---------------------------------------------------------

def test_the_wide_realization_states_its_pricing_absent_and_a_run_on_it_refuses_absolute_pricing_by_name():
    from mimarsinan.mapping.platform.target_bundles import absolute_pricing_refusal, get_target_bundle
    from mimarsinan.mapping.platform.target_expansion import expand_target
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    real = json.loads(get_target_bundle("odin").realizations_json)["odin_wide_1024x256_mb16"]
    assert "28 nm ASIC" in real["pricing_absent"] and "refuses absolute pricing by name" in real["pricing_absent"]
    dp, pc = expand_target({}, {"target": WIDE}, write_derived=True)
    resolved = build_platform_constraints_resolved(merge_pipeline_config(dp, pc))
    assert resolved.get("platform_physics_resolved") is None
    name, reason = absolute_pricing_refusal(resolved)
    assert name == "absolute_pricing" and WIDE in reason


def test_a_realization_that_states_its_own_profile_is_priced_by_it(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import load_bundle

    def own(spec, _):
        spec.pop("pricing_absent")
        spec["keys"]["platform_physics_profile"] = "generic_estimated_22nm"
    path = tmp_path / "odin.json"
    path.write_text(json.dumps(_odin_with(own)), encoding="utf-8")
    real = json.loads(load_bundle(path).realizations_json)["odin_wide_1024x256_mb16"]
    assert real["values"]["platform_physics_profile"] == "generic_estimated_22nm"


def test_the_wide_realization_s_provenance_says_how_its_capacity_was_set():
    raw = json.loads((TARGETS_DIR / "odin.json").read_text(encoding="utf-8"))
    provenance = raw["realizations"]["odin_wide_1024x256_mb16"]["geometry"]["provenance"]
    assert "set on the session's command line and not read back from the card" in provenance
    assert "150 of 150 exact certificates" in provenance


# --- [U-CHIPKEYS2] technology_blocks reads every target name through the one lookup (v-m7) --------------------------

def test_technology_blocks_reads_a_realization_through_the_one_lookup(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import load_bundles, technology_blocks
    raw = json.loads((TARGETS_DIR / "neurram.json").read_text(encoding="utf-8"))
    raw["realizations"] = {"r": {
        "geometry": {"name": "neurram_r", "cores": [{"max_axons": 256, "max_neurons": 256, "count": 2,
                                                       "has_bias": False}],
                     "weight_bits": 4, "provenance": "a test realization", "claim_class": "quarantined"},
        "keys": {"platform_physics_profile": "generic_estimated_22nm"}, "computation": {},
        "record": {"report": "r.json", "sha256": "0" * 64, "result": "a test record"}}}
    (tmp_path / "neurram.json").write_text(json.dumps(raw), encoding="utf-8")
    blocks = technology_blocks(load_bundles(tmp_path))
    assert blocks["generic_estimated_22nm"] == (("neurram@r", 2, 2),)
    assert blocks["neurram"] == (("neurram", 2, 2),)


# --- [U-CHIPKEYS3] the realization's own fields, each accepted, and its faithfulness ---------------------------------

def test_a_realization_takes_each_of_its_fields(tmp_path):
    """display_name, geometry, keys, computation, record (report, sha256, design, session, result), notes and
    pricing_absent together."""
    from mimarsinan.mapping.platform.target_bundles import load_bundle

    def every(spec, _):
        spec["display_name"] = "the display name of the test"  # [U-CHIPKEYS7] the alphabet
        spec["notes"] = "the test's"
        spec["record"].update({"design": "d", "session": "s"})
    path = tmp_path / "odin.json"
    path.write_text(json.dumps(_odin_with(every)), encoding="utf-8")
    real = json.loads(load_bundle(path).realizations_json)["odin_wide_1024x256_mb16"]
    assert real["display_name"] == "the display name of the test" and set(real["record"]) == {"report", "sha256", "design", "session",
                                                                             "result"}


def test_a_realization_s_departing_statement_is_printed_in_its_label_and_not_in_its_bundle_s(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle, load_bundles

    def departs(spec, data):
        data["faithfulness"] = {"class": "approximate", "named": ["a departure of the bundle"],
                                "given": data["faithfulness"]["given"]}
        spec["computation"]["membrane_bits"] = {"departs": {
            "computation": "a 16-bit register the card holds", "source": "s", "location": "l",
            "quote": "the membrane is a 16-bit register", "reads": "a 16-bit register", "assumed": 16,
            "reason": "the test's"}}
    (tmp_path / "odin.json").write_text(json.dumps(_odin_with(departs)), encoding="utf-8")
    known = load_bundles(tmp_path)
    wide = get_target_bundle(WIDE, known)
    given = 'approximate, given "leakage and SDSP learning off by the deployment"'  # [U-CHIPKEYS6] quoted fields
    assert wide.faithfulness.label() == (f'{given}: named "a departure of the bundle"; membrane_bits departs from '
                                         'its source\'s computation "a 16-bit register the card holds", assumed "16"')
    assert get_target_bundle("odin", known).faithfulness.label() == f'{given}: named "a departure of the bundle"'
    assert wide.faithfulness.given == get_target_bundle("odin").faithfulness.given


def test_a_realization_inherits_its_bundle_s_departure_unless_it_states_the_key(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import get_target_bundle, load_bundles
    raw = json.loads((TARGETS_DIR / "truenorth.json").read_text(encoding="utf-8"))
    raw["realizations"] = {"r": {
        "geometry": {"name": "truenorth_r", "cores": [{"max_axons": 256, "max_neurons": 256, "count": 2,
                                                         "has_bias": True}],
                     "weight_bits": 1, "provenance": "a test realization", "claim_class": "quarantined"},
        "keys": {}, "computation": {}, "pricing_absent": "a test realization no profile prices",
        "record": {"report": "r.json", "sha256": "0" * 64, "result": "a test record"}}}
    (tmp_path / "truenorth.json").write_text(json.dumps(raw), encoding="utf-8")
    known = load_bundles(tmp_path)
    assert get_target_bundle("truenorth@r", known).faithfulness.label() == get_target_bundle("truenorth").faithfulness.label()


# --- [U-CHIPKEYS3] the lookup and the names over a bundle whose realizations are stated empty -----------------------

def test_a_bundle_whose_realizations_are_empty_text_names_none():
    from mimarsinan.mapping.platform.descriptions.realizations import lookup, names
    from mimarsinan.mapping.platform.target_bundles import Faithfulness, TargetBundle
    bundle = TargetBundle("x", "g", "c", Faithfulness("exact"), "{}", "{}", "")
    assert names({"x": bundle}) == ["x"]
    assert lookup({"x": bundle}, "x@r") is None and lookup({"x": bundle}, "x") is bundle


def test_a_realization_with_no_keys_field_loads(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import load_bundle

    def no_keys(spec, _):
        spec.pop("keys")
        spec["computation"].pop("membrane_bits")
    path = tmp_path / "odin.json"
    path.write_text(json.dumps(_odin_with(no_keys)), encoding="utf-8")
    real = json.loads(load_bundle(path).realizations_json)["odin_wide_1024x256_mb16"]
    assert real["values"]["membrane_bits"] == json.loads((TARGETS_DIR / "odin.json").read_text())["keys"]["membrane_bits"]


@pytest.mark.parametrize("field", ["geometry", "computation", "record"])
def test_a_realization_without_a_required_field_refuses_by_name(field, tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError, load_bundle
    path = tmp_path / "odin.json"
    path.write_text(json.dumps(_odin_with(_drop(field))), encoding="utf-8")
    with pytest.raises(TargetBundleError, match=rf"the realization has unknown fields \[\] or lacks \['{field}'\]"):
        load_bundle(path)


def test_a_realization_that_is_no_object_refuses(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError, load_bundle
    path = tmp_path / "odin.json"
    path.write_text(json.dumps(_odin_with(_realizations({"odin_wide_1024x256_mb16": ["geometry"]}))), encoding="utf-8")
    with pytest.raises(TargetBundleError, match=r"the realization must be an object"):
        load_bundle(path)


def test_the_names_list_each_realization_after_its_bundle(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import load_bundles, target_names
    raw = json.loads((TARGETS_DIR / "odin.json").read_text(encoding="utf-8"))
    (tmp_path / "odin.json").write_text(json.dumps(raw), encoding="utf-8")
    assert target_names(load_bundles(tmp_path)) == ["odin", WIDE]


# --- [U-CHIPKEYS4] 3(e): the realization record judged by form at load; what cannot be judged is carried --------------

@pytest.mark.parametrize("report", ["r.json", "a/b-c_d.e/deployment_report.json", ".hidden/r.json", "a/..b/r.json",
                                    "Results_2026-11/R1.json", "a/Results_2026-11/R1.json"])
def test_a_record_of_each_form_loads_and_is_carried_as_stated(report, tmp_path):
    """A report path of names joined by '/', none of them '.' or '..'; design and session may be left out. The record
    is carried as stated: whether the report at that path holds that digest is not judged at load."""
    from mimarsinan.mapping.platform.target_bundles import load_bundle

    def form(spec, _):
        spec["record"] = {"report": report, "sha256": "f" * 64, "result": "a test record"}
    path = tmp_path / "odin.json"
    path.write_text(json.dumps(_odin_with(form)), encoding="utf-8")
    record = json.loads(load_bundle(path).realizations_json)["odin_wide_1024x256_mb16"]["record"]
    assert record == {"report": report, "sha256": "f" * 64, "result": "a test record"}


def test_a_record_s_every_defect_is_named_in_one_refusal(tmp_path):
    from mimarsinan.mapping.platform.target_bundles import TargetBundleError, load_bundle

    def two(spec, _):
        spec["record"].update({"report": "/r.json", "result": ""})
    path = tmp_path / "odin.json"
    path.write_text(json.dumps(_odin_with(two)), encoding="utf-8")
    with pytest.raises(TargetBundleError, match=rf"{_REPORT}.*; record.result is a non-empty text, got ''"):
        load_bundle(path)


def test_the_record_forms_are_the_record_s_closed_fields():
    from mimarsinan.mapping.platform.descriptions.realizations import RECORD_FORMS, _RECORD_FIELDS
    assert set(RECORD_FORMS) == set(_RECORD_FIELDS)
