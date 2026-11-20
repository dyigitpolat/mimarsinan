"""[U-HOST2] The value rule's other clauses: the derivation's one declared form (``hosts.derivation``), the registry's
contract (``hosts.registry``), the planes and link ends the description states, and compositions (verify m-4).

A row naming a host is admitted as the named mode's own row, or as a derivation of the mode's rows declared in one form
(``{"of": {<row>: <factor>, ...}, "combine": "sum" | "slower"}``) which the loader recomputes from the description and
compares, else refused by name. W22's scaled host rows and a server-host sensitivity are written in that form here. A
test that needs other host numbers registers a description of its own through the registry's contract. No
description, mode, host or chip is named here (the first read description, its modes, the registry's profiles).
"""

from __future__ import annotations

import copy
import math

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import apply_overrides, available_profiles, get_platform_physics
from mimarsinan.deployment_record.platform_physics.constants import spec_for
from mimarsinan.deployment_record.platform_physics.hosts import derivation as DV
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.admission import (
    described, host_derivation_block, register_host_description, unregister_host_description,
)
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    ENERGY_PLANE_FIELD, HOST_IDENTITY_FIELD, SYSTEM, HostDescriptionError, host_attach_block, host_description,
    host_description_block, host_descriptions,
)
from mimarsinan.deployment_record.platform_physics.hosts.link_ends import LINK_ENDS_FIELD
from mimarsinan.deployment_record.platform_physics.hosts.registry import HostRegistryError
from mimarsinan.deployment_record.platform_physics.hosts.rows import HOST
from mimarsinan.deployment_record.platform_physics.loader import OVERRIDE_FIELDS
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.units import POWER, TIME

from unit.deployment_record.g3_fixtures import quantities
from unit.deployment_record.registered_hosts import declared_rows, description_data, registered

NAME = host_descriptions()[0]
DESC = host_description(NAME)
DEFAULT = f"{NAME} {DESC.default_mode}"
CORNERS = ("low", "nominal", "high")
VIEW = dict(synaptic_events=1e6, host_macs=1e8, compute_op_count=3, cores_physical=8, latency_steps=32, timesteps=4,
            neurons_used=100, segment_cores=1)


def _block():
    return host_description_block(NAME)


def _chip():
    for name in available_profiles():
        try:
            resolve_platform_physics(name, merge_override_blocks({"host": _block()}))
        except ToolchainRefusal:
            continue
        return name
    raise AssertionError("no registry profile takes the identified host block")


def _resolve(rows, chip=None):
    return resolve_platform_physics(chip or _chip(), merge_override_blocks({"host": rows}))


def _refused(call):
    with pytest.raises(ToolchainRefusal) as refused:
        call()
    return str(refused.value)


def _si(band):
    return tuple(getattr(band, end) for end in CORNERS)


def _own(key):
    d, _why = described(DEFAULT)
    return d.rows[key]


DECLARED = sorted(described(DEFAULT)[0].rows)


class TestTheOneForm:
    def test_the_form_is_stated_once_and_read_by_the_loader(self):
        """The field, its two combinations and the form's text live in hosts.derivation; the loader reads the field."""
        assert DV.DERIVATION_FIELD == "derived_from" and DV.COMBINES == ("sum", "slower")
        assert '"derived_from": {"of": {"<row the named mode declares>": <factor>, ...}' in DV.__doc__
        assert DV.DERIVATION_FIELD in OVERRIDE_FIELDS

    @pytest.mark.parametrize("key", DECLARED)
    @pytest.mark.parametrize("factor", [0.5, 3.0])
    def test_a_scaled_row_is_admitted_at_its_recomputed_band_and_seals_its_declaration(self, key, factor):
        """One row at one factor: the band is the factor times the description's at every corner; the declaration is
        sealed with the row and the sealed physics reads back."""
        rows = {**_block(), **host_derivation_block(NAME, "default", {key: factor})}
        physics = _resolve(rows)
        assert _si(physics.band(key)) == pytest.approx(tuple(factor * v for v in _si(_own(key).band)), rel=1e-12)
        assert physics.constants[key].derived_from and physics.constants[key].evidence_kind == "derived"
        payload = physics.to_dict()
        assert payload["constants"][key][DV.DERIVATION_FIELD] == {"combine": "sum", "of": {key: factor}}
        assert PlatformPhysics.from_dict(payload).to_dict() == payload

    @pytest.mark.parametrize("key", DECLARED)
    @pytest.mark.parametrize("factor", [0.2, 5.0])
    def test_w22s_scaled_host_row_is_written_in_the_form(self, key, factor):
        """W22's scaling sites (TR/e2e/e7/w22_fidelity.py scaled_from and scaled_block, lines 388-431 at 5cad1cb7):
        the resolved row's band times the factor at all three corners, its unit kept, class derived, its citation,
        its host, energy plane, link ends and note kept. Declared with the one form it is admitted; the same row
        without the declaration is another value under the host's name and refuses by name."""
        physics = _resolve(_block())
        v = physics.constants[key]
        base = {"low": v.low, "nominal": v.nominal, "high": v.high, "unit": v.unit, "evidence_kind": v.evidence_kind,
                "citation": v.citation, HOST_IDENTITY_FIELD: v.host}
        if key in physics.system_energy_rows:
            base[ENERGY_PLANE_FIELD] = SYSTEM
        if key in physics.link_ends:
            base[LINK_ENDS_FIELD] = {c: list(e) for c, e in physics.link_ends[key].items()}
        scaled = {"nominal": base["nominal"] * factor, "low": base["low"] * factor, "high": base["high"] * factor,
                  "unit": base["unit"], "evidence_kind": "derived", "citation": base["citation"],
                  "derivation": f"W22 robustness: the resolved {key} ({base['evidence_kind']}) x {factor:g}",
                  **{f: base[f] for f in (HOST_IDENTITY_FIELD, ENERGY_PLANE_FIELD, LINK_ENDS_FIELD) if f in base},
                  "note": "W22 robustness scaling (test)"}
        point = apply_overrides(physics, {key: {**scaled, DV.DERIVATION_FIELD: {"of": {key: factor}}}})
        assert _si(point.band(key)) == pytest.approx(tuple(factor * x for x in _si(v.band)), rel=1e-12)
        assert key in _refused(lambda: apply_overrides(physics, {key: scaled}))

    def test_a_server_host_sensitivity_is_a_declared_derivation_of_the_identified_hosts_rows(self):
        """A faster, hungrier host stated against the identified one: its rate, dispatch and powers as factors of
        the description's rows. Admitted, and the priced host time is the description's arithmetic at those
        factors (the edge host's own rows, never a development machine's numbers)."""
        factors = {"host_macs_per_s": 2.0, "t_host_op_overhead": 0.5, "p_host": 20.0, "p_host_idle": 10.0}
        server = {**_block(), **host_derivation_block(NAME, "default", factors)}
        physics = _resolve(server)
        terms = {t.name: t for t in price_absolute(quantities(**VIEW), physics).terms}
        rate, dispatch = _own("host_macs_per_s").band, _own("t_host_op_overhead").band
        expected = VIEW["host_macs"] / (2.0 * rate.nominal) + VIEW["compute_op_count"] * 0.5 * dispatch.nominal
        assert terms["latency_host_s"].band.nominal == pytest.approx(expected, rel=1e-9)

    def test_a_declared_band_unlike_its_recomputation_refuses(self):
        key = DECLARED[0]
        row = host_derivation_block(NAME, "default", {key: 2.0})[key]
        row.update({end: row[end] * 1.5 for end in CORNERS})
        assert key in _refused(lambda: _resolve({**_block(), key: row}))

    def test_a_derivation_of_a_row_the_mode_refuses_refuses_with_the_descriptions_reason(self):
        """A row for a mode the description refuses is refused by name, quoting the description's own reason."""
        for mode, facts in sorted(DESC.modes.items()):
            for key, reason in sorted((facts.get("refused") or {}).items()):
                base = host_description_block(NAME, DESC.default_mode)[key]
                row = {**base, HOST_IDENTITY_FIELD: f"{NAME} {mode}", "evidence_kind": "derived",
                       "derivation": "x 1 (test)", DV.DERIVATION_FIELD: {"of": {key: 1.0}}}
                why = _refused(lambda: _resolve({key: row}))
                assert key in why and reason in why
                own = {**base, HOST_IDENTITY_FIELD: f"{NAME} {mode}"}
                assert reason in _refused(lambda: _resolve({key: own}))

    @pytest.mark.parametrize("bad", [
        lambda k: {"of": {k: 0.0}}, lambda k: {"of": {k: -1.0}}, lambda k: {"of": {k: math.inf}},
        lambda k: {"of": {k: math.nan}}, lambda k: {"of": {k: True}}, lambda k: {"of": {k: "2"}},
        lambda k: {"of": {"no_such_row": 1.0}}, lambda k: {"of": {}}, lambda k: {"of": {k: 1.0}, "combine": "max"},
        lambda k: {"of": {k: 1.0}, "scale": 2.0}, lambda k: {"factor": 2.0}, lambda k: [(k, 2.0)]],
        ids=["zero", "negative", "inf", "nan", "bool", "str", "unknown_row", "empty", "combine", "extra_key",
             "no_of", "not_a_mapping"])
    def test_a_declaration_outside_the_form_refuses_by_name(self, bad):
        key = DECLARED[0]
        row = {**_block()[key], "evidence_kind": "derived", "derivation": "test", DV.DERIVATION_FIELD: bad(key)}
        assert key in _refused(lambda: _resolve({**_block(), key: row}))

    def test_slower_composes_times_only_and_a_derivation_keeps_its_dimension(self):
        powers = [k for k in DECLARED if spec_for(k).dimension == POWER]
        times = [k for k in DECLARED if spec_for(k).dimension == TIME]
        others = [k for k in DECLARED if spec_for(k).dimension != spec_for(powers[0]).dimension]
        with pytest.raises(DV.HostDerivationError, match="composes times"):
            DV.judged(powers[0], {"of": {powers[0]: 1.0}, "combine": "slower"})
        with pytest.raises(DV.HostDerivationError, match="dimension"):
            DV.judged(powers[0], {"of": {powers[0]: 1.0, others[0]: 1.0}})
        assert DV.judged(times[0], {"of": {t: 1.0 for t in times}, "combine": "slower"})[1] == "slower"

    def test_a_host_rows_derivation_names_that_row(self):
        powers = [k for k in DECLARED if spec_for(k).dimension == POWER]
        key, other = powers[0], powers[1]
        row = host_derivation_block(NAME, "default", {key: {"of": {key: 1.0, other: 1.0}}})[key]
        row[DV.DERIVATION_FIELD] = {"of": {other: 1.0}}
        row.update({end: getattr(_own(other), end) for end in CORNERS})
        assert "derivation of that row" in _refused(lambda: _resolve({**_block(), key: row}))

    def test_a_derivation_is_class_derived(self):
        key = DECLARED[0]
        row = {**host_derivation_block(NAME, "default", {key: 2.0})[key], "evidence_kind": "published",
               "citation": "test"}
        assert "class 'published'" in _refused(lambda: _resolve({**_block(), key: row}))

    @pytest.mark.parametrize("combine", ["sum", "slower"])
    def test_a_stated_combination_of_rows_is_recomputed(self, combine):
        times = [k for k in DECLARED if spec_for(k).dimension == TIME]
        key, other = times[0], times[1]
        block = host_derivation_block(NAME, "default", {key: {"of": {key: 1.0, other: 2.0}, "combine": combine}})
        physics = _resolve({**_block(), **block})
        pick = max if combine == "slower" else sum
        expected = tuple(pick([a, 2.0 * b]) for a, b in zip(_si(_own(key).band), _si(_own(other).band)))
        assert _si(physics.band(key)) == pytest.approx(expected, rel=1e-12)

    def test_a_sealed_derivation_is_judged_by_the_same_rule(self):
        key = DECLARED[0]
        payload = _resolve({**_block(), **host_derivation_block(NAME, "default", {key: 2.0})}).to_dict()
        for change in ({DV.DERIVATION_FIELD: {"combine": "sum", "of": {key: 3.0}}},
                       {end: payload["constants"][key][end] * 1.01 for end in CORNERS}):
            changed = copy.deepcopy(payload)
            changed["constants"][key].update(change)
            assert key in _refused(lambda: PlatformPhysics.from_dict(changed))
        dropped = copy.deepcopy(payload)
        del dropped["constants"][key][DV.DERIVATION_FIELD]
        assert key in _refused(lambda: PlatformPhysics.from_dict(dropped))

    def test_the_own_row_restated_in_another_unit_refuses_by_its_unit(self):
        """The rule compares the unit as well as the band: the description's own value restated in another unit of
        the same dimension (the same SI band) is not the description's row."""
        from mimarsinan.deployment_record.units import to_canonical, unit_for

        for key in DECLARED:
            row = _block()[key]
            others = [u for u in ("W", "mW", "G/s", "M/s", "us", "ns", "pJ", "nJ")
                      if u != row["unit"] and unit_for(u).dimension == unit_for(row["unit"]).dimension]
            if not others:
                continue
            scale = to_canonical(1.0, row["unit"]) / to_canonical(1.0, others[0])
            restated = {**row, "unit": others[0], **{end: row[end] * scale for end in CORNERS}}
            assert "unit" in _refused(lambda: _resolve({**_block(), key: restated}))
            return
        raise AssertionError("no declared row has a second unit of its dimension")

    def test_a_derivation_on_a_row_naming_no_host_refuses(self):
        key = DECLARED[0]
        row = host_derivation_block(NAME, "default", {key: 2.0})[key]
        del row[HOST_IDENTITY_FIELD]
        assert key in _refused(lambda: _resolve({key: row}))


class TestPlanesAndLinkEnds:
    def test_a_host_energy_row_under_another_plane_refuses(self):
        """Each energy row's plane is the description's statement: a system row declared attributable, or an
        attributable row declared system, would move its energy between the headline and the system line."""
        d, _why = described(DEFAULT)
        for key, plane in sorted(d.planes.items()):
            row = copy.deepcopy(_block()[key])
            if plane == SYSTEM:
                del row[ENERGY_PLANE_FIELD]
            else:
                row[ENERGY_PLANE_FIELD] = SYSTEM
            why = _refused(lambda: _resolve({**_block(), key: row}))
            assert key in why and ENERGY_PLANE_FIELD in why

    def test_a_link_row_with_other_ends_refuses(self):
        d, _why = described(DEFAULT)
        for key, ends in sorted(d.ends.items()):
            row = copy.deepcopy(_block()[key])
            row[LINK_ENDS_FIELD] = {c: ["host", "chip"] for c in CORNERS}
            if {c: tuple(v) for c, v in row[LINK_ENDS_FIELD].items()} == dict(ends):
                row[LINK_ENDS_FIELD] = {c: ["host"] for c in CORNERS}
            assert LINK_ENDS_FIELD in _refused(lambda: _resolve({**_block(), key: row}))
            del row[LINK_ENDS_FIELD]
            assert LINK_ENDS_FIELD in _refused(lambda: _resolve({**_block(), key: row}))


#: A test-only description's rows (test numbers, no machine's): a rate and a dispatch time.
TEST_ROWS = {"host_macs_per_s": {"nominal": 20.0, "unit": "G/s", "evidence_kind": "estimated", "note": "test"},
             "t_host_op_overhead": {"nominal": 30.0, "unit": "us", "evidence_kind": "estimated", "note": "test"}}


class TestTheRegistry:
    def test_a_registered_description_is_read_while_registered_and_only_then(self, tmp_path):
        with registered(tmp_path, "test_registered_host", TEST_ROWS) as host:
            physics = _resolve(declared_rows("test_registered_host", TEST_ROWS))
            assert {physics.constants[k].host for k in TEST_ROWS} == {host}
            assert "test_registered_host" in host_descriptions()
        assert "test_registered_host" not in host_descriptions()
        assert "no read host description" in _refused(
            lambda: _resolve(declared_rows("test_registered_host", TEST_ROWS)))

    def test_only_the_registered_descriptions_own_values_are_admitted(self, tmp_path):
        with registered(tmp_path, "test_registered_host", TEST_ROWS):
            rows = declared_rows("test_registered_host", TEST_ROWS)
            rows["host_macs_per_s"]["nominal"] = 40.0
            assert "host_macs_per_s" in _refused(lambda: _resolve(rows))

    def test_a_description_is_judged_whole_before_it_is_read(self, tmp_path):
        measured = {**TEST_ROWS, "host_macs_per_s": {**TEST_ROWS["host_macs_per_s"], "evidence_kind": "measured"}}
        import json

        (tmp_path / "test_measured_host.md").write_text("test")
        path = tmp_path / "test_measured_host.json"
        path.write_text(json.dumps(description_data("test_measured_host", measured)))
        with pytest.raises(HostDescriptionError, match="measured"):
            register_host_description(path)
        assert "test_measured_host" not in host_descriptions()

    def test_a_name_already_read_and_an_unknown_name_refuse(self, tmp_path):
        import json

        (tmp_path / f"{NAME}.md").write_text("test")
        (tmp_path / f"{NAME}.json").write_text(json.dumps(description_data(NAME, TEST_ROWS)))
        with pytest.raises(HostRegistryError, match="already read"):
            register_host_description(tmp_path / f"{NAME}.json")
        with pytest.raises(HostRegistryError, match="cannot be taken out"):
            unregister_host_description(NAME)
        with pytest.raises(HostRegistryError):
            register_host_description(tmp_path / "missing.json")


def _store_chip():
    """The first registry profile that declares its backing store (a host attach row composes on it once the run
    restates the store as host-owned)."""
    return next(name for name in available_profiles() if get_platform_physics(name).backing_store is not None)


class TestCompositions:
    def test_a_measured_row_with_combine_slower_is_judged_before_composition(self):
        """m-4 (the verify's a10): on an identified physics, a measured dispatch row naming the same host with
        combine slower is judged as declared (measured, not the description's), never relabelled 'derived' by the
        composition; and a host row composes with no other stage through combine."""
        physics = _resolve(_block())
        row = {"nominal": 13.48, "unit": _own("t_host_op_overhead").unit, "evidence_kind": "measured",
               "note": "a measurement (test)", "combine": "slower", HOST_IDENTITY_FIELD: DEFAULT}
        why = _refused(lambda: apply_overrides(physics, {"t_host_op_overhead": row}))
        assert "class 'measured'" in why and "combine" in why
        own = {**_block()["t_host_op_overhead"], "combine": "slower"}
        assert "combine" in _refused(lambda: apply_overrides(physics, {"t_host_op_overhead": own}))

    def test_the_attach_row_is_a_declared_derivation_and_the_composed_row_is_the_chips(self):
        """The programming channel's host stage is declared as a derivation of the host's link row; it composes with
        the chip's own port (slower), and the composed chip row carries no host and no derivation (verify m-1's
        'identity on non-host rows')."""
        attach = host_attach_block(NAME)
        (key,) = attach
        ((source, factor),) = attach[key][DV.DERIVATION_FIELD]["of"].items()
        assert factor == 1.0 and spec_for(source).group == HOST and spec_for(key).group != HOST
        blocks = {"host": _block(), "host_attach": attach, "store": ST.store_block(ST.HOST, "a variant (test)")}
        physics = resolve_platform_physics(_store_chip(), merge_override_blocks(blocks))
        assert physics.constants[key].host == "" and physics.constants[key].derived_from == ""
        assert "host" not in physics.constants[key].to_dict() and "derived_from" not in physics.constants[key].to_dict()
        chip = get_platform_physics(_store_chip())
        for i, _end in enumerate(CORNERS):
            assert _si(physics.band(key))[i] == pytest.approx(max(_si(chip.band(key))[i], _si(physics.band(source))[i]))

    def test_an_attach_row_with_a_foreign_value_or_no_derivation_refuses(self):
        attach = host_attach_block(NAME)
        (key,) = attach
        blocks = {"host": _block(), "store": ST.store_block(ST.HOST, "a variant (test)")}
        foreign = {key: {**attach[key], "nominal": attach[key]["nominal"] / 100.0, "low": attach[key]["low"] / 100.0,
                         "high": attach[key]["high"] / 100.0}}
        assert key in _refused(lambda: resolve_platform_physics(
            _store_chip(), merge_override_blocks({**blocks, "host_attach": foreign})))
        plain = {key: {k: v for k, v in attach[key].items() if k != DV.DERIVATION_FIELD}}
        assert key in _refused(lambda: resolve_platform_physics(
            _store_chip(), merge_override_blocks({**blocks, "host_attach": plain})))

    def test_an_attach_row_of_another_host_beside_the_host_rows_refuses(self, tmp_path):
        attach = host_attach_block(NAME)
        (key,) = attach
        ((source, _factor),) = attach[key][DV.DERIVATION_FIELD]["of"].items()
        rows = {source: {"nominal": 1.0, "unit": attach[key]["unit"], "evidence_kind": "estimated", "note": "test"}}
        with registered(tmp_path, "test_attach_host", rows) as other:
            stage = {**attach[key], HOST_IDENTITY_FIELD: other, "low": 1.0, "nominal": 1.0, "high": 1.0}
            blocks = {"host": _block(), "host_attach": {key: stage}, "store": ST.store_block(ST.HOST, "a (test)")}
            assert "one host" in _refused(lambda: resolve_platform_physics(_store_chip(),
                                                                           merge_override_blocks(blocks)))
