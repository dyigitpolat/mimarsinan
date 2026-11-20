"""[U-HOST4] The third verify's m-2 as a class (rulings 2026-11-11, U-HOST3 item 1): a refusal clause in the unit's scope
that no case pins. The four clauses whose mutants survived the unit's tests (the verify's v0378, v0380, v0439, v0442,
v0444 and v0578), each pinned by a hand-made case red on its mutant; the per-condition harness (steps/
condition_mutants.py) now also replaces a compound condition as a whole and removes each raise statement, and the
cases its survivors asked for follow. Each case names the clause it pins. No description, mode, host or chip is named
here: the first read description, its default mode, and the first registry profile that takes its block.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics import available_profiles, get_platform_physics
from mimarsinan.deployment_record.platform_physics.constants import PHYSICS_CONSTANTS
from mimarsinan.deployment_record.platform_physics.hosts import admission as A
from mimarsinan.deployment_record.platform_physics.hosts import registry as REG
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.derivation import (
    DERIVATION_FIELD, HostDerivationError, judged, parsed, text_refusal,
)
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    host_attach_block, host_description, host_description_block, host_descriptions,
)
from mimarsinan.deployment_record.platform_physics.loader import apply_overrides
from mimarsinan.deployment_record.platform_physics.profile import PHYSICS_FORMAT_VERSION, PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics

from unit.deployment_record.registered_hosts import registered_copy

NAME = host_descriptions()[0]
STORE_CHIP = next(p for p in available_profiles() if get_platform_physics(p).backing_store is not None)


def _chip():
    for name in available_profiles():
        try:
            resolve_platform_physics(name, merge_override_blocks({"host": host_description_block(NAME)}))
        except ToolchainRefusal:
            continue
        return name
    raise AssertionError("no registry profile takes the identified host block")


CHIP = _chip()


def _sealed():
    return resolve_platform_physics(CHIP, merge_override_blocks({"host": host_description_block(NAME)})).to_dict()


def _store_physics():
    blocks = {"host": host_description_block(NAME), "store": ST.store_block(ST.HOST, "a variant (test)")}
    return resolve_platform_physics(STORE_CHIP, merge_override_blocks(blocks))


def _refused(call, kind=ToolchainRefusal):
    with pytest.raises(kind) as why:
        call()
    return str(why.value)


class TestASealedDerivationTextThatIsNoDeclaration:
    """derivation.py:93 (the verify's v0378, the raise removed): the text is refused by name, never read on."""

    def test_parsed_refuses_it_by_name(self):
        key = next(iter(host_description_block(NAME)))
        assert "is not a declaration" in _refused(lambda: parsed(key, "no declaration (test)"), HostDerivationError)

    def test_a_sealed_physics_carrying_it_refuses_by_name(self):
        payload = _sealed()
        key = next(k for k, value in sorted(payload["constants"].items()) if value.get("host"))
        payload["constants"][key][DERIVATION_FIELD] = "no declaration (test)"
        why = _refused(lambda: PlatformPhysics.from_dict(payload))
        assert key in why and "is not a declaration" in why


class TestADerivedChipRowsTextNamingItsOwnRow:
    """derivation.py:129 (the verify's v0380, the row's own key dropped from the rows a text may name): a derived chip
    row's text may name the row it derives (the attach row names no host row of its own key)."""

    def test_the_text_judge_admits_it(self):
        key, row = next(iter(host_attach_block(NAME).items()))
        terms, _combine = judged(key, row[DERIVATION_FIELD])
        assert key not in dict(terms)
        assert text_refusal(key, f"{key}: {row['derivation']}", terms) is None

    def test_a_run_declaring_it_is_admitted(self):
        block = copy.deepcopy(host_attach_block(NAME))
        key = next(iter(block))
        block[key]["derivation"] = f"{key}: {block[key]['derivation']}"
        physics = apply_overrides(_store_physics(), block)
        assert block[key]["derivation"] in physics.constants[key].derivation

    def test_a_text_naming_another_vocabulary_row_still_refuses(self):
        block = copy.deepcopy(host_attach_block(NAME))
        key = next(iter(block))
        other = next(k for k in sorted(host_description_block(NAME)) if k not in block[key][DERIVATION_FIELD]["of"])
        block[key]["derivation"] = f"{other}: {block[key]['derivation']}"
        assert other in _refused(lambda: apply_overrides(_store_physics(), block))


class TestADescriptionWithoutItsProseFile:
    """description.py:137-139 (the verify's v0439 and v0442, the compound condition replaced as a whole by False, and
    v0444, the raise removed): a description read as a file pair refuses by name when its prose file is not beside
    it."""

    def test_registering_it_refuses_by_name_and_registers_nothing(self, tmp_path):
        name = "test_no_prose_host"
        data = json.loads(REG.description_path(NAME).read_text())
        data.update(name=name, description_file=f"{name}.md")
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data, indent=1))
        why = _refused(lambda: A.register_host_description(path))
        assert name in why and "missing beside it" in why
        assert name not in REG.description_paths()

    def test_a_registered_description_whose_prose_file_goes_refuses_by_name(self, tmp_path):
        with registered_copy(tmp_path, NAME, lambda data: None, as_name="test_prose_gone_host") as name:
            (tmp_path / f"{name}.md").unlink()
            why = _refused(lambda: host_description(name))
            assert name in why and "missing beside it" in why

    def test_a_held_description_is_judged_as_its_json_alone(self):
        payload = _sealed()
        assert PlatformPhysics.from_dict(payload).to_dict() == payload


class TestTheOriginOfARowThatDoesNotResolve:
    """admission.py:232-233 (the harness's h0136, the test replaced by False, and h0139, the raise removed): the one
    admission rule's entry refuses by name a host its caller names (``also_named``) or a row its caller counts as
    judged (``judged``) when that host resolves to no read description, never an AttributeError."""

    def test_a_named_host_no_description_declares_refuses_by_name(self):
        why = _refused(lambda: A.admit_host_rows({}, "a test", also_named=["test_no_such_host test"]),
                       A.HostRowAdmissionError)
        assert "test_no_such_host" in why and "no read host description declares" in why

    def test_a_row_counted_as_judged_naming_no_read_description_refuses_by_name(self):
        payload = _sealed()
        key = next(k for k, value in sorted(payload["constants"].items()) if value.get("host"))
        stray = dataclasses.replace(PlatformPhysics.from_dict(payload).constants[key], host="test_no_such_host test")
        why = _refused(lambda: A.admit_host_rows({key: stray}, "a test", judged=[key]), A.HostRowAdmissionError)
        assert "test_no_such_host" in why and "no read host description declares" in why


class TestTheAttachRowsNoteOfALinkRowWithoutADerivation:
    """description.py:296 (the harness's h0413, the empty text's ``or`` replaced as a whole by True): the attach row's
    note states the link row's own derivation, nothing where the link row states none (a description whose link row
    is an estimate)."""

    def test_the_note_states_no_derivation_where_the_link_row_states_none(self, tmp_path):
        link = next(iter(host_attach_block(NAME).values()))[DERIVATION_FIELD]["of"]
        (key,) = link

        def estimate(data):
            for rows in [data.get("constants") or {}] + [m.get("constants") or {} for m in data["modes"].values()]:
                if key in rows:
                    rows[key].pop("derivation", None)
                    rows[key].update(evidence_kind="estimated", note="a test's estimate of the link (test)")

        with registered_copy(tmp_path, NAME, estimate, as_name="test_link_estimate_host") as name:
            note = next(iter(host_attach_block(name).values()))["note"]
        assert f"(its {key}, derived as: ); a test's estimate of the link (test)" in note


class TestTheLoadersOwnClauses:
    """loader.py:152 (h0446, the empty override set's test replaced by False) and loader.py:201 (h0460, h0480, h0482,
    the compound condition replaced as a whole by True)."""

    def test_no_override_returns_the_physics_as_given(self):
        """A call that declares nothing re-judges nothing and seals nothing: even a physics built in code with host
        rows and no origin comes back as given."""
        physics = dataclasses.replace(PlatformPhysics.from_dict(_sealed()), host_origin=None)
        assert apply_overrides(physics, {}) is physics and physics.host_origin is None

    def test_a_chip_row_deriving_from_host_rows_and_naming_no_host_is_refused_on_the_result(self):
        """Only a row naming a host, or a host row, is judged before composition; a chip row that names no host and
        states a derivation is refused by name on the composed result (the one admission rule's judgment of the
        physics with the run's overrides)."""
        key, row = next(iter(copy.deepcopy(host_attach_block(NAME)).items()))
        row.pop("host")
        why = _refused(lambda: apply_overrides(_store_physics(), {key: row}))
        assert key in why and "names no host" in why
        assert "with the run's overrides" in why and "the run's declared rows" not in why


class TestThePhysicsOwnClauses:
    """profile.py:198 (h0562) and profile.py:200 (h0558), the tests replaced by False: R5's clauses of the sealed
    physics' constructor, in the scope since the version rule's reader beside them is the unit's."""

    def test_a_constant_keyed_under_another_key_refuses_by_name(self):
        physics = PlatformPhysics.from_dict(_sealed())
        first, second = sorted(physics.constants)[:2]
        constants = {**physics.constants, first: physics.constants[second]}
        why = _refused(lambda: dataclasses.replace(physics, constants=constants), ValueError)
        assert first in why and second in why and "carries key" in why

    def test_a_system_level_row_naming_no_declared_constant_refuses_by_name(self):
        physics = PlatformPhysics.from_dict(_sealed())
        stray = next(k for k in sorted(PHYSICS_CONSTANTS) if k not in physics.constants)
        why = _refused(lambda: dataclasses.replace(physics, system_energy_rows=physics.system_energy_rows + (stray,)),
                       ValueError)
        assert stray in why and "system_energy_rows" in why


class TestASealedPhysicsOfAnotherVersion:
    """profile.py:195 (the verify's v0578, the raise removed): a sealed physics of another integer version refuses by
    name (the unit's sweep tried non-integer spellings only)."""

    @pytest.mark.parametrize("version", [PHYSICS_FORMAT_VERSION - 1, PHYSICS_FORMAT_VERSION + 1],
                             ids=["below", "above"])
    def test_it_refuses_by_name(self, version):
        payload = resolve_platform_physics(CHIP, None).to_dict()
        payload["format_version"] = version
        why = _refused(lambda: PlatformPhysics.from_dict(payload), ValueError)
        assert "format_version" in why and str(version) in why
