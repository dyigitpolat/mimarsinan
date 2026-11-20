"""[U-HOST3] m-5: the words a derived host row prints are tied to its declaration.

The printed basis of a row that declares ``derived_from`` states its declaration (its rows and factors) before the
row's citation, so a reader of a key's basis can tell a scaled host row from the description's own row by more than
the class word (the second verify's c1: a x20 rate printed only the description's citations). A derivation text that
contradicts the declaration (a vocabulary row that is no term of it, or a stated factor, a number beside x, × or *,
that no term declares) is refused by name at every entry (the verify's c2: a text stating 'x 1' beside a declared
factor 2 was admitted). W22's scaled rows (their text 'the resolved <row> (<class>) x <factor>') stay admitted with
the one added field. No description, mode, host or chip is named here.
"""

from __future__ import annotations

import copy

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics import available_profiles, get_platform_physics
from mimarsinan.deployment_record.platform_physics.hosts import admission as A
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    host_attach_block, host_description_block, host_descriptions,
)
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics

NAME = host_descriptions()[0]


def _chip():
    for name in available_profiles():
        try:
            resolve_platform_physics(name, merge_override_blocks({"host": host_description_block(NAME)}))
        except ToolchainRefusal:
            continue
        return name
    raise AssertionError("no registry profile takes the identified host block")


CHIP = _chip()


def _block():
    return copy.deepcopy(host_description_block(NAME))


def _resolve(rows):
    return resolve_platform_physics(CHIP, merge_override_blocks({"host": rows}))


def _refused(call):
    with pytest.raises(ToolchainRefusal) as why:
        call()
    return str(why.value)


def _rows():
    return sorted(_block())


def _w22_row(key, factor):
    """W22's scaled_from form (w22_fidelity.py:388-408 at 5cad1cb7) with the one added field."""
    base = _block()[key]
    row = {**base, "low": base["low"] * factor, "nominal": base["nominal"] * factor, "high": base["high"] * factor,
           "evidence_kind": "derived", "derivation": f"W22 robustness: the resolved {key} ({base['evidence_kind']}) "
                                                     f"x {factor:g}",
           "note": "W22 robustness scaling", "derived_from": {"of": {key: factor}}}
    return row


class TestThePrintedBasis:
    @pytest.mark.parametrize("factor", [20.0, 0.2])
    def test_a_derived_rows_basis_states_its_rows_and_factors(self, factor):
        """c1: a x20 rate's basis named only the description's citations."""
        key = _rows()[0]
        physics = _resolve({**_block(), **A.host_derivation_block(NAME, "default", {key: factor})})
        basis = physics.constants[key].basis
        assert f"{factor!r} x {key}" in basis and "derived_from" in basis, basis
        assert physics.constants[key].band.basis == basis

    def test_a_combined_derivations_basis_states_every_term(self):
        rows = [k for k in _rows() if k.startswith("t_")]
        key, other = rows[0], rows[1]
        rows_block = A.host_derivation_block(NAME, "default", {key: {"of": {key: 1.0, other: 2.0}}})
        basis = _resolve({**_block(), **rows_block}).constants[key].basis
        assert f"1.0 x {key}" in basis and f"2.0 x {other}" in basis and "the sum of" in basis

    def test_the_descriptions_own_row_prints_no_declaration(self):
        key = _rows()[0]
        assert "derived_from" not in _resolve(_block()).constants[key].basis

    def test_the_sealed_row_prints_the_same_basis(self):
        key = _rows()[0]
        physics = _resolve({**_block(), **A.host_derivation_block(NAME, "default", {key: 20.0})})
        again = PlatformPhysics.from_dict(physics.to_dict())
        assert again.constants[key].basis == physics.constants[key].basis


class TestAContradictingDerivationText:
    def test_a_text_stating_another_factor_refuses_by_name(self):
        """c2: 'identity ... x 1' beside a declared factor 2."""
        key = _rows()[0]
        row = {**A.host_derivation_block(NAME, "default", {key: 2.0})[key],
               "derivation": "identity: the description's own row x 1 (test)"}
        why = _refused(lambda: _resolve({**_block(), key: row}))
        assert key in why and "1" in why

    @pytest.mark.parametrize("text", ["the row x 3 (test)", "3 x the row (test)", "the row × 3 (test)",
                                      "the row *3 (test)"], ids=["x_after", "x_before", "times_sign", "star"])
    def test_each_written_form_of_a_factor_is_read(self, text):
        key = _rows()[0]
        row = {**A.host_derivation_block(NAME, "default", {key: 2.0})[key], "derivation": text}
        _refused(lambda: _resolve({**_block(), key: row}))

    def test_a_text_naming_a_row_that_is_no_term_refuses_by_name(self):
        key, other = _rows()[0], _rows()[1]
        row = {**A.host_derivation_block(NAME, "default", {key: 2.0})[key],
               "derivation": f"twice the description's {other} (test)"}
        assert other in _refused(lambda: _resolve({**_block(), key: row}))

    def test_a_sealed_text_edited_to_contradict_refuses_at_read_back(self):
        key = _rows()[0]
        payload = _resolve({**_block(), **A.host_derivation_block(NAME, "default", {key: 2.0})}).to_dict()
        payload["constants"][key]["derivation"] = "the description's own row x 1 (test)"
        assert key in _refused(lambda: PlatformPhysics.from_dict(payload))

    def test_a_text_without_numbers_or_rows_is_no_contradiction(self):
        key = _rows()[0]
        row = {**A.host_derivation_block(NAME, "default", {key: 2.0})[key], "derivation": "a stated scaling (test)"}
        assert _resolve({**_block(), key: row}).constants[key].derived_from

    def test_a_factor_written_to_fewer_digits_is_its_factor(self):
        key = _rows()[0]
        row = {**A.host_derivation_block(NAME, "default", {key: 1.0 / 3.0})[key],
               "derivation": f"the row x {1.0 / 3.0:g} (test)"}
        assert _resolve({**_block(), key: row}).constants[key].derived_from


class TestTheWritersTexts:
    @pytest.mark.parametrize("factor", [0.2, 5.0])
    def test_w22s_scaled_rows_stay_admitted_with_the_one_field(self, factor):
        for key in _rows():
            physics = _resolve({**_block(), key: _w22_row(key, factor)})
            assert physics.constants[key].derived_from, key

    def test_the_block_writers_text_is_admitted(self):
        for key in _rows():
            physics = _resolve({**_block(), **A.host_derivation_block(NAME, "default", {key: 0.5})})
            assert physics.constants[key].derived_from, key

    def test_the_attach_rows_text_states_its_declaration_and_is_admitted(self):
        (key, row), = host_attach_block(NAME).items()
        assert "1.0 x t_link_per_byte" in row["derivation"], row["derivation"]
        chip = next(p for p in available_profiles() if get_platform_physics(p).backing_store is not None)
        blocks = {"host": _block(), "host_attach": host_attach_block(NAME),
                  "store": ST.store_block(ST.HOST, "a variant (test)")}
        assert resolve_platform_physics(chip, merge_override_blocks(blocks)).has(key)
