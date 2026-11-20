"""[U-HOST3] m-1 as a class, "a refusal expression has a condition no test pins": one hand-made case per condition of the
admission rule and the registry (the wall's readers: ``test_host_wall_readers_uhost3``).

The unit's per-condition mutant harness (FU/uhost_pricing3/steps/condition_mutants.py) mutates each operand of each
boolean expression and each element of each closed set in ``hosts/admission.py``, ``hosts/derivation.py``,
``hosts/registry.py``, the description judge and the loader's entries; each mutant is killed by a case of the unit's own
test files. The second verify's surviving admission mutants come first: x01 (the tolerance loosened) and x02 (the
nominal corner only), x04 (a zero factor admitted), x14 (an unknown mode falls back to the default), x41 (a chip row's
derivation naming no host admitted) and x32 (a sealed declaration not judged again). No description, mode, host or chip
is named here.
"""

from __future__ import annotations

import copy

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics import available_profiles
from mimarsinan.deployment_record.platform_physics.hosts import admission as A
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    HOST_IDENTITY_FIELD, host_description, host_description_block, host_descriptions,
)
from mimarsinan.deployment_record.platform_physics.loader import apply_overrides
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics

NAME = host_descriptions()[0]
DEFAULT = host_description(NAME).default_mode


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


def _sealed_with(derivations):
    return _resolve({**_block(), **A.host_derivation_block(NAME, "default", derivations)}).to_dict()


# ------------------------------------------------------------------------------- the second verify's six
class TestTheVerifysAdmissionSurvivors:
    @pytest.mark.parametrize("corner", ["low", "nominal", "high"])
    def test_x01_x02_one_corner_off_by_a_part_in_a_billion_refuses(self, corner):
        """x01 (REL_TOL 1e-12 loosened to 1e-3) and x02 (the nominal corner compared alone) admitted it."""
        key = next(k for k, r in sorted(_block().items()) if r["low"] < r["nominal"] < r["high"])
        row = dict(_block()[key])
        row[corner] = row[corner] * (1 - 1e-9 if corner == "low" else 1 + 1e-9)  # the band stays ordered
        assert key in _refused(lambda: _resolve({**_block(), key: row}))

    def test_x04_a_zero_factor_with_its_zero_band_refuses(self):
        key = sorted(_block())[0]
        row = {**_block()[key], "low": 0.0, "nominal": 0.0, "high": 0.0, "evidence_kind": "derived",
               "derivation": "zero (test)", "derived_from": {"of": {key: 0.0}}}
        assert key in _refused(lambda: _resolve({**_block(), key: row}))

    @pytest.mark.parametrize("mode", ["no_such_mode", DEFAULT.lower() + "_", DEFAULT.upper() + "x"])
    def test_x14_an_unknown_mode_refuses_by_name(self, mode):
        key = sorted(_block())[0]
        row = {**_block()[key], HOST_IDENTITY_FIELD: f"{NAME} {mode}"}
        why = _refused(lambda: _resolve({key: row}))
        assert mode in why

    def test_x41_a_chip_row_declaring_a_derivation_but_no_host_refuses(self):
        physics = _resolve(_block())
        link = _block()["t_link_per_byte"]
        row = {**{k: link[k] for k in ("low", "nominal", "high", "unit")}, "evidence_kind": "derived",
               "derivation": "the host's link row (test)", "derived_from": {"of": {"t_link_per_byte": 1.0}}}
        assert "names no host" in _refused(lambda: apply_overrides(physics, {"t_program_per_byte": row}))

    def test_x32_a_sealed_zero_factor_refuses_at_read_back(self):
        key = sorted(k for k in _block() if k.startswith("t_"))[0]
        payload = _sealed_with({key: 1.0})
        payload["constants"][key].update(low=0.0, nominal=0.0, high=0.0)
        payload["constants"][key]["derived_from"] = {"combine": "sum", "of": {key: 0.0}}
        assert key in _refused(lambda: PlatformPhysics.from_dict(payload))

    def test_x32_a_sealed_slower_on_a_power_refuses_at_read_back(self):
        key = sorted(k for k in _block() if k.startswith("p_"))[0]
        payload = _sealed_with({key: 1.0})
        payload["constants"][key]["derived_from"] = {"combine": "slower", "of": {key: 1.0}}
        assert key in _refused(lambda: PlatformPhysics.from_dict(payload))

    def test_x32_a_sealed_declaration_across_dimensions_refuses_at_read_back(self):
        power = sorted(k for k in _block() if k.startswith("p_"))[0]
        energy = sorted(k for k in _block() if k.startswith("e_"))[0]
        payload = _sealed_with({power: 1.0})
        row, other = payload["constants"][power], _block()[energy]
        for end in ("low", "nominal", "high"):
            row[end] = row[end] + other[end]
        row["derived_from"] = {"combine": "sum", "of": {power: 1.0, energy: 1.0}}
        assert power in _refused(lambda: PlatformPhysics.from_dict(payload))
