"""[U-E3 Z-3] The ladder's declaration keys and their one check live with its refusals (``scaling_ladder_types``).

A priced platform may differ from its scheduled program only in the row keys and the physics (``PRICE_KEYS``), a
row's platform from its priced program only in the row keys (``ROW_KEYS``); anything else refuses by name
(``LadderDeclarationError``). The keys and the check moved from ``scaling_ladder`` beside the refusal they raise
(the module budget's 300 lines, with U-C2 and U-E combined); the ladder re-exports the keys and reads the one check,
and both refusals print what they printed before."""

from __future__ import annotations

import pytest

import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
import mimarsinan.deployment_record.introspection.scaling_ladder_types as types

PLATFORM = {"cores": [{"max_axons": 8, "max_neurons": 8, "count": 2}], "weight_bits": 8, "inference_batch_K": 1,
            "inference_rate_hz": 30.0, "min_lifetime_hours": 43_800.0, "platform_physics_profile": "neurram"}
PRICED = ("a priced platform may differ from its scheduled program only in inference_batch_K, inference_rate_hz, "
          "min_lifetime_hours, platform_physics_profile, platform_physics_overrides, platform_physics_resolved")
ROW = "a row's platform may differ from its priced program only in inference_batch_K, inference_rate_hz, " \
      "min_lifetime_hours"


class TestTheKeysAndTheirCheck:
    def test_the_keys_live_with_the_refusal_and_the_ladder_re_exports_them(self):
        assert types.ROW_KEYS == ("inference_batch_K", "inference_rate_hz", "min_lifetime_hours")
        assert types.PRICE_KEYS == types.ROW_KEYS + (
            "platform_physics_profile", "platform_physics_overrides", "platform_physics_resolved")
        assert ladder.ROW_KEYS is types.ROW_KEYS and ladder.PRICE_KEYS is types.PRICE_KEYS

    def test_the_check_admits_a_change_of_its_keys_and_refuses_any_other_by_name(self):
        changed = {**PLATFORM, "inference_batch_K": 64, "platform_physics_profile": "odin"}
        types.require_declared_change(changed, PLATFORM, types.PRICE_KEYS, "a priced platform")
        with pytest.raises(types.LadderDeclarationError, match=r"^a row's platform only in inference_batch_K, "):
            types.require_declared_change(changed, PLATFORM, types.ROW_KEYS, "a row's platform")


class TestTheRefusalsPrintAsBefore:
    """Pins of the two call sites (no behavior change): the same refusal class and message."""

    def test_a_priced_platform_that_changes_its_cores_refuses_by_name(self):
        scheduled = ladder.ScheduledProgram(platform=PLATFORM, packed_tiles=2)
        moved = {**PLATFORM, "cores": [{"max_axons": 8, "max_neurons": 8, "count": 3}]}
        with pytest.raises(ladder.LadderDeclarationError) as refused:
            ladder.price_scheduled(None, None, scheduled, moved)
        assert str(refused.value) == PRICED
        assert ladder.price_scheduled(None, None, ladder.ScheduledProgram(
            platform=PLATFORM, packed_tiles=2, refusal={"name": "x", "reason": "y"}),
            {**PLATFORM, "inference_batch_K": 16}).refusal == {"name": "x", "reason": "y"}

    def test_a_row_that_changes_its_physics_refuses_by_name(self):
        priced = ladder.PricedProgram(platform=PLATFORM, packed_tiles=2)
        with pytest.raises(ladder.LadderDeclarationError) as refused:
            ladder.ladder_row(None, None, priced, {**PLATFORM, "platform_physics_profile": "odin"}, None)
        assert str(refused.value) == ROW
