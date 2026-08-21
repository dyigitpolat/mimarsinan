"""The wizard renders the soma axes from the registry, and holds NO second table.

``firing_modes_by_spiking`` used to be a hardcoded literal duplicating
``legal_firing_modes``; a legality change would have gone stale in the browser
only. The dict must now BE the legality SSOT's answer, key for key.
"""

import pytest

from mimarsinan.chip_simulation.spiking_semantics import (
    ALL_SPIKING_MODES,
    legal_firing_modes,
)
from mimarsinan.config_schema.registry import REGISTRY, serialize_registry
from mimarsinan.config_schema.resolve import legal_values_view
from mimarsinan.gui.wizard.schema import get_wizard_defaults
from mimarsinan.gui.wizard.schema_api import resolve_payload

_SOMA_KEYS = ("firing_granularity", "membrane_arithmetic", "membrane_bits",
              "weight_sign_granularity")


class TestTheFiringModeTableIsTheLegalitySsot:
    def test_the_dict_equals_the_legality_derived_table(self):
        served = get_wizard_defaults()["firing_modes_by_spiking"]
        assert served == {
            mode: list(legal_firing_modes(mode)) for mode in served
        }

    def test_every_spiking_mode_is_covered(self):
        served = get_wizard_defaults()["firing_modes_by_spiking"]
        assert set(served) == set(ALL_SPIKING_MODES)

    def test_the_historical_answer_is_unchanged(self):
        served = get_wizard_defaults()["firing_modes_by_spiking"]
        assert served == {
            "lif": ["Default", "Novena"],
            "ttfs": ["TTFS"],
            "ttfs_quantized": ["TTFS"],
            "ttfs_cycle_based": ["TTFS"],
        }


class TestTheSomaKeysRenderGenerically:
    @pytest.mark.parametrize("key", _SOMA_KEYS)
    def test_the_key_serializes_into_the_wizard_schema(self, key):
        record = serialize_registry()["keys"][key]
        assert record["label"] and len(record["doc"]) >= 15
        assert record["group"] in ("spiking", "hardware")
        assert record["category"] == "advanced"

    @pytest.mark.parametrize("key", ("firing_granularity", "membrane_arithmetic"))
    def test_the_legal_set_drives_the_widget(self, key):
        """|legal| == 1 LOCKS the field; the enum options are the superset."""
        options = set(REGISTRY[key].resolved_options() or ())
        for cfg in ({"spiking_family": "lif", "spiking_variant": "streamed"},
                    {"spiking_family": "lif", "spiking_variant": "synchronized"},
                    {"spiking_family": "ttfs", "spiking_variant": "analytical"}):
            legal = legal_values_view(cfg)[key]
            assert legal and set(legal) <= options

    def test_the_platform_widths_ship_in_the_wizard_defaults(self):
        platform = get_wizard_defaults()["platform_constraints"]
        assert platform["membrane_bits"] == 0
        assert platform["weight_sign_granularity"] == "per_synapse"


class TestTheResolveEndpointAnswersInRowsNotExceptions:
    """``/api/config/resolve`` is a live keystroke channel: a half-typed core
    grid under a declared soma point must come back as keyed rows, never as a
    500 from the cross-key contract."""

    @pytest.mark.parametrize("cores", ["nope", [{}], 7, [{"count": 2}],
                                       {"count": 2}, ["a"], []])
    def test_a_per_event_draft_over_a_malformed_grid_returns_keyed_rows(self, cores):
        payload = resolve_payload({
            "deployment_parameters": {"spiking_family": "lif",
                                      "spiking_variant": "streamed",
                                      "firing_granularity": "per_event"},
            "platform_constraints": {"cores": cores},
        })
        assert payload["ok"] is False
        assert payload["errors"]
        for row in payload["errors"]:
            assert row["rule_id"] and row["message"]

    def test_a_declared_bias_lane_surfaces_the_remediable_soma_row(self):
        payload = resolve_payload({
            "deployment_parameters": {"spiking_family": "lif",
                                      "spiking_variant": "streamed",
                                      "firing_granularity": "per_event"},
            "platform_constraints": {"cores": [{}]},
        })
        rows = [row for row in payload["errors"]
                if row["rule_id"] == "soma_law_contract"]
        assert [row["key"] for row in rows] == ["firing_granularity"]
