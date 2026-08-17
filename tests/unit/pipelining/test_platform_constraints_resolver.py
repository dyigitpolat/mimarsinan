"""Resolved platform constraints must carry the scheduling keys end-to-end."""

import pytest

from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)


def test_scheduling_keys_forward():
    pcfg = build_platform_constraints_resolved({
        "allow_scheduling": True,
        "max_schedule_passes": 128,
    })
    assert pcfg["allow_scheduling"] is True
    assert pcfg["max_schedule_passes"] == 128


def test_scheduling_defaults_are_the_historical_build_defaults():
    # Absent keys resolve to the builder's historical defaults — a dropped
    # key here silently disarms the declared schedule (t0_44 measured).
    pcfg = build_platform_constraints_resolved({})
    assert pcfg["allow_scheduling"] is False
    assert pcfg["max_schedule_passes"] == 8


def test_retired_weight_reuse_knob_never_resolves():
    # Weight reuse is always on (W1.1): the retired knob is NOT resolved even
    # when an old flat config still carries it — the config-schema layer serves
    # such documents the keyed retirement remedy instead.
    pcfg = build_platform_constraints_resolved({"allow_weight_reuse": True})
    assert "allow_weight_reuse" not in pcfg
    assert "allow_weight_reuse" not in build_platform_constraints_resolved({})


def test_sanafe_floorplan_keys_forward():
    # W1.2: the declared floorplan keys ride platform_constraints_resolved so
    # the SANA-FE step reads them from the same resolved surface as capacity.
    # The declared platform must be consistent with the grid (2x5 tiles x 4
    # cores = 40 slots): an undersized grid now fails loud at this seam.
    pcfg = build_platform_constraints_resolved({
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 40}],
        "cores_per_tile": 4, "tile_grid_rows": 2, "tile_grid_cols": 5,
    })
    assert pcfg["cores_per_tile"] == 4
    assert pcfg["tile_grid_rows"] == 2
    assert pcfg["tile_grid_cols"] == 5


def test_sanafe_floorplan_keys_default_to_derived():
    pcfg = build_platform_constraints_resolved({})
    assert pcfg["cores_per_tile"] == 0
    assert pcfg["tile_grid_rows"] == 0
    assert pcfg["tile_grid_cols"] == 0


def test_resolved_floorplan_is_concrete_for_a_derived_declaration():
    # The DERIVATION is visible in the resolved surface, not just the
    # declaration: default platform (1000 cores) on the default loihi preset
    # (4 cores/tile) -> 250 tiles -> most-square exact grid 10x25. Declared
    # keys stay verbatim (0 = derived) next to the concrete *_resolved values.
    pcfg = build_platform_constraints_resolved({})
    assert (pcfg["cores_per_tile"], pcfg["tile_grid_rows"],
            pcfg["tile_grid_cols"]) == (0, 0, 0)
    assert pcfg["cores_per_tile_resolved"] == 4
    assert (pcfg["tile_grid_rows_resolved"], pcfg["tile_grid_cols_resolved"]) \
        == (10, 25)


def test_resolved_floorplan_carries_an_explicit_declaration_verbatim():
    pcfg = build_platform_constraints_resolved({
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 40}],
        "cores_per_tile": 4, "tile_grid_rows": 4, "tile_grid_cols": 4,
    })
    # Declared keys verbatim AND the concrete resolution (here identical:
    # an explicit oversized 4x4 grid is a legitimate fixed physical chip).
    assert (pcfg["cores_per_tile"], pcfg["tile_grid_rows"],
            pcfg["tile_grid_cols"]) == (4, 4, 4)
    assert pcfg["cores_per_tile_resolved"] == 4
    assert pcfg["tile_grid_rows_resolved"] == 4
    assert pcfg["tile_grid_cols_resolved"] == 4


def test_resolved_floorplan_reads_the_declared_preset():
    # sanafe_arch_preset is a flat-config key and IS in scope at the resolver
    # seam: truenorth's physical wiring is 1 core/tile.
    pcfg = build_platform_constraints_resolved({
        "cores": [{"max_axons": 64, "max_neurons": 64, "count": 12}],
        "sanafe_arch_preset": "truenorth",
    })
    assert pcfg["cores_per_tile_resolved"] == 1
    assert (pcfg["tile_grid_rows_resolved"] * pcfg["tile_grid_cols_resolved"]
            ) == 12


def test_count_less_core_declarations_resolve_as_one_core_each():
    # Minimal core dicts without "count" (e.g. resolve_bias_mode queries)
    # follow the imc_platforms convention: a declared type exists once.
    pcfg = build_platform_constraints_resolved({
        "cores": [{"max_axons": 8, "max_neurons": 8},
                  {"max_axons": 8, "max_neurons": 8}],
    })
    # 2 cores on loihi's 4/tile wiring -> one tile.
    assert pcfg["cores_per_tile_resolved"] == 4
    assert (pcfg["tile_grid_rows_resolved"], pcfg["tile_grid_cols_resolved"]) \
        == (1, 1)


def test_undersized_explicit_grid_fails_loud_at_the_resolver_seam():
    # The capacity invariant fires at resolution time (model configuration),
    # not first at the SANA-FE step: 2x2 tiles x 4 cores = 16 < 40 declared.
    with pytest.raises(ValueError, match="capacity"):
        build_platform_constraints_resolved({
            "cores": [{"max_axons": 256, "max_neurons": 256, "count": 40}],
            "cores_per_tile": 4, "tile_grid_rows": 2, "tile_grid_cols": 2,
        })


def test_platform_physics_is_absent_by_default():
    # A run that declares no profile must carry NO physics, so the absolute
    # objectives report "none declared" instead of a framework default.
    assert build_platform_constraints_resolved({})["platform_physics_resolved"] is None


def test_platform_physics_resolves_onto_the_surface_the_record_carries():
    pcfg = build_platform_constraints_resolved({"platform_physics_profile": "truenorth"})
    physics = pcfg["platform_physics_resolved"]
    assert physics["name"] == "truenorth"
    assert physics["constants"]["t_cycle"]["nominal"] == 1000.0


def test_platform_physics_overrides_ride_the_resolved_surface():
    pcfg = build_platform_constraints_resolved({
        "platform_physics_profile": "truenorth",
        "platform_physics_overrides": {
            "t_cycle": {"nominal": 500.0, "unit": "us", "note": "overclocked"}
        },
    })
    physics = pcfg["platform_physics_resolved"]
    assert physics["constants"]["t_cycle"]["nominal"] == 500.0
    assert physics["constants"]["t_cycle"]["overridden"] is True


def test_overrides_alone_resolve_to_a_custom_target():
    pcfg = build_platform_constraints_resolved({
        "platform_physics_overrides": {
            "t_cycle": {"nominal": 1.0, "unit": "ms", "note": "vendor datasheet"}
        },
    })
    assert pcfg["platform_physics_resolved"]["name"] == "custom"


def test_an_unknown_physics_profile_fails_loud_at_resolution_time():
    # The floorplan discipline: an invalid declaration is refused HERE, not
    # discovered as a crash deep in the cost model.
    with pytest.raises(KeyError, match="nosuchchip"):
        build_platform_constraints_resolved({"platform_physics_profile": "nosuchchip"})


def test_the_resolved_physics_is_json_safe():
    import json

    pcfg = build_platform_constraints_resolved({"platform_physics_profile": "truenorth"})
    assert json.loads(json.dumps(pcfg["platform_physics_resolved"]))["name"] == "truenorth"
