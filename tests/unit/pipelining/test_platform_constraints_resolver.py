"""Resolved platform constraints must carry the scheduling keys end-to-end."""

from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)


def test_scheduling_keys_forward():
    pcfg = build_platform_constraints_resolved({
        "allow_scheduling": True,
        "schedule_policy": "bank_clustered", "max_schedule_passes": 128,
    })
    assert pcfg["allow_scheduling"] is True
    assert pcfg["schedule_policy"] == "bank_clustered"
    assert pcfg["max_schedule_passes"] == 128


def test_scheduling_defaults_are_the_historical_build_defaults():
    # Absent keys resolve to the builder's historical defaults — a dropped
    # key here silently disarms the declared schedule (t0_44 measured).
    pcfg = build_platform_constraints_resolved({})
    assert pcfg["allow_scheduling"] is False
    assert pcfg["schedule_policy"] == "pool"
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
    pcfg = build_platform_constraints_resolved({
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
