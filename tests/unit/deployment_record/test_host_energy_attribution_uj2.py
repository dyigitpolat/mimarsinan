"""[U-J2 K-1, K-2] The host's energy split by what the deployment causes, and the idle term's wait without a crash.

Owner answer of 2026-10-29, verbatim choice "Host's work only (Recommended)": the main energy charges the energy the
host spends because of the deployment, its active power above idle while it works ((p_host - p_host_idle) x host
time) plus every byte over the chip link and through host memory ((e_link_per_byte + e_host_mem_per_byte) x crossing
bytes, both directions); host time always adds to latency; the whole-system energy, with the host's idle power over
the inference's wait, is printed beside as a labelled system line, never in the main energy.

The composition is generic: a host description states, per energy row it declares (a power or an energy per byte),
whether the row's energy is attributable (the deployment's work) or system-level (the host's own draw), and every
row of its block carries that statement (``energy_plane``). A system-level power is a floor: the host draws it in
every state, so an attributable power is charged above it over its own time and the floor's energy over the whole
inference is system-level. The headline sums the attributable terms (``energy_host_mj``); the system line
(``energy_system_mj``) is the chip side plus the whole host (``energy_host_system_mj``: p_host x host time + p_host_idle
x wait + the link and host memory, at the whole host's corners, the former whole-host composition).

[targeted M1] The work's power is a difference of two bands, so its band ends are the cost ends of the difference: the
high end the power's high less the floor's low, the low end the power's low less the floor's high (never below zero:
the host draws at least its idle while it works), the nominal one setup's. Corner by corner (high less high) the
work's high end read 9.31 W on the shipped host instead of 15 - 4.0 = 11.0 W. At the nominal the system line is the
headline plus the floor under the work and the idle over the wait; at every end the headline lies at or below it.

K-2: the wait the idle power multiplies is the latency's parts the host does not execute (the chip, the link, the
exposed writes), summed directly. Formed as e2e - host time corner by corner, float rounding on a point or near-point
chip latency (neurram, isaac_like, the study chips) unordered the band and Band raised ValueError."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import vocabulary
from mimarsinan.deployment_record.platform_physics.hosts import description as described
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    HostDescriptionError, host_description, host_description_block,
)
from mimarsinan.deployment_record.platform_physics.hosts.store import HOST as HOST_STORE, store_block
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.schema.provenance import Band

NAME = "jetson_orin_nano_8gb"
HOSTS = Path(vocabulary.__file__).resolve().parent / "hosts"
CORNERS = ("low", "nominal", "high")
EDGE = resolve_platform_physics("generic_estimated_22nm", merge_override_blocks({"host": host_description_block(NAME)}))
#: [U-J3, amended] The edge host beside a host-owned store: the pass carry crosses the link only to one (U-J's model).
EDGE_HOST_STORE = resolve_platform_physics("generic_estimated_22nm", merge_override_blocks(
    {"host": host_description_block(NAME), "store": store_block(HOST_STORE, "U-J's model (test)")}))
BASE = dict(synaptic_events=1e6, cores_physical=8, latency_steps=32, timesteps=4, neurons_used=100, segment_cores=1)
NO_LINK = dict(carry_in_bytes=0, carry_out_bytes=0, segment_in_bytes=0, segment_out_bytes=0, host_link_transfers=0)
#: The rows whose energy the statement sorts: the host's powers and its energies per crossing byte.
ENERGY_ROWS = {"p_host": "attributable", "p_host_idle": "system", "e_link_per_byte": "attributable",
               "e_host_mem_per_byte": "attributable"}
#: CL0's frozen EX-D gen record's view (cost_identity_w1.FROZEN_RECORDS, sha256-pinned) as it prices on a chip with
#: the edge host's block: one host operator (200,704 MACs) beside 3 cores, no pass carry, no segment census.
EX_D = {"axons_physical": 7776.0, "axons_used": 640.0, "boundary_events": 602.0, "carried_raster_bytes": 0.0,
        "carry_in_bytes": 0.0, "carry_out_bytes": 0.0, "carry_peak_live_bytes": 0.0, "cells_physical": 1990656.0,
        "cells_used": 68096.0, "compute_op_count": 1.0, "connectivity_entries": 6.0, "cores_allocated": 3.0,
        "cores_physical": 9.0, "host_macs": 200704.0, "host_ops_s": 0.0048624980263412, "host_params": 200966.0,
        "latency_steps": 15.0, "macs": 68096.0, "neurons_physical": 2304.0, "neurons_used": 394.0,
        "noc_inter_tile_packets": 0.0, "noc_intra_tile_packets": 1877.0, "noc_total_hops": 0.0,
        "noc_total_packets": 1877.0, "onchip_macs": 68096.0, "onchip_params": 68890.0, "pass_count": 0.0,
        "reprogram_passes": 1.0, "reprogrammed_bytes": 42560.0, "reprogrammed_cores": 3.0, "segment_cores": 3.0,
        "synaptic_events": 29700.4, "sync_count": 0.0, "tiles": 3.0, "timesteps": 4.0, "total_macs": 268800.0,
        "total_params": 269856.0, "total_spikes": 106329.0, "weight_bits": 5.0}


def _q(**values):
    return Quantities({key: QuantityValue(float(value), "static") for key, value in values.items()})


def _priced(physics, **values):
    pricing = price_absolute(_q(**values), physics)
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


def _c(term, corner):
    return getattr(term.band, corner)


def _k(physics, key, corner):
    return getattr(physics.band(key), corner)


def _edge(profile):
    return resolve_platform_physics(profile, merge_override_blocks({"host": host_description_block(NAME)}))


def _work_power(physics):
    """[targeted M1] The cost ends of p_host - p_host_idle (W): low the power's low less the floor's high, never below
    zero; nominal one setup's; high the power's high less the floor's low."""
    return {"low": max(0.0, _k(physics, "p_host", "low") - _k(physics, "p_host_idle", "high")),
            "nominal": _k(physics, "p_host", "nominal") - _k(physics, "p_host_idle", "nominal"),
            "high": _k(physics, "p_host", "high") - _k(physics, "p_host_idle", "low")}


def _with_description(tmp_path, monkeypatch, edit):
    """The shipped description with ``edit`` applied to its JSON, served from ``tmp_path``."""
    data = json.loads((HOSTS / f"{NAME}.json").read_text())
    edit(data)
    (tmp_path / f"{NAME}.json").write_text(json.dumps(data))
    (tmp_path / data["description_file"]).write_text((HOSTS / data["description_file"]).read_text())
    monkeypatch.setattr(described, "_DIR", tmp_path)


class TestTheDescriptionStatesTheAttribution:
    def test_every_energy_row_is_stated_once_and_the_idle_draw_is_system_level(self):
        description = host_description(NAME)
        assert dict(description.energy_attribution) == ENERGY_ROWS
        assert "Host's work only" in description.energy_attribution_basis

    def test_every_energy_row_of_the_block_carries_its_plane_and_no_other_row_does(self):
        block = host_description_block(NAME)
        for key, row in block.items():
            assert row.get(described.ENERGY_PLANE_FIELD) == ENERGY_ROWS.get(key), key

    def test_the_resolved_physics_names_its_system_level_rows_and_keeps_them_through_a_record(self):
        assert EDGE.system_energy_rows == ("p_host_idle",)
        assert PlatformPhysics.from_dict(EDGE.to_dict()).system_energy_rows == ("p_host_idle",)

    def test_a_physics_without_a_stated_plane_serializes_as_before(self):
        chip = resolve_platform_physics("generic_estimated_22nm", None)
        assert chip.system_energy_rows == () and "system_energy_rows" not in chip.to_dict()

    def test_an_energy_row_without_its_statement_refuses_by_name(self, tmp_path, monkeypatch):
        _with_description(tmp_path, monkeypatch, lambda d: d["energy_attribution"]["rows"].pop("e_link_per_byte"))
        with pytest.raises(HostDescriptionError, match="e_link_per_byte"):
            host_description(NAME)

    @pytest.mark.parametrize("row, plane", [("t_link_per_byte", "attributable"), ("no_such_row", "system"),
                                            ("p_host", "shared")])
    def test_a_statement_of_no_energy_row_or_of_no_plane_refuses_by_name(self, tmp_path, monkeypatch, row, plane):
        _with_description(tmp_path, monkeypatch, lambda d: d["energy_attribution"]["rows"].__setitem__(row, plane))
        with pytest.raises(HostDescriptionError, match=row if row != "p_host" else plane):
            host_description(NAME)

    def test_a_floor_above_the_power_it_lies_under_refuses_by_name(self, tmp_path, monkeypatch):
        def edit(data):
            data["modes"]["15W"]["constants"]["p_host_idle"]["high"] = 16.0  # above p_host's 15 W at the high corner
        _with_description(tmp_path, monkeypatch, edit)
        with pytest.raises(HostDescriptionError, match="p_host.*p_host_idle.*high"):
            host_description_block(NAME)

    def test_an_override_stating_a_plane_on_a_row_that_is_not_a_host_energy_row_refuses_by_name(self):
        with pytest.raises(ValueError, match="t_cycle.*energy_plane"):
            resolve_platform_physics("generic_estimated_22nm", {"t_cycle": {
                "nominal": 1.0, "unit": "us", "evidence_kind": "estimated", "note": "n", described.ENERGY_PLANE_FIELD: "system"}})

    def test_an_override_stating_an_unknown_plane_refuses_by_name(self):
        block = host_description_block(NAME)
        block["p_host_idle"] = {**block["p_host_idle"], described.ENERGY_PLANE_FIELD: "shared"}
        with pytest.raises(ValueError, match="shared"):
            resolve_platform_physics("generic_estimated_22nm", block)


class TestTheHeadlineChargesTheHostsWorkOnly:
    """One MAC-free host operator (a pool): its dispatch and its bytes at the host memory time, no crossing byte."""

    VALUES = dict(BASE, host_macs=0, compute_op_count=1, host_memory_bound_bytes=6656, **NO_LINK)

    def test_the_work_is_active_power_above_idle_over_the_hosts_time_at_the_cost_ends_of_the_difference(self):
        """[targeted M1] The work's band ends are the cost ends of (p_host - p_host_idle), each x the host time's end."""
        terms, _ = _priced(EDGE, **self.VALUES)
        power = _work_power(EDGE)
        for corner in CORNERS:
            busy = _c(terms["latency_host_s"], corner)
            work = power[corner] * busy * 1e3
            assert _c(terms["energy_host_work_mj"], corner) == pytest.approx(work, rel=1e-12, abs=1e-18), corner
            assert _c(terms["energy_host_mj"], corner) == pytest.approx(work, rel=1e-12, abs=1e-18), corner

    def test_the_headline_is_the_chip_plus_the_attributable_host_and_no_idle(self):
        terms, _ = _priced(EDGE, **self.VALUES)
        for corner in CORNERS:
            chip = _c(terms["energy_dynamic_mj"], corner) + _c(terms["energy_static_mj"], corner)
            assert _c(terms["energy_per_inference_mj"], corner) == pytest.approx(
                chip + _c(terms["energy_host_work_mj"], corner), rel=1e-12), corner
        source = terms["energy_per_inference_mj"].source
        assert "x (e2e_latency_s - host time)" not in source and "idle floor under" not in source

    def test_the_idle_draw_is_system_level_under_the_work_and_over_the_wait(self):
        """[targeted M1] The floor and the idle stay at the whole host's corners; the system-level host is the whole
        host there (p_host x host time + p_host_idle x wait), no longer the floor and the idle alone."""
        terms, _ = _priced(EDGE, **self.VALUES)
        for corner in CORNERS:
            busy, e2e = _c(terms["latency_host_s"], corner), _c(terms["e2e_latency_s"], corner)
            idle = _k(EDGE, "p_host_idle", corner)
            assert _c(terms["energy_host_floor_mj"], corner) == pytest.approx(idle * busy * 1e3, rel=1e-12)
            assert _c(terms["energy_host_idle_mj"], corner) == pytest.approx(idle * (e2e - busy) * 1e3, rel=1e-9)
            whole_host = (_k(EDGE, "p_host", corner) * busy + idle * (e2e - busy)) * 1e3
            assert _c(terms["energy_host_system_mj"], corner) == pytest.approx(whole_host, rel=1e-9), corner

    def test_the_system_line_is_the_chip_side_plus_the_whole_host_at_its_corners(self):
        """[targeted M1] At every end the system line is the chip plus the whole host at the host's corners (the
        former whole-host composition); at the nominal, one setup, it is the headline plus the floor and the idle."""
        terms, _ = _priced(EDGE, **self.VALUES)
        for corner in CORNERS:
            busy, e2e = _c(terms["latency_host_s"], corner), _c(terms["e2e_latency_s"], corner)
            chip = _c(terms["energy_dynamic_mj"], corner) + _c(terms["energy_static_mj"], corner)
            assert _c(terms["energy_system_mj"], corner) == pytest.approx(
                chip + _c(terms["energy_host_system_mj"], corner), rel=1e-12), corner
            whole_host = (_k(EDGE, "p_host", corner) * busy + _k(EDGE, "p_host_idle", corner) * (e2e - busy)) * 1e3
            assert _c(terms["energy_system_mj"], corner) == pytest.approx(chip + whole_host, rel=1e-9), corner
        assert _c(terms["energy_system_mj"], "nominal") == pytest.approx(
            sum(_c(terms[name], "nominal") for name in ("energy_per_inference_mj", "energy_host_floor_mj",
                                                        "energy_host_idle_mj")), rel=1e-12)
        source = terms["energy_system_mj"].source
        assert "system line" in source and "not the headline" in source and "p_host_idle" in source

    def test_the_headline_lies_at_or_below_the_system_line_at_every_end(self):
        terms, _ = _priced(EDGE, **self.VALUES)
        for corner in CORNERS:
            assert _c(terms["energy_per_inference_mj"], corner) <= _c(terms["energy_system_mj"], corner), corner
            assert _c(terms["energy_host_mj"], corner) <= _c(terms["energy_host_system_mj"], corner), corner

    def test_host_time_adds_to_the_latency_whatever_the_attribution(self, tmp_path, monkeypatch):
        attributed, _ = _priced(EDGE, **self.VALUES)
        _with_description(tmp_path, monkeypatch,
                          lambda d: d["energy_attribution"]["rows"].__setitem__("p_host_idle", "attributable"))
        whole = resolve_platform_physics("generic_estimated_22nm",
                                         merge_override_blocks({"host": host_description_block(NAME)}))
        terms, _ = _priced(whole, **self.VALUES)
        assert terms["e2e_latency_s"].band == attributed["e2e_latency_s"].band
        assert terms["latency_host_s"].band == attributed["latency_host_s"].band


class TestTheCrossingBytesBrokenOut:
    """[U-J3, amended] The pass carry crosses the link to a host-owned store (``EDGE_HOST_STORE``)."""

    VALUES = dict(BASE, host_macs=0, compute_op_count=0, host_memory_bound_bytes=0,
                  **dict(NO_LINK, carry_in_bytes=300, carry_out_bytes=200, segment_in_bytes=50, segment_out_bytes=10,
                         host_link_transfers=4))

    def test_the_link_and_host_memory_are_each_their_own_attributable_term(self):
        terms, _ = _priced(EDGE_HOST_STORE, **self.VALUES)
        for corner in CORNERS:
            link = 560 * _k(EDGE_HOST_STORE, "e_link_per_byte", corner) * 1e3
            memory = 560 * _k(EDGE_HOST_STORE, "e_host_mem_per_byte", corner) * 1e3
            assert _c(terms["energy_host_link_mj"], corner) == pytest.approx(link, rel=1e-12)
            assert _c(terms["energy_host_mem_mj"], corner) == pytest.approx(memory, rel=1e-12)
            assert _c(terms["energy_host_mj"], corner) == pytest.approx(link + memory, rel=1e-12)
        assert "energy_host_work_mj" not in terms and "energy_host_floor_mj" not in terms  # no host time

    def test_the_host_idles_over_the_whole_link_bound_inference_in_the_system_line_only(self):
        terms, _ = _priced(EDGE_HOST_STORE, **self.VALUES)
        for corner in CORNERS:
            e2e = _c(terms["e2e_latency_s"], corner)
            assert _c(terms["energy_host_idle_mj"], corner) == pytest.approx(
                _k(EDGE_HOST_STORE, "p_host_idle", corner) * e2e * 1e3, rel=1e-12)
            assert _c(terms["energy_system_mj"], corner) == pytest.approx(
                _c(terms["energy_per_inference_mj"], corner) + _c(terms["energy_host_idle_mj"], corner), rel=1e-12)


class TestTheStatementDecides:
    def test_a_host_stating_its_idle_draw_attributable_charges_it_in_the_headline_and_prints_no_system_line(
            self, tmp_path, monkeypatch):
        _with_description(tmp_path, monkeypatch,
                          lambda d: d["energy_attribution"]["rows"].__setitem__("p_host_idle", "attributable"))
        whole = resolve_platform_physics("generic_estimated_22nm",
                                         merge_override_blocks({"host": host_description_block(NAME)}))
        assert whole.system_energy_rows == ()
        terms, refused = _priced(whole, **TestTheHeadlineChargesTheHostsWorkOnly.VALUES)
        for corner in CORNERS:
            busy, e2e = _c(terms["latency_host_s"], corner), _c(terms["e2e_latency_s"], corner)
            expected = (_k(whole, "p_host", corner) * busy + _k(whole, "p_host_idle", corner) * (e2e - busy)) * 1e3
            assert _c(terms["energy_host_mj"], corner) == pytest.approx(expected, rel=1e-9), corner
        assert not {"energy_system_mj", "energy_host_system_mj", "energy_host_floor_mj"} & set(terms)
        assert "energy_system_mj" not in refused

    def test_a_server_block_states_no_plane_and_prices_its_work_as_before(self):
        server = {"t_host_op_overhead": {"low": 13.37, "nominal": 13.48, "high": 13.64, "unit": "us",
                                         "evidence_kind": "measured", "note": "a server's dispatch wall (test)"},
                  "host_macs_per_s": {"nominal": 22.0, "unit": "G/s", "evidence_kind": "measured", "note": "n"},
                  "p_host": {"low": 4.98, "nominal": 37.34, "high": 280.0, "unit": "W", "evidence_kind": "published",
                             "citation": "a band over two hosts (test)"}}
        physics = resolve_platform_physics("study_compass", merge_override_blocks({"host": server}))
        terms, refused = _priced(physics, **dict(BASE, host_macs=1_000_000, compute_op_count=3))
        for corner in CORNERS:
            assert _c(terms["energy_host_mj"], corner) == pytest.approx(
                _k(physics, "p_host", corner) * _c(terms["latency_host_s"], corner) * 1e3, rel=1e-12)
        new = {"energy_host_work_mj", "energy_host_floor_mj", "energy_host_mem_mj", "energy_host_system_mj",
               "energy_system_mj"}
        assert not new & set(terms) and not new & set(refused)

    def test_without_a_priced_latency_the_system_line_refuses_by_name_and_the_headline_prices(self):
        """No compute time (the operator declares no t_cycle): the wait cannot be priced, so the idle draw it
        multiplies cannot either; the headline (dynamic and the host's work) prices without it."""
        block = merge_override_blocks({"host": host_description_block(NAME)})
        block["e_mac"] = {"nominal": 1.0, "unit": "pJ", "evidence_kind": "estimated", "note": "a test MAC"}
        physics = resolve_platform_physics(None, block)
        terms, refused = _priced(physics, **dict(BASE, host_macs=0, compute_op_count=1, host_memory_bound_bytes=0,
                                                 **NO_LINK))
        assert "e2e_latency_s" in refused and "energy_per_inference_mj" in terms
        assert "energy_host_work_mj" in terms and "energy_system_mj" not in terms
        assert "p_host_idle" in refused["energy_system_mj"] and "latency" in refused["energy_system_mj"]


class TestTheWaitIsTheChipSideLatency:
    """[K-2] CL0's frozen EX-D gen record on point- and near-point-band chips: it raised ValueError (the wait's
    corners unordered by float rounding) and now prices, its refusals by name unchanged."""

    @pytest.mark.parametrize("profile", ["neurram", "isaac_like", "study_compass", "study_vr_like", "study_isaac_like"])
    def test_the_ex_d_view_prices_with_a_monotone_wait(self, profile):
        physics = _edge(profile)
        terms, refused = _priced(physics, **EX_D)
        idle = terms["energy_host_idle_mj"].band
        assert idle.low <= idle.nominal <= idle.high
        for corner in CORNERS:
            wait = _c(terms["e2e_latency_s"], corner) - _c(terms["latency_host_s"], corner)
            assert _c(terms["energy_host_idle_mj"], corner) == pytest.approx(
                _k(physics, "p_host_idle", corner) * wait * 1e3, rel=1e-6, abs=1e-15), corner

    #: The headlines the host joins: its energy, the latency and its inverse (the chip alone prices no host).
    HOST_JOINS = ("energy", "e2e_latency_s", "throughput_inferences_s")

    @pytest.mark.parametrize("profile", ["neurram", "isaac_like"])
    def test_the_point_band_chip_keeps_its_refusals_by_name(self, profile):
        _terms, refused = _priced(_edge(profile), **EX_D)
        _terms_chip, refused_chip = _priced(resolve_platform_physics(profile, None), **EX_D)
        assert set(refused) <= set(refused_chip)  # the edge host adds no refusal
        for name, reason in refused_chip.items():
            if not name.startswith(self.HOST_JOINS):
                assert refused.get(name) == reason, name

    @pytest.mark.parametrize("profile", ["neurram", "isaac_like", "study_compass"])
    @pytest.mark.parametrize("host_macs", [1.0, 977.0, 200704.0, 3.3e7])
    def test_no_host_share_unorders_the_wait_on_a_point_band_chip(self, profile, host_macs):
        terms, _ = _priced(_edge(profile), **dict(EX_D, host_macs=host_macs))
        for name in ("energy_host_idle_mj", "energy_host_system_mj", "energy_system_mj", "energy_per_inference_mj"):
            band = terms[name].band
            assert band.low <= band.nominal <= band.high, name


class TestTheWorksBandEndsAreTheCostEndsOfTheDifference:
    """[targeted M1] A power above a floor is a difference of two bands: its cost-high end pairs the power's high with
    the floor's low, its cost-low end the power's low with the floor's high (never below zero), its nominal is one
    setup's. The description's own rule (an attributable power at or above its floor at every corner) still refuses
    by name."""

    def test_the_difference_takes_its_cost_ends(self):
        power, floor = Band(6.0, 10.0, 15.0, "p"), Band(4.0, 5.0, 5.69, "f")
        band, why = described.power_above(power, floor, "p - f")
        assert why is None
        assert (band.low, band.nominal, band.high) == pytest.approx((6.0 - 5.69, 5.0, 11.0), rel=1e-12)
        assert band.basis == "p - f"

    def test_the_low_end_never_falls_below_zero(self):
        band, why = described.power_above(Band(4.0, 11.01, 15.0, "p"), Band(4.0, 5.685, 5.69, "f"), "p - f")
        assert why is None and band.low == 0.0
        assert (band.nominal, band.high) == pytest.approx((11.01 - 5.685, 15.0 - 4.0), rel=1e-12)

    def test_point_bands_give_a_point(self):
        band, why = described.power_above(Band(9.0, 9.0, 9.0, "p"), Band(4.0, 4.0, 4.0, "f"), "p - f")
        assert why is None and (band.low, band.nominal, band.high) == (5.0, 5.0, 5.0)

    def test_a_power_below_its_floor_at_a_corner_still_refuses_by_name(self):
        band, why = described.power_above(Band(4.0, 5.0, 15.0, "p"), Band(4.0, 5.5, 5.69, "f"), "p - f")
        assert band is None and "below the floor at the nominal corner" in why

    def test_the_shipped_host_charges_its_work_up_to_its_budget_less_its_lowest_idle(self):
        """The shipped host's work power: 0 at the low end (its low equals the floor's low, below the floor's high),
        the preprint's one-setup difference at the nominal, the vendor's budget less the peer-reviewed idle at the
        high end; each read from the block, not typed."""
        block = host_description_block(NAME)
        p, f = block["p_host"], block["p_host_idle"]
        power = _work_power(EDGE)
        assert power["high"] == pytest.approx(p["high"] - f["low"], rel=1e-12)
        assert power["low"] == max(0.0, p["low"] - f["high"]) == 0.0
        assert power["high"] > p["high"] - f["high"]  # the corner-by-corner pairing understated it

    #: Host MACs and crossing bytes on the generic chip, and CL0's EX-D view on a point-band and a study chip.
    VIEWS = {"generic_estimated_22nm": dict(BASE, host_macs=200704, compute_op_count=2, host_memory_bound_bytes=0,
                                            **dict(NO_LINK, carry_in_bytes=300, carry_out_bytes=200,
                                                   host_link_transfers=2)),
             "neurram": EX_D, "study_compass": EX_D}

    @pytest.mark.parametrize("profile", sorted(VIEWS))
    def test_with_host_macs_the_work_and_the_system_line_keep_their_ends(self, profile):
        # [U-J3, amended] the generic view's pass carry crosses the link to a host-owned store
        physics = EDGE_HOST_STORE if profile == "generic_estimated_22nm" else _edge(profile)
        terms, _ = _priced(physics, **self.VIEWS[profile])
        power = _work_power(physics)
        for corner in CORNERS:
            busy = _c(terms["latency_host_s"], corner)
            assert _c(terms["energy_host_work_mj"], corner) == pytest.approx(power[corner] * busy * 1e3, rel=1e-12)
            chip = sum(_c(terms[name], corner) for name in ("energy_dynamic_mj", "energy_static_mj") if name in terms)
            host = _k(physics, "p_host", corner) * busy * 1e3 + sum(
                _c(terms[name], corner) for name in ("energy_host_idle_mj", "energy_host_link_mj", "energy_host_mem_mj")
                if name in terms)
            assert _c(terms["energy_system_mj"], corner) == pytest.approx(chip + host, rel=1e-12), corner
            assert _c(terms["energy_per_inference_mj"], corner) <= _c(terms["energy_system_mj"], corner)
