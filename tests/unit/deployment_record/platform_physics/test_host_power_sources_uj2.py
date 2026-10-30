"""[U-J2 K-3] The edge host's power rows and what each end rests on.

Owner rule of 2026-10-28: the host's constants come from its published documentation or peer-reviewed measurements.
U-J's review (M3) found both power rows on an arXiv preprint (cao2026vlaulap). The search of 2026-10-29
(``RA/paper_pdfs/fetch_20261028_host/SOURCES.md``, section U-J2) found:

- vendor: NVIDIA Jetson Linux r36.5, Power Management, the TDP budget table for "Jetson Orin Nano 8GB", mode 15W:
  "VDD_IN Average Power: 15W" (OC2 throttling) and "VDD_IN Instantaneous Power: 20W" (OC3). It bounds the module's
  average power while it works; it states no idle or active figure;
- peer-reviewed: tsuyuki2026efficient (ICASSP 2026; read in its arXiv version 2603.26145v1), Table 5, a Jetson Orin
  Nano "at 15 W power mode", power "directly at the power supply input" (a 5 kHz measurement board): idle 4.0 W,
  ResNet 19.9 W, MobileViT-XXS 14.0 W. The active readings are the board's under GPU inference, above the module's
  15 W budget, so they are not the module's active power;
- no vendor or peer-reviewed figure of the module's own active power in the 15 W mode was reachable (the module data
  sheet DS-11105-001 v1.7 timed out; the MDPI and IEEE Access full texts refuse this host).

Hence: ``p_host_idle``'s low end rests on the peer-reviewed idle (4.0 W); its nominal and high stay the preprint's
VDD_IN readings, class stated, because the attributable work is a difference of two readings (p_host - p_host_idle)
and only the preprint states both in one setup (module VDD_IN, the 15 W mode). ``p_host``'s high end is the vendor's
VDD_IN average-power budget (15 W); its nominal stays the preprint's (11.01 W); its low end is the idle floor's (an
operator drawing nothing beyond idle). [targeted M1] The work above idle keeps its one-setup nominal (5.325 W); its
band ends are the cost ends of the difference, each a bound across sources and never one setup: the high end the
vendor's budget less the peer-reviewed idle (15 - 4.0 = 11.0 W, where the corner-by-corner pairing read 9.31 W), the
low end the power's low less the floor's high, never below zero (0 W)."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.platform_physics.hosts.description import (
    host_description, host_description_block, power_above,
)
from mimarsinan.deployment_record.schema.provenance import Band

NAME = "jetson_orin_nano_8gb"
VENDOR_TDP = "VDD_IN Average Power: 15W"
#: tsuyuki2026efficient Table 5 (15 W power mode, power at the board's supply input): idle, ResNet, MobileViT-XXS.
PEER_IDLE_W, PEER_ACTIVE_W = 4.0, (19.9, 14.0)
#: cao2026vlaulap Table 4 (15 W mode, VDD_IN): active and idle, single and ensemble.
PREPRINT_ACTIVE_W, PREPRINT_IDLE_W = (11.01, 11.12), (5.69, 5.68)
BUDGET_W = 15.0


def _row(key):
    return host_description(NAME).modes["15W"]["constants"][key]


def test_the_idle_floors_low_end_is_the_peer_reviewed_reading():
    row = _row("p_host_idle")
    assert (row["low"], row["nominal"], row["high"]) == pytest.approx(
        (PEER_IDLE_W, sum(PREPRINT_IDLE_W) / 2, max(PREPRINT_IDLE_W)), rel=1e-9)
    assert row["evidence_kind"] == "published"
    assert "tsuyuki2026efficient" in row["citation"] and "cao2026vlaulap" in row["citation"]


def test_the_work_powers_high_end_is_the_vendors_average_power_budget():
    row = _row("p_host")
    assert (row["low"], row["nominal"], row["high"]) == pytest.approx(
        (PEER_IDLE_W, PREPRINT_ACTIVE_W[0], BUDGET_W), rel=1e-9)
    assert VENDOR_TDP in row["citation"] and "cao2026vlaulap" in row["citation"]


def test_the_work_above_the_floor_keeps_its_single_setup_nominal_and_takes_the_cost_ends():
    """[targeted M1] The nominal is the preprint's one-setup difference; the ends are the cost ends of the difference
    (the high end the vendor's budget less the peer-reviewed idle; the low end never below zero)."""
    block = host_description_block(NAME)
    ends = ("low", "nominal", "high")
    band, why = power_above(Band(*(block["p_host"][end] for end in ends), "p_host"),
                            Band(*(block["p_host_idle"][end] for end in ends), "p_host_idle"), "p_host - p_host_idle")
    assert why is None
    assert (band.low, band.nominal, band.high) == pytest.approx(
        (0.0, PREPRINT_ACTIVE_W[0] - sum(PREPRINT_IDLE_W) / 2, BUDGET_W - PEER_IDLE_W), abs=1e-9)


def test_each_power_rows_departure_says_what_each_end_rests_on():
    departures = host_description(NAME).departures
    for key in ("p_host", "p_host_idle"):
        text = departures[key]
        for word in ("tsuyuki2026efficient", "ICASSP 2026", "supply input", "cao2026vlaulap", "preprint",
                     VENDOR_TDP, "one setup", "cost ends"):
            assert word in text, (key, word)
    assert "19.9" in departures["p_host"] and "above the module's 15 W budget" in departures["p_host"]
