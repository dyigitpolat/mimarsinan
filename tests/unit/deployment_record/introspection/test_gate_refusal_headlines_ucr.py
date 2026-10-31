"""[U-CR C-2, SD-11 SD11-D2] A pricing gate refusal names every headline and axis it suppresses, and 'no physics
declared' is printed only where no profile exists.

The SRAM-CIM macro's profile states an external component (its backing store); a run that leaves the store undeclared
prices nothing and ``price_absolute`` refused it under the gate's own name (``absolute_pricing``) only. Every reader
that looks a headline's refusal up by the headline's name then found none: the scaling ladder's energy and time axes
printed 'no physics declared' with ``headline_refusals`` null on a declared profile (SD-11: 72 rows of
``p:resnet18_w1q:sram_cim_macro``, variant host), and the cross-platform comparison listed the macro's headlines as
neither priced nor unavailable. The gate now refuses each headline it suppresses by name, beside its own refusal, with
the gate's reason; a reader that still finds no named refusal on a declared profile prints the gap by name."""

from __future__ import annotations

import pytest

import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
import mimarsinan.deployment_record.introspection.scaling_ladder_law as law
from mimarsinan.deployment_record.cost.absolute import ABSOLUTE_TERM_NAMES, price_absolute
from mimarsinan.deployment_record.platform_physics import apply_overrides, get_platform_physics
from mimarsinan.deployment_record.study import compare_platforms
from unit.deployment_record.g3_fixtures import quantities
from unit.deployment_record.introspection.test_scaling_ladder import READING, _facts, _network, _platform

STORE = "sram_cim_macro is read only with its backing store declared: e_dma_per_byte, t_program_per_byte undeclared"
_STORE_ROWS = {"e_dma_per_byte": {"nominal": 100.0, "unit": "pJ", "evidence_kind": "published", "citation": "t p.1"},
               "t_program_per_byte": {"nominal": 0.1, "unit": "ns", "evidence_kind": "published", "citation": "t p.1"}}


def _macro(store: bool = False):
    macro = get_platform_physics("sram_cim_macro")
    return apply_overrides(macro, _STORE_ROWS) if store else macro


def _row(physics, *, k=1, **declared):
    """A ladder row of the two-conv test network on the test cores under ``physics`` (None: a geometry-only bundle)."""
    network = _network(4)
    platform = {**_platform(64), "platform_physics_resolved": None if physics is None else physics.to_dict(),
                **({} if k is None else {"inference_batch_K": k}), **declared}
    census = ladder.shape_census(network, platform)
    priced = ladder.price_program(network, census, platform, _facts)
    return ladder.ladder_row(network, census, priced, platform, READING)


class TestTheGate:
    def test_each_suppressed_headline_is_refused_by_name_with_the_gate_reason(self):
        pricing = price_absolute(quantities(synaptic_events=10, host_macs=0), _macro())
        refused = {r.name: r.reason for r in pricing.refusals}
        assert pricing.terms == ()
        assert refused["absolute_pricing"].startswith(STORE)
        for headline in ABSOLUTE_TERM_NAMES:
            assert refused[headline] == "absolute_pricing: " + refused["absolute_pricing"], headline

    def test_a_declared_store_prices_the_headlines_and_refuses_none_of_them_by_the_gate(self):
        pricing = price_absolute(quantities(synaptic_events=10, host_macs=0), _macro(store=True))
        assert not [r for r in pricing.refusals if r.reason.startswith("absolute_pricing")]
        assert "absolute_pricing" not in {r.name for r in pricing.refusals}

    def test_the_cross_platform_comparison_names_the_suppressed_headlines(self):
        table = compare_platforms(quantities(synaptic_events=10, host_macs=0), ["sram_cim_macro"])
        row, = table.rows
        assert row.values == {}
        assert set(row.unavailable) == set(ABSOLUTE_TERM_NAMES)
        assert all(STORE in reason for reason in row.unavailable.values())


class TestTheLadderRow:
    """SD-11's host-only macro row: the macro's profile declared, its backing store not."""

    @pytest.fixture(scope="class")
    def host_only(self):
        return _row(_macro(), k=1)

    def test_the_axes_name_the_gate_refusal_not_an_absent_profile(self, host_only):
        for axis in ("energy", "time"):
            absent = host_only[axis]["absent"]
            assert absent.startswith("absolute_pricing: " + STORE), (axis, absent)
            assert "no physics declared" not in absent
        assert host_only["axes_priced"] == []

    def test_the_headline_refusals_name_the_gate(self, host_only):
        assert host_only["headlines"] == {"e2e_latency_s": None, "energy_per_inference_mj": None}
        for headline, reason in host_only["headline_refusals"].items():
            assert reason is not None and reason.startswith("absolute_pricing: " + STORE), headline

    def test_the_write_law_keeps_naming_its_own_terms(self, host_only):
        refusals = host_only["write_law"]["refusals"]
        assert refusals["energy_write_mj"].startswith(STORE)

    def test_a_geometry_only_bundle_still_reads_no_physics_declared(self):
        row = _row(None, k=1)
        for axis in ("energy", "time"):
            assert row[axis]["absent"] == "no physics declared", axis
        assert row["headline_refusals"] == {"e2e_latency_s": None, "energy_per_inference_mj": None}
        assert row["write_law"] == {"absent": "no physics declared"}

    def test_a_declared_profile_without_a_batch_names_the_batch(self):
        row = _row(_macro(store=True), k=None)
        assert row["write_law"] == {"absent": "no inference_batch_K declared"}

    def test_a_declared_profile_with_no_named_refusal_names_the_gap(self):
        """A reader that finds neither the term nor a refusal by the headline's name on a declared profile names the
        unpriced term and the pricing's refusals, never 'no physics declared'."""
        terms = {"program": {}, "compute_only": {}, "no_host": {}, "once": {}, "period": {}}
        refusals = {"program": {}, "compute_only": {"some_constant": "unpriced by the test"}, "no_host": {},
                    "once": {}, "period": {}}
        names = ("energy_dynamic_mj", "energy_per_inference_mj", "energy_write_mj", "energy_host_mj")
        column = law.axis_columns(terms, refusals, names, k=1, reading=READING, multiplexed=False, x=1.0, passes=1,
                                  axis="energy")
        assert "no physics declared" not in column["absent"]
        assert "energy_dynamic_mj" in column["absent"] and "some_constant" in column["absent"]
