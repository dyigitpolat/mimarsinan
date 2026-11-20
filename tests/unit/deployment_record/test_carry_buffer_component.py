"""[U-F4 targeted pass, M2] A declared on-chip carry buffer is a chip component, priced on the capacity it declares.

Beside its access energy (``e_carry_buffer_per_byte`` x the bytes it holds), the buffer leaks while the chip is
powered and occupies silicon whether or not a program fills it, so both are priced on the declared capacity
(``carry_buffer_bytes``, the quantity ``carry_buffer_capacity_bytes`` on both planes): its static power
``p_carry_buffer_static_per_byte`` x the capacity over the priced latency joins the static energy, and its area
``area_carry_buffer_per_byte`` x the capacity joins the chip area on either area path. A target declaring the buffer
without either constant refuses that headline by name, as it refuses the energy without the access constant. A chip
that declares no buffer prices exactly what it priced before."""

from __future__ import annotations

import math

import pytest

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
from mimarsinan.deployment_record.platform_physics.constants import spec_for
from mimarsinan.deployment_record.quantities.from_candidate import from_candidate
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.quantities.spec import QUANTITY_SPECS, Quantities, QuantityValue
from mimarsinan.deployment_record.units import AREA, POWER

from unit.deployment_record.g3_fixtures import constant, physics, record_with
from unit.deployment_record.printed_names import NOT_PRODUCED, named, without
from unit.deployment_record.record_fixtures import make_schedule

KEY, QUANTITY = "carry_buffer_bytes", "carry_buffer_capacity_bytes"
STATIC, AREA_CONSTANT = "p_carry_buffer_static_per_byte", "area_carry_buffer_per_byte"
ACCESS = "e_carry_buffer_per_byte"
CAPACITY = 1_000_000
#: 1,000 steps at the generic profile's 1,000 ns cycle: a 1 ms priced latency, no host work.
BASE = {"synaptic_events": 1e6, "latency_steps": 1000, "host_macs": 0, "compute_op_count": 0, "cores_physical": 4,
        "cells_physical": 4 * 65536, "carry_out_bytes": 16, "carry_in_bytes": 24, "carry_onchip_bytes": 64}
BUFFER = {STATIC: constant(2.0, unit="nW", note="test: a cited buffer leakage"),
          AREA_CONSTANT: constant(0.5, unit="um^2", note="test: a cited buffer cell"),
          ACCESS: constant(4.0, unit="pJ", note="test: a cited buffer access")}


def _q(**values: float) -> Quantities:
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _price(quantities: Quantities, **constants):
    pricing = price_absolute(quantities, physics("generic_estimated_22nm", **constants))
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


class TestTheDeclaration:
    def test_its_static_power_and_area_are_physics_constants_on_the_declared_capacity(self):
        assert (spec_for(STATIC).dimension, spec_for(STATIC).display_unit) == (POWER, "nW")
        assert (spec_for(AREA_CONSTANT).dimension, spec_for(AREA_CONSTANT).display_unit) == (AREA, "um^2")
        assert QUANTITY in spec_for(STATIC).multiplicand and "e2e_latency_s" in spec_for(STATIC).multiplicand
        assert spec_for(AREA_CONSTANT).multiplicand == QUANTITY and QUANTITY in QUANTITY_SPECS

    def test_both_planes_state_the_declared_capacity(self):
        context = candidate_context_from_platform({KEY: CAPACITY})
        candidate = from_candidate(layout=None, chip_param_capacity=None, total_params=None,
                                   host_side_segment_count=None, context=context)
        assert candidate.get(QUANTITY).value == CAPACITY
        assert from_record(record_with(make_schedule(), **{KEY: CAPACITY})).get(QUANTITY).value == CAPACITY
        assert not from_candidate(layout=None, chip_param_capacity=None, total_params=None,
                                  host_side_segment_count=None,
                                  context=candidate_context_from_platform({})).has(QUANTITY)
        assert not from_record(record_with(make_schedule())).has(QUANTITY)


class TestThePricer:
    def test_the_buffer_leaks_on_its_declared_capacity_over_the_priced_latency(self):
        terms, refusals = _price(_q(**BASE, **{QUANTITY: CAPACITY}), **BUFFER)
        plain, _ = _price(_q(**BASE), **BUFFER)
        static = terms["energy_static_mj"]
        assert math.isclose(static.value - plain["energy_static_mj"].value, 2.0e-9 * CAPACITY * 1e-3 * 1e3)
        assert STATIC in static.source and "energy_per_inference_mj" in terms, refusals

    def test_it_leaks_while_it_holds_no_carried_value(self):
        idle = {**BASE, "carry_onchip_bytes": 0}
        terms, _ = _price(_q(**idle, **{QUANTITY: CAPACITY}), **BUFFER)
        plain, _ = _price(_q(**idle), **BUFFER)
        assert terms["energy_static_mj"].value > plain["energy_static_mj"].value

    def test_the_buffer_adds_its_area_on_its_declared_capacity(self):
        terms, refusals = _price(_q(**BASE, **{QUANTITY: CAPACITY}), **BUFFER)
        plain, _ = _price(_q(**BASE), **BUFFER)
        area = terms["chip_area_mm2"]
        assert math.isclose(area.value - plain["chip_area_mm2"].value, 0.5e-12 * CAPACITY * 1e6), refusals
        assert AREA_CONSTANT in area.source

    def test_a_buffer_without_a_declared_static_power_refuses_the_energy_by_name(self):
        terms, refusals = _price(_q(**BASE, **{QUANTITY: CAPACITY}),
                                 **{k: v for k, v in BUFFER.items() if k != STATIC})
        assert "energy_per_inference_mj" not in terms and STATIC in refusals["energy_per_inference_mj"]
        assert f"{CAPACITY} B" in refusals["energy_per_inference_mj"]
        assert "energy_dynamic_mj" in terms and "chip_area_mm2" in terms

    def test_a_buffer_without_a_declared_area_refuses_the_area_by_name(self):
        terms, refusals = _price(_q(**BASE, **{QUANTITY: CAPACITY}),
                                 **{k: v for k, v in BUFFER.items() if k != AREA_CONSTANT})
        assert "chip_area_mm2" not in terms and AREA_CONSTANT in refusals["chip_area_mm2"]
        assert "energy_per_inference_mj" in terms

    @pytest.mark.parametrize("constants", [{}, BUFFER], ids=["undeclared", "declared"])
    def test_a_chip_without_a_buffer_prices_what_it_priced_before(self, constants):
        """[U-ENERGY class 2] The buffer's declared rows over a census the view does not produce are named beside the
        estimate with the plane's reason, never priced: the values are the same, the texts the same but for those
        names."""
        plain = {k: v for k, v in BASE.items() if k != "carry_onchip_bytes"}
        terms, refusals = _price(_q(**plain), **constants)
        base_terms, base_refusals = _price(_q(**plain))
        rows = tuple(constants)
        assert all(NOT_PRODUCED in n for t in terms.values() for r in rows for n in named(t.source, r))
        assert {n: (t.value, without(t.source, rows)) for n, t in terms.items()} == {
            n: (t.value, t.source) for n, t in base_terms.items()}
        assert refusals == base_refusals
