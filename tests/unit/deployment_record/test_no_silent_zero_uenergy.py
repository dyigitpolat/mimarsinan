"""[U-ENERGY class 2] No declared constant prices zero unnamed.

Every factor of every component table is, on each plane, produced (a value, a known zero among them), withheld with
the view's reason, not produced on this plane with the plane's reason (named beside the estimate, never a refusal), or
zero by the platform's declaration (its reason printed in the basis). The ratchet reads views built by the planes
themselves (a candidate view through the ladder's own calls, a sealed record through ``from_record``), not a table of
keys. Every constant here is a test constant."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.formulas import (
    AREA_COMPONENTS, CHIP_AREA_ADDERS, ENERGY_COMPONENTS, LATENCY_COMPONENTS, DerivedFactor,
)
from mimarsinan.deployment_record.platform_physics import profile_from_dict
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

BASE = {"e_mac": 0.5, "t_cycle": 100.0, "e_neuron_update": 2.0}
TABLES = (ENERGY_COMPONENTS, LATENCY_COMPONENTS, AREA_COMPONENTS, CHIP_AREA_ADDERS)


def _physics(constants, name="silent"):
    return profile_from_dict({
        "format_version": 1, "name": name, "display_name": name, "description_file": "t.md",
        "validity": {"measurement_kind": "projection"},
        "constants": {k: {"nominal": v, "evidence_kind": "estimated", "note": "test"} for k, v in constants.items()},
    })


def _values(**extra):
    base = {"synaptic_events": 1e6, "cores_physical": 4.0, "latency_steps": 10.0, "host_macs": 0.0,
            "compute_op_count": 0.0, "timesteps": 1.0,
            "neuron_cycles_per_window": 1.0, **extra}  # [composed head, rule 12] U-LIF2 W12
    return {k: QuantityValue(float(v), "static") for k, v in base.items()}


def _source(pricing, name):
    return next(t.source for t in pricing.terms if t.name == name)


def _keys(factor):
    return set(factor.over + factor.per) if isinstance(factor, DerivedFactor) else {factor}


class TestTheFourStates:
    def test_a_declared_constant_over_a_census_the_plane_never_counts_is_named_with_the_reason(self):
        q = Quantities(_values(), unproduced={"neurons_used": "the plane counts no outputs"})
        pricing = price_absolute(q, _physics(BASE))
        assert "e_neuron_update(neuron_update; not produced on this plane: the plane counts no outputs)" in \
            _source(pricing, "energy_dynamic_mj")

    def test_a_declared_constant_over_an_absent_census_is_named_even_without_a_reason(self):
        pricing = price_absolute(Quantities(_values()), _physics(BASE))
        assert "e_neuron_update(neuron_update; not produced on this plane" in _source(pricing, "energy_dynamic_mj")

    def test_the_headline_still_prints(self):
        q = Quantities(_values(), unproduced={"neurons_used": "the plane counts no outputs"})
        pricing = price_absolute(q, _physics(BASE))
        assert "energy_per_inference_mj" in {t.name for t in pricing.terms}
        assert "energy_per_inference_mj" not in {r.name for r in pricing.refusals}

    def test_a_zero_by_declaration_prints_its_reason_in_the_basis(self):
        q = Quantities(_values(neurons_used=10.0), known_zero={"lane_ops": "the chip declares no lane block"})
        pricing = price_absolute(q, _physics({**BASE, "e_lane_op": 1.0}))
        source = _source(pricing, "energy_dynamic_mj")
        assert "zero by declaration: e_lane_op (the chip declares no lane block)" in source
        assert "e_lane_op(lane" not in source  # never named unpriced, never priced

    def test_a_dataflow_without_converters_prints_its_converter_zero(self):
        pricing = price_absolute(Quantities(_values(neurons_used=10.0)), _physics({**BASE, "e_adc_conversion": 1.0}))
        assert ("zero by declaration: e_adc_conversion (the declared dataflow (digital) has no converter)"
                in _source(pricing, "energy_dynamic_mj"))

    def test_a_produced_census_prices_as_before(self):
        pricing = price_absolute(Quantities(_values(neurons_used=100.0)), _physics(BASE))
        dyn = next(t for t in pricing.terms if t.name == "energy_dynamic_mj")
        assert dyn.value == pytest.approx((0.5e-12 * 1e6 + 2e-12 * 100.0) * 1e3, rel=1e-12)


class TestTheCountsThePlanesProduce:
    def test_a_value_domain_program_sends_no_spike_packets(self):
        from mimarsinan.deployment_record.quantities.value_domain import ValueDomainFacts, apply_value_rule
        values = {}
        apply_value_rule(values, ValueDomainFacts(mac_events=10, latency_steps=2, carry_out_bytes=0,
                                                  carry_in_bytes=0))
        assert values["noc_intra_tile_packets"].value == 0.0 and values["noc_total_hops"].value == 0.0

    def test_the_candidate_census_counts_outputs_and_the_rows_read(self):
        from mimarsinan.deployment_record.build.payload_sizes import candidate_programming_census
        from mimarsinan.mapping.noc.pass_program import PassProgram

        class _Noc:
            pass_programs = (
                PassProgram(core_cells=(12 * 5, 8 * 3), resident=True, core_used=((12, 5), (8, 3))),
                PassProgram(core_cells=(10 * 4,), resident=False, core_used=((10, 4),)),
            )
        census = candidate_programming_census(_Noc(), weight_bits=8)
        assert census.output_neurons == 5 + 3 + 4
        assert census.local_mem_read_elements == 12 + 8 + 10
        assert census.reprogrammed_bytes == 40


def _every_row_physics():
    """A description that declares every row of every component table (test constants)."""
    rows = {constant: 1.0 for table in TABLES for constant, _factors, _label in table}
    return _physics({**rows, "e_mac": 0.5, "t_cycle": 100.0}, name="every_row")


def _check_view(view, plane_zeros):
    """The ratchet on one plane's view: (1) only the platform's declared absences are zero by declaration; (2) after
    the pricer's own derivation (the conversion model) every absent factor of every component table is in exactly
    one absent state; (3) every declared row over an absent factor is named in the component table's notes: with the
    view's or the plane's reason when withheld or not produced, with the declaration's reason when zero by
    declaration. Returns how many (row, view) pairs it checked."""
    from mimarsinan.deployment_record.cost.absolute.context import PricingContext
    raw_zero = getattr(view, "known_zero", {})
    assert set(raw_zero) <= set(plane_zeros), \
        f"zero by declaration beyond the platform's: {sorted(set(raw_zero) - set(plane_zeros))}"
    ctx = PricingContext.create(view, _every_row_physics())
    q = ctx.quantities
    checked = 0
    for table in TABLES:
        _bands, evidence, unpriced = ctx.component_bands(table, 1.0)
        notes = " | ".join(evidence + unpriced)
        for constant, factors, label in table:
            if not ctx.priced(constant):
                continue
            keys = set().union(*(_keys(f) for f in factors))
            absent = sorted(k for k in keys if not q.has(k))
            if not absent:
                continue
            for key in absent:
                states = [s for s in (q.withheld_reason(key), q.unproduced_reason(key), q.known_zero_reason(key)) if s]
                assert len(states) == 1, f"{constant}: {key} is absent in {len(states)} states"
            checked += 1
            zero = next((q.known_zero_reason(k) for k in absent if q.known_zero_reason(k)), None)
            if zero is not None:
                assert f"zero by declaration: {constant} ({zero})" in notes, f"{constant}: zero not printed"
            else:
                reason = next(s for k in absent for s in (q.withheld_reason(k), q.unproduced_reason(k)) if s)
                assert f"{constant}({label}" in notes and reason in notes, \
                    f"{constant} over {absent} is not named with its reason"
    return checked


class TestTheRatchetReadsViews:
    """The class-2 ratchet on views the planes build: a candidate view whose unproduced factors were turned into
    zeros by declaration is red here."""

    def test_a_candidate_view_names_every_declared_constant_over_an_absent_census(self):
        import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
        from mimarsinan.deployment_record.introspection.scaling_ladder_lanes import host_census
        from mimarsinan.deployment_record.objectives import CandidateStaticView, declared_core_capacity
        from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
        from mimarsinan.deployment_record.quantities.planes.states import declared_zeros
        from unit.deployment_record.introspection.test_scaling_ladder import _facts, _network, _platform
        network = _network(4)
        point = {**_platform(8), "inference_batch_K": 16}
        census = ladder.shape_census(network, point)
        scheduled = ladder.schedule_program(network, census, point, _facts)
        facts, host = scheduled.facts, host_census(network, census)
        context = candidate_context_from_platform(
            point, latency_steps=facts.latency_steps, programming=facts.programming, carry=facts.carry,
            passes=facts.passes, **{k: host.get(k) for k in ("host_macs", "onchip_macs", "host_params",
                                                              "onchip_params", "compute_op_count")})
        view = CandidateStaticView(layout=scheduled.stats, chip_param_capacity=declared_core_capacity(dict(point)),
                                   total_params=network.total_params, host_side_segment_count=census.host_segments,
                                   physics=None, quantity_context=context, noc_fragments=scheduled.noc).quantities
        zeros = declared_zeros(context.output_stage is not None, context.carry_buffer_capacity_bytes is not None,
                               context.value_rule is not None)
        assert view.unproduced, "the candidate plane states what it does not produce"
        assert _check_view(view, zeros) > 0

    def test_a_sealed_record_names_every_declared_constant_over_an_absent_census(self):
        from mimarsinan.deployment_record.quantities.planes.states import declared_zeros
        from mimarsinan.deployment_record.quantities.from_record import from_record
        from mimarsinan.deployment_record.quantities.value_domain import declared_buffer_bytes, names_value_domain
        from unit.deployment_record.record_fixtures import make_full_record
        record = make_full_record()
        view = from_record(record)
        platform = record.identity.platform
        zeros = declared_zeros(record.output_stage is not None, declared_buffer_bytes(platform) is not None,
                               names_value_domain(platform))
        assert _check_view(view, zeros) > 0



class TestTheShares:
    def test_the_share_pricing_keeps_the_plane_reason_in_its_text(self):
        from mimarsinan.deployment_record.cost.absolute.host_split import price_shares
        q = Quantities(_values(), unproduced={"neurons_used": "the plane counts no outputs"})
        terms, _ = price_shares(q, _physics(BASE))
        for variant in terms:
            band = terms[variant].get("energy_dynamic_mj")
            assert band is not None, variant
            assert "not produced on this plane: the plane counts no outputs" in band.basis, variant
