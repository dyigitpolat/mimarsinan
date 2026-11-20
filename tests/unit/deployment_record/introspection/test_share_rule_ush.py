"""[U-SH, phase C defect G3-D1] One MAC and carry share rule for the record plane and the exploration plane.

The dynamic energy's MAC share is the compute-only pricing of a program's quantities (no carry, programming, sync, op
count or host quantity; host_macs 0) and its carry share is the no-host pricing (no op count or host quantity;
host_macs 0) less the compute-only one, corner by corner. The scaling ladder priced both inside ``price_scheduled``
and subtracted them in ``axis_columns``; the record plane (``design_point.pricing.record_variant``) priced neither, so
Fig. 1's record-plane row printed both shares null with no name. Both planes now price the shares through one function
(``cost.absolute.host_split.price_shares``) and read them through one rule (``host_split.shares``, which the ladder's
law module re-exports), which names every absent share by the compute-absence rule (U-CR C-2) on the pricing that
lacks it."""

from __future__ import annotations

import inspect

import mimarsinan.deployment_record.design_point.pricing as pricing
import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
import mimarsinan.deployment_record.introspection.scaling_ladder_law as law
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.host_terms import HOST_SIDE_QUANTITIES
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.schema.provenance import Band
from unit.deployment_record.g3_fixtures import quantities
from unit.deployment_record.introspection.test_scaling_ladder import READING

DYNAMIC, HEADLINE = "energy_dynamic_mj", "energy_per_inference_mj"
NAMES = (DYNAMIC, HEADLINE, "energy_write_mj", "energy_host_mj")
#: The share variants as the ladder of R defined them (``scaling_ladder.price_scheduled``), pinned by name.
CARRY = ("carried_raster_bytes", "carry_peak_live_bytes", "carry_out_bytes", "carry_in_bytes")
PROGRAMMING = ("reprogrammed_bytes", "reprogrammed_cores", "reprogram_passes")
DROPS = {"compute_only": CARRY + PROGRAMMING + ("sync_count", "compute_op_count") + HOST_SIDE_QUANTITIES,
         "no_host": ("compute_op_count",) + HOST_SIDE_QUANTITIES}


def _band(low: float, nominal: float, high: float) -> Band:
    return Band(low, nominal, high, "test")


def _columns(terms, refusals):
    return law.axis_columns(terms, refusals, NAMES, k=1, reading=READING, multiplexed=False, x=1.0, passes=1,
                            axis="energy")


def _priced(**extra):
    once = {"energy_write_mj": _band(1.0, 2.0, 3.0)}
    terms = {"program": {}, "compute_only": {DYNAMIC: _band(1.0, 2.0, 4.0)}, "no_host": {DYNAMIC: _band(2.0, 3.0, 4.5)},
             "once": once, "period": once}
    refusals = {name: {} for name in terms}
    for name, value in extra.items():
        terms[name] = value
    return terms, refusals


class TestOneRule:
    def test_the_share_pricings_are_the_ladders_variants(self):
        """``price_shares`` prices exactly the ladder's two variants of R (drop lists and host_macs 0)."""
        from mimarsinan.deployment_record.cost.absolute.host_split import price_shares
        q = quantities(synaptic_events=1.0e6, host_macs=5.0e5, sync_count=7, compute_op_count=3, carry_in_bytes=4096,
                       carry_out_bytes=4096, reprogrammed_bytes=1.0e5, reprogrammed_cores=4, reprogram_passes=2,
                       segment_in_bytes=1024, segment_out_bytes=512, cores_physical=4, weight_bits=8)
        physics = get_platform_physics("neurram")
        terms, refusals = price_shares(q, physics)
        assert list(terms) == list(refusals) == ["compute_only", "no_host"]
        for name, drop in DROPS.items():
            values = {k: q.get(k) for k in q.keys() if k not in drop}
            values["host_macs"] = QuantityValue(value=0.0, provenance="static")
            by_hand = price_absolute(Quantities(values), physics)
            assert terms[name] == {t.name: t.band for t in by_hand.terms if t.band is not None}, name
            assert refusals[name] == {r.name: r.reason for r in by_hand.refusals}, name
        assert terms["compute_only"][DYNAMIC] is not None and terms["no_host"][DYNAMIC] is not None

    def test_both_planes_price_and_read_their_shares_through_the_one_rule(self):
        """Neither plane holds share variants of its own: the ladder and the record plane both price them by the one
        rule (``share_pricings``, which ``price_shares`` reads; [V-M1] the ladder reads the variants' pricings whole, for
        each cost's static legs), and ``axis_columns`` and the record plane both read the shares with ``law.shares``."""
        ladder_source = inspect.getsource(ladder.price_scheduled)
        record_source = inspect.getsource(pricing.record_variant)
        assert "share_pricings(" in ladder_source and "price_shares(" in record_source
        assert "law.shares(" in record_source or " shares(" in record_source
        assert " shares(" in inspect.getsource(law.axis_columns)
        for module in (ladder, pricing):
            text = inspect.getsource(module)
            assert "HOST_SIDE_QUANTITIES" not in text and "_CARRY" not in text, module.__name__
        assert not hasattr(ladder, "_variant")


class TestTheShareReading:
    def test_priced_shares_are_the_compute_only_and_the_corner_wise_boundary(self):
        terms, refusals = _priced()
        column = _columns(terms, refusals)
        assert column["terms"]["compute_only"] == [1.0, 2.0, 4.0]
        assert column["terms"]["boundary"] == [1.0, 1.0, 0.5]  # corner-aligned, no re-sorting
        assert column["absent"] is None and "boundary_absent" not in column

    def test_an_absent_compute_only_share_names_both_by_the_compute_absence_rule(self):
        terms, refusals = _priced(compute_only={})
        refusals["compute_only"] = {HEADLINE: "no compute-energy source is declared: test"}
        column = _columns(terms, refusals)
        assert column["terms"]["compute_only"] is None and column["terms"]["boundary"] is None
        assert column["absent"] == "no compute-energy source is declared: test"
        assert "boundary_absent" not in column

    def test_a_boundary_the_no_host_pricing_refuses_alone_is_named_by_its_refusal(self):
        terms, refusals = _priced(no_host={})
        refusals["no_host"] = {DYNAMIC: "a converter constant is unstated at 6 bits"}
        column = _columns(terms, refusals)
        assert column["terms"]["compute_only"] == [1.0, 2.0, 4.0] and column["terms"]["boundary"] is None
        assert column["absent"] is None
        assert column["boundary_absent"] == "a converter constant is unstated at 6 bits"

    def test_a_no_host_gap_no_refusal_names_is_named_as_a_gap(self):
        terms, refusals = _priced(no_host={})
        refusals["no_host"] = {"some_constant": "unpriced by the test"}
        column = _columns(terms, refusals)
        named = column["boundary_absent"]
        assert DYNAMIC in named and "some_constant" in named and "no-host pricing" in named
        assert "no physics declared" not in named

    def test_the_shares_of_a_program_no_pricing_ran_on_read_no_physics_declared(self):
        share = law.shares({}, {}, DYNAMIC, HEADLINE)
        assert share["compute_only"] is None and share["boundary"] is None
        assert share["absent"] == law.NO_PHYSICS and "boundary_absent" not in share
