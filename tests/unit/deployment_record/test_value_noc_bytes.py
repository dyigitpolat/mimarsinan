"""CH-D2: value-domain NoC byte-hops on the candidate plane, priced by their own per-byte-hop row."""

from types import SimpleNamespace

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.objectives.views import CandidateStaticView
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities.from_candidate import CandidateQuantityContext
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.value_domain import ValueDomainFacts
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.noc import ValueNocCensusError, value_noc_bytes

from unit.deployment_record.g3_fixtures import quantities
from unit.deployment_record.record_fixtures import make_full_record

#: A 2 x 2-tile chip, one core per tile, column-major tiles: (0,0) (0,1) (1,0) (1,1).
_PLACEMENTS = (((0, 0), (1, 1), (2, 2), (3, 3)),)
#: (producer, consumer) -> distinct wires; softcore 4 runs in another stage (a carry, not mesh traffic).
_WIRES = {(0, 1): 10, (0, 3): 4, (1, 2): 6, (2, 3): 2, (4, 0): 7}


def _bytes(bits):
    return value_noc_bytes(pair_wires=_WIRES, stage_placements=_PLACEMENTS, cores_per_tile=1, mesh_height=2,
                           activation_bits=bits)


class TestTheCensus:
    def test_the_byte_hops_equal_the_hand_census(self):
        # 10 x 1 hop + 4 x 2 hops + 6 x 2 hops + 2 x 1 hop = 32 value-hops, one byte each at 8 bits.
        assert _bytes(8) == 32.0
        assert _bytes(4) == 16.0

    def test_a_wire_inside_one_tile_crosses_no_link(self):
        assert value_noc_bytes(pair_wires={(0, 1): 9}, stage_placements=(((0, 0), (1, 1)),),
                               cores_per_tile=4, mesh_height=1, activation_bits=8) == 0.0

    def test_an_undeclared_grid_refuses_by_name(self):
        with pytest.raises(ValueNocCensusError) as info:
            _bytes(0)
        assert isinstance(info.value, ToolchainRefusal)


_RULE = ValueDomainFacts(mac_events=None, latency_steps=None, carry_out_bytes=None, carry_in_bytes=None)


def _view(value_bits, *, value_domain=None):
    value_domain = value_bits is not None if value_domain is None else value_domain
    context = CandidateQuantityContext(value_bits=value_bits, cores_per_tile=1, tile_mesh_height=2,
                                       value_rule=_RULE if value_domain else None)
    fragments = SimpleNamespace(census=SimpleNamespace(pair_wires=_WIRES), pass_placements=_PLACEMENTS,
                                stage_placements=_PLACEMENTS)
    return CandidateStaticView(layout=None, chip_param_capacity=None, total_params=None,
                               host_side_segment_count=None, quantity_context=context, noc_fragments=fragments)


class TestBothPlanes:
    def test_a_value_domain_candidate_carries_the_byte_hops(self):
        values = _view(8).quantities
        assert values.get("noc_value_bytes").value == 32.0
        assert values.get("noc_value_bytes").provenance == "static"

    def test_a_spiking_candidate_carries_none(self):
        assert not _view(None).quantities.has("noc_value_bytes")
        assert _view(None).quantities.withheld_reason("noc_value_bytes") is None

    def test_a_value_candidate_without_the_grid_withholds_the_byte_hops_by_name(self):
        values = _view(None, value_domain=True).quantities
        assert not values.has("noc_value_bytes")
        assert "activation_bits" in values.withheld_reason("noc_value_bytes")

    def test_a_value_candidate_whose_wires_stay_in_their_tile_states_zero_without_the_grid(self):
        context = CandidateQuantityContext(cores_per_tile=4, tile_mesh_height=1, value_rule=_RULE)
        fragments = SimpleNamespace(census=SimpleNamespace(pair_wires={(0, 1): 9}),
                                    pass_placements=(((0, 0), (1, 1)),), stage_placements=(((0, 0), (1, 1)),))
        view = CandidateStaticView(layout=None, chip_param_capacity=None, total_params=None,
                                   host_side_segment_count=None, quantity_context=context, noc_fragments=fragments)
        assert view.quantities.get("noc_value_bytes").value == 0.0

    def test_a_spiking_record_carries_none(self):
        assert not from_record(make_full_record()).has("noc_value_bytes")


def _energy_terms(physics, **extra):
    values = quantities(synaptic_events=1000, host_macs=0, **extra)
    return {t.name: t for t in price_absolute(values, physics).terms}


class TestPricing:
    @pytest.mark.parametrize("profile", ["isaac_like", "generic_estimated_22nm"])
    def test_the_value_noc_energy_is_its_row_times_the_byte_hops(self, profile):
        physics = get_platform_physics(profile)
        with_noc = _energy_terms(physics, noc_value_bytes=32)["energy_dynamic_mj"]
        without = _energy_terms(physics)["energy_dynamic_mj"]
        expected = physics.band("e_noc_value_per_byte_hop").nominal * 32 * 1e3
        assert with_noc.value - without.value == pytest.approx(expected, rel=1e-12)
        assert "noc_value: e_noc_value_per_byte_hop" in with_noc.source

    def test_a_profile_without_the_row_names_it_unpriced(self):
        physics = get_platform_physics("odin")
        terms = _energy_terms(physics, noc_value_bytes=32, cores_physical=1, latency_steps=1)
        assert "unpriced: e_noc_value_per_byte_hop(noc_value)" in terms["energy_per_inference_mj"].source

    def test_the_dynamic_term_names_the_undeclared_row_too(self):
        """energy_dynamic_mj names every component of its own sum it cannot price, as the headline does."""
        terms = _energy_terms(get_platform_physics("odin"), noc_value_bytes=32, cores_physical=1, latency_steps=1)
        assert "unpriced: e_noc_value_per_byte_hop(noc_value)" in terms["energy_dynamic_mj"].source

    @pytest.mark.parametrize("profile", ["isaac_like", "generic_estimated_22nm", "odin"])
    def test_withheld_byte_hops_are_named_on_both_energy_terms(self, profile):
        """t0_42-class: the value NoC is withheld without activation_bits, whether or not the target
        declares its row; the dynamic term never drops it silently."""
        reason = "activation_bits undeclared: the value NoC traffic has no grid"
        values = Quantities({"synaptic_events": QuantityValue(1000.0, "static"),
                             "host_macs": QuantityValue(0.0, "static"),
                             "cores_physical": QuantityValue(1.0, "static"),
                             "latency_steps": QuantityValue(1.0, "static")},
                            withheld={"noc_value_bytes": reason})
        pricing = price_absolute(values, get_platform_physics(profile))
        terms = {t.name: t for t in pricing.terms}
        named = f"e_noc_value_per_byte_hop(noc_value; {reason})"
        assert named in terms["energy_dynamic_mj"].source
        if "energy_per_inference_mj" in terms:
            assert named in terms["energy_per_inference_mj"].source
