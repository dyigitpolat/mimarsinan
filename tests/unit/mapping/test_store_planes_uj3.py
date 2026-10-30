"""[U-J3 R-1] Both planes route the store traffic by the one declaration their resolved physics carries.

The candidate plane forms the host-chip boundary's quantities (``objectives.host_link``): a pass boundary that carries
values adds one transfer per direction only where the store is host-owned (a chip-owned store keeps the carry on the
chip's own DRAM, so only the segment boundaries transfer). The record plane states the carry from the sealed record
and no segment census. On either plane the pricer routes the carry by the store (``hosts.store``): on a chip-owned
store the chip's rows price it and the link carries none of it; on a host-owned store the link and host memory price
it and the chip's store rows none. Vehicles: the carried widths of U-J's by-hand case and U-F4's residual pipelined
fixture."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

CORNERS = ("low", "nominal", "high")
HOST = "jetson_orin_nano_8gb"


def _physics(store):
    blocks = {"host": host_description_block(HOST)}
    if store is not None:
        blocks["store"] = ST.store_block(store, "test")
    return resolve_platform_physics("generic_estimated_22nm", merge_override_blocks(blocks))


STORES = {ST.CHIP: _physics(ST.CHIP), ST.HOST: _physics(ST.HOST), None: _physics(None)}


def _transfers(physics, passes, segment_transfers=0, **platform):
    from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
    from mimarsinan.mapping.support.host_boundary import HostBoundaryCensus

    census = HostBoundaryCensus(to_chip_elements=0, from_chip_elements=0, transfers=segment_transfers,
                                memory_bound_elements=0)
    pcfg = {"platform_physics_resolved": physics.to_dict(), "simulation_steps": 1, **platform}
    context = candidate_context_from_platform(pcfg, passes=passes, host_boundary=census)
    return dict(context.host_link_quantities)["host_link_transfers"].value


class TestTheCandidatePlanesTransfers:
    """U-J's by-hand boundaries: 10 values out after pass 0 and 5 after pass 1; 10 in before pass 2, 5 before 3."""

    WIDTHS = ((0, 10, 0), (1, 5, 10), (2, 0, 5))

    def _passes(self):
        from mimarsinan.deployment_record.build.write_census import ProgramPasses

        return ProgramPasses(rows=(), boundary_widths=self.WIDTHS, onchip_boundary_widths=None)

    def test_a_chip_owned_store_transfers_at_the_segment_boundaries_only(self):
        assert _transfers(STORES[ST.CHIP], self._passes(), segment_transfers=2) == 2

    def test_a_host_owned_store_adds_each_carried_boundary_per_direction(self):
        assert _transfers(STORES[ST.HOST], self._passes(), segment_transfers=2) == 2 + 4

    def test_a_chip_owned_store_needs_no_pass_rows_to_count_its_transfers(self):
        assert _transfers(STORES[ST.CHIP], None, segment_transfers=3) == 3


@pytest.fixture(scope="module", params=["together", "groups"])
def pipelined(request):
    """U-F4's residual pipelined fixture (as U-J's targeted pass reads it)."""
    from unit.mapping.pipelined_multiplexing.test_planes import _ladder_noc, _pipelined

    platform, program = _pipelined("residual", request.param)
    scheduled, _specs = _ladder_noc(program, platform, 10)
    return platform, scheduled


class TestThePipelinedFixture:
    def test_its_carried_boundaries_transfer_on_a_host_owned_store_only(self, pipelined):
        platform, scheduled = pipelined
        widths = scheduled.facts.passes.boundary_widths
        carried = sum(int(bool(o)) + int(bool(i)) for _p, o, i in widths)
        assert carried > 0
        base = {k: v for k, v in platform.items() if k != "platform_physics_resolved"}
        assert _transfers(STORES[ST.HOST], scheduled.facts.passes, **base) == carried
        assert _transfers(STORES[ST.CHIP], scheduled.facts.passes, **base) == 0


def _q(**values):
    return Quantities({key: QuantityValue(float(v), "static") for key, v in values.items()})


#: The same program's carry on both planes: the candidate plane states its segment census (here none crosses), the
#: record plane states none (the sealed record carries no host-chip boundary census).
VIEW = dict(synaptic_events=1e6, cores_physical=8, latency_steps=32, timesteps=4, neurons_used=100, segment_cores=1,
            host_macs=0, compute_op_count=0, host_memory_bound_bytes=0, carry_in_bytes=300, carry_out_bytes=200)
CANDIDATE = dict(VIEW, segment_in_bytes=0, segment_out_bytes=0, host_link_transfers=2)
RECORD = dict(VIEW)


def _terms(physics, values):
    pricing = price_absolute(_q(**values), physics)
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


class TestBothPlanesRouteTheCarryAlike:
    @pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
    def test_the_chips_dynamic_energy_is_the_same_on_both_planes(self, store):
        candidate, _ = _terms(STORES[store], CANDIDATE)
        record, _ = _terms(STORES[store], RECORD)
        for corner in CORNERS:
            assert getattr(candidate["energy_dynamic_mj"].band, corner) == pytest.approx(
                getattr(record["energy_dynamic_mj"].band, corner), rel=1e-12)

    def test_a_host_owned_store_prices_the_carry_on_the_link_on_both_planes(self):
        candidate, _ = _terms(STORES[ST.HOST], CANDIDATE)
        record, _ = _terms(STORES[ST.HOST], RECORD)
        for name in ("energy_host_link_mj", "energy_host_mem_mj"):
            for corner in CORNERS:
                assert getattr(candidate[name].band, corner) == pytest.approx(
                    getattr(record[name].band, corner), rel=1e-12), name
        assert "e_link_per_byte(segment I/O" in record["energy_per_inference_mj"].source

    def test_a_chip_owned_store_puts_none_of_the_carry_on_the_link_on_either_plane(self):
        candidate, _ = _terms(STORES[ST.CHIP], CANDIDATE)
        record, _ = _terms(STORES[ST.CHIP], RECORD)
        assert "energy_host_link_mj" not in candidate and "energy_host_link_mj" not in record
        assert "e_link_per_byte(segment I/O" in record["energy_per_inference_mj"].source
        assert "pass carry" not in record["energy_per_inference_mj"].source

    def test_no_store_refuses_by_the_same_name_on_both_planes(self):
        _c, candidate = _terms(STORES[None], CANDIDATE)
        _r, record = _terms(STORES[None], RECORD)
        assert candidate["e2e_latency_s"] == record["e2e_latency_s"] and "backing_store" in record["e2e_latency_s"]
