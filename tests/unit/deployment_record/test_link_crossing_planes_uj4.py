"""[U-J4 K-1] Both planes price an operand program's host-sourced payload on the host-chip link alone.

The vehicle is N2's attention program packed on 64 x 64 cores (E-6's record-plane case): its operand regions are
written per inference from the host. The built program's passes (the mapping plane, ``mapping_program_passes``) and
the sealed record's (``record_write_census``) state the same period census, so beside the edge host each plane's
write term is the cells' commit alone (the same as an on-chip write-back of the same cells), the payload's bytes are
the program's crossing on both, and the headline pays them at the link and host memory once, on either store."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.deployment_record.build.program_census import mapping_program_passes
from mimarsinan.deployment_record.cost import host_terms as HT
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_inputs
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.write_law import plane_writes, record_write_census
from unit.chip_simulation.value_run.operand_programs import attention_program

CORNERS = ("low", "nominal", "high")
GEOMETRY = dict(axons=64, neurons=64, count=3, passes=128)
WRITE = {"t_write_per_unit": {"nominal": 10.0, "unit": "ns", "evidence_kind": "estimated", "note": "test constant"},
         "endurance_cycles": {"nominal": 1e6, "unit": "cycles", "evidence_kind": "estimated", "note": "test constant"}}
VALUES = dict(synaptic_events=1e6, cores_physical=8, latency_steps=32, timesteps=4, neurons_used=100, segment_cores=1,
              host_macs=0, compute_op_count=0, host_memory_bound_bytes=0, carry_in_bytes=0, carry_out_bytes=0,
              segment_in_bytes=0, segment_out_bytes=0, host_link_transfers=0)


def _physics(store):
    blocks = {"write": WRITE, "host": host_description_block("jetson_orin_nano_8gb"),
              "store": ST.store_block(store, "test")}
    return resolve_platform_physics("generic_estimated_22nm", merge_override_blocks(blocks))


PHYSICS = {store: _physics(store) for store in (ST.CHIP, ST.HOST)}


def _platform(source):
    return {"cores": [{"max_axons": 64, "max_neurons": 64, "count": 10_000, "write_granularity": "cell",
                       "operand": {"payload_bits": 8, "write_source": source}}],
            "inference_batch_K": 1, "inference_rate_hz": 30.0, "min_lifetime_hours": 43_800.0}


@pytest.fixture(scope="module")
def mapping():
    return attention_program("softmax").packed_mapping(**GEOMETRY)


def _record(mapping, platform):
    schedule = schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0)
    return SimpleNamespace(schedule=schedule, timing=SimpleNamespace(per_segment=(), s_global=1),
                           identity=SimpleNamespace(platform=platform),
                           utilization=SimpleNamespace(crossbar=SimpleNamespace(weight_bits=8)))


def _planes(mapping, source):
    platform = _platform(source)
    built = plane_writes(mapping_program_passes(mapping), platform, weight_bits=8, timesteps=1, label="mapping")
    sealed = record_write_census(_record(mapping, platform))
    assert built is not None and sealed is not None and built.census is not None and sealed.census is not None
    return {"mapping": WriteInputs.of(built), "record": WriteInputs.of(sealed)}


@pytest.fixture(scope="module")
def planes(mapping):
    return {source: _planes(mapping, source) for source in ("host", "on_chip")}


def _q(**values):
    return Quantities({key: QuantityValue(float(value), "static") for key, value in values.items()})


@pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
@pytest.mark.parametrize("plane", ["mapping", "record"])
class TestEachPlane:
    def test_the_write_term_is_the_cells_commit_alone(self, planes, store, plane):
        host = {t.name: t.band for t in price_write_inputs(planes["host"][plane], PHYSICS[store]).terms}
        own = {t.name: t.band for t in price_write_inputs(planes["on_chip"][plane], PHYSICS[store]).terms}
        for corner in CORNERS:
            assert getattr(host["energy_write_mj"], corner) == pytest.approx(
                getattr(own["energy_write_mj"], corner), rel=1e-12), corner

    def test_the_headline_pays_the_payload_at_the_link_and_host_memory_once(self, planes, store, plane):
        physics = PHYSICS[store]
        moved = HT.write_link_bytes(planes["host"][plane]).program
        assert moved > 0
        priced = {source: {t.name: t.band for t in price_absolute(_q(**VALUES), physics,
                                                                  writes=planes[source][plane]).terms}
                  for source in ("host", "on_chip")}
        for corner in CORNERS:
            per_byte = getattr(physics.band("e_link_per_byte"), corner) + getattr(
                physics.band("e_host_mem_per_byte"), corner)
            delta = (getattr(priced["host"]["energy_per_inference_mj"], corner)
                     - getattr(priced["on_chip"]["energy_per_inference_mj"], corner))
            static = (getattr(priced["host"]["energy_static_mj"], corner)  # [U-J5 L-5] over the priced link time
                      - getattr(priced["on_chip"]["energy_static_mj"], corner))
            assert delta - static == pytest.approx(moved * per_byte * 1e3, rel=1e-9), corner


def test_both_planes_move_the_same_payload_bytes(planes):
    mapped, sealed = (HT.write_link_bytes(planes["host"][plane]) for plane in ("mapping", "record"))
    assert mapped == sealed and mapped.program > 0


def test_a_write_back_on_chip_moves_nothing_over_the_link(planes):
    assert all(HT.write_link_bytes(planes["on_chip"][plane]).program == 0.0 for plane in ("mapping", "record"))


def test_the_record_plane_splits_the_operand_writes_without_the_transfer(planes):
    """The record pricer's static/operand split (design_point.pricing): the operand writes are the cells' commit."""
    from mimarsinan.deployment_record.design_point.pricing import _without_operand_writes

    physics = PHYSICS[ST.CHIP]

    def operand_writes(inputs):
        total, static = (next(t.band for t in price_write_inputs(view, physics).terms if t.name == "energy_write_mj")
                         for view in (inputs, _without_operand_writes(inputs)))
        return total.nominal - static.nominal

    host, own = (operand_writes(planes[source]["record"]) for source in ("host", "on_chip"))
    assert own > 0 and host == pytest.approx(own, rel=1e-12)
