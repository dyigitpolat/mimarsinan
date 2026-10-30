"""[U-J5 L-4, U-J4 review m4] With one of ``e_link_per_byte`` and ``e_host_mem_per_byte`` declared, the residency
step choice names the other in its step's basis, never summing the declared row silently.

Beside the host-chip link the step choice prices the bytes it moves over the link (a host-owned store's carry and
write transfer, a host-sourced payload on either store) at ``hosts.store.store_byte_energy``: the link plus host
memory where declared. A host description declares both (``REQUIRED_HOST_ROWS``); a hand block may declare one. The
missing row is named in the step's basis with the shared helper of U-J4's targeted pass
(``hosts.store.unpriced_link_row``), the headline's and the lowering price's text; neither declared still refuses by
name (``StepChoiceError``). No price changes."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost.absolute.step_choice import carry_energy, cheapest_step, price_step
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.period_census import PassBoundaryCensus, PassCensus, PassPeriodCensus

HOST = "jetson_orin_nano_8gb"
GENERIC = "generic_estimated_22nm"
CORNERS = ("low", "nominal", "high")
LINK_ENERGY = ("e_link_per_byte", "e_host_mem_per_byte")
WRITE = {"t_write_per_unit": {"nominal": 10.0, "unit": "ns", "evidence_kind": "estimated", "note": "test constant"}}
UNPRICED = "; unpriced: "


def _physics(store, drop=()):
    block = {key: row for key, row in host_description_block(HOST).items() if key not in drop}
    return resolve_platform_physics(GENERIC, merge_override_blocks(
        {"write": WRITE, "host": block, "store": ST.store_block(store, "test")}))


def _policy(source="host"):
    return WritePolicy(batch_k=4, producer_write_source=source, payload_bits=8, write_granularity="cell")


def _census(payload=400, producer=10, carry=(200, 300)):
    passes = (PassCensus(position=0, segment_index=0, resident=False, cores=1, resident_cores=0,
                         written_cells_backing_store=100, written_cells_producer=producer, payload_bytes=payload,
                         compute_steps=3),
              PassCensus(position=1, segment_index=0, resident=True, cores=1, resident_cores=1,
                         written_cells_backing_store=0, written_cells_producer=0, payload_bytes=0, compute_steps=3))
    boundary = (PassBoundaryCensus(after_pass=0, carry_out_bytes=carry[0], carry_in_bytes=carry[1]),)
    return PassPeriodCensus(passes=passes, boundaries=boundary)


def _named(basis):
    return basis.split(UNPRICED, 1)[1] if UNPRICED in basis else ""


@pytest.mark.parametrize("drop", [("e_link_per_byte",), ("e_host_mem_per_byte",)])
class TestOneLinkEnergyRowDeclared:
    def test_the_host_sourced_payload_names_the_missing_row_on_a_chip_owned_store(self, drop):
        write = price_step(_census(payload=0, carry=(0, 0)), _physics(ST.CHIP, drop), _policy()).write
        assert f"{drop[0]}(the host-sourced payload bytes per inference" in _named(write.basis), write.basis

    def test_a_host_owned_stores_write_transfer_and_payload_name_it(self, drop):
        write = price_step(_census(), _physics(ST.HOST, drop), _policy()).write
        named = _named(write.basis)
        assert f"{drop[0]}(the period's payload bytes / K" in named, write.basis
        assert f"{drop[0]}(the host-sourced payload bytes per inference" in named, write.basis

    def test_a_host_owned_stores_carry_names_it(self, drop):
        carry = carry_energy(_census(), _physics(ST.HOST, drop))
        assert f"{drop[0]}(the carried bytes" in _named(carry.basis), carry.basis

    def test_no_price_changes(self, drop):
        """The step's price is the declared row's alone, as before: the name is text only."""
        physics, census = _physics(ST.HOST, drop), _census()
        kept = [key for key in LINK_ENERGY if key not in drop][0]
        step, alone = price_step(census, physics, _policy()), price_step(census, physics, _policy("on_chip"))
        for corner in CORNERS:
            assert getattr(step.write, corner) - getattr(alone.write, corner) == pytest.approx(
                10 * getattr(physics.band(kept), corner) * 1e3, rel=1e-9), corner
            assert getattr(step.carry, corner) == pytest.approx(500 * getattr(physics.band(kept), corner) * 1e3,
                                                                rel=1e-9), corner
        assert cheapest_step([census, census], physics, _policy()) == 0


@pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
def test_both_rows_declared_name_nothing(store):
    step = price_step(_census(), _physics(store), _policy())
    assert UNPRICED not in step.write.basis and UNPRICED not in step.carry.basis, (step.write.basis, step.carry.basis)


def test_neither_row_declared_still_refuses_by_name():
    with pytest.raises(Exception) as caught:
        price_step(_census(payload=0, carry=(0, 0)), _physics(ST.CHIP, LINK_ENERGY), _policy())
    assert "declares neither e_link_per_byte nor e_host_mem_per_byte" in str(caught.value)
