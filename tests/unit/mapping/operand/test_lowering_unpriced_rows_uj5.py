"""[U-J5 L-6 and L-4] The lowering price names every row it reads and the physics does not declare.

L-6 (U-J4's open boundary): on a chip's own store (and without the host-chip link) N2's lowering price injects the
streamed vectors at the chip's ``e_dma_per_byte`` (``hosts.store.injection_byte_energy``). Six shipped profiles declare
no such row, so the vectors price at 0 and the tie rule decides (fewer bank cells, the lowering streaming more), which
the printed reason did not say. The reason now names the row, with the shared helper of U-J4's targeted pass
(``hosts.store.unpriced_link_row``); no choice and no price changes. On a host-owned store the vectors are injected
from the host's memory over the link, so the chip's row is not read and nothing new is named.

L-4, the lowering half: with one link energy row declared, the lowering's printed reason already names the other
(U-J4's targeted pass, ``LoweringPrice.unpriced``); pinned here beside the step choice's (``test_step_choice_link_rows``).

Vehicle (U-J4's): one cost-only product (batch 2, m 3, k 4, n 5): B stationary holds 40 bank cells and streams 24
elements, A^T stationary 24 and 40."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.mapping.support.operand_form import AT_STATIONARY, B_STATIONARY, Lowering, OperandForm, stationary_operand
from mimarsinan.mapping.support.operand_price import lowering_price

HOST = "jetson_orin_nano_8gb"
GENERIC = "generic_estimated_22nm"
FORM = OperandForm(batch=(2,), m=3, k=4, n=5, a_perm=(0, 1, 2), b_perm=(0, 1, 2), out_perm=(0, 1, 2))
B, AT = Lowering(FORM, B_STATIONARY), Lowering(FORM, AT_STATIONARY)
WRITE = {"t_write_per_unit": {"nominal": 10.0, "unit": "ns", "evidence_kind": "estimated", "note": "test constant"}}
WITHOUT_DMA = ("hermes_64", "loihi", "neurram", "odin", "sram_cim_macro", "truenorth")
SOURCES = ("host", "on_chip")
UNPRICED = "; unpriced: "
STREAMED = "e_dma_per_byte(the lowering's streamed vectors"


def _physics(chip, store=None, host=True, drop=()):
    blocks = {"write": WRITE}
    if host:
        blocks["host"] = {key: row for key, row in host_description_block(HOST).items() if key not in drop}
    if store is not None:
        blocks["store"] = ST.store_block(store, "test")
    return resolve_platform_physics(chip, merge_override_blocks(blocks))


def _platform(physics, source):
    return {"platform_physics_resolved": physics.to_dict(), "inference_batch_K": 1, "activation_bits": 8,
            "cores": [{"max_axons": 64, "max_neurons": 64, "count": 64, "write_granularity": "cell",
                       "operand": {"payload_bits": 8, "write_source": source}}]}


def _choice(physics, source, cost_only=True):
    price = lowering_price(_platform(physics, source))
    chosen, reason = stationary_operand(FORM, price, cost_only)
    return chosen.stationary, reason, price


def _named(reason):
    return reason.rsplit(UNPRICED, 1)[1] if UNPRICED in reason else ""


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("chip", WITHOUT_DMA)
class TestTheStreamedVectorsUnpricedRow:
    def test_without_the_link_the_reason_names_it_and_the_tie_still_decides(self, chip, source):
        chosen, reason, price = _choice(_physics(chip, host=False), source)
        assert chosen == AT_STATIONARY and price(B) == price(AT), reason
        # [chain 5b test-pin amendment] The transfer part is zero: the head declares NeuRRAM's e_write_per_cell
        # (480828a8, after the unit's base), which prices both lowerings' cells alike; without it both are 0.0.
        bare = _platform(_physics(chip, host=False), source)
        bare["platform_physics_resolved"]["constants"].pop("e_write_per_cell", None)
        transfer = lowering_price(bare)
        assert transfer(B) == transfer(AT) == 0.0, (transfer(B), transfer(AT))
        assert _named(reason).startswith(STREAMED) and "the physics declares no e_dma_per_byte" in reason, reason

    def test_on_the_chips_own_store_beside_the_link_the_reason_names_it(self, chip, source):
        chosen, reason, price = _choice(_physics(chip, ST.CHIP), source)
        assert chosen == AT_STATIONARY and price(B) == price(AT), reason
        assert STREAMED in _named(reason), reason
        assert "the physics prices the link and declares no e_dma_per_byte" in reason, reason

    def test_on_a_host_owned_store_the_chips_row_is_not_read(self, chip, source):
        chosen, reason, _price = _choice(_physics(chip, ST.HOST), source)
        assert chosen == B_STATIONARY and STREAMED not in reason, reason

    def test_a_function_bearing_choice_names_it_too(self, chip, source):
        chosen, reason, _price = _choice(_physics(chip, host=False), source, cost_only=False)
        assert chosen == B_STATIONARY and STREAMED in _named(reason), reason


@pytest.mark.parametrize("store", [None, ST.CHIP, ST.HOST])
def test_a_chip_declaring_its_dma_row_names_nothing(store):
    _chosen, reason, _price = _choice(_physics(GENERIC, store, host=store is not None), "on_chip")
    assert UNPRICED not in reason, reason


@pytest.mark.parametrize("drop", [("e_link_per_byte",), ("e_host_mem_per_byte",)])
@pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
def test_one_link_energy_row_declared_is_named_in_the_lowerings_reason(store, drop):
    """L-4's lowering half, closed by U-J4's targeted pass: the missing row of the two is named."""
    _chosen, reason, _price = _choice(_physics(GENERIC, store, drop=drop), "host")
    assert f"{drop[0]}(the lowering's host-sourced payload" in _named(reason), reason
