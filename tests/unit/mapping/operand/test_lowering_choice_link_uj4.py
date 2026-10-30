"""[U-J4 targeted pass, review M1] N2's cost-only lowering choice under the link rule, priced by hand.

The writes never order the two lowerings (both place the same cells, ``placed_operand_cells``), and a host-sourced
payload moves the same bytes under either, so the streamed vectors do: they are injected at the carry's per-byte energy
on the declared store (``hosts.store.injection_byte_energy``). Where both lowerings price equal, the tie rule picks
fewer bank cells, which is the lowering streaming more (``operand_form.stationary_operand``). U-J4 prices the injected
vectors on a host-owned store at the link and host memory, where they were at the chip's ``e_dma_per_byte`` before, so
the choice moves exactly where one of the two per-byte energies is zero (its rows undeclared) and the other is not:

- a chip that declares no ``e_dma_per_byte``, on a host-owned store beside the edge host: from A^T stationary (the tie
  at the write energy) to B stationary (it streams less), because the injected carry now pays the link and host memory;
- a chip that declares it, on a host-owned store beside a hand block declaring neither link energy row: from B to A^T,
  and the lowering's printed reason names both rows unpriced (never a silent 0, as on the headline).

On a chip-owned store, and without the link, nothing moves. Vehicle: one cost-only product (batch 2, m 3, k 4, n 5):
B stationary holds 40 bank cells and streams 24 elements, A^T stationary 24 and 40, at 8-bit activations (1 B each)."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.platform_physics import available_profiles
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.mapping.support.operand_form import (
    AT_STATIONARY, B_STATIONARY, Lowering, OperandForm, stationary_operand,
)
from mimarsinan.mapping.support.operand_price import lowering_price

HOST = "jetson_orin_nano_8gb"
GENERIC = "generic_estimated_22nm"
FORM = OperandForm(batch=(2,), m=3, k=4, n=5, a_perm=(0, 1, 2), b_perm=(0, 1, 2), out_perm=(0, 1, 2))
B, AT = Lowering(FORM, B_STATIONARY), Lowering(FORM, AT_STATIONARY)
WRITE = {"t_write_per_unit": {"nominal": 10.0, "unit": "ns", "evidence_kind": "estimated", "note": "test constant"}}
LINK_ENERGY = ("e_link_per_byte", "e_host_mem_per_byte")
SOURCES = ("host", "on_chip")
#: The shipped profiles that declare no e_dma_per_byte (pinned by ``test_the_profiles_without_a_dma_row``).
WITHOUT_DMA = ("hermes_64", "loihi", "neurram", "odin", "sram_cim_macro", "truenorth")
UNPRICED = "; unpriced: "


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


def _choice(physics, source):
    price = lowering_price(_platform(physics, source))
    chosen, reason = stationary_operand(FORM, price, True)
    return chosen.stationary, reason, price


def _named(reason):
    return reason.rsplit(UNPRICED, 1)[1] if UNPRICED in reason else ""


def test_the_vehicles_two_lowerings_order_bank_cells_and_streamed_vectors_oppositely():
    assert (B.cells, B.streamed_elements, AT.cells, AT.streamed_elements) == (40, 24, 24, 40)


def test_the_profiles_without_a_dma_row():
    """The set the moved choice reaches among the shipped profiles (a new profile states its row or joins it)."""
    assert tuple(p for p in available_profiles() if not _physics(p, host=False).has("e_dma_per_byte")) == WITHOUT_DMA


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("chip", WITHOUT_DMA)
class TestAChipWithoutADmaRow:
    """No ``e_dma_per_byte``: the injected carry is unpriced on the chip's own store (the tie rule decides, as on
    RC1), and on a host-owned store beside the link it pays the link and host memory, so the choice moves to B."""

    def test_on_a_host_owned_store_the_injected_carry_moves_the_choice_to_b(self, chip, source):
        physics = _physics(chip, ST.HOST)
        chosen, reason, price = _choice(physics, source)
        per_byte = sum(physics.band(key).nominal for key in LINK_ENERGY)
        assert chosen == B_STATIONARY, reason
        assert price(AT) - price(B) == pytest.approx((AT.streamed_elements - B.streamed_elements) * 1.0 * per_byte,
                                                     rel=1e-9)
        assert UNPRICED not in reason  # every row the price reads is declared

    def test_on_the_chips_own_store_the_tie_rule_still_decides(self, chip, source):
        chosen, reason, price = _choice(_physics(chip, ST.CHIP), source)
        assert chosen == AT_STATIONARY and price(B) == price(AT), reason

    def test_without_the_link_the_tie_rule_still_decides(self, chip, source):
        """[U-J5 L-6] The tie still decides; the reason now names the streamed vectors' undeclared e_dma_per_byte
        (``tests/unit/mapping/operand/test_lowering_unpriced_rows_uj5.py``), and nothing else."""
        chosen, reason, price = _choice(_physics(chip, host=False), source)
        assert chosen == AT_STATIONARY and price(B) == price(AT)
        assert _named(reason).startswith("e_dma_per_byte(the lowering's streamed vectors") and "e_link" not in reason
        # [chain 5b test-pin amendment] The transfer part is zero: the head declares NeuRRAM's e_write_per_cell
        # (480828a8, after the unit's base), which prices both lowerings' cells alike; without it both are 0.0.
        bare = _platform(_physics(chip, host=False), source)
        bare["platform_physics_resolved"]["constants"].pop("e_write_per_cell", None)
        transfer = lowering_price(bare)
        assert transfer(B) == transfer(AT) == 0.0, (transfer(B), transfer(AT))


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("store", [None, ST.CHIP, ST.HOST])
def test_a_chip_declaring_its_dma_row_keeps_b_on_every_store(store, source):
    """With ``e_dma_per_byte`` and the link rows both declared, the injection is positive on either store: B."""
    chosen, reason, _price = _choice(_physics(GENERIC, store, host=store is not None), source)
    assert chosen == B_STATIONARY and UNPRICED not in reason, reason


class TestAnUndeclaredLinkEnergyRowInTheLoweringPrice:
    """Beside the link, a byte the lowering's price reads at the link and host memory (its host-sourced payload on
    either store, its streamed vectors on a host-owned one) whose link energy row the physics does not declare (a
    hand block; a host description declares every row, ``REQUIRED_HOST_ROWS``) is named unpriced in the printed
    reason, as on the headline, never read as a silent 0."""

    @pytest.mark.parametrize("source", SOURCES)
    def test_neither_row_on_a_host_owned_store_moves_the_choice_to_the_tie_and_names_both(self, source):
        physics = _physics(GENERIC, ST.HOST, drop=LINK_ENERGY)
        chosen, reason, price = _choice(physics, source)
        assert chosen == AT_STATIONARY and price(B) == price(AT), reason
        named = _named(reason)
        for key in LINK_ENERGY:
            assert f"{key}(the lowering's streamed vectors" in named, (key, reason)
            assert (f"{key}(the lowering's host-sourced payload" in named) == (source == "host"), (key, reason)

    @pytest.mark.parametrize("drop", [("e_link_per_byte",), ("e_host_mem_per_byte",), LINK_ENERGY])
    @pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
    def test_the_host_sourced_payload_names_each_undeclared_row(self, store, drop):
        _chosen, reason, _price = _choice(_physics(GENERIC, store, drop=drop), "host")
        named = _named(reason)
        for key in LINK_ENERGY:
            assert (f"{key}(the lowering's host-sourced payload" in named) == (key in drop), (key, reason)

    @pytest.mark.parametrize("drop", [("e_link_per_byte",), ("e_host_mem_per_byte",)])
    def test_one_row_on_a_host_owned_store_names_it_and_b_stands(self, drop):
        chosen, reason, _price = _choice(_physics(GENERIC, ST.HOST, drop=drop), "on_chip")
        assert chosen == B_STATIONARY, reason
        assert _named(reason).startswith(f"{drop[0]}(the lowering's streamed vectors"), reason

    def test_nothing_crossing_names_nothing(self):
        """An on-chip write-back on the chip's own store moves no byte over the link: nothing is named."""
        chosen, reason, _price = _choice(_physics(GENERIC, ST.CHIP, drop=LINK_ENERGY), "on_chip")
        assert chosen == B_STATIONARY and UNPRICED not in reason, reason

    @pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
    def test_a_declared_row_is_never_named(self, store):
        _chosen, reason, _price = _choice(_physics(GENERIC, store), "host")
        assert UNPRICED not in reason, reason
