"""[U-J5 L-2, U-J4 review m2] The host description declares, per corner, which ends of the host-chip link its
``e_link_per_byte`` row counts, and the link's energy names the chip's own end unpriced only where it is not counted.

U-J4 printed the boundary ("the chip's own end of the host-chip link is unpriced") on every link energy term, keyed on
the row's name, at every corner; the Jetson description's own note says its low and nominal ends count the host's PHY
end alone and its high end counts both ends. Generic form: a host description declares ``link_ends`` for its link row
(``{"rows": {"e_link_per_byte": {"low": [...], "nominal": [...], "high": [...]}}, "basis": ...}``, each corner's
ends among ``host`` and ``chip``); the block's row carries it into the resolved physics (so a record reprices alike),
and the boundary names the chip's end unpriced at the corners where it is not counted. A description (or a hand
block) that declares none keeps U-J4's text at every corner, unchanged."""

from __future__ import annotations

import json
import shutil

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs
from mimarsinan.deployment_record.platform_physics.hosts import description as D
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
from mimarsinan.deployment_record.platform_physics.hosts.registry import description_path
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

from unit.deployment_record.registered_hosts import registered_copy

HOST = "jetson_orin_nano_8gb"
GENERIC = "generic_estimated_22nm"
FIELD = "link_ends"
ROW = "e_link_per_byte"
JETSON_ENDS = {"low": ["host"], "nominal": ["host"], "high": ["host", "chip"]}
#: U-J4's text, printed at every corner where a description declares no link ends.
EVERY_CORNER = ("boundary: the chip's own end of the host-chip link is unpriced (no row of the vocabulary prices it: "
                "a crossing byte pays the link and host memory, never the chip's backing-store rows)")
END = "the chip's own end of the host-chip link"
VALUES = dict(synaptic_events=1e6, cores_physical=8, latency_steps=32, timesteps=4, neurons_used=100, segment_cores=1,
              host_macs=0, compute_op_count=0, host_memory_bound_bytes=0, carry_in_bytes=0, carry_out_bytes=0,
              segment_in_bytes=50, segment_out_bytes=10, host_link_transfers=4)


def _block(ends="keep"):
    """The shipped block, or a hand block whose link row states ``ends`` (a declaration the loader judges)."""
    block = {key: dict(row) for key, row in host_description_block(HOST).items()}
    if ends != "keep":
        block[ROW].pop(FIELD, None)
        if ends is not None:
            block[ROW][FIELD] = ends
    return block


_COPIES = []


@pytest.fixture(autouse=True)
def _copies(tmp_path):
    """[U-HOST2] Where a test's description of other link ends is written; each is taken out when the test ends."""
    from contextlib import ExitStack

    with ExitStack() as stack:
        _COPIES.append((tmp_path, stack))
        yield
        _COPIES.pop()


def _described_block(ends):
    """[U-HOST2] The block of a test-only description: the shipped one with its link row's ``ends`` declared (None:
    no declaration), registered through the registry's contract (a host row's ends are its description's statement,
    so another statement is another description)."""
    def edit(data):
        data[FIELD]["rows"].pop(ROW, None)
        if ends is not None:
            data[FIELD]["rows"][ROW] = ends
        if not data[FIELD]["rows"]:
            del data[FIELD]
    directory, stack = _COPIES[-1]
    tag = "none" if ends is None else "_".join("".join(e[0] for e in ends[c]) for c in sorted(ends))
    name = stack.enter_context(registered_copy(directory / tag, HOST, edit, as_name=f"test_{HOST}_ends_{tag}"))
    return {key: dict(row) for key, row in host_description_block(name).items()}


def _physics(block=None, store=ST.CHIP):
    try:
        return resolve_platform_physics(GENERIC, merge_override_blocks(
            {"host": _block() if block is None else block, "store": ST.store_block(store, "test")})), None
    except Exception as exc:  # noqa: BLE001 - the refusal (or the base's unknown field) is the subject
        return None, exc


def _resolved(block=None, store=ST.CHIP):
    physics, error = _physics(block, store)
    assert physics is not None, f"the host block does not resolve: {error}"
    return physics


def _terms(physics, loads=None, **extra):
    pricing = price_absolute(Quantities({k: QuantityValue(float(v), "static") for k, v in {**VALUES, **extra}.items()}),
                             physics, loads=loads)
    return {t.name: t for t in pricing.terms}


class TestTheJetsonDescriptionDeclaresItsEnds:
    def test_the_description_declares_the_ends_its_note_states(self):
        data = json.loads(description_path(HOST).read_text())
        declared = (data.get(FIELD) or {}).get("rows", {}).get(ROW)
        assert declared == JETSON_ENDS, data.get(FIELD)
        assert "both ends" in data["constants"][ROW]["note"] and "host's end alone" in data["constants"][ROW]["note"]
        assert (data.get(FIELD) or {}).get("basis"), "the declaration states its basis"
        assert FIELD in description_path(HOST).with_suffix(".md").read_text(), "the md states it"

    def test_the_block_row_carries_it_into_the_resolved_physics(self):
        assert host_description_block(HOST)[ROW].get(FIELD) == JETSON_ENDS
        sealed = _resolved().to_dict()
        assert (sealed.get(FIELD) or {}).get(ROW) == JETSON_ENDS, sealed.get(FIELD)
        assert PlatformPhysics.from_dict(sealed).to_dict() == sealed


@pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
class TestTheBoundaryFollowsTheDeclaredEnds:
    def test_the_jetson_link_energy_names_the_chips_end_at_the_low_and_nominal_corners_only(self, store):
        source = _terms(_resolved(store=store))["energy_host_link_mj"].source
        assert EVERY_CORNER not in source, source
        assert f"{END} is unpriced at the low and nominal corners" in source, source
        assert "counts both ends at the high corner" in source, source

    def test_a_description_without_the_declaration_keeps_the_text_at_every_corner(self, store):
        source = _terms(_resolved(_described_block(None), store))["energy_host_link_mj"].source
        assert EVERY_CORNER in source, source

    def test_both_ends_counted_at_every_corner_names_nothing(self, store):
        both = {corner: ["host", "chip"] for corner in ("low", "nominal", "high")}
        terms = _terms(_resolved(_described_block(both), store))
        assert not any(END in term.source for term in terms.values()), terms["energy_host_link_mj"].source


def test_a_program_load_over_the_link_names_the_same_corners():
    """[U-ENERGY] The program load is the one load law over the program's census (1000 B rewritten per round)."""
    from unit.deployment_record.load_census_fixtures import written_program

    terms = _terms(_resolved(store=ST.HOST), loads=written_program((500,), (500,)))
    source = terms["energy_programming_mj"].source
    assert f"{END} is unpriced at the low and nominal corners" in source and EVERY_CORNER not in source, source


def test_the_headline_carries_the_link_terms_text():
    source = _terms(_resolved())["energy_per_inference_mj"].source
    assert f"{END} is unpriced at the low and nominal corners" in source, source


def test_the_values_do_not_move():
    """The declaration changes text only: every term's band equals the undeclared block's."""
    census = PassPeriodCensus(passes=(PassCensus(position=0, segment_index=0, resident=False, cores=1, resident_cores=0,
                                                 written_cells_backing_store=0, written_cells_producer=10,
                                                 payload_bytes=0, compute_steps=3),),
                              boundaries=())
    writes = WriteInputs(census, WritePolicy(batch_k=1, producer_write_source="host", payload_bits=8,
                                             write_granularity="cell"))
    declared, plain = _resolved(), _resolved(_described_block(None))
    q = Quantities({k: QuantityValue(float(v), "static") for k, v in VALUES.items()})
    a = {t.name: t.band for t in price_absolute(q, declared, writes=writes).terms}
    b = {t.name: t.band for t in price_absolute(q, plain, writes=writes).terms}
    assert a.keys() == b.keys()
    for name, band in a.items():
        assert (band.low, band.nominal, band.high) == (b[name].low, b[name].nominal, b[name].high), name


class TestARefusedDeclaration:
    @pytest.mark.parametrize("ends, why", [
        ({"low": ["host"], "nominal": ["host"]}, "high"),
        ({"low": ["host"], "nominal": ["host"], "high": ["host", "router"]}, "router"),
        ({"low": [], "nominal": ["host"], "high": ["host"]}, "low"),
        ({"low": ["host", "host"], "nominal": ["host"], "high": ["host"]}, "low"),
        (["host"], "corner"),
    ])
    def test_an_ill_formed_declaration_refuses_by_name(self, ends, why):
        physics, error = _physics(_block(ends=ends))
        assert physics is None and FIELD in str(error) and why in str(error), error
        assert "unknown fields" not in str(error), error  # refused for what it states, not as an unknown field

    def test_a_declaration_on_a_row_that_prices_no_link_end_refuses(self):
        block = _block()
        block["e_host_mem_per_byte"][FIELD] = JETSON_ENDS
        physics, error = _physics(block)
        assert physics is None and FIELD in str(error) and "e_host_mem_per_byte" in str(error), error
        assert "unknown fields" not in str(error), error

    def test_the_description_refuses_ends_for_a_row_that_prices_no_link_end(self, tmp_path):
        """[U-HOST2] The edited copy enters through the registry's contract, whose judge refuses it."""
        def edit(data):
            data[FIELD] = {"rows": {"p_host": JETSON_ENDS}, "basis": "test"}
        try:
            with registered_copy(tmp_path, HOST, edit):
                pass
            error = None
        except Exception as exc:  # noqa: BLE001 - the refusal is the subject
            error = exc
        assert error is not None and FIELD in str(error) and "p_host" in str(error), error
