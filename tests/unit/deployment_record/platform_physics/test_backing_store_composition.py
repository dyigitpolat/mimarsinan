"""[consts] A backing store composes with every chip description: one rule, no chip named.

A backing-store block (CL5's DDR4, HBM2 and 1 MB on-chip SRAM blocks) declares the store's side of the programming
link: its I/O energy per byte (``e_dma_per_byte``) and its channel time per byte (``t_program_per_byte``). A chip that
declares its own programming port (TrueNorth's 10 MHz scan chain, the donor-scaled ports of Loihi, ODIN and
ISAAC-like, the 22 nm point's estimated port) is the other stage of that link, and a streamed payload moves no faster
than its slower stage: the store's channel time composes with the chip's port by ``combine: slower`` (corner by
corner, the larger time), where a plain override would replace the port and price TrueNorth's reload through a DDR4
channel 20,000 times faster than its scan chain. A chip without its own port (the SRAM-CIM macro's external component,
NeuRRAM, HERMES-64) takes the store's time alone. The store's I/O energy replaces a chip's own transfer energy (the
default composition): the vocabulary prices one channel energy per byte.

What the store cannot supply stays refused by name: NeuRRAM and HERMES-64 state no commit energy, so a program that
rewrites their cells prices its transfer and refuses its write energy.

Every constant here is a test constant (``(test)`` in its citation or note).
"""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost.absolute.writes import ENERGY_WRITE, LATENCY_WRITE, price_write_terms
from mimarsinan.deployment_record.platform_physics import apply_overrides, available_profiles, get_platform_physics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

from unit.deployment_record.g3_fixtures import constant
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block, host_descriptions

#: A DDR4-like store: I/O energy per byte and a channel time per byte (test constants).
_DDR4 = {"e_dma_per_byte": {"nominal": 162.5, "unit": "pJ", "evidence_kind": "published",
                            "citation": "DDR3/4 I/O per 64 bits over 8 bytes (test)"},
         "t_program_per_byte": {"low": 0.0278, "nominal": 0.0393, "high": 0.0556, "unit": "ns",
                                "evidence_kind": "published", "citation": "DDR4 channel throughput (test)",
                                "combine": "slower"}}
#: A slow store whose channel is slower than some ports at some corners.
_SLOW = {"t_program_per_byte": {"low": 10.0, "nominal": 100.0, "high": 500.0, "unit": "ns",
                                "evidence_kind": "published", "citation": "a slow serial store (test)",
                                "combine": "slower"}}
#: [U-HOST2] A host's power and rate: the first read description's own rows, as its block declares them.
_HOST = {key: row for key, row in host_description_block(host_descriptions()[0]).items()
         if key in ("p_host", "host_macs_per_s")}
#: The shipped chips that state no commit energy (``e_write_per_cell`` or ``e_program_per_byte``).
#: [U-C3, B3] NeuRRAM declares its derived write energy per pulse; HERMES-64's is not determinable from its primaries.
_NO_COMMIT_ENERGY = {"hermes_64"}


def _ends(band):
    return band.low, band.nominal, band.high


def _composed(name, store=_DDR4):
    return resolve_platform_physics(name, merge_override_blocks({"host": _HOST, "backing_store": store}))


def _rotating_census():
    """Two passes on one core, each rewriting 4,096 cells (4,096 B at 8 bits) every period."""
    passes = tuple(PassCensus(position=q, segment_index=0, resident=False, cores=1, resident_cores=0,
                              written_cells_backing_store=4096, written_cells_producer=0, payload_bytes=4096,
                              compute_steps=4) for q in range(2))
    return PassPeriodCensus(passes=passes, label="test: two rotating passes on one core")


class TestTheSlowerComposition:
    def test_the_store_channel_bounds_the_ports_time_corner_by_corner(self):
        port = get_platform_physics("generic_estimated_22nm")          # its port: 4 / 80 / 1000 ns per byte
        own = _ends(port.band("t_program_per_byte"))
        fast = apply_overrides(port, {"t_program_per_byte": _DDR4["t_program_per_byte"]})
        assert _ends(fast.band("t_program_per_byte")) == pytest.approx(own)     # the port is the slower stage
        slow = apply_overrides(port, _SLOW)
        assert _ends(slow.band("t_program_per_byte")) == pytest.approx((10e-9, 100e-9, 1000e-9))

    def test_a_chip_without_its_own_port_takes_the_store_channel(self):
        macro = apply_overrides(get_platform_physics("sram_cim_macro"), _DDR4)
        assert _ends(macro.band("t_program_per_byte")) == pytest.approx((0.0278e-9, 0.0393e-9, 0.0556e-9))

    def test_the_composed_value_states_both_stages_and_is_marked_a_departure(self):
        value = apply_overrides(get_platform_physics("truenorth"), _DDR4).constants["t_program_per_byte"]
        assert value.overridden and value.evidence_kind == "derived"
        assert "slower" in value.derivation
        assert "DDR4 channel throughput (test)" in value.derivation and "scan chain" in value.derivation

    def test_replace_is_the_default_composition(self):
        plain = {k: {kk: vv for kk, vv in v.items() if kk != "combine"} for k, v in _DDR4.items()}
        replaced = apply_overrides(get_platform_physics("truenorth"), plain)
        assert _ends(replaced.band("t_program_per_byte")) == pytest.approx((0.0278e-9, 0.0393e-9, 0.0556e-9))

    @pytest.mark.parametrize("declaration, match", [
        ({"e_dma_per_byte": {"nominal": 1.0, "unit": "pJ", "note": "t", "combine": "slower"}}, "time"),
        ({"t_program_per_byte": {"nominal": 1.0, "unit": "ns", "note": "t", "combine": "faster"}}, "combine"),
    ], ids=["slower-on-an-energy", "unknown-combine"])
    def test_a_combine_the_rule_cannot_apply_refuses_by_name(self, declaration, match):
        with pytest.raises(ValueError, match=match):
            apply_overrides(get_platform_physics("truenorth"), declaration)

    def test_the_combine_field_survives_the_block_merge(self):
        merged = merge_override_blocks({"backing_store": _DDR4})
        assert merged["t_program_per_byte"]["combine"] == "slower"
        assert merged["t_program_per_byte"]["note"] == "[backing_store]"


class TestTheDeclarationFields:
    """[fix round, review R3] A declaration states only the closed field set; a field outside it refuses by name, so
    a misspelled ``combine`` can never silently replace a chip's port."""

    @pytest.mark.parametrize("field", ["combin", "source_ref", "measured_on"])
    def test_a_field_outside_the_declaration_refuses_by_name(self, field):
        declaration = {k: v for k, v in _DDR4["t_program_per_byte"].items() if k != "combine"}
        declaration[field] = "slower"
        with pytest.raises(ValueError, match=rf"unknown fields \['{field}'\]"):
            apply_overrides(get_platform_physics("truenorth"), {"t_program_per_byte": declaration})

    def test_every_field_of_the_closed_set_is_read(self):
        declaration = {"low": 1.0, "nominal": 2.0, "high": 3.0, "unit": "ns", "evidence_kind": "derived",
                       "citation": "c (test)", "derivation": "d (test)", "note": "n (test)", "combine": "replace"}
        value = apply_overrides(get_platform_physics("truenorth"), {"t_program_per_byte": declaration}
                                ).constants["t_program_per_byte"]
        assert (value.low, value.nominal, value.high, value.unit) == (1.0, 2.0, 3.0, "ns")
        assert (value.citation, value.derivation, value.note) == ("c (test)", "d (test)", "n (test)")


class TestTheComposedEvidence:
    """[fix round, review R4] The composition is arithmetic (derived), and an arithmetic cannot strengthen its
    inputs: where an estimated stage prevails at some corner, the composed value stays estimated."""

    def test_an_estimated_port_that_prevails_keeps_its_class(self):
        value = apply_overrides(get_platform_physics("generic_estimated_22nm"),
                                {"t_program_per_byte": _DDR4["t_program_per_byte"]}).constants["t_program_per_byte"]
        assert value.evidence_kind == "estimated" and value.note.strip()
        assert "slower" in value.derivation and "pipelined" in value.derivation

    def test_the_class_is_the_weakest_of_the_stages_that_prevail(self):
        """The slow store (published) prevails at the low and nominal corners, the 22 nm port (estimated) at the
        high one."""
        value = apply_overrides(get_platform_physics("generic_estimated_22nm"), _SLOW).constants["t_program_per_byte"]
        assert _ends(value.band) == pytest.approx((10e-9, 100e-9, 1000e-9))
        assert value.evidence_kind == "estimated"

    def test_a_published_store_that_prevails_at_every_corner_composes_as_derived(self):
        store = {"t_program_per_byte": {"low": 5e3, "nominal": 6e3, "high": 7e3, "unit": "ns",
                                        "evidence_kind": "published", "citation": "a very slow store (test)",
                                        "combine": "slower"}}
        value = apply_overrides(get_platform_physics("generic_estimated_22nm"), store).constants["t_program_per_byte"]
        assert _ends(value.band) == pytest.approx((5e-6, 6e-6, 7e-6))
        assert value.evidence_kind == "derived"


class TestEveryChipComposesWithEveryStore:
    @pytest.mark.parametrize("name", available_profiles())
    @pytest.mark.parametrize("store", [_DDR4, {**_DDR4, **_SLOW}], ids=["fast-store", "slow-store"])
    def test_every_shipped_chip_takes_the_store_by_one_rule(self, name, store):
        own = get_platform_physics(name)
        physics = _composed(name, store)
        assert physics.undeclared_external() == ()
        assert {"e_dma_per_byte", "t_program_per_byte", "p_host"} <= set(physics.overridden_keys())
        assert _ends(physics.band("e_dma_per_byte")) == pytest.approx((162.5e-12,) * 3)    # the store's I/O
        channel = apply_overrides(get_platform_physics("sram_cim_macro"),
                                  {"t_program_per_byte": store["t_program_per_byte"]}).band("t_program_per_byte")
        expected = (tuple(max(a, b) for a, b in zip(_ends(own.band("t_program_per_byte")), _ends(channel)))
                    if own.has("t_program_per_byte") else _ends(channel))
        assert _ends(physics.band("t_program_per_byte")) == pytest.approx(expected)

    def test_truenorths_reload_keeps_its_scan_chain_behind_a_ddr4_store(self):
        """The serial write wall of a rotating census: 4,096 B per pass at TrueNorth's 800 ns per byte, not at the
        store's channel."""
        policy = WritePolicy(batch_k=1, overlap_mode="serial", physical_cores=1)
        pricing = price_write_terms(_rotating_census(), _composed("truenorth"), policy, most_written=2)
        wall = {t.name: t for t in pricing.terms}[LATENCY_WRITE].band
        assert wall.nominal == pytest.approx(2 * 4096 * 800e-9, rel=1e-9)

    @pytest.mark.parametrize("name", available_profiles())
    def test_the_store_prices_the_transfer_and_a_missing_commit_energy_stays_refused(self, name):
        pricing = price_write_terms(_rotating_census(), _composed(name), WritePolicy(batch_k=1), most_written=2)
        refused = {r.name: r.reason for r in pricing.refusals}
        energy = {t.name: t for t in pricing.terms}.get(ENERGY_WRITE)
        if name in _NO_COMMIT_ENERGY:
            assert energy is None
            assert refused[ENERGY_WRITE].startswith("cells are written per inference but the target declares no "
                                                    "commit energy")
        else:
            assert energy is not None, refused
            assert "e_dma_per_byte x period_rewrite_bytes / K" in energy.band.basis
            assert "unpriced: e_dma_per_byte" not in energy.band.basis

    def test_the_chips_without_a_commit_energy_are_exactly_hermes(self):
        missing = {name for name in available_profiles()
                   if not {"e_write_per_cell", "e_program_per_byte"} & set(get_platform_physics(name).constants)}
        assert missing == _NO_COMMIT_ENERGY
