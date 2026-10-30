"""[U-J J-1] The edge host priced by hand: one host operator and one boundary transfer, corner by corner.

The edge host block adds what the server's block never priced: the host's idle power over the part of each
inference it waits (on the chip and on the link), and the host-chip link. Every byte that crosses the boundary, in
either direction (the pass-boundary carry and the segment I/O at host operators, the network input and its readout),
is priced at the link and at host memory: energy per byte at both, time at the slower stage per byte plus a fixed
latency per transfer. Host time (operator dispatch, MACs at the host rate, and a MAC-free operator's bytes at the
host memory time) adds to the latency and, through it, to the static energy, as the model already composes it.
Without the edge rows nothing moves: no term, no source, no quantity.
[U-J3, amended] The pass carry and the write law's store payload cross the link only to a host-owned store (U-J's
model), priced once there; on a chip-owned store the chip's rows price them (``test_store_routing_uj3``). The pins that
price store traffic over the link declare the host-owned store (``EDGE_HOST_STORE``); beside the link a chip that
states no store refuses its store traffic by name."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
from mimarsinan.deployment_record.platform_physics.hosts.store import HOST as HOST_STORE, store_block
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

CORNERS = ("low", "nominal", "high")
EDGE = resolve_platform_physics("generic_estimated_22nm", merge_override_blocks(
    {"host": host_description_block("jetson_orin_nano_8gb")}))
CHIP = resolve_platform_physics("generic_estimated_22nm", None)
#: [U-J3, amended] The edge host beside a host-owned store: every crossing byte, store traffic included, on the link.
EDGE_HOST_STORE = resolve_platform_physics("generic_estimated_22nm", merge_override_blocks(
    {"host": host_description_block("jetson_orin_nano_8gb"), "store": store_block(HOST_STORE, "U-J's model (test)")}))
BASE = dict(synaptic_events=1e6, cores_physical=8, latency_steps=32, timesteps=4, neurons_used=100, segment_cores=1)
NO_LINK = dict(carry_in_bytes=0, carry_out_bytes=0, segment_in_bytes=0, segment_out_bytes=0, host_link_transfers=0)


def _q(**values):
    return Quantities({key: QuantityValue(float(value), "static") for key, value in values.items()})


def _terms(physics, **values):
    pricing = price_absolute(_q(**values), physics)
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


def _c(band, corner):
    return getattr(band, corner)


def _k(key, corner):
    return getattr(EDGE.band(key), corner)


def _rate(corner):
    """The host MAC rate (MACs per second) is a dividing constant: the low time corner reads its high end."""
    return getattr(EDGE.band("host_macs_per_s"), {"low": "high", "nominal": "nominal", "high": "low"}[corner])


class TestOneHostOperator:
    """A MAC-free host operator (a pool): one dispatch and its bytes at the host memory time."""

    VALUES = dict(BASE, host_macs=0, compute_op_count=1, host_memory_bound_bytes=6656, **NO_LINK)

    def test_its_host_time_is_the_dispatch_plus_its_bytes_at_the_memory_time(self):
        terms, _ = _terms(EDGE, **self.VALUES)
        host = terms["latency_host_s"].band
        for corner in CORNERS:
            expected = _k("t_host_op_overhead", corner) + 6656 * _k("t_host_mem_per_byte", corner)
            assert _c(host, corner) == pytest.approx(expected, rel=1e-12), corner
        assert "t_host_mem_per_byte x host_memory_bound_bytes" in terms["latency_host_s"].source

    def test_host_time_adds_to_the_latency_and_to_the_static_energy(self):
        edge, _ = _terms(EDGE, **self.VALUES)
        chip, _ = _terms(CHIP, **self.VALUES)
        for corner in CORNERS:
            e2e = _c(edge["e2e_latency_s"].band, corner)
            assert e2e == pytest.approx(_c(chip["e2e_latency_s"].band, corner)
                                        + _c(edge["latency_host_s"].band, corner), rel=1e-12)
            static_power = _c(chip["energy_static_mj"].band, corner) / _c(chip["e2e_latency_s"].band, corner)
            assert _c(edge["energy_static_mj"].band, corner) == pytest.approx(static_power * e2e, rel=1e-12)

    def test_its_energy_is_active_power_over_its_time_and_idle_power_over_the_wait(self):
        """[U-J2 K-1, amended] The host's work above idle is the headline's host energy; the idle power over the
        wait (and under the work) is the host's own draw, in the whole-system line (owner answer of 2026-10-29).
        [U-J2 targeted M1, amended] The work's band ends are the cost ends of p_host - p_host_idle (low: p_host's low
        less p_host_idle's high, never below zero; high: p_host's high less p_host_idle's low); the system line is the
        chip plus the whole host at its corners (p_host over the host's time, p_host_idle over the wait)."""
        terms, _ = _terms(EDGE, **self.VALUES)
        work_power = {"low": max(0.0, _k("p_host", "low") - _k("p_host_idle", "high")),
                      "nominal": _k("p_host", "nominal") - _k("p_host_idle", "nominal"),
                      "high": _k("p_host", "high") - _k("p_host_idle", "low")}
        for corner in CORNERS:
            busy = _c(terms["latency_host_s"].band, corner)
            e2e = _c(terms["e2e_latency_s"].band, corner)
            idle = _k("p_host_idle", corner) * (e2e - busy) * 1e3
            assert _c(terms["energy_host_idle_mj"].band, corner) == pytest.approx(idle, rel=1e-9)
            assert _c(terms["energy_host_mj"].band, corner) == pytest.approx(
                work_power[corner] * busy * 1e3, rel=1e-12, abs=1e-18)
            chip = _c(terms["energy_dynamic_mj"].band, corner) + _c(terms["energy_static_mj"].band, corner)
            assert _c(terms["energy_system_mj"].band, corner) == pytest.approx(
                chip + _k("p_host", corner) * busy * 1e3 + idle, rel=1e-9)
        assert "energy_host_link_mj" not in terms  # no byte crossed

    def test_the_headline_is_the_chip_plus_the_host(self):
        terms, _ = _terms(EDGE, **self.VALUES)
        for corner in CORNERS:
            parts = sum(_c(terms[name].band, corner)
                        for name in ("energy_dynamic_mj", "energy_static_mj", "energy_host_mj"))
            assert _c(terms["energy_per_inference_mj"].band, corner) == pytest.approx(parts, rel=1e-12)

    def test_host_macs_are_priced_at_the_host_rate_beside_the_dispatch(self):
        terms, _ = _terms(EDGE, **dict(self.VALUES, host_macs=1_000_000))
        for corner in CORNERS:
            expected = (1e6 / _rate(corner) + _k("t_host_op_overhead", corner)
                        + 6656 * _k("t_host_mem_per_byte", corner))
            assert _c(terms["latency_host_s"].band, corner) == pytest.approx(expected, rel=1e-12), corner


class TestOneBoundaryTransfer:
    """One readout of 1,000 carried bytes off the chip: the link and host memory, one fixed latency. [U-J3, amended]
    The carry crosses the link to a host-owned store (``EDGE_HOST_STORE``)."""

    VALUES = dict(BASE, host_macs=0, compute_op_count=0, host_memory_bound_bytes=0,
                  **dict(NO_LINK, carry_out_bytes=1000, host_link_transfers=1))

    def test_its_time_is_the_slower_stage_per_byte_plus_one_transfer_latency(self):
        terms, _ = _terms(EDGE_HOST_STORE, **self.VALUES)
        link = terms["latency_host_link_s"].band
        for corner in CORNERS:
            per_byte = max(_k("t_link_per_byte", corner), _k("t_host_mem_per_byte", corner))
            assert _c(link, corner) == pytest.approx(1000 * per_byte + _k("t_link_transfer", corner), rel=1e-12)

    def test_its_energy_is_the_link_and_host_memory_per_byte(self):
        """[U-J2 K-1, amended] The link and host memory are each their own term."""
        terms, _ = _terms(EDGE_HOST_STORE, **self.VALUES)
        for corner in CORNERS:
            assert _c(terms["energy_host_link_mj"].band, corner) == pytest.approx(
                1000 * _k("e_link_per_byte", corner) * 1e3, rel=1e-12)
            assert _c(terms["energy_host_mem_mj"].band, corner) == pytest.approx(
                1000 * _k("e_host_mem_per_byte", corner) * 1e3, rel=1e-12)

    def test_the_link_time_adds_to_the_latency_and_the_host_idles_over_it(self):
        edge, _ = _terms(EDGE_HOST_STORE, **self.VALUES)
        chip, _ = _terms(CHIP, **self.VALUES)
        for corner in CORNERS:
            e2e = _c(edge["e2e_latency_s"].band, corner)
            assert e2e == pytest.approx(_c(chip["e2e_latency_s"].band, corner)
                                        + _c(edge["latency_host_link_s"].band, corner), rel=1e-12)
            # [U-J2 K-1, amended] the host idles over the whole inference in the system line; the headline's host
            # energy is the bytes' link and host memory
            idle = _k("p_host_idle", corner) * e2e * 1e3
            assert _c(edge["energy_host_idle_mj"].band, corner) == pytest.approx(idle, rel=1e-12)
            assert _c(edge["energy_host_mj"].band, corner) == pytest.approx(
                _c(edge["energy_host_link_mj"].band, corner) + _c(edge["energy_host_mem_mj"].band, corner), rel=1e-12)

    def test_the_carry_is_priced_once_on_the_host_owned_store(self):
        """[U-J3, amended] Formerly ``test_the_chips_own_end_of_the_boundary_stays_the_chips``: it pinned the readout
        on the chip's e_readout_per_byte beside the link and host memory, the double count U-J3 removes. On the
        host-owned store the chip's dynamic energy carries no readout; the link and host memory price it."""
        edge, _ = _terms(EDGE_HOST_STORE, **self.VALUES)
        chip, _ = _terms(CHIP, **self.VALUES)
        for corner in CORNERS:
            readout = 1000 * _k("e_readout_per_byte", corner) * 1e3
            assert _c(edge["energy_dynamic_mj"].band, corner) == pytest.approx(
                _c(chip["energy_dynamic_mj"].band, corner) - readout, rel=1e-9)


def test_every_byte_crosses_in_both_directions_at_the_link_and_host_memory():
    values = dict(BASE, host_macs=0, compute_op_count=0, host_memory_bound_bytes=0, carry_in_bytes=300,
                  carry_out_bytes=200, segment_in_bytes=50, segment_out_bytes=10, host_link_transfers=4)
    terms, _ = _terms(EDGE_HOST_STORE, **values)  # [U-J3, amended] the carry over the link: a host-owned store
    for corner in CORNERS:  # [U-J2 K-1, amended] the link and host memory each their own term
        assert _c(terms["energy_host_link_mj"].band, corner) == pytest.approx(
            560 * _k("e_link_per_byte", corner) * 1e3, rel=1e-12)
        assert _c(terms["energy_host_mem_mj"].band, corner) == pytest.approx(
            560 * _k("e_host_mem_per_byte", corner) * 1e3, rel=1e-12)
        per_byte = max(_k("t_link_per_byte", corner), _k("t_host_mem_per_byte", corner))
        time = 560 * per_byte + 4 * _k("t_link_transfer", corner)
        assert _c(terms["latency_host_link_s"].band, corner) == pytest.approx(time, rel=1e-12)


def test_a_view_without_the_boundary_census_names_the_link_unpriced():
    terms, _ = _terms(EDGE_HOST_STORE,  # [U-J3, amended] the carry over the link: a host-owned store
                      **dict(BASE, host_macs=0, compute_op_count=1, carry_in_bytes=100, carry_out_bytes=100))
    assert "e_link_per_byte(segment I/O" in terms["energy_per_inference_mj"].source
    assert "t_link_transfer(" in terms["e2e_latency_s"].source
    assert "t_host_mem_per_byte(" in terms["e2e_latency_s"].source


def test_the_edge_host_never_reads_a_measured_wall_of_another_machine():
    """A sealed record's host wall was measured on the machine that ran the deployment, not on this host: an
    identified host prices its own time from its declared rows."""
    values = dict(BASE, host_macs=1_000_000, compute_op_count=3, host_ops_s=0.002, **NO_LINK)
    terms, _ = _terms(EDGE, **values)
    for corner in CORNERS:
        expected = 1e6 / _rate(corner) + 3 * _k("t_host_op_overhead", corner)
        assert _c(terms["latency_host_s"].band, corner) == pytest.approx(expected, rel=1e-12), corner


class TestAServerBlockKeepsTheMeasuredWallsRule:
    """[targeted M1] Only a host that describes its own whole model (a host description's block: compute, active and
    idle power, memory, the link) prices its own time beside a measured wall. A server's measured calibration block
    (CL0's: rate, dispatch, power, no ``host_compute_rate``) on a profile that relates no machine to the deployment
    host keeps the measured wall's rule as before: host MACs refuse by name for ``host_compute_rate``, and a
    dispatch-only record prices no host time."""

    @staticmethod
    def _server_block():
        from mimarsinan.deployment_record.platform_physics.host_calibration import calibration_overrides, host_block

        p_host = {"p_host": {"low": 5.0, "nominal": 37.4, "high": 280.0, "unit": "W", "evidence_kind": "published",
                             "citation": "a band over two hosts (test)"}}
        measured = calibration_overrides(macs_per_s=1e9, op_overhead_s=(1e-5, 2e-5, 4e-5), p_host_w=None,
                                         identity="the experiment server (test)", declared=p_host)
        return merge_override_blocks({"host": host_block({"platform_physics_overrides": measured})})

    RECORD = dict(BASE, compute_op_count=3, host_ops_s=0.002)

    @pytest.mark.parametrize("profile", ["neurram", "hermes_64"])
    def test_host_macs_on_a_profile_without_the_relation_refuse_by_name(self, profile):
        server = resolve_platform_physics(profile, self._server_block())
        assert not server.has("host_compute_rate")
        terms, refused = _terms(server, **dict(self.RECORD, host_macs=1_000_000))
        assert "latency_host_s" not in terms
        assert "pricing it needs host_compute_rate" in refused["energy_per_inference_mj"]

    @pytest.mark.parametrize("profile", ["neurram", "hermes_64"])
    def test_a_dispatch_only_record_prices_no_host_time(self, profile):
        server = resolve_platform_physics(profile, self._server_block())
        terms, _ = _terms(server, **dict(self.RECORD, host_macs=0))
        assert "latency_host_s" not in terms

    def test_the_edge_host_describes_its_own_host_and_the_server_block_does_not(self):
        from mimarsinan.deployment_record.platform_physics.hosts.description import describes_host

        assert describes_host(EDGE) and not describes_host(CHIP)
        assert not describes_host(resolve_platform_physics("neurram", self._server_block()))


def test_without_the_edge_rows_nothing_moves():
    """The chip's own physics with the edge host's quantities in the view: the terms and their sources are the
    terms of the view without them."""
    plain = dict(BASE, host_macs=0, compute_op_count=1, carry_in_bytes=100, carry_out_bytes=100)
    with_census = dict(plain, segment_in_bytes=50, segment_out_bytes=10, host_link_transfers=4,
                       host_memory_bound_bytes=6656)
    before, refused_before = _terms(CHIP, **plain)
    after, refused_after = _terms(CHIP, **with_census)
    assert {n: (t.band, t.source) for n, t in before.items()} == {n: (t.band, t.source) for n, t in after.items()}
    assert refused_before == refused_after


class TestTheWritePayloadCrossesTheLink:
    """A backing-store payload streams from host memory over the link into the chip's programming port: its bytes per
    inference (the period's payload over K) pay the link and host memory, and its time is the write law's at the
    slower of the chip's port and the link (the attach block, ``combine: slower``). [U-J3, amended] On a host-owned
    store (``EDGE_HOST_STORE``); a chip-owned store's write port reads the chip's own DRAM, and the attach block
    refuses by name there."""

    @staticmethod
    def _census(payload):
        from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

        return PassPeriodCensus(passes=(PassCensus(
            position=0, segment_index=0, resident=False, cores=1, resident_cores=0, written_cells_backing_store=100,
            written_cells_producer=0, payload_bytes=payload, compute_steps=3),))

    def test_the_period_payload_over_k_pays_the_link_and_host_memory(self):
        from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs
        from mimarsinan.deployment_record.quantities.write_law import WritePolicy

        values = dict(BASE, host_macs=0, compute_op_count=0, host_memory_bound_bytes=0, **NO_LINK)
        pricing = price_absolute(_q(**values), EDGE_HOST_STORE,
                                 writes=WriteInputs(self._census(400), WritePolicy(batch_k=4)))
        terms = {t.name: t for t in pricing.terms}  # [U-J2 K-1, amended] the link and host memory each their own term
        for corner in CORNERS:
            assert _c(terms["energy_host_link_mj"].band, corner) == pytest.approx(
                400 / 4 * _k("e_link_per_byte", corner) * 1e3, rel=1e-12)
            assert _c(terms["energy_host_mem_mj"].band, corner) == pytest.approx(
                400 / 4 * _k("e_host_mem_per_byte", corner) * 1e3, rel=1e-12)

    def test_the_attach_block_feeds_the_programming_port_over_the_link(self):
        from mimarsinan.deployment_record.platform_physics.hosts.description import host_attach_block

        block = host_attach_block("jetson_orin_nano_8gb")
        assert set(block) == {"t_program_per_byte"} and block["t_program_per_byte"]["combine"] == "slower"
        chip = resolve_platform_physics("study_compass", None)
        edge = resolve_platform_physics("study_compass", merge_override_blocks(  # [U-J3, amended] a host-owned store
            {"host": host_description_block("jetson_orin_nano_8gb"), "host_attach": block,
             "store": store_block(HOST_STORE, "the host-owned variant (test)")}))
        for corner in CORNERS:
            assert _c(edge.band("t_program_per_byte"), corner) == pytest.approx(max(
                _c(chip.band("t_program_per_byte"), corner), _c(edge.band("t_link_per_byte"), corner)), rel=1e-12)


class TestACarryBufferHoldsItsBoundariesOffTheLink:
    """[targeted M2] A transfer is counted for a pass boundary's crossing part only: a declared on-chip carry buffer
    (``carry_buffer_bytes``) keeps a wavefront group's own values on chip (``onchip_carry``), so their boundaries
    cross the link no more than their bytes do. Without the buffer every carried boundary is a transfer per
    direction, as before."""

    #: Three pass boundaries: 10 values out after pass 0 and 5 after pass 1; 10 in before pass 2, 5 before pass 3.
    WIDTHS = ((0, 10, 0), (1, 5, 10), (2, 0, 5))
    #: The group's own share: the 10 values produced in pass 0 and read in pass 2 stay in the buffer.
    HELD = ((0, 10, 0), (1, 0, 10))

    @staticmethod
    def _transfers(passes, **platform):
        from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
        from mimarsinan.mapping.support.host_boundary import HostBoundaryCensus

        nothing = HostBoundaryCensus(to_chip_elements=0, from_chip_elements=0, transfers=0, memory_bound_elements=0)
        pcfg = {"platform_physics_resolved": EDGE.to_dict(), "simulation_steps": 1, **platform}
        context = candidate_context_from_platform(pcfg, passes=passes, host_boundary=nothing)
        quantities, withheld = dict(context.host_link_quantities), dict(context.host_link_withheld)
        return (quantities["host_link_transfers"].value if "host_link_transfers" in quantities else None), withheld

    def _passes(self, held):
        from mimarsinan.deployment_record.build.write_census import ProgramPasses

        return ProgramPasses(rows=(), boundary_widths=self.WIDTHS, onchip_boundary_widths=held)

    def test_by_hand_a_boundary_the_buffer_holds_whole_is_no_transfer(self):
        # crossing parts with the buffer: (0, 0, 0), (1, 5, 0), (2, 0, 5): 5 out after pass 1, 5 in before pass 3
        assert self._transfers(self._passes(self.HELD), carry_buffer_bytes=1 << 20)[0] == 2
        assert self._transfers(self._passes(self.HELD))[0] == 4  # no buffer: every boundary, each direction

    def test_a_plane_stating_no_on_chip_split_withholds_the_transfers_by_name(self):
        transfers, withheld = self._transfers(self._passes(None), carry_buffer_bytes=1 << 20)
        assert transfers is None and "carry_buffer_bytes" in withheld["host_link_transfers"]


@pytest.fixture(scope="module", params=["together", "groups"])
def pipelined(request):
    """U-F4's residual pipelined fixture: copies of every level in one wavefront group (together) or in several
    (groups), scheduled by the ladder's planner."""
    from unit.mapping.pipelined_multiplexing.test_planes import _ladder_noc, _pipelined

    platform, program = _pipelined("residual", request.param)
    scheduled, specs = _ladder_noc(program, platform, 10)
    return request.param, platform, program, scheduled, specs


def _pipelined_link(platform, facts, buffer=None):
    from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
    from mimarsinan.mapping.support.host_boundary import HostBoundaryCensus

    nothing = HostBoundaryCensus(to_chip_elements=0, from_chip_elements=0, transfers=0, memory_bound_elements=0)
    pcfg = {**platform, "platform_physics_resolved": EDGE.to_dict()}
    if buffer is not None:
        pcfg["carry_buffer_bytes"] = buffer
    context = candidate_context_from_platform(pcfg, carry=facts.carry, passes=facts.passes, host_boundary=nothing)
    return context, dict(context.host_link_quantities)["host_link_transfers"].value


def _crossing_by_hand(specs, noc):
    """The candidate's carried spans and delayed slices (the carry census's inputs) less those one wavefront group
    produces and reads, and the transfers they make: one per direction at each (pass, segment) row boundary."""
    from mimarsinan.mapping.noc.fragments import delayed_carry_wires, spec_segments
    from mimarsinan.mapping.support.schedule.pass_carry import carried_softcore_spans
    from mimarsinan.mapping.support.wavefront_stages import program_groups

    segment = spec_segments(specs, getattr(noc, "runs", None))
    groups = program_groups(noc.pass_programs)
    spans = carried_softcore_spans(specs, noc.pass_placements, noc.census.pair_wires, segment)
    inside = {n: groups[a] for n, _w, a, b in spans if groups[a] is not None and groups[a] == groups[b]}
    crossing = [(n, w, a, b) for n, w, a, b in spans if n not in inside]
    replays = [(n, w, r) for n, w, r in delayed_carry_wires(specs, noc) if not (n in inside and groups[r] == inside[n])]
    rows = sorted({(p, segment[s]) for p, placements in enumerate(noc.pass_placements) for s, _h in placements})
    before = {key: rows[i - 1] for i, key in enumerate(rows) if i}
    out = {(a, segment[n]) for n, _w, a, _b in crossing}
    inbound = {before[(b, segment[n])] for n, _w, _a, b in crossing} | {before[(r, segment[n])] for n, _w, r in replays}
    return crossing, replays, len(out) + len(inbound)


class TestTheCarryBufferOnTheResidualPipelinedFixture:
    def test_without_the_buffer_every_carried_boundary_is_a_transfer(self, pipelined):
        _name, platform, _program, scheduled, _specs = pipelined
        _context, transfers = _pipelined_link(platform, scheduled.facts)
        assert transfers == sum(int(bool(o)) + int(bool(i)) for _p, o, i in scheduled.facts.passes.boundary_widths)

    def test_with_the_buffer_only_the_crossing_part_transfers(self, pipelined):
        from mimarsinan.mapping.support.schedule.pass_carry import boundary_transfer_bytes
        from unit.mapping.lanes.lane_planes import STEPS

        name, platform, _program, scheduled, specs = pipelined
        crossing, replays, expected = _crossing_by_hand(specs, scheduled.noc)
        context, transfers = _pipelined_link(platform, scheduled.facts, buffer=10 ** 9)
        assert transfers == expected < _pipelined_link(platform, scheduled.facts)[1]
        # the transfers' crossing part is the bytes' crossing part (onchip_carry's split of the census)
        assert context.carry_out_bytes == sum(boundary_transfer_bytes(width=w, timesteps=STEPS) for _n, w, _a, _b
                                              in crossing)
        assert context.carry_in_bytes == context.carry_out_bytes + sum(
            boundary_transfer_bytes(width=w, timesteps=STEPS) for _n, w, _r in replays)
        if name == "together":  # every copy in one group: every carried value stays on chip, no boundary crosses
            assert transfers == 0 and context.carry_out_bytes == 0 and context.carry_onchip_bytes > 0

    def test_both_planes_state_the_same_on_chip_split(self, pipelined):
        from mimarsinan.deployment_record.build.program_census import mapping_program_passes
        from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

        _name, platform, program, scheduled, _specs = pipelined
        built = mapping_program_passes(build_hybrid_mapping_for_pipeline(program.ir, platform))
        candidate = scheduled.facts.passes
        assert candidate.onchip_boundary_widths and built.onchip_boundary_widths == candidate.onchip_boundary_widths
        assert built.boundary_widths == candidate.boundary_widths
