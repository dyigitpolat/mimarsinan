"""[consts fix round, review R1] Pi and M read the write law: one source for what writing a tile costs.

Pi (law.md §2.1) is the cost of writing every tile of the program once over the compute-only cost of one inference.
The ladder prices that numerator with the write law (``cost/absolute/writes.py``), the pricer of every per-inference
write term, on the deployment load of the program (``period_census(every_core=True)``: every core of every pass
written once from an empty chip, nothing kept), at K = 1 and serial; and M's period writes with the same law on the
program's own period census. So a write-verify wall (``t_write_per_unit`` x iterations) and a cell commit energy
(``e_write_per_cell`` x iterations) are Pi's whenever the write law prices them, and where the write law refuses (a
chip that declares no commit energy), the Pi axis is absent with that refusal's reason, never a transfer-only number.
[consts, owner decision of 2026-10-26] The write law charges the per-core programming energy (``e_core_program``) once
per rewritten core, so Pi's numerator carries it once per tile and M's period writes once per core the period rewrites.

The hand point is ``test_scaling_ladder``'s (two convs and a head, one softcore per 15 x 16 core, 4-bit weights).
Every constant here is a test constant.
"""

from __future__ import annotations

import pytest

import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses, period_census
from mimarsinan.deployment_record.platform_physics import profile_from_dict
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.write_law import plane_write_census, written_cores
from unit.deployment_record.introspection.test_scaling_ladder import (
    ACTIVITY, BITS, CONSTANTS, CONV1, CONV2, F_HAND, HEAD, READING, SEGMENTS, T, _facts, _network, _platform,
    _positions,
)

#: The hand point's as-mapped cells and payload (every softcore alone on its core, 4-bit weights).
CELLS = _positions(4) * (CONV1 + CONV2) + HEAD
TILES = 2 * _positions(4) + 1
PAYLOAD = _positions(4) * (CONV1 * BITS // 8 + CONV2 * BITS // 8) + HEAD * BITS // 8
STEPS = SEGMENTS * (T + 1)
PJ, NS = 1e-12, 1e-9


def _physics(name: str, constants: dict):
    return profile_from_dict({
        "format_version": 1, "name": name, "display_name": name, "description_file": "t.md",
        "validity": {"measurement_kind": "projection"},
        "constants": {k: {"nominal": v, "evidence_kind": "estimated", "note": "test"} for k, v in constants.items()},
    })


def _row(physics=None, *, cores: int = 64, k: int = 16, granularity=None, payload=None):
    network = _network(4)
    platform = _platform(cores)
    if granularity is not None:
        platform["cores"] = [{**platform["cores"][0], "write_granularity": granularity}]
    if physics is not None or payload is not None:
        platform["platform_physics_resolved"] = payload if payload is not None else physics.to_dict()
    point = {**platform, "inference_batch_K": k}
    census = ladder.shape_census(network, point)
    priced = ladder.price_program(network, census, point, _facts)
    return ladder.ladder_row(network, census, priced, point, READING)


def _compute_energy_pj(c):
    return c["e_mac"] * CELLS * T * ACTIVITY + c.get("e_core_init", 0.0) * TILES


def _compute_time_ns(c):
    return c["t_cycle"] * STEPS + c.get("t_core_init", 0.0) * TILES


class TestTheDeploymentLoad:
    def _passes(self):
        row = PassRow(segment_index=0, resident=True, core_cells=(10, 20), marks=frozenset({0}),
                      core_extents=((2, 5), (4, 5)), compute_steps=3)
        return ProgramPasses(rows=(row,))

    def test_a_lone_resident_pass_writes_nothing_per_period_and_every_core_once_at_load(self):
        period = period_census(self._passes(), weight_bits=4, capacity={(8, 8): 4})
        load = period_census(self._passes(), weight_bits=4, capacity={(8, 8): 4}, every_core=True)
        assert [p.payload_bytes for p in period.census.passes] == [0]
        (written,) = load.census.passes
        assert (written.resident, written.resident_cores, written.written_cells_backing_store) == (False, 0, 30)
        assert written.payload_bytes == 10 * 4 // 8 + 20 * 4 // 8
        assert load.written_extents == (((2, 5), (4, 5)),)
        assert "every core of every pass" in load.census.label

    def test_the_load_writes_the_rows_programmed_bytes_and_cores_once(self):
        row = _row(cores=8)
        c = CONSTANTS
        assert (row["programmed_bytes_once"], row["programmed_cores_once"]) == (PAYLOAD, TILES)
        assert row["energy"]["terms"]["once"] == pytest.approx(
            [((c["e_dma_per_byte"] + c["e_program_per_byte"]) * PAYLOAD + c["e_core_program"] * TILES) * PJ * 1e3] * 3,
            rel=1e-12)


class TestPiReadsTheWriteLaw:
    def test_pi_on_the_hand_point_is_the_write_laws_fetch_commit_and_core_programming_over_compute(self):
        """The write law prices a tile write as its fetch (e_dma_per_byte) and commit (e_program_per_byte) per byte
        and, [consts] once per written core, the per-core programming energy e_core_program."""
        row = _row(cores=64, k=16)
        c = CONSTANTS
        pi_e = ((c["e_dma_per_byte"] + c["e_program_per_byte"]) * PAYLOAD + c["e_core_program"] * TILES) / (
            _compute_energy_pj(c))
        pi_t = c["t_program_per_byte"] * PAYLOAD / _compute_time_ns(c)
        for axis, expected in (("energy", pi_e), ("time", pi_t)):
            assert row[axis]["pi"] == pytest.approx([expected] * 3, rel=1e-12)
            assert row[axis]["absent"] is None

    def test_a_write_verify_wall_enters_the_time_pi(self):
        """t_write_per_unit x iterations per written cell: the wall the write law prices, beside which the transfer
        (t_program_per_byte x bytes) is not the write time."""
        c = {**CONSTANTS, "t_write_per_unit": 100.0, "write_verify_iterations": 3.0}
        row = _row(_physics("ladder_wv", c), granularity="cell")
        wall = c["t_write_per_unit"] * c["write_verify_iterations"] * CELLS
        assert row["time"]["pi"] == pytest.approx([wall / _compute_time_ns(c)] * 3, rel=1e-12)
        assert row["time"]["terms"]["once"] == pytest.approx([wall * NS] * 3, rel=1e-12)

    def test_a_cell_commit_energy_enters_the_energy_pi(self):
        """e_write_per_cell x iterations per written cell: the write law's commit, which takes precedence over a
        per-byte commit."""
        c = {**CONSTANTS, "e_write_per_cell": 0.5, "write_verify_iterations": 2.0}
        row = _row(_physics("ladder_cell", c))
        once = (c["e_dma_per_byte"] * PAYLOAD + c["e_write_per_cell"] * c["write_verify_iterations"] * CELLS
                + c["e_core_program"] * TILES)
        assert row["energy"]["pi"] == pytest.approx([once / _compute_energy_pj(c)] * 3, rel=1e-12)

    def test_a_missing_commit_energy_leaves_the_energy_axis_absent_by_name(self):
        c = {k: v for k, v in CONSTANTS.items() if k != "e_program_per_byte"}
        row = _row(_physics("ladder_no_commit", c), cores=8)
        energy = row["energy"]
        assert energy["pi"] is None and energy["regime"] is None and energy["pi_over_k"] is None
        assert energy["absent"].startswith("the write law refuses writing every tile once: ")
        assert "declares no commit energy" in energy["absent"]
        assert energy["terms"]["compute_only"] is not None  # the compute is priced; only the write is refused
        assert row["time"]["pi"] is not None and "energy" not in row["axes_priced"]


class TestMReadsTheWriteLaw:
    @pytest.mark.parametrize("axis, term", [("energy", "energy_write_mj"), ("time", "latency_write_exposed_s")])
    def test_the_period_writes_are_the_write_laws_at_the_rows_batch(self, axis, term):
        """Serial writes: the write law's term at K is the period's tile writes over K, so M's period term and the
        write law are one number."""
        row = _row(cores=8, k=16)
        assert row["schedule_reading"].startswith("U1: multiplexed")
        period, at_k = row[axis]["terms"]["period"], row["write_law"]["terms"][term]
        assert [p / 16 for p in period] == pytest.approx(at_k, rel=1e-12)
        m = row[axis]["overhead"]
        compute = row[axis]["terms"]["compute_only"]
        assert m["toolchain_priced"] == pytest.approx(
            [1.0 + ka + w / c for ka, w, c in zip(m["kappa"], at_k, compute)], rel=1e-12)

    def test_the_period_writes_charge_each_core_the_period_rewrites_once(self):
        """[consts] M's period energy: the period's rewrite bytes fetched and committed, and e_core_program once per
        core the period census rewrites (``written_cores``, counted on the program's own census here)."""
        network, point = _network(4), {**_platform(8), "inference_batch_K": 16}
        census = ladder.shape_census(network, point)
        scheduled = ladder.schedule_program(network, census, point, _facts)
        period, _ = plane_write_census(scheduled.facts.passes, point, weight_bits=BITS, timesteps=None, label="hand")
        assert period is not None
        rewritten = written_cores(period.census)
        row = ladder.ladder_row(network, census, ladder.price_scheduled(network, census, scheduled, point), point,
                                READING)
        c = CONSTANTS
        assert 0 < rewritten <= TILES and row["reprogrammed_bytes_per_period"] > 0
        want = (c["e_dma_per_byte"] + c["e_program_per_byte"]) * row["reprogrammed_bytes_per_period"] + (
            c["e_core_program"] * rewritten)
        assert row["energy"]["terms"]["period"] == pytest.approx([want * PJ * 1e3] * 3, rel=1e-12)


class TestAChipWithoutACommitEnergyBehindAStore:
    """Review R1's case on the shipped descriptions: a store prices the transfer, the time Pi carries the
    write-verify wall the write law prices, and the energy Pi is absent by the write law's refusal on a chip that
    declares no commit energy. [U-C3, B3] NeuRRAM now declares its derived write energy per pulse, so its energy Pi
    prices; HERMES-64 (not determinable from its primaries) is the shipped chip without one."""

    STORE = {"e_dma_per_byte": {"nominal": 162.5, "unit": "pJ", "evidence_kind": "published",
                                "citation": "DDR4 I/O (test)"},
             "t_program_per_byte": {"low": 0.0278, "nominal": 0.0393, "high": 0.0556, "unit": "ns",
                                    "evidence_kind": "published", "citation": "DDR4 channel (test)",
                                    "combine": "slower"}}

    def test_neurram_behind_a_store_prices_the_wall_and_the_energy(self):
        physics = resolve_platform_physics("neurram", merge_override_blocks({"backing_store": self.STORE}))
        assert physics is not None
        row = _row(payload=physics.to_dict(), cores=F_HAND - 1, k=1, granularity="cell")
        wall = physics.band("t_write_per_unit")
        iterations = physics.band("write_verify_iterations")
        want = [w * i * CELLS for w, i in ((wall.low, iterations.low), (wall.nominal, iterations.nominal),
                                           (wall.high, iterations.high))]
        assert row["time"]["terms"]["once"] == pytest.approx(want, rel=1e-12)
        assert row["time"]["regime"]["reading"] == "resident-dominant"
        # [U-C3] the fetch (the store's e_dma_per_byte) plus the commit, (e_write + e_verify) x iterations per cell
        write, verify = physics.band("e_write_per_cell"), physics.band("e_verify_read_per_cell")
        dma = physics.band("e_dma_per_byte").nominal
        once = [(dma * PAYLOAD + (w + v) * i * CELLS) * 1e3 for w, v, i in (
            (write.low, verify.low, iterations.low), (write.nominal, verify.nominal, iterations.nominal),
            (write.high, verify.high, iterations.high))]
        assert row["energy"]["terms"]["once"] == pytest.approx(once, rel=1e-12)
        assert row["energy"]["pi"] is not None and row["energy"]["absent"] is None

    def test_hermes_behind_a_store_prices_the_wall_and_refuses_the_energy_by_name(self):
        physics = resolve_platform_physics("hermes_64", merge_override_blocks({"backing_store": self.STORE}))
        assert physics is not None
        row = _row(payload=physics.to_dict(), cores=F_HAND - 1, k=1, granularity="core")
        assert row["time"]["pi"] is not None and row["time"]["absent"] is None
        assert row["energy"]["pi"] is None and row["energy"]["regime"] is None
        assert "declares no commit energy" in row["energy"]["absent"]
