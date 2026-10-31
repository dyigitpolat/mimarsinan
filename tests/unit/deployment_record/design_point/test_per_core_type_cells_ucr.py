"""[U-CR C-3] The per-core-type write blocks of a row count each written weight's cells, as the row's own summary and
the write law do: once Figure 1's mixed row states NeuRRAM's block on its static core type (2 cells per signed weight),
its static type's ``static_written_cells_per_period`` is twice the weights its share writes, the types sum to the row's
written cells, and each type's cells times its technology's per-cell constants are the write energy the row prices."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.design_point.timeline import program_writes
from mimarsinan.deployment_record.platform_physics import PlatformPhysics
from unit.chip_simulation.value_run.integer_programs import integer_inputs

from . import programs as P

ROLES = {"neurram": "static", "sram_cim_macro": "operand"}


@pytest.fixture(scope="module")
def rotated():
    from mimarsinan.deployment_record.design_point.rows import judged_row
    program, hybrid, platform = P.attention_on_hybrid(1, 1, True)
    platform = {**platform, "weight_bits": 8, "inference_batch_K": 1}
    row = judged_row(nf=program.model, ir_graph=program.ir, hybrid=hybrid, x_int=integer_inputs(program, 8),
                     platform=platform, physics=PlatformPhysics.from_dict(platform["platform_physics_resolved"]),
                     k_grid=(1, 16, 64), weight_bits=8, roles=ROLES)
    return row, platform, hybrid


def test_the_static_type_counts_two_cells_per_written_weight(rotated):
    row, platform, hybrid = rotated
    static_core = next(c for c in platform["cores"] if c.get("write_physics") == "neurram")
    assert (static_core["cells_per_weight"], static_core["weight_cell_rows"]) == (2, 2)
    writes = program_writes(hybrid, platform, weight_bits=8)
    share = next(p for p in writes.by_technology.parts if p.technology.write_physics == "neurram")
    weights = sum(p.written_cells_backing_store for p in share.census.passes)
    assert weights > 0
    entry = next(e for e in row["writes"]["per_core_type"] if e["write_technology"] == "neurram")
    assert entry["static_written_cells_per_period"] == 2 * weights


def test_the_core_types_sum_to_the_row_s_written_cells(rotated):
    row, _platform, _hybrid = rotated
    entries = row["writes"]["per_core_type"]
    for field in ("static_written_cells_per_period", "operand_written_cells_per_inference"):
        assert sum(e[field] for e in entries) == row["writes"][field], field
