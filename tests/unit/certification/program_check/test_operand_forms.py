"""[E6] N2's operand banks and operand cores under W15. A bank bound to a payload holds a NaN placeholder in the IR;
the value executor writes the payload's codes into every placement of it per sample (``operand_blocks``,
``fill_operand_blocks``). The check compares a core's static weights with its operand blocks cleared, and each
payload placement's cells per neuron (bank, bank row and column, and the source of each row) with the identity
program's; the program's binding table must be the IR's, every payload write produced before the stage that reads it,
the payload's codes on its grid, and the payload's width bounds the accumulation (the codes' range is the operand
certificate's, an assumed executor clause)."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch

from mimarsinan.certification.program_check.report import ESTABLISHED, FAILED
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow

from unit.certification.program_check import programs as P
from unit.certification.program_check import unit_vehicles as UV
from unit.chip_simulation.value_run.operand_programs import image_inputs


def _failed(report, condition: str, rule: str) -> None:
    result = report.results[condition]
    assert result.status == FAILED, (condition, result.status, result.note)
    assert rule in {v.rule for v in result.violations}, [v.text() for v in result.violations]
    assert report.theorem1.startswith("failed") and condition in report.theorem1, report.theorem1


def _payload_records(program):
    """(stage, core index, record) of every placement of a bound bank."""
    banks = set(program.dynamic_bank_bindings)
    for stage in P.neural(program):
        hcm = stage.hard_core_mapping
        for h, records in enumerate(hcm.soft_core_placements_per_hard_core):
            for record in records:
                if record.get("weight_bank_id") is not None and int(record["weight_bank_id"]) in banks:
                    yield stage, h, record


@pytest.mark.parametrize("name", ["operand_softmax", "operand_relu", "operand_identity"])
def test_an_operand_program_is_established_with_every_payload_cell_compared(name):
    program, ir, decl = UV.BUILDERS[name]()
    report = P.run(program, ir, decl)
    assert report.established("theorem1"), (report.theorem1, report.failures_by_condition())
    assert report.counts["payload_cells"] > 0 and report.results["tiles"].status == ESTABLISHED
    assert "operand payload" in report.theorem1


def test_a_value_flow_run_leaves_the_refcount_memo_fresh():
    """The value flow's refcounts count each stage's payload writes (N2); the check derives the memo the same way."""
    program, ir, decl = UV.operand("softmax")
    with torch.no_grad():
        ValueHybridCoreFlow(program, dtype=torch.float64)(image_inputs(2))
    assert program._consumer_counts_cache is not None
    report = P.run(program, ir, decl)
    assert report.results["H9"].status == ESTABLISHED, [v.text() for v in report.results["H9"].violations]
    assert report.established("theorem1"), report.theorem1


def test_a_payload_placement_reading_another_heads_bank():
    """One placement's bank swapped for the other head's bank of the same shape: the static weights are unchanged
    (both placeholders), the executor writes the other head's payload."""
    program, ir, decl = UV.operand("softmax")
    bad = copy.deepcopy(program)
    bindings = bad.dynamic_bank_bindings
    _stage, _h, record = next(iter(_payload_records(bad)))
    bank = int(record["weight_bank_id"])
    other = next(b for b, binding in bindings.items()
                 if b != bank and binding.payload_shape == bindings[bank].payload_shape)
    record["weight_bank_id"] = other
    _failed(P.run(bad, ir, decl), "tiles", "payload_cells")


def test_a_payload_row_reading_another_source():
    """Two rows of one payload block swapped on the core: their weights are the payload's, so only the payload
    comparison sees the rows exchange their sources."""
    program, ir, decl = UV.operand("softmax")
    bad = copy.deepcopy(program)
    stage, h, record = next(iter(_payload_records(bad)))
    core = stage.hard_core_mapping.cores[h]
    a = int(record["axon_offset"])
    sources = list(core.axon_sources)
    sources[a], sources[a + 1] = sources[a + 1], sources[a]
    core.axon_sources = sources
    core._axon_source_spans = None
    _failed(P.run(bad, ir, decl), "tiles", "payload_cells")


def test_a_binding_table_that_is_not_the_irs():
    program, ir, decl = UV.operand("softmax")
    bad = copy.deepcopy(program)
    bank = next(iter(bad.dynamic_bank_bindings))
    binding = bad.dynamic_bank_bindings[bank]
    bad.dynamic_bank_bindings[bank] = dataclasses.replace(binding, payload_offset=binding.payload_offset + 1)
    _failed(P.run(bad, ir, decl), "tiles", "operand_binding")


def test_a_program_that_lost_its_binding_table():
    program, ir, decl = UV.operand("softmax")
    bad = copy.deepcopy(program)
    del bad.dynamic_bank_bindings
    _failed(P.run(bad, ir, decl), "tiles", "operand_binding")


def _rebound(field: str, value_of):
    """The operand program and its IR with every binding's ``field`` replaced in both (so both programs agree)."""
    program, ir, decl = UV.operand("softmax")
    bad_program, bad_ir = copy.deepcopy(program), copy.deepcopy(ir)
    for table in (bad_program.dynamic_bank_bindings, bad_ir.dynamic_bank_bindings):
        for bank, binding in list(table.items()):
            table[bank] = dataclasses.replace(binding, **{field: value_of(bad_ir, binding)})
    return bad_program, bad_ir, decl


def test_a_payload_write_produced_after_the_stage_that_reads_it():
    def last_host_op(ir, _binding):
        return int([n for n in ir.nodes if type(n).__name__ == "ComputeOp"][-1].id)

    bad, bad_ir, decl = _rebound("source_node_id", last_host_op)
    _failed(P.run(bad, bad_ir, decl), "H9", "payload_unproduced")


def test_a_payload_whose_values_are_no_codes():
    """A binding without a payload step writes the values themselves (a float program): no integer charge."""
    bad, bad_ir, decl = _rebound("payload_step", lambda _ir, _binding: None)
    _failed(P.run(bad, bad_ir, decl), "H8", "off_lattice")


def test_a_payload_wider_than_the_float32_upload():
    """The payload's declared width bounds every cell: at 30 bits a payload code is no longer exact in the value
    census's float32 upload."""
    bad, bad_ir, decl = _rebound("payload_bits", lambda _ir, _binding: 30)
    _failed(P.run(bad, bad_ir, decl), "H8", "upload_precision")


def test_an_operand_block_on_a_fused_core():
    """The executor refuses an operand bank on a fused core by name (``OperandPlacementError``); the check names it."""
    program, ir, decl = UV.operand("softmax")
    bad = copy.deepcopy(program)
    stage, h, _record = next(iter(_payload_records(bad)))
    core = stage.hard_core_mapping.cores[h]
    core.fused_component_axons = [int(core.axons_per_core)]
    _failed(P.run(bad, ir, decl), "tiles", "operand_on_fused_core")


def test_the_operand_instances_carry_their_instance_as_output_column():
    """N2's emission names each instance of a bank by its position (as a conv's bank positions are), so every
    operand neuron has its own identity: ``ir-tiling`` keys it."""
    program, ir, decl = UV.operand("softmax")
    instances = {}
    for node in ir.get_neural_cores():
        if node.weight_bank_id is not None and int(node.weight_bank_id) in ir.dynamic_bank_bindings:
            instances.setdefault(int(node.weight_bank_id), []).append(node.perceptron_output_column)
    assert instances and all(columns == list(range(len(columns))) for columns in instances.values()), instances
    report = P.run(program, ir, decl, unbounded_ir=ir)
    assert report.results["ir-tiling"].status == ESTABLISHED, [v.text() for v in report.results["ir-tiling"].violations]


def test_the_static_part_of_an_operand_core_is_compared_as_any_core():
    """A static weight beside the payload blocks: the operand core's static grid entry outside its blocks moved."""
    program, ir, decl = UV.operand("softmax")
    bad = P.clone(program)
    stage, h, record = next(iter(_payload_records(bad)))
    core = stage.hard_core_mapping.cores[h]
    grid = np.asarray(core.core_matrix)
    rows = range(int(record["axon_offset"]), int(record["axon_offset"]) + int(record["axons"]))
    other = [r for r in range(core.axons_per_core - core.available_axons) if r not in rows]
    if not other:
        pytest.skip("the payload block fills the core's used rows")
    grid[other[0], int(record["neuron_offset"])] += 1.0
    _failed(P.run(bad, ir, decl), "tiles", "fan_in")
