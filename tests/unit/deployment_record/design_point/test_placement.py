"""Per-operator placement of an executed program: every IR node in one operator, each with its place, role and the
requirement table's reason, and the passes or host stage it runs in."""

from __future__ import annotations

from mimarsinan.deployment_record.design_point.placement import (
    DYNAMIC_BANKS, OPERAND_REASON, operator_of_nodes, operator_placement,
)
from mimarsinan.mapping.capability.requirements import requirement_of
from mimarsinan.mapping.ir import NeuralCore

from . import programs as P


def _rows(count=3):
    program = P.residual_program()
    hybrid = P.built(count)
    return program, hybrid, operator_placement(program.ir, hybrid, names=P.perceptron_names(program))


def test_every_ir_node_belongs_to_one_operator():
    program, _hybrid, rows = _rows()
    assert sum(r["ir_nodes"] for r in rows) == len(program.ir.nodes)
    assert len(set(operator_of_nodes(program.ir, P.perceptron_names(program)))) == len(program.ir.nodes)


def test_core_operators_are_static_with_the_crossbar_reason_and_run_in_passes():
    _program, hybrid, rows = _rows()
    passes = sum(1 for s in hybrid.stages if s.kind == "neural")
    cores = [r for r in rows if r["placement"] == "core"]
    assert cores and all(r["role"] == "static" for r in cores)
    assert all(r["reason"] == requirement_of("mm").reason for r in cores)
    assert all(r["passes"] and set(r["passes"]) <= set(range(passes)) for r in cores)
    assert len({p for r in cores for p in r["passes"]}) == passes


def test_host_operators_carry_their_kind_reason_and_stage():
    _program, hybrid, rows = _rows()
    host = [r for r in rows if r["placement"] == "host"]
    assert {r["kind"] for r in host} == {"clamp", "add", "pool"}
    for r in host:
        assert r["reason"] == requirement_of(r["kind"]).reason
        assert hybrid.stages[r["host_stage"]].kind == "compute"
        assert r["passes"] == []


def test_a_core_whose_bank_is_bound_per_inference_is_an_operand_core():
    program = P.residual_program()
    core = next(n for n in program.ir.nodes if isinstance(n, NeuralCore) and n.weight_bank_id is not None)
    try:
        setattr(program.ir, DYNAMIC_BANKS, {core.weight_bank_id: "test binding"})
        rows = operator_placement(program.ir, P.built(3), names=P.perceptron_names(program))
    finally:
        delattr(program.ir, DYNAMIC_BANKS)
    marked = [r for r in rows if r["role"] == "operand"]
    assert len(marked) == 1 and marked[0]["reason"] == OPERAND_REASON and marked[0]["placement"] == "core"
    assert all(r["role"] == "static" for r in rows if r["placement"] == "core" and r is not marked[0])
