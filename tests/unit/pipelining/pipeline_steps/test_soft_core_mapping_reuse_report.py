"""The reuse-phase report is ALWAYS ON: a pure, total IR read (no config gate).

Weight reuse stopped being configuration (W1.1): banks are built
unconditionally by the mappers, so the schedule split is a fact of every
mapped IR graph — a bankless graph honestly reads all-reprogram.
"""

import inspect

import numpy as np

import mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step as step_module
from mimarsinan.mapping.ir.graph import IRGraph
from mimarsinan.mapping.ir.source import IRSource
from mimarsinan.mapping.ir.types import NeuralCore
from mimarsinan.mapping.ir.weight_bank import WeightBank
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
    print_weight_reuse_report,
)


def _sources(n):
    return np.array([IRSource(node_id=-2, index=i) for i in range(n)], dtype=object)


def _owned_core(core_id, axons=4, neurons=3):
    return NeuralCore(
        id=core_id, name=f"owned{core_id}", input_sources=_sources(axons),
        core_matrix=np.zeros((axons, neurons), dtype=np.float32),
    )


def _bank_core(core_id, bank_id, axons=4, neurons=3):
    return NeuralCore(
        id=core_id, name=f"bank{core_id}", input_sources=_sources(axons),
        core_matrix=None, weight_bank_id=bank_id, weight_row_slice=(0, neurons),
    )


def test_bankless_graph_reports_all_reprogram(capsys):
    graph = IRGraph(
        nodes=[_owned_core(0), _owned_core(1)],
        output_sources=np.array([IRSource(node_id=1, index=0)], dtype=object),
    )
    print_weight_reuse_report(graph)
    out = capsys.readouterr().out
    assert "[SoftCoreMappingStep] Weight-reuse schedule: " in out
    assert "2 reprogram + 0 reuse phases" in out


def test_banked_graph_reports_the_reuse_split(capsys):
    bank = WeightBank(id=0, core_matrix=np.zeros((4, 3), dtype=np.float32))
    graph = IRGraph(
        nodes=[_bank_core(i, 0) for i in range(3)],
        output_sources=np.array([IRSource(node_id=2, index=0)], dtype=object),
        weight_banks={0: bank},
    )
    print_weight_reuse_report(graph)
    assert "1 reprogram + 2 reuse phases" in capsys.readouterr().out


def test_report_wiring_has_no_config_gate():
    # process() prints the report unconditionally; the retired
    # allow_weight_reuse gate must not return in any form.
    src = inspect.getsource(step_module)
    assert "print_weight_reuse_report(ir_graph)" in src
    assert "allow_weight_reuse" not in src
