"""N2 design §2.4: a dynamic operand bank's binding to the host op that writes its payload, validated at finalize.

One binding per bank, the bank's shape, the source op before every instance of the bank, the value domain only
(a spiking contract refuses by name), an unpruned bank; an IR pickled before the field reads an empty table.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.ir.dynamic_bank import (
    DynamicBankBinding,
    OperandBindingError,
    dynamic_bank_ids,
    validate_bindings,
)
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.mapping.support.operand_write import OperandWrite

AXONS, NEURONS = 3, 2


def _sources(pairs):
    return np.array([IRSource(node_id=int(n), index=int(i)) for n, i in pairs], dtype=object)


def _graph(*, write_first: bool = True, bank_shape=(AXONS, NEURONS), pruned: bool = False) -> IRGraph:
    """A producer core (node 0, 6 outputs), the payload write (node 1) reading them, and two instances of bank 0
    streaming the producer's first three outputs; ``write_first=False`` puts the write after the instances."""
    producer = NeuralCore(id=0, name="producer", input_sources=_sources([(-2, i) for i in range(4)]),
                          core_matrix=np.ones((4, AXONS * NEURONS)))
    write = ComputeOp(id=1, name="write", input_sources=_sources([(0, i) for i in range(AXONS * NEURONS)]),
                      op_type="OperandWrite", params={"module": OperandWrite(banks=1, payload_shape=(AXONS, NEURONS))},
                      input_shape=(1, AXONS, NEURONS), output_shape=(1, AXONS, NEURONS))
    instances = [NeuralCore(id=2 + j, name=f"instance{j}", input_sources=_sources([(0, i) for i in range(AXONS)]),
                            core_matrix=None, weight_bank_id=0, weight_row_slice=(0, NEURONS))
                 for j in range(2)]
    if pruned:
        instances[1].pruned_row_mask = [False, True, False]
        instances[1].pruned_col_mask = [False, False]
    nodes = [producer, write, *instances] if write_first else [producer, *instances, write]
    graph = IRGraph(nodes=nodes, output_sources=_sources([(2, 0), (3, 1)]),
                    weight_banks={0: WeightBank(id=0, core_matrix=np.full(bank_shape, np.nan))})
    graph.dynamic_bank_bindings = {0: DynamicBankBinding(
        bank_id=0, source_node_id=1, payload_shape=(AXONS, NEURONS), transpose=False, batch_index=(0,),
        payload_bits=8)}
    return graph


def test_a_well_formed_binding_validates():
    graph = _graph()
    validate_bindings(graph, packaging=MVM_PACKAGING)
    assert dynamic_bank_ids(graph) == frozenset({0})


def test_the_refusal_is_a_toolchain_refusal_and_a_value_error():
    assert issubclass(OperandBindingError, ToolchainRefusal) and issubclass(OperandBindingError, ValueError)


def test_one_binding_per_bank_keyed_by_its_own_bank():
    graph = _graph()
    graph.dynamic_bank_bindings = {5: graph.dynamic_bank_bindings[0]}
    with pytest.raises(OperandBindingError, match="bank"):
        validate_bindings(graph, packaging=MVM_PACKAGING)


def test_a_binding_of_a_bank_the_graph_does_not_hold_refuses():
    graph = _graph()
    graph.weight_banks = {}
    with pytest.raises(OperandBindingError, match="bank 0"):
        validate_bindings(graph, packaging=MVM_PACKAGING)


def test_the_payload_shape_is_the_bank_shape():
    with pytest.raises(OperandBindingError, match="shape"):
        validate_bindings(_graph(bank_shape=(NEURONS, AXONS)), packaging=MVM_PACKAGING)


def test_the_source_precedes_every_instance_of_the_bank():
    with pytest.raises(OperandBindingError, match="precede"):
        validate_bindings(_graph(write_first=False), packaging=MVM_PACKAGING)


def test_the_source_is_a_host_op():
    graph = _graph()
    graph.dynamic_bank_bindings = {0: DynamicBankBinding(
        bank_id=0, source_node_id=0, payload_shape=(AXONS, NEURONS), transpose=False, batch_index=(0,),
        payload_bits=8)}
    with pytest.raises(OperandBindingError, match="host op"):
        validate_bindings(graph, packaging=MVM_PACKAGING)


def test_a_spiking_contract_refuses_a_binding_by_name():
    with pytest.raises(OperandBindingError, match="value domain"):
        validate_bindings(_graph(), packaging=SPIKING_PACKAGING)


@pytest.mark.parametrize("config", [{}, {"spiking_mode": "lif"}, {"spiking_mode": "ttfs"}])
def test_an_operand_core_type_under_a_spiking_contract_refuses_at_the_lowering_step(config):
    """The step applies wherever a core type declares ``operand`` and refuses every spiking contract by name
    before it reads the model (ODIN and every LIF or TTFS chip)."""
    from mimarsinan.pipelining.pipeline_steps.mapping.operand_lowering_step import OperandLoweringStep

    class _Pipeline:
        def __init__(self, declared):
            self.config = declared

        def get_target_metric(self):
            return 0.0

    cores = [{"max_axons": 64, "max_neurons": 64, "count": 2, "operand": {"payload_bits": 8}}]
    pipeline = _Pipeline({**config, "cores": cores})
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
    assert OperandLoweringStep.applies_to(DeploymentPlan.of(pipeline))
    with pytest.raises(OperandBindingError, match="spiking contract"):
        OperandLoweringStep(pipeline).process()


def test_a_pruned_operand_bank_refuses():
    with pytest.raises(OperandBindingError, match="pruned"):
        validate_bindings(_graph(pruned=True), packaging=MVM_PACKAGING)


def test_a_graph_without_bindings_validates_under_every_contract():
    graph = _graph()
    graph.dynamic_bank_bindings = {}
    for packaging in (MVM_PACKAGING, SPIKING_PACKAGING):
        validate_bindings(graph, packaging=packaging)
    assert dynamic_bank_ids(graph) == frozenset()


def test_a_graph_has_an_empty_table_by_default_and_after_a_legacy_unpickle():
    graph = IRGraph(nodes=[], output_sources=np.array([], dtype=object))
    assert graph.dynamic_bank_bindings == {}
    state = dict(_graph().__dict__)
    state.pop("dynamic_bank_bindings")
    legacy = IRGraph.__new__(IRGraph)
    legacy.__dict__.update(state)
    assert legacy.dynamic_bank_bindings == {}
    assert pickle.loads(pickle.dumps(legacy)).dynamic_bank_bindings == {}


def test_a_binding_survives_a_pickle_round_trip():
    graph = pickle.loads(pickle.dumps(_graph()))
    assert graph.dynamic_bank_bindings[0].payload_shape == (AXONS, NEURONS)
    validate_bindings(graph, packaging=MVM_PACKAGING)
