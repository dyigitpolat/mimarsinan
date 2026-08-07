"""[P3] Streamed-lif structural contract: interior host ops fail loud."""

import numpy as np
import pytest
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.verification.streamed import (
    NotStreamableError,
    assert_streamable_ir,
    assert_streamable_model_or_raise,
)

N = 2


def _core(node_id, name, sources, matrix):
    return NeuralCore(
        id=node_id,
        name=name,
        input_sources=np.asarray(sources, dtype=object),
        core_matrix=np.asarray(matrix, dtype=np.float64),
        threshold=1.0,
    )


def _op(node_id, name, src_node):
    return ComputeOp(
        id=node_id, name=name,
        input_sources=np.asarray(
            [IRSource(src_node, i) for i in range(N)], dtype=object,
        ),
        op_type="identity", params={"module": nn.Identity()},
    )


def _graph(nodes, out_node):
    return IRGraph(
        nodes=nodes,
        output_sources=np.asarray(
            [IRSource(out_node, i) for i in range(N)], dtype=object,
        ),
    )


class TestIrGate:
    def test_prefix_and_suffix_hosts_are_streamable(self):
        eye = np.eye(N)
        a = _op(0, "encode", -2)
        b = _core(1, "L0", [IRSource(0, i) for i in range(N)], eye)
        c = _core(2, "L1", [IRSource(1, i) for i in range(N)], eye * 0.5)
        d = _op(3, "readout", 2)
        assert_streamable_ir(_graph([a, b, c, d], 3))

    def test_pure_neural_chain_is_streamable(self):
        eye = np.eye(N)
        a = _core(0, "L0", [IRSource(-2, i) for i in range(N)], eye)
        b = _core(1, "L1", [IRSource(0, i) for i in range(N)], eye)
        assert_streamable_ir(_graph([a, b], 1))

    def test_interior_host_op_fails_loud_naming_the_op(self):
        eye = np.eye(N)
        a = _core(0, "L0", [IRSource(-2, i) for i in range(N)], eye)
        pool = _op(1, "maxpool", 0)
        b = _core(2, "L1", [IRSource(1, i) for i in range(N)], eye)
        with pytest.raises(NotStreamableError, match="maxpool"):
            assert_streamable_ir(_graph([a, pool, b], 2))

    def test_no_neural_span_fails_loud(self):
        only = _op(0, "host_only", -2)
        with pytest.raises(NotStreamableError, match="no neural cores"):
            assert_streamable_ir(_graph([only], 0))


class TestStaticModelGate:
    def test_simplemlp_is_streamable(self):
        from mimarsinan.models.perceptron_mixer.simple_mlp import SimpleMLP

        model = SimpleMLP("cpu", (1, 8, 8), 4, 16, 8)
        assert_streamable_model_or_raise(
            model, (1, 8, 8), 4, encoding_placement="subsume",
        )

    def test_lenet5_interior_pools_fail_loud(self):
        from mimarsinan.models.lenet5 import LeNet5

        model = LeNet5((1, 28, 28), 10)
        with pytest.raises(NotStreamableError):
            assert_streamable_model_or_raise(
                model, (1, 28, 28), 10, encoding_placement="subsume",
            )
