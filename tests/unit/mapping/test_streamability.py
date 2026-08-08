"""[P6] Streamed span topology: per-segment streaming, end-to-end reported."""

import numpy as np
import pytest
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.verification.streamed import (
    NotStreamableError,
    streamed_span_report_ir,
    streamed_span_report_model,
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


class TestIrSpanReport:
    def test_prefix_and_suffix_hosts_are_end_to_end(self):
        eye = np.eye(N)
        a = _op(0, "encode", -2)
        b = _core(1, "L0", [IRSource(0, i) for i in range(N)], eye)
        c = _core(2, "L1", [IRSource(1, i) for i in range(N)], eye * 0.5)
        d = _op(3, "readout", 2)
        report = streamed_span_report_ir(_graph([a, b, c, d], 3))
        assert report.segments == 1
        assert report.interior_host_ops == ()
        assert report.end_to_end

    def test_pure_neural_chain_is_end_to_end(self):
        eye = np.eye(N)
        a = _core(0, "L0", [IRSource(-2, i) for i in range(N)], eye)
        b = _core(1, "L1", [IRSource(0, i) for i in range(N)], eye)
        report = streamed_span_report_ir(_graph([a, b], 1))
        assert report.segments == 1
        assert report.end_to_end

    def test_interior_host_op_reports_two_segments_no_raise(self):
        eye = np.eye(N)
        a = _core(0, "L0", [IRSource(-2, i) for i in range(N)], eye)
        pool = _op(1, "maxpool", 0)
        b = _core(2, "L1", [IRSource(1, i) for i in range(N)], eye)
        report = streamed_span_report_ir(_graph([a, pool, b], 2))
        assert report.segments == 2
        assert report.interior_host_ops == ("maxpool",)
        assert not report.end_to_end
        assert "maxpool" in report.describe()

    def test_two_interior_hops_report_three_segments(self):
        eye = np.eye(N)
        a = _core(0, "L0", [IRSource(-2, i) for i in range(N)], eye)
        p1 = _op(1, "pool1", 0)
        b = _core(2, "L1", [IRSource(1, i) for i in range(N)], eye)
        p2 = _op(3, "pool2", 2)
        c = _core(4, "L2", [IRSource(3, i) for i in range(N)], eye)
        report = streamed_span_report_ir(_graph([a, p1, b, p2, c], 4))
        assert report.segments == 3
        assert report.interior_host_ops == ("pool1", "pool2")

    def test_adjacent_interior_hosts_count_one_boundary(self):
        eye = np.eye(N)
        a = _core(0, "L0", [IRSource(-2, i) for i in range(N)], eye)
        p1 = _op(1, "pool", 0)
        p2 = _op(2, "flatten", 1)
        b = _core(3, "L1", [IRSource(2, i) for i in range(N)], eye)
        report = streamed_span_report_ir(_graph([a, p1, p2, b], 3))
        assert report.segments == 2
        assert report.interior_host_ops == ("pool", "flatten")

    def test_no_neural_span_fails_loud(self):
        only = _op(0, "host_only", -2)
        with pytest.raises(NotStreamableError, match="no neural cores"):
            streamed_span_report_ir(_graph([only], 0))


class TestStaticModelReport:
    def test_simplemlp_is_end_to_end(self):
        from mimarsinan.models.perceptron_mixer.simple_mlp import SimpleMLP

        model = SimpleMLP("cpu", (1, 8, 8), 4, 16, 8)
        report = streamed_span_report_model(
            model, (1, 8, 8), 4, encoding_placement="subsume",
        )
        assert report.end_to_end

    def test_lenet5_reports_multi_span_without_raising(self):
        from mimarsinan.models.lenet5 import LeNet5

        model = LeNet5((1, 28, 28), 10)
        report = streamed_span_report_model(
            model, (1, 28, 28), 10, encoding_placement="subsume",
        )
        assert report.segments >= 2
        assert "features_5" in report.interior_host_ops
        assert not report.end_to_end

    def test_lenet5_offload_places_both_pools_interior(self):
        from mimarsinan.models.lenet5 import LeNet5

        model = LeNet5((1, 28, 28), 10)
        report = streamed_span_report_model(
            model, (1, 28, 28), 10, encoding_placement="offload",
        )
        assert "features_2" in report.interior_host_ops
        assert "features_5" in report.interior_host_ops
        assert report.segments >= 3
