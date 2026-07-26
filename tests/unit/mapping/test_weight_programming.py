"""[wsm V0'] WeightProgrammingReport: the weight-programming boundary, measured."""

import numpy as np
import pytest

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.weight_programming import (
    weight_programming_report,
)


def _owned_two_core():
    w1 = np.ones((5, 4), dtype=np.float32)
    s1 = np.array([IRSource(-2, i) for i in range(4)] + [IRSource(-3, 0)], dtype=object)
    c1 = NeuralCore(id=0, name="a", input_sources=s1, core_matrix=w1, latency=0)
    w2 = np.ones((5, 2), dtype=np.float32)
    s2 = np.array([IRSource(0, i) for i in range(4)] + [IRSource(-3, 0)], dtype=object)
    c2 = NeuralCore(id=1, name="b", input_sources=s2, core_matrix=w2, latency=1)
    out = np.array([IRSource(1, 0), IRSource(1, 1)], dtype=object)
    return IRGraph(nodes=[c1, c2], output_sources=out)


def _banked_tokens(n_tokens=3, in_features=4, out_features=4):
    rows = in_features + 1
    bank = WeightBank(
        id=0, core_matrix=np.ones((rows, out_features), dtype=np.float32)
    )
    nodes = []
    for tok in range(n_tokens):
        srcs = np.array(
            [IRSource(-2, tok * in_features + i) for i in range(in_features)]
            + [IRSource(-3, 0)],
            dtype=object,
        )
        nodes.append(NeuralCore(
            id=tok, name=f"col{tok}", input_sources=srcs, core_matrix=None,
            weight_bank_id=0, weight_row_slice=(0, out_features), latency=0,
            perceptron_index=0, perceptron_output_column=tok,
            perceptron_output_slice=(0, out_features),
        ))
    out = np.array(
        [IRSource(t, j) for t in range(n_tokens) for j in range(out_features)],
        dtype=object,
    )
    return IRGraph(nodes=nodes, output_sources=out, weight_banks={0: bank})


class TestOwnedProgram:
    def test_every_param_programs_once(self):
        hybrid = build_identity_hybrid_mapping(ir_graph=_owned_two_core())
        report = weight_programming_report(hybrid)
        assert report.neural_stages == 1
        assert report.programming_events == 2
        assert report.params_programmed == 5 * 4 + 5 * 2
        assert report.params_unique == report.params_programmed
        assert report.reuse_factor == pytest.approx(1.0)


class TestBankSharedProgram:
    def test_fresh_pool_truth_programs_every_instance(self):
        # Today's regime: each placed instance programs its bank copy — the
        # report exposes the gap to the weight-stationary ideal (unique).
        hybrid = build_identity_hybrid_mapping(ir_graph=_banked_tokens(3))
        report = weight_programming_report(hybrid)
        assert report.programming_events == 3
        assert report.params_programmed == 3 * (5 * 4)
        assert report.params_unique == 5 * 4
        assert report.reuse_factor == pytest.approx(1.0 / 3.0)

    def test_packed_pool_counts_match_identity(self):
        ir = _banked_tokens(4)
        hybrid = build_hybrid_hard_core_mapping(
            ir_graph=ir,
            cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 8}],
        )
        report = weight_programming_report(hybrid)
        assert report.programming_events == 4
        assert report.params_programmed == 4 * (5 * 4)
        assert report.params_unique == 5 * 4

    def test_mixed_owned_and_banked(self):
        ir = _banked_tokens(2)
        w = np.ones((3, 2), dtype=np.float32)
        srcs = np.array(
            [IRSource(0, 0), IRSource(1, 0), IRSource(-3, 0)], dtype=object
        )
        ir.nodes.append(NeuralCore(
            id=7, name="head", input_sources=srcs, core_matrix=w, latency=1,
        ))
        ir.output_sources = np.array(
            [IRSource(7, 0), IRSource(7, 1)], dtype=object
        )
        hybrid = build_identity_hybrid_mapping(ir_graph=ir)
        report = weight_programming_report(hybrid)
        assert report.programming_events == 3
        assert report.params_programmed == 2 * 20 + 6
        assert report.params_unique == 20 + 6

    def test_summary_is_loud_and_complete(self):
        report = weight_programming_report(
            build_identity_hybrid_mapping(ir_graph=_banked_tokens(3))
        )
        text = report.summary()
        assert "reuse_factor=0.33" in text
        assert "params_programmed=60" in text
        assert "unique=20" in text
