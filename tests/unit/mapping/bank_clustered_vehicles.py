"""Hand-built IR vehicles for the bank-clustered schedule policy.

One place builds the graphs the policy tests reason about, so the shape-only
answer and the deployed program are always compared on the SAME structure:

* :func:`token_graph` — ONE shared bank streamed over N spatial positions (the
  conv shape): the class the policy is proven on.
* :func:`two_layer_dependency_graph` — two banks where layer 1 CONSUMES layer 0
  inside one segment: outside the policy's class (the builder declines it too).
* :func:`multi_segment_graph` — N host-separated one-core segments, each of
  which deployment programs as its own pass.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from mimarsinan.mapping.ir import (
    ComputeOp,
    IRGraph,
    IRSource,
    NeuralCore,
    WeightBank,
)
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType,
    LayoutSoftCoreSpec,
)
from mimarsinan.mapping.layout.softcore_spec_adapter import spec_from_neural_core
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)

TWO_CORES = [{"max_axons": 32, "max_neurons": 32, "count": 2}]


def _bank(bank_id: int, in_features: int, out_features: int, seed: int) -> WeightBank:
    rng = np.random.default_rng(seed)
    return WeightBank(
        id=bank_id,
        core_matrix=rng.normal(size=(in_features + 1, out_features)).astype(np.float32),
    )


def token_graph(n_tokens: int = 7, in_features: int = 4, out_features: int = 4):
    """One shared bank streamed over ``n_tokens`` spatial positions (the conv shape)."""
    bank = _bank(0, in_features, out_features, seed=7)
    nodes = []
    for tok in range(n_tokens):
        srcs = np.array(
            [IRSource(-2, tok * in_features + i) for i in range(in_features)]
            + [IRSource(-3, 0)],
            dtype=object,
        )
        nodes.append(NeuralCore(
            id=tok, name=f"b0_col{tok}", input_sources=srcs, core_matrix=None,
            weight_bank_id=0, weight_row_slice=(0, out_features), latency=0,
            perceptron_index=0, perceptron_output_column=tok,
            perceptron_output_slice=(0, out_features),
        ))
    out = np.array(
        [IRSource(n.id, j) for n in nodes for j in range(out_features)],
        dtype=object,
    )
    return IRGraph(nodes=nodes, output_sources=out, weight_banks={0: bank})


def two_layer_dependency_graph(n_tokens: int = 3, features: int = 4):
    """Two banks, layer 1 consuming layer 0 — one segment WITH a dependency."""
    banks = {b: _bank(b, features, features, seed=5 + b) for b in (0, 1)}
    nodes = []
    for tok in range(n_tokens):
        nodes.append(NeuralCore(
            id=tok, name=f"L0_t{tok}",
            input_sources=np.array(
                [IRSource(-2, tok * features + j) for j in range(features)]
                + [IRSource(-3, 0)], dtype=object),
            core_matrix=None, weight_bank_id=0, weight_row_slice=(0, features),
            latency=0, perceptron_index=0, perceptron_output_column=tok,
            perceptron_output_slice=(0, features),
        ))
    for tok in range(n_tokens):
        nodes.append(NeuralCore(
            id=n_tokens + tok, name=f"L1_t{tok}",
            input_sources=np.array(
                [IRSource(tok, j) for j in range(features)] + [IRSource(-3, 0)],
                dtype=object),
            core_matrix=None, weight_bank_id=1, weight_row_slice=(0, features),
            latency=1, perceptron_index=1, perceptron_output_column=tok,
            perceptron_output_slice=(0, features),
        ))
    out = np.array(
        [IRSource(n_tokens + tok, j) for tok in range(n_tokens) for j in range(features)],
        dtype=object,
    )
    return IRGraph(nodes=nodes, output_sources=out, weight_banks=banks)


def multi_segment_graph(n_segments: int = 6, features: int = 4):
    """``n_segments`` host-separated segments, one neural core each."""
    bank = _bank(0, features, features, seed=3)
    nodes: List = []
    previous = -2
    for seg in range(n_segments):
        host = ComputeOp(
            id=1000 + seg, name=f"host{seg}",
            input_sources=np.array(
                [IRSource(previous, j) for j in range(features)], dtype=object),
            op_type="identity", input_shape=(features,), output_shape=(features,),
        )
        nodes.append(host)
        nodes.append(NeuralCore(
            id=seg, name=f"n{seg}",
            input_sources=np.array(
                [IRSource(host.id, j) for j in range(features)] + [IRSource(-3, 0)],
                dtype=object),
            core_matrix=None, weight_bank_id=0, weight_row_slice=(0, features),
            latency=seg, perceptron_index=seg, perceptron_output_column=0,
            perceptron_output_slice=(0, features),
        ))
        previous = seg
    out = np.array([IRSource(previous, j) for j in range(features)], dtype=object)
    return IRGraph(nodes=nodes, output_sources=out, weight_banks={0: bank})


def softcores_of(graph) -> List[LayoutSoftCoreSpec]:
    """The shape-only specs a layout answer packs, straight from the IR cores."""
    return [
        spec_from_neural_core(
            core, hardware_bias=False, fallback_residency_class_id=-(index + 1),
        )
        for index, core in enumerate(graph.get_neural_cores())
    ]


def hard_core_types(cores: Sequence[dict]) -> List[LayoutHardCoreType]:
    return [LayoutHardCoreType(**dict(ct)) for ct in cores]


def deployed_pass_count(graph, policy: str, cores: Sequence[dict] = TWO_CORES) -> int:
    """Neural stages the hard-core builder actually emits — the deployed program."""
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=graph,
        cores_config=[dict(ct) for ct in cores],
        strategy=MappingStrategy.resolve(
            ChipCapabilities(allow_scheduling=True, schedule_policy=policy)
        ),
    )
    return len([stage for stage in hybrid.stages if stage.kind == "neural"])
