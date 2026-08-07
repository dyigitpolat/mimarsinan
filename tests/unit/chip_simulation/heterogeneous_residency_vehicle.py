"""A residency chain whose per-ordinal weights AND biases genuinely differ.

The single-bank token vehicle cannot catch an ordinal permutation: every
resident core holds the same payload, so reversing the aliased weight list is
invisible. This vehicle gives each physical-core ordinal a DIFFERENT bank
(distinct matrix and distinct hardware bias) for every pass of one residency
chain, which is what makes ``resident_from.weights[:n]`` order-sensitive and
therefore testable.
"""

import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)

__all__ = ["build_heterogeneous_residency_mapping", "N_BANKS", "N_INSTANCES"]

N_BANKS = 2
N_INSTANCES = 3
_ROWS, _COLS = 5, 4


def build_heterogeneous_residency_mapping():
    """Two banks x three instances over a two-core pool.

    ``try_bank_clustered_passes`` grants one core per bank, so every pass
    places bank 0 on ordinal 0 and bank 1 on ordinal 1: three passes, one
    residency chain, ordinals that must NOT be swapped.
    """
    rng = np.random.default_rng(19)
    banks = {}
    for b in range(N_BANKS):
        # Distinct magnitude per bank so a swap cannot cancel out.
        mat = (rng.normal(size=(_ROWS, _COLS)) + 10.0 * (b + 1)).astype(np.float64)
        banks[b] = WeightBank(id=b, core_matrix=mat)

    nodes = []
    node_id = 0
    for b in range(N_BANKS):
        for inst in range(N_INSTANCES):
            srcs = np.array(
                [IRSource(-2, (b * N_INSTANCES + inst) * (_ROWS - 1) + i)
                 for i in range(_ROWS - 1)]
                + [IRSource(-3, 0)],
                dtype=object,
            )
            nodes.append(NeuralCore(
                id=node_id, name=f"b{b}_i{inst}", input_sources=srcs,
                core_matrix=None, weight_bank_id=b,
                weight_row_slice=(0, _COLS),
                perceptron_index=b, perceptron_output_column=inst,
                latency=0,
                # Bias is per-BANK too: an ordinal swap must move biases as
                # well as weights, and this is what pins that.
                hardware_bias=np.full(_COLS, float(b + 1) * 3.5),
            ))
            node_id += 1

    graph = IRGraph(
        nodes=nodes,
        output_sources=np.array(
            [IRSource(n.id, j) for n in nodes for j in range(_COLS)],
            dtype=object,
        ),
        weight_banks=banks,
    )
    strategy = MappingStrategy.resolve(ChipCapabilities(
        allow_scheduling=True, schedule_policy="bank_clustered",
    ))
    return build_hybrid_hard_core_mapping(
        ir_graph=graph,
        cores_config=[{"max_axons": 8, "max_neurons": 8, "count": N_BANKS}],
        strategy=strategy,
    )
