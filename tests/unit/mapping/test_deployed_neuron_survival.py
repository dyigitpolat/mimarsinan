"""``DeployedNeuronSurvival`` — the deployed per-neuron reality.

Which ORIGINAL output-neuron indices of each perceptron survive pruning (liveness-dead
cores removed + zeroed columns compacted) into the deployed mapping. Reconstructed from
the pruned ir_graph (the deployment authority), it lets per-neuron behavioral gates
project their full (NF) records onto the neurons that are ACTUALLY deployed, so a pruned
deployment is compared apples-to-apples instead of failing a raw shape check.
"""

import numpy as np

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.ir.types import IRSource
from mimarsinan.mapping.pruning.deployed_neuron_survival import (
    DeployedNeuronSurvival,
    derive_deployed_neuron_survival,
)


def _core(node_id, perceptron_index, out_slice, pre_pruning_col_mask, n_survivors):
    n_axons = 4
    return NeuralCore(
        id=node_id,
        name=f"c{node_id}",
        input_sources=np.array([IRSource(node_id=-2, index=i) for i in range(n_axons)]),
        core_matrix=np.zeros((n_axons, n_survivors), dtype=np.float64),
        perceptron_index=perceptron_index,
        perceptron_output_slice=out_slice,
        pre_pruning_col_mask=pre_pruning_col_mask,
    )


def _graph(nodes):
    return IRGraph(
        nodes=nodes,
        output_sources=np.array([IRSource(node_id=nodes[0].id, index=0)]),
    )


def test_survivors_drop_pruned_columns():
    # perceptron 0 has 6 original output neurons; columns 1 and 4 pruned.
    mask = [False, True, False, False, True, False]
    surv = derive_deployed_neuron_survival(_graph([_core(0, 0, (0, 6), mask, 4)]))
    np.testing.assert_array_equal(surv.survivors[0], [0, 2, 3, 5])


def test_full_survival_when_no_mask():
    # No pruning ran -> pre_pruning_col_mask None -> the whole slice survives.
    surv = derive_deployed_neuron_survival(_graph([_core(0, 0, (0, 4), None, 4)]))
    np.testing.assert_array_equal(surv.survivors[0], [0, 1, 2, 3])


def test_multiple_tiles_of_one_perceptron_concatenate():
    surv = derive_deployed_neuron_survival(_graph([
        _core(0, 0, (0, 3), [False, True, False], 2),   # survivors 0, 2
        _core(1, 0, (3, 6), [True, False, False], 2),   # survivors 4, 5
    ]))
    np.testing.assert_array_equal(surv.survivors[0], [0, 2, 4, 5])


def test_survivor_count_must_match_compacted_matrix_width():
    # mask keeps 4 but matrix width claims 3 -> inconsistency must fail loud.
    import pytest
    bad = _core(0, 0, (0, 6), [False, True, False, False, True, False], 3)
    with pytest.raises(AssertionError):
        derive_deployed_neuron_survival(_graph([bad]))


def test_project_keeps_the_deployed_count_largest_per_sample():
    # M=4 survivors; pruned/dead neurons contribute 0, so projection keeps the 4
    # largest values per sample (dropping the N-M smallest, which are zeros).
    surv = DeployedNeuronSurvival(survivors={0: np.array([0, 2, 3, 5])})
    nf = {0: np.array([
        [0.0, 0.9, 0.0, 0.5, 0.0, 0.7],
        [0.1, 0.2, 0.0, 0.0, 0.8, 0.3],
    ])}
    proj = surv.project(nf)
    np.testing.assert_array_equal(np.sort(proj[0][0]), [0.0, 0.5, 0.7, 0.9])
    np.testing.assert_array_equal(np.sort(proj[0][1]), [0.1, 0.2, 0.3, 0.8])


def test_project_is_identity_when_full_width():
    surv = DeployedNeuronSurvival(survivors={0: np.array([0, 1, 2, 3])})
    nf = {0: np.arange(8, dtype=float).reshape(2, 4)}
    proj = surv.project(nf)
    np.testing.assert_array_equal(proj[0], nf[0])


def test_project_passes_through_unknown_perceptron():
    # A perceptron with no survival entry (e.g. gate covers it another way) is untouched.
    surv = DeployedNeuronSurvival(survivors={})
    nf = {7: np.ones((2, 5))}
    assert np.array_equal(surv.project(nf)[7], nf[7])
