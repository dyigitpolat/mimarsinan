"""``DeployedNeuronSurvival`` — the deployed per-neuron reality.

Which ORIGINAL output-neuron indices of each perceptron survive pruning (liveness-dead
cores removed + zeroed columns compacted) into the deployed mapping. Reconstructed from
the pruned ir_graph (the deployment authority), it lets per-neuron behavioral gates
project their full (NF) records onto the neurons that are ACTUALLY deployed, so a pruned
deployment is compared apples-to-apples instead of failing a raw shape check.
"""

import numpy as np

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.ir.source import IRSource
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


def test_project_selects_the_surviving_original_columns():
    # Projection is INDEX-BASED: it keeps exactly the surviving ORIGINAL columns,
    # regardless of the values pruned neurons carry (streamed LIF membrane init /
    # folded half-step bias can make a pruned zero-input neuron emit).
    surv = DeployedNeuronSurvival(survivors={0: np.array([0, 2, 3, 5])})
    nf = {0: np.array([
        [0.0, 0.9, 0.0, 0.5, 0.0, 0.7],
        [0.1, 0.2, 0.0, 0.0, 0.8, 0.3],
    ])}
    proj = surv.project(nf)
    np.testing.assert_array_equal(proj[0][0], [0.0, 0.0, 0.5, 0.7])
    np.testing.assert_array_equal(proj[0][1], [0.1, 0.0, 0.0, 0.3])


def test_project_matches_top_m_when_pruned_neurons_are_exactly_zero():
    # On the old top-M premise (pruned neurons contribute exactly 0, all values
    # non-negative) index selection is multiset-identical to top-M-by-value.
    surv = DeployedNeuronSurvival(survivors={0: np.array([0, 2, 3, 5])})
    nf = {0: np.array([
        [0.4, 0.0, 0.9, 0.5, 0.0, 0.7],
        [0.1, 0.0, 0.2, 0.6, 0.0, 0.3],
    ])}
    proj = surv.project(nf)
    m = 4
    top_m = np.sort(nf[0], axis=1)[:, nf[0].shape[1] - m:]
    np.testing.assert_array_equal(np.sort(proj[0], axis=1), top_m)


def test_project_is_correct_where_top_m_would_misselect():
    # A NEGATIVE kept value breaks the top-M premise: top-M would keep the
    # pruned column's 0.0 and drop the kept -1.0. Index selection is correct.
    surv = DeployedNeuronSurvival(survivors={0: np.array([0, 2])})
    nf = {0: np.array([[-1.0, 0.0, 3.0]])}
    proj = surv.project(nf)
    np.testing.assert_array_equal(proj[0], [[-1.0, 3.0]])
    m = 2
    top_m = np.sort(nf[0], axis=1)[:, nf[0].shape[1] - m:]
    assert not np.array_equal(np.sort(proj[0], axis=1), top_m)


def test_project_is_identity_when_full_width():
    surv = DeployedNeuronSurvival(survivors={0: np.array([0, 1, 2, 3])})
    nf = {0: np.arange(8, dtype=float).reshape(2, 4)}
    proj = surv.project(nf)
    np.testing.assert_array_equal(proj[0], nf[0])


def test_project_is_identity_at_deployed_width_even_with_original_indices():
    # A record already at deployed width M stays untouched even when the survivor
    # indices reference the ORIGINAL (wider) numbering — the record was captured
    # post-compaction and reindexing it would be wrong.
    surv = DeployedNeuronSurvival(survivors={0: np.array([1, 4, 5])})
    nf = {0: np.array([[7.0, 8.0, 9.0]])}
    np.testing.assert_array_equal(surv.project(nf)[0], nf[0])


def test_project_passes_through_unknown_perceptron():
    # A perceptron with no survival entry (e.g. gate covers it another way) is untouched.
    surv = DeployedNeuronSurvival(survivors={})
    nf = {7: np.ones((2, 5))}
    assert np.array_equal(surv.project(nf)[7], nf[7])


def test_project_fails_loud_on_unindexable_record():
    # Width neither the deployed count nor indexable by the survivor set: the
    # record is structurally inconsistent with the deployment — never guess.
    import pytest
    surv = DeployedNeuronSurvival(survivors={3: np.array([0, 5])})
    nf = {3: np.ones((2, 4))}  # 4 > 2 survivors but column 5 does not exist
    with pytest.raises(AssertionError, match=r"perceptron 3"):
        surv.project(nf)


def test_project_fails_loud_when_narrower_than_deployed_width():
    # A record NARROWER than the deployed width can never be the deployed
    # observable; identity here would hide a capture bug (pins == vs <=).
    import pytest
    surv = DeployedNeuronSurvival(survivors={1: np.array([0, 1, 2])})
    nf = {1: np.ones((2, 2))}
    with pytest.raises(AssertionError, match=r"perceptron 1"):
        surv.project(nf)


def test_project_fails_loud_on_empty_survivor_set():
    # An empty survivor entry means the perceptron deployed zero neurons; a
    # non-empty record for it is structurally inconsistent.
    import pytest
    surv = DeployedNeuronSurvival(survivors={2: np.array([], dtype=np.int64)})
    nf = {2: np.ones((2, 3))}
    with pytest.raises(AssertionError, match=r"perceptron 2"):
        surv.project(nf)
