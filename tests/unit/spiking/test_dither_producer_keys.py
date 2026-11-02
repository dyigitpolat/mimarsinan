"""[U-DOOR] The phase-dithered encode keys every element by its IR node's channel at unbounded geometry, on both twins.

The deployed program rotates a stage-input position's comb by the channel at unbounded geometry of the IR node output it
holds (``phase_channels.channels_of``: a core node's output channel, a host node's output index). Which IR node an
element is an output of is the mapping's decision alone: a token-wise subsumed encoding is one host node per token
(keyed 0..5 each), a host add over the same token value one node (keyed 0..23), a convolution's cores one node per
position (keyed by output channel). The NF twin read a rule by the value's shape (the last-dim index; then U-DOOR's
first build: a perceptron's channel axis, else a host value's flat index), and no such rule is the deployed one
(``test_exactness_door``'s token forms; ``final_units/udoor/report.md``, Targeted Pass). It now reads its keys from the
IR the program is built from (``map_model_to_ir.mapper_phase_keys``: each element's IR node and output index through
the same channel table, ``phase_channels.source_channels``), so both twins key from one rule.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import unit.spiking.host_gather_forms as hgf
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping import map_model_to_ir as MMI
from mimarsinan.mapping.packing import phase_channels as PCH
from mimarsinan.spiking.cycle_trains import EncodedTrain
from mimarsinan.spiking.spike_trains import uniform_spike_train

T = 15
THETA_ENC = 2.185
#: Forms whose mappers map to several IR nodes per sample (tokens, convolution positions) or gather several producers.
KEYED_FORMS = ("token_skip", "token_residual", "vit_tiny", "pool_join_max", "chain_2", "dense_concat",
               "nested_gather_core_read", "projection")


def _deployed_sources(program) -> dict:
    """Every mapper node's IR sources in the program's own IR (the build leaves its walk memo on the nodes)."""
    out = {}
    for node in program.repr_.execution_order():
        sources = getattr(node, "_ir_sources", None)
        if sources is not None:
            out[id(node)] = np.asarray(sources, dtype=object)
    assert out, "the build left no IR memo to compare with"
    return out


@pytest.mark.parametrize("form", KEYED_FORMS)
def test_every_stage_input_slice_reads_the_keys_the_nf_walk_gives_its_elements(form):
    """The deployed program's channels of each stage-input slice (``channels_of``) equal, output for output, the NF
    keys (``mapper_phase_keys``) of the mapper elements that are that node's outputs."""
    program = hgf.build(form, 0, THETA_ENC, "windowed_lif", False)
    deployed = _deployed_sources(program)
    nf = MMI.mapper_phase_keys(program.repr_)
    held: dict = {}  # IR node id -> {output index: NF key}
    for node_key, sources in deployed.items():
        keys = nf[node_key].reshape(-1)
        for src, key in zip(sources.reshape(-1), keys):
            if int(src.node_id) >= 0:
                assert held.setdefault(int(src.node_id), {}).setdefault(int(src.index), int(key)) == int(key)
    compared = 0
    for stage in program.hybrid.stages:
        for s in getattr(stage, "input_map", None) or ():
            if int(s.node_id) not in held:
                continue
            channels = PCH.channels_of(program.hybrid, s.node_id, s.size).tolist()
            outputs = held[int(s.node_id)]
            for k, channel in enumerate(channels):
                if k in outputs:
                    assert outputs[k] == channel, (form, int(s.node_id), k)
                    compared += 1
    assert compared, f"{form}: no stage-input slice holds a mapper element"


@pytest.mark.parametrize("form", ("token_residual", "pool_join_max", "chain_2"))
def test_the_keys_do_not_move_with_the_core_geometry(form):
    """E1: the NF keys (unbounded cores) equal the channels of the same elements in an IR whose cores hold 4 neurons
    (every perceptron's outputs tiled), a cost-only decision."""
    program = hgf.build(form, 0, THETA_ENC, "windowed_lif", False)
    repr_ = program.repr_
    nf = MMI.mapper_phase_keys(repr_)
    tiled = IRMapping(q_max=1.0, firing_mode="Default", max_axons=256, max_neurons=4).map(repr_)
    try:
        table = PCH.node_phase_channels(tiled)
        for node in repr_.execution_order():
            sources = getattr(node, "_ir_sources", None)
            if sources is not None:
                assert np.array_equal(PCH.source_channels(table, sources), nf[id(node)]), type(node).__name__
    finally:
        repr_.clear_ir_caches()


def test_a_token_wise_encoding_and_a_host_add_over_its_token_value_are_keyed_by_their_ir_nodes():
    program = hgf.build("token_residual", 0, THETA_ENC, "streamed_lif", False)
    nf = MMI.mapper_phase_keys(program.repr_)
    order = program.repr_.execution_order()
    stem = next(n for n in order if getattr(getattr(n, "perceptron", None), "is_encoding_layer", False))
    add = next(n for n in order if type(n).__name__ == "ComputeOpMapper" and "add" in str(getattr(n, "name", "")))
    assert np.array_equal(nf[id(stem)], np.tile(np.arange(6), (4, 1)))  # one host node per token
    assert np.array_equal(nf[id(add)], np.arange(24).reshape(4, 6))  # one host node over the token value


def test_a_convolution_core_is_keyed_by_its_output_channel_and_a_subsumed_one_by_its_flat_index():
    program = hgf.build("pool_join_max", 0, THETA_ENC, "streamed_lif", False)
    nf = MMI.mapper_phase_keys(program.repr_)
    convs = [n for n in program.repr_.execution_order() if getattr(n, "perceptron", None) is not None]
    stem, core = convs[0], convs[1]
    assert getattr(stem.perceptron, "is_encoding_layer", False) and not getattr(core.perceptron, "is_encoding_layer")
    assert np.array_equal(nf[id(stem)], np.arange(4 * 8 * 8).reshape(4, 8, 8))  # one host node, its output index
    assert np.array_equal(nf[id(core)], np.broadcast_to(np.arange(4).reshape(4, 1, 1), (4, 4, 4)))  # per position


def test_the_mapping_reads_once_per_graph_and_leaves_no_ir_memo(monkeypatch):
    program = hgf.build("token_skip", 0, THETA_ENC, "streamed_lif", False)
    program.repr_.clear_ir_caches()
    calls = []
    original = IRMapping.map
    monkeypatch.setattr(IRMapping, "map", lambda self, r: calls.append(1) or original(self, r))
    first = MMI.mapper_phase_keys(program.repr_)
    assert MMI.mapper_phase_keys(program.repr_) is first and len(calls) == 1
    assert all(getattr(n, "_ir_sources", None) is None for n in program.repr_.execution_order())


def test_an_element_no_channel_keys_refuses_by_name_as_its_stage_input_would():
    sources = np.array([IRSource(3, 0), IRSource(3, 1), IRSource(7, 0)], dtype=object)
    keys = PCH.source_channels({3: 10}, sources)
    assert keys.tolist() == [10, 11, PCH.UNKEYED]
    node = object()
    with pytest.raises(PCH.PhaseChannelRefusal, match="no channel at unbounded geometry"):
        PCH.element_channels({id(node): keys}, node, (3,))
    with pytest.raises(PCH.PhaseChannelRefusal, match="IR sources"):
        PCH.element_channels({id(node): np.zeros(4, dtype=np.int64)}, node, (3,))
    with pytest.raises(PCH.PhaseChannelRefusal, match="IR sources"):
        PCH.element_channels({}, node, (3,))


def test_a_constant_source_takes_no_rotation():
    """An off or an always-on axon is no stage input's position (count 0 or T: no rotation moves it)."""
    sources = np.array([IRSource(-1, 0), IRSource(-3, 0), IRSource(-2, 5)], dtype=object)
    assert PCH.source_channels({PCH.RAW_INPUT: 0}, sources).tolist() == [0, 0, 5]


def test_keys_zero_to_n_minus_one_keep_the_last_dim_encode_bit_for_bit():
    rate = torch.rand(5, 18)
    assert torch.equal(uniform_spike_train(rate, T, phase_dither=True, channels=torch.arange(18)),
                       uniform_spike_train(rate, T, phase_dither=True))


def test_per_sample_keys_broadcast_through_the_streamed_row_blocks():
    """The streamed encode builds a cycle in row blocks of its batch; keys of one sample broadcast over each block."""
    torch.manual_seed(1)
    rate = torch.rand(64, 2, 6)
    keys = torch.as_tensor(np.arange(12).reshape(2, 6))
    streamed = EncodedTrain(rate, T, phase_dither=True, channels=keys)
    assert streamed._rows_per_block() < 64
    eager = uniform_spike_train(rate, T, phase_dither=True, channels=keys)
    for t in range(T):
        assert torch.equal(streamed.cycle(t), eager[t]), t
