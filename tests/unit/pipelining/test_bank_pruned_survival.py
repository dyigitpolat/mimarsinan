"""[U-IN A-2, F-SURV] The parity gates project the NF through the program's own kept-column rule.

``derive_deployed_neuron_survival`` read only ``pre_pruning_col_mask``, the mask IR pruning leaves on an OWNED core it
compacts at once, while a bank-backed core (a convolution position, a token-wise linear) keeps its pruned output channels
in ``pruned_col_mask`` until Soft Core Mapping compacts it physically (``ir_pruning_compact._attach_bank_metadata``,
``compaction.compact_soft_core_mapping``). Every exact program whose pruning removed a bank output channel was refused
with a false ``NfScmParityError: perceptron N neuron-count mismatch`` (d_tok: 16 of 16 per weight type on chain_2,
pool_join_max and token_skip with killed channels, 8 of 8 on vit_tiny and lenet5, 4 of 4 at t0_51's declared geometry).
It also keyed a multi-column perceptron's survivors by channel alone, so a convolution's or a token-wise perceptron's
record could not be projected even with the right masks. Now one function derives each core's surviving outputs from
the packer's own rule (the IR compaction's ``pre_pruning_col_mask`` composed with ``compaction_keep_indices`` over the
core's current masks), survivors are keyed by (column, channel) as the IR's provenance places them, and each record is
projected through its perceptron's declared ``output_channel_axis``; a core or record the rule cannot derive refuses by
name.

Denominators: the bank-pruned fixtures of d_tok (chain_2, pool_join_max, token_skip and vit_tiny with killed channels,
chain_2's natural pruning at 4-bit seed 0, lenet5 through the model-mask path at wide cores and at t0_51's declared
geometry), 8 seeds, both LIF semantics, float and 4-bit; a TTFS convolution with a pruned bank channel for the TTFS
per-neuron gate; the event-serial law's raster comparison on a pruned program.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import unit.spiking.host_gather_forms as hgf
from conftest import MockPipeline
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.common.refusal.toolchain_refusal import is_toolchain_refusal
from mimarsinan.config_schema.defaults import get_default_deployment_parameters, get_default_platform_constraints
from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.ir.source import IRSource
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper, ReshapeMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.pruning import DeployedNeuronSurvival, derive_deployed_neuron_survival
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM
from mimarsinan.models.nn.activations.ttfs_cycle import TTFSCycleActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.core.nf_scm_parity import (
    assert_nf_scm_parity_or_raise,
    assert_streamed_nf_scm_exact_or_raise,
    compare_normalized_records,
)
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver

THETA_ENC = 2.185
SEEDS = range(8)
SEMANTICS = ("streamed_lif", "windowed_lif")
_KILL = dict(kill=hgf.first_core_kill, prune=True)
#: ``name -> (form, scm_sequence's pruning keywords, wq bits)``: d_tok's bank-pruned fixtures.
CASES = {
    "chain_2_kill": ("chain_2", _KILL, hgf.WQ_BITS),
    "pool_join_max_kill": ("pool_join_max", _KILL, hgf.WQ_BITS),
    "token_skip_kill": ("token_skip", _KILL, hgf.WQ_BITS),
    "vit_tiny_kill": ("vit_tiny", _KILL, hgf.WQ_BITS),
    "chain_2_natural": ("chain_2", dict(prune=True), hgf.WQ_BITS),
    "lenet5_masks": ("lenet5", dict(_KILL, masks=True, geometry=dict(max_axons=2048, max_neurons=2048)), hgf.WQ_BITS),
    # t0_51's declared geometry: 784 x 512 cores, axon coalescing, 5-bit weights.
    "lenet5_masks_t0_51": ("lenet5", dict(_KILL, masks=True,
                                          geometry=dict(max_axons=784, max_neurons=512, allow_coalescing=True)), 5),
}
#: The one natural case that prunes a live bank channel (d_tok 6.1): chain_2, 4-bit, seed 0.
NATURAL_LIVE = ("chain_2_natural", True, 0)


def _survival_api():
    """``(record_layouts, SurvivalRefusal)``, imported where a test reads them (the module collects on any tree)."""
    from mimarsinan.mapping.pruning import SurvivalRefusal, record_layouts

    return record_layouts, SurvivalRefusal


def _program(case: str, semantics: str, wq: bool, seed: int) -> hgf.Program:
    form, pruning, bits = CASES[case]
    return hgf.build(form, seed, THETA_ENC, semantics, wq, wq_bits=bits, **pruning)


def _bank_pruned(program: hgf.Program) -> int:
    """Bank-backed cores whose pruned output channels wait for Soft Core Mapping's compaction."""
    return sum(1 for n in program.ir.get_neural_cores() if n.weight_bank_id is not None
               and n.pruned_col_mask is not None and any(n.pruned_col_mask))


class _NF(nn.Module):
    def __init__(self, program: hgf.Program, policy):
        super().__init__()
        self.repr_ = program.repr_
        self._perceptrons = nn.ModuleList(list(program.perceptrons))
        self._policy = policy

    def get_perceptrons(self):
        return list(self._perceptrons)

    def forward(self, x):
        return SegmentForwardDriver(self.repr_, hgf.T, self._policy())(x)


def _streamed_stub(program: hgf.Program, **extra):
    cfg = get_default_deployment_parameters()
    cfg.update(get_default_platform_constraints())
    cfg.update({"spiking_family": "lif", "spiking_variant": "streamed", "simulation_steps": hgf.T,
                "input_shape": program.shape, "device": "cpu", "weight_quantization": bool(program.wq), **extra})
    return SimpleNamespace(config=cfg)


def _readings(case: str, semantics: str, wq: bool, seed: int) -> dict:
    """The projection's widths against the record's, the projected NF against the SCM at atol 0, and (streamed) the
    pipeline's streamed exactness gate."""
    program = _program(case, semantics, wq, seed)
    x = hgf.samples(program, 32, seed)
    out: dict = {"bank_pruned": _bank_pruned(program)}
    if semantics == "streamed_lif":
        try:
            assert_streamed_nf_scm_exact_or_raise(_streamed_stub(program), _NF(program, hgf.SEMANTICS[semantics][2]),
                                                  program.ir, x)
            out["gate"] = "pass"
        except AssertionError as exc:
            out["gate"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    _o, nf = hgf.nf_window_counts(program, x, semantics)
    _o, scm = hgf.scm_window_counts(program, x, semantics)
    try:
        record_layouts, _refusal = _survival_api()
    except ImportError as exc:  # a tree without the projection's layouts: the gate's reading above is the red one
        out["projection"] = f"{type(exc).__name__}: {exc}"
        return out
    projected = derive_deployed_neuron_survival(program.ir).project(nf, layouts=record_layouts(program.perceptrons))
    out["widths"] = {pi: (int(projected[pi].shape[1]), int(scm[pi].shape[1])) for pi in scm
                     if projected[pi].shape[1] != scm[pi].shape[1]}
    if not out["widths"]:
        out["differ"] = compare_normalized_records({pi: projected[pi] for pi in scm}, scm, atol=0.0)[0]
    return out


def _failures(case: str, semantics: str, wq: bool, seeds) -> dict:
    failing = {}
    for seed in seeds:
        r = _readings(case, semantics, wq, seed)
        if (case, wq, seed) == NATURAL_LIVE or not case.endswith("natural"):
            assert r["bank_pruned"] > 0, f"{case}: no bank-backed core carries a pruned channel (vacuous case)"
        if "projection" in r or r["widths"] or r.get("differ") or r.get("gate", "pass") != "pass":
            failing[seed] = r
    return failing


@pytest.mark.parametrize("case,semantics,wq", [("chain_2_kill", "streamed_lif", False),
                                               ("token_skip_kill", "windowed_lif", True),
                                               ("lenet5_masks", "streamed_lif", True),
                                               ("chain_2_natural", "streamed_lif", True)])
@pytest.mark.timeout(300)
def test_a_bank_pruned_program_projects_to_its_record_at_one_case(case, semantics, wq):
    assert _failures(case, semantics, wq, (0,)) == {}


@pytest.mark.parametrize("wq", (False, True), ids=("float", "wq4"))
@pytest.mark.parametrize("semantics", SEMANTICS)
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_every_bank_pruned_program_projects_to_its_record_and_passes_the_gate(case, semantics, wq):
    assert _failures(case, semantics, wq, SEEDS) == {}


@pytest.mark.timeout(600)
def test_the_event_serial_raster_comparison_projects_too():
    """Under ``firing_granularity: per_event`` the streamed gate compares per-cycle rasters beside the counts; a pruned
    program's rasters project through the same survivors (each neuron's T cycles)."""
    law = SomaLaw.resolve({"firing_granularity": "per_event"})
    semantics = "per_event_streamed"
    hgf.SEMANTICS[semantics] = (False, dict(spiking_mode="lif", cycle_accurate_lif_forward=True,
                                            pass_transfer=VERBATIM, soma_law=law, firing_mode=law.firing_mode),
                                lambda: LifSegmentPolicy(soma_law=law))
    try:
        for seed in range(2):
            program = hgf.build("chain_2", seed, THETA_ENC, semantics, False, **_KILL)
            assert _bank_pruned(program) > 0
            x = hgf.samples(program, 16, seed)
            assert_streamed_nf_scm_exact_or_raise(_streamed_stub(program, firing_granularity="per_event"),
                                                  _NF(program, hgf.SEMANTICS[semantics][2]), program.ir, x)
    finally:
        hgf.SEMANTICS.pop(semantics, None)


# ----- the TTFS per-neuron gate on a convolution with a pruned bank channel -----

_TTFS_T = 4


class _ToyFlow(nn.Module):
    def __init__(self, repr_, perceptrons):
        super().__init__()
        self._repr = repr_
        self._perceptrons = nn.ModuleList(perceptrons)

    def get_perceptrons(self):
        return list(self._perceptrons)

    def get_mapper_repr(self):
        return self._repr

    def forward(self, x):
        return self._repr(x)


def _ttfs_conv_toy(kill: bool):
    """input (1, 6, 6) -> a convolution core (4 channels, a weight bank over 36 positions) -> flatten -> a dense core;
    with ``kill``, channel 1 is zeroed and IR pruning removes it from the bank."""
    from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph

    torch.manual_seed(0)
    conv = Conv2DPerceptronMapper(InputMapper((1, 6, 6)), 1, 4, 3, padding=1, use_batchnorm=False, name="c1")
    head = Perceptron(4, 4 * 6 * 6, normalization=nn.Identity())
    out = PerceptronMapper(ReshapeMapper(conv, (4 * 6 * 6,)), head)
    perceptrons = [conv.perceptron, head]
    for p in perceptrons:
        act = TTFSCycleActivation(T=_TTFS_T, activation_scale=1.0)
        p.base_activation = act
        p.activation = act
        p.set_activation_scale(1.0)
    with torch.no_grad():
        conv.perceptron.layer.weight.mul_(3.0)
        conv.perceptron.layer.bias.copy_(torch.linspace(0.05, 0.3, 4))
        if kill:
            conv.perceptron.layer.weight[1] = 0.0
            conv.perceptron.layer.bias[1] = 0.0
    repr_ = ModelRepresentation(out)
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)
    ir = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024).map(repr_)
    for node in ir.nodes:
        if isinstance(node, NeuralCore):
            node.threshold = 1.0
            node.parameter_scale = torch.tensor(1.0)
    if kill:
        ir = prune_ir_graph(ir, store_heatmap=False, simulation_steps=_TTFS_T, spiking_mode="ttfs_quantized")
    IRLatency(ir).calculate()
    return _ToyFlow(repr_, perceptrons).double().eval(), ir


def _ttfs_pipeline():
    p = MockPipeline()
    p.config.update({"spiking_mode": "ttfs_quantized", "firing_mode": "TTFS", "spike_generation_mode": "TTFS",
                     "thresholding_mode": "<=", "simulation_steps": _TTFS_T, "input_shape": (1, 6, 6)})
    return p


@pytest.mark.parametrize("kill", (False, True), ids=("unpruned", "bank_pruned"))
def test_the_ttfs_per_neuron_gate_projects_a_pruned_bank_channel(kill):
    model, ir = _ttfs_conv_toy(kill)
    if kill:
        assert any(n.weight_bank_id is not None and n.pruned_col_mask is not None and any(n.pruned_col_mask)
                   for n in ir.get_neural_cores())
    torch.manual_seed(1)
    samples = torch.rand(3, 1, 6, 6, dtype=torch.float64)
    assert assert_nf_scm_parity_or_raise(_ttfs_pipeline(), model, ir, samples, atol=1e-9) == 0.0


# ----- the shared rule on hand-built cores -----

def _owned(node_id, out_slice, pre_mask, width, column=None, pi=0):
    return NeuralCore(id=node_id, name=f"c{node_id}",
                      input_sources=np.array([IRSource(node_id=-2, index=i) for i in range(4)]),
                      core_matrix=np.zeros((4, width)), perceptron_index=pi, perceptron_output_slice=out_slice,
                      perceptron_output_column=column, pre_pruning_col_mask=pre_mask)


def _banked(node_id, column, out_slice, pruned_cols, pi=0):
    n = out_slice[1] - out_slice[0]
    return NeuralCore(id=node_id, name=f"b{node_id}",
                      input_sources=np.array([IRSource(node_id=-2, index=i) for i in range(4)]),
                      core_matrix=None, weight_bank_id=0, weight_row_slice=(0, n), perceptron_index=pi,
                      perceptron_output_slice=out_slice, perceptron_output_column=column,
                      pruned_row_mask=[False] * 4, pruned_col_mask=pruned_cols)


def _graph(nodes):
    return IRGraph(nodes=nodes, output_sources=np.array([IRSource(node_id=nodes[0].id, index=0)]))


def test_a_bank_core_survives_by_its_pending_compaction():
    """Two positions of one bank (channels 0..3, channel 2 pruned on both): survivors are (column, channel) pairs."""
    survival = derive_deployed_neuron_survival(_graph([_banked(0, 0, (0, 4), [False, False, True, False]),
                                                       _banked(1, 1, (0, 4), [False, False, True, False])]))
    assert list(survival.survivors[0]) == [0, 1, 3, 0, 1, 3]
    assert list(survival.columns[0]) == [0, 0, 0, 1, 1, 1]


def test_a_multi_column_record_projects_through_its_channel_axis():
    survival = derive_deployed_neuron_survival(_graph([_banked(0, 0, (0, 3), [False, True, False]),
                                                       _banked(1, 1, (0, 3), [False, True, False])]))
    # channel-first (C=3, P=2): element c * 2 + p; channels-last: p * 3 + c.
    record = np.arange(6, dtype=float).reshape(1, 6)
    first = survival.project({0: record}, layouts={0: (0, 3)})[0]
    last = survival.project({0: record}, layouts={0: (-1, 3)})[0]
    assert sorted(first[0].tolist()) == [0.0, 1.0, 4.0, 5.0]
    assert sorted(last[0].tolist()) == [0.0, 2.0, 3.0, 5.0]


def test_a_multi_column_record_without_its_layout_refuses_by_name():
    survival = derive_deployed_neuron_survival(_graph([_banked(0, 0, (0, 3), [False, True, False]),
                                                       _banked(1, 1, (0, 3), [False, True, False])]))
    _layouts, refusal = _survival_api()
    with pytest.raises(refusal) as caught:
        survival.project({0: np.zeros((1, 6))})
    assert is_toolchain_refusal(caught.value) and "perceptron 0" in str(caught.value)


def test_a_core_whose_kept_set_cannot_be_derived_refuses_by_name():
    # a pre-pruning mask over 5 columns on a tile of 6
    _layouts, refusal = _survival_api()
    with pytest.raises(refusal) as caught:
        derive_deployed_neuron_survival(_graph([_owned(0, (0, 6), [False] * 5, 5)]))
    assert is_toolchain_refusal(caught.value) and "c0" in str(caught.value)


def test_an_owned_tile_composes_its_ir_compaction_with_the_packer_rule():
    # columns 1 and 4 of 6 compacted away by IR pruning; nothing pending for the packer
    survival = derive_deployed_neuron_survival(_graph([_owned(0, (0, 6), [False, True, False, False, True, False], 4)]))
    assert list(survival.survivors[0]) == [0, 2, 3, 5]
    assert list(survival.columns[0]) == [0, 0, 0, 0]


def test_record_layouts_read_each_perceptrons_declared_channel_axis():
    conv = Conv2DPerceptronMapper(InputMapper((1, 6, 6)), 1, 4, 3, padding=1, use_batchnorm=False)
    dense = Perceptron(5, 8, normalization=nn.Identity())
    record_layouts, _refusal = _survival_api()
    assert record_layouts([conv.perceptron, dense]) == {0: (0, 4), 1: (-1, 5)}
    assert DeployedNeuronSurvival(survivors={}).project({3: np.ones((2, 2))}, layouts={})[3].shape == (2, 2)
