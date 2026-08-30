"""The twin folds a CONVOLUTION the way the mapper unfolds it.

A conv hop whose receptive field is a PART of its input is mapped to one core
per output position over a shared weight bank; the twin therefore has to fold
per mapped slot table, with a membrane per position, or it is folding a
different network than the one that deploys. Before this, the twin only ever
folded a hop whose effective weight spanned the whole input — so every vehicle
with a body block (a 3x3 conv over a 14x14 map feeds 2744 cells into a 126-slot
weight) was refused by name, which is exactly the class of vehicle that
pretrains highest.

The gate is the same one the whole-input chain answers to: NF window counts and
NF per-cycle RASTERS must equal the packed hard-core executor's at atol=0.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.spiking.serial import SerialFoldUnsupportedError
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

from .test_serial_executors import PER_EVENT_UNBOUNDED, T, build_flow, run_counts
from .test_serial_nf_twin import _nf_model

INPUT_SHAPE = (2, 10, 10)
#: (in, out, kernel, stride, padding) per hop. Hop 1 is the BODY BLOCK: its
#: 27-slot receptive field is a part of the 75-cell map it reads.
STEM = (2, 3, 3, 2, 1)
BODY = (3, 4, 3, 2, 1)
COLLAPSE = (4, 5, 3, 3, 0)


def _int_grid_(module: nn.Linear, low: int, high: int) -> None:
    """A small-integer weight grid: the deployed core's own currency, and wide
    enough that a neuron actually multi-spikes within one cycle."""
    with torch.no_grad():
        module.weight.copy_(
            torch.randint(low, high, module.weight.shape, dtype=module.weight.dtype)
        )
        if module.bias is not None:
            module.bias.zero_()


def build_conv_chain(theta: float = 90.0, *, enc_scale: float = 16.0,
                     n_cores: int = 32, max_axons: int = 64,
                     max_neurons: int = 32):
    """A subsumed conv stem, ONE body block, and a collapse conv.

    The body block is the whole point: a partial receptive field over a map the
    hop does not consume whole. The collapse hop keeps the whole-input case in
    the same fixture, so one witness covers both unfolds.
    """
    torch.manual_seed(11)
    mapper = InputMapper(INPUT_SHAPE)
    convs = []
    for idx, (c_in, c_out, k, s, p) in enumerate((STEM, BODY, COLLAPSE)):
        mapper = Conv2DPerceptronMapper(
            mapper, in_channels=c_in, out_channels=c_out, kernel_size=k,
            stride=s, padding=p, bias=False, use_batchnorm=False,
            name=f"conv{idx}",
        )
        convs.append(mapper)

    repr_ = ModelRepresentation(mapper)
    mark_encoding_layers(repr_)

    upstream_scale = 1.0
    for conv in convs:
        p = conv.perceptron
        encoding = bool(getattr(p, "is_encoding_layer", False))
        _int_grid_(p.layer, *((0, 3) if encoding else (-1, 3)))
        scale = enc_scale if encoding else theta
        p.set_activation_scale(scale)
        p.per_input_scales = torch.full(
            (int(p.layer.weight.shape[1]),), float(upstream_scale)
        )
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
        upstream_scale = scale

    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=max_axons,
        max_neurons=max_neurons,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": max_axons, "max_neurons": max_neurons,
                       "count": n_cores}],
    )
    return repr_, hybrid, convs


def _sample(n: int = 3):
    torch.manual_seed(5)
    return torch.rand(n, *INPUT_SHAPE) * 0.9


def _nf_rates(repr_, x, *, model=None):
    """The NF read in the deployed executor's own dtype.

    The comparator is a tie-sensitive integer decision, so a float32 twin
    against a float64 executor measures DTYPE, not fold order (measured on this
    fixture: at a hot operating point the two disagree by exactly one spike per
    window in float32 and are bit-equal in float64). Matching precision is what
    makes an atol=0 claim about the FOLD.
    """
    model = _nf_model(repr_).double() if model is None else model
    with torch.no_grad():
        out = chip_aligned_segment_forward(
            model, x.double(), T, soma_law=PER_EVENT_UNBOUNDED)
    scale = torch.as_tensor(
        model.get_perceptrons()[-1].activation_scale, dtype=out.dtype)
    return (out / scale).reshape(out.shape[0], -1).to(torch.float64)


def _hcm_rates(hybrid, x):
    flow = build_flow(hybrid, law=PER_EVENT_UNBOUNDED, packed=True)
    out, _ = run_counts(flow, x)
    return (out / float(T)).reshape(out.shape[0], -1).to(torch.float64)


HOT = dict(theta=40.0, enc_scale=8.0)
"""A hot operating point: the collapse hop emits up to 30 spikes in ONE cycle,
so a cycle-atomic fold could not reproduce it even by accident."""


class TestTheFixtureIsTheVehicleShapeThatWasRefused:
    def test_the_body_block_reads_a_map_it_does_not_consume_whole(self):
        _, _, convs = build_conv_chain()
        body = convs[1]
        gather = body.patch_gather((BODY[0], 5, 5))
        assert gather.patch_size == 27
        assert gather.source_size == 75
        assert gather.n_positions == 9
        assert gather.patch_size < gather.source_size, (
            "a degenerate fixture whose conv spans its whole input would prove "
            "nothing new"
        )

    def test_the_mapping_really_tiles_it_into_one_core_per_position(self):
        _, hybrid, _ = build_conv_chain()
        cores = [
            core
            for stage in hybrid.stages
            if stage.hard_core_mapping is not None
            for core in stage.hard_core_mapping.cores
        ]
        assert len(cores) >= 9, (
            f"the body block alone maps to 9 positions; got {len(cores)} cores"
        )

    def test_the_whole_input_twin_refuses_this_fixture(self):
        """Teeth: without the mapper's unfold the twin cannot fold this hop at
        all — the refusal this phase exists to remove."""
        from mimarsinan.models.nn.activations.lif_serial import SerialFoldSlot

        slot = SerialFoldSlot(
            soma_law=PER_EVENT_UNBOUNDED,
            weight=torch.zeros(4, 27), bias=None, theta=1.0, membrane_init=0.0,
        )
        with pytest.raises(SerialFoldUnsupportedError, match="carries 75 slots"):
            slot.feed(torch.zeros(3, 3, 5, 5))


class TestTheConvTwinIsTheDeployment:
    @pytest.mark.parametrize("point", [{}, HOT], ids=["nominal", "hot"])
    def test_nf_equals_packed_hcm_at_atol_zero(self, point):
        repr_, hybrid, _ = build_conv_chain(**point)
        x = _sample()
        nf = _nf_rates(repr_, x)
        hcm = _hcm_rates(hybrid, x)
        assert nf.shape == hcm.shape
        assert torch.equal(nf, hcm)

    def test_the_witness_multi_spikes_and_a_theta_mutation_turns_it_red(self):
        """A gate that stays green under a mutated deployed threshold proves
        nothing; and a fixture that never emits 2 spikes in one cycle cannot
        tell the per-event law from the per-cycle one."""
        repr_, hybrid, _ = build_conv_chain(**HOT)
        x = _sample()
        nf = _nf_rates(repr_, x)
        assert torch.equal(nf, _hcm_rates(hybrid, x))

        rasters = _nf_emission_rasters(repr_, x)
        body_peak = float(rasters[0].max())
        assert body_peak >= 2.0, (
            f"degenerate witness: the BODY BLOCK's peak per-cycle emission is "
            f"{body_peak}, so the per-event law is indistinguishable from the "
            f"per-cycle law exactly where this phase changed the fold"
        )

        for stage in hybrid.stages:
            if stage.hard_core_mapping is not None:
                for core in stage.hard_core_mapping.cores:
                    core.threshold = float(core.threshold) * 2.0
        assert not torch.equal(nf, _hcm_rates(hybrid, x))

    def test_each_mapped_position_carries_its_own_membrane(self):
        """One shared membrane across positions would make the fold depend on
        the order positions were visited in; the counts must be position-wise."""
        repr_, _, _ = build_conv_chain(**HOT)
        x = _sample(2)
        rasters = _nf_emission_rasters(repr_, x)
        body = rasters[0]
        assert body.shape[2:] == (4, 3, 3)
        per_position = body.sum(dim=(0, 1, 2)).flatten()
        assert float(per_position.max()) > float(per_position.min()), (
            "every position emitting identically would hide a shared membrane"
        )


def _nf_emission_rasters(repr_, x):
    """Per-hop ``(T, B, *out)`` emission MULTIPLICITY trains, in hop order."""
    captured: list = []
    from mimarsinan.spiking import segment_policies as sp

    original = sp.run_streamed_lif_cycles

    def spy(policy, **kwargs):
        train, events = original(policy, **kwargs)
        if events is not None:
            captured.append(events.detach().clone())
        return train, events

    sp.run_streamed_lif_cycles = spy
    try:
        _nf_rates(repr_, x)
    finally:
        sp.run_streamed_lif_cycles = original
    return captured
