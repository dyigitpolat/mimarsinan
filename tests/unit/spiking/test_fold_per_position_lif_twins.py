"""[U-PFOLD] The LIF twins over a fold per position, and the LIF activation's channel axis (corrections 8 and 11).

The NF's per-cycle deployed fold (``segment_policy_lif_fold``, inside the measurement plane) and its event-serial fold
(``segment_policy_lif_serial``, under a per-event soma law) decompose a hop through the mapper's slot tables against
the perceptron's effective weight; over a fold per position each core's table is folded against its own position's
weight (a convolution's cores are its output positions). The LIF activation divides a per-channel scale along the
producer's declared channel axis (a convolution's is its channel-first axis), not the last axis (S-12: a crash where
the width differs from the channels, a silent mis-scaling where it equals them).

Red on U-TGAUGE4's tree: the convolution seam's count read was silently off (one gauge per producer) or failed unnamed
(drawn per-channel gauges); the LIF activation divided a channel-first output on its last axis.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import unit.spiking.host_gather_forms as hg
import unit.spiking.position_fold_forms as pf

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


@pytest.mark.parametrize("seed", (0, 1, 2))
@pytest.mark.parametrize("form", pf.CONV_FORMS)
def test_the_per_cycle_deployed_fold_decomposes_each_core_by_its_position(form, seed, monkeypatch):
    """Inside the measurement plane the convolution seam's reader arms the per-cycle deployed fold, which agrees with
    the twin's own pre-activation on every cycle (each core's table against its own position's weight), and the
    count walk equals the deployed read."""
    import mimarsinan.spiking.segment_policy_lif_serial as serial

    armed: list = []

    class Recording(serial.CycleFoldSlot):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            armed.append(self)

    monkeypatch.setattr(serial, "CycleFoldSlot", Recording)
    program = pf.count_program(form, seed, "streamed_lif")
    read = pf.count_read(program, "streamed_lif", seed)
    reader = program.perceptrons[-1]
    mine = [slot for slot in armed if getattr(slot, "perceptron", None) is reader]
    assert mine and all(slot.agrees for slot in mine)
    pf.assert_equal_reads(read)


@pytest.mark.parametrize("seed", (0, 1))
def test_the_event_serial_fold_decomposes_each_core_by_its_position(seed):
    """Under a per-event soma law the event-serial twin folds the convolution seam's reader per core, each core
    against its own position's weight, and the count walk equals the deployed read under the same law."""
    from mimarsinan.chip_simulation.soma_law import SomaLaw
    from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
    from mimarsinan.chip_simulation.hybrid_run.output_decode import executor_output_values

    law = SomaLaw(firing_mode="Novena", thresholding_mode="<=", firing_granularity="per_event",
                  membrane_arithmetic="unbounded", membrane_bits=0)
    program = pf.count_program("conv_seam", seed, "streamed_lif")
    read = pf.count_read(program, "streamed_lif", seed, soma_law=law)
    flow = SpikingHybridCoreFlow(program.shape, program.hybrid, hg.T, firing_mode=law.firing_mode,
                                 spike_mode="Uniform", thresholding_mode=law.thresholding_mode, spiking_mode="lif",
                                 cycle_accurate_lif_forward=True, lif_execution_synchronized=False,
                                 soma_law=law).eval()
    with torch.no_grad():
        deployed = executor_output_values(flow, flow(read["x"])).double().reshape(read["walk"].shape)
    pf.assert_equal_reads({"walk": read["walk"], "deployed": deployed})


@pytest.mark.parametrize("width", (3, 4))
def test_the_lif_activation_divides_a_per_channel_scale_along_the_declared_channel_axis(width):
    """A convolution's LIF activation (channel-first output ``(B, C, H, W)``) divides each channel by its own scale:
    with the width unequal to the channels it crashed, with the width equal it divided each column by a channel's
    scale."""
    p = Perceptron(4, 6)
    p.output_channel_axis = 1
    scale = torch.tensor([0.5, 1.0, 2.0, 4.0])
    p.set_activation_scale(scale)
    lif = LIFActivation(T=8, activation_scale=p.activation_scale)
    p.base_activation = lif
    p.activation = lif
    x = torch.rand(2, 4, 3, width) * 4.0
    with torch.no_grad():
        out = p.activation(x)
    want = LIFActivation(T=8, activation_scale=torch.tensor(1.0))
    for c in range(4):
        expected = want(x[:, c] / scale[c]) * scale[c]
        torch.testing.assert_close(out[:, c], expected)


@pytest.mark.parametrize("semantics", ("streamed_lif", "windowed_lif"))
@pytest.mark.parametrize("seed", (0, 1, 2, 3))
def test_the_convolution_seam_reads_at_drawn_per_channel_gauges(semantics, seed):
    """Correction 11's test: the convolution seam with drawn per-channel gauges in the count family (on the design's
    probe 4 of 6 failed unnamed at the LIF activation's last-axis division)."""
    program = pf.count_program("conv_seam", seed, semantics, gauges="drawn")
    pf.assert_equal_reads(pf.count_read(program, semantics, seed))


@pytest.mark.parametrize("width", (3, 4))
def test_the_timing_activation_reads_a_per_channel_scale_along_the_declared_channel_axis(width):
    """The timing activation of a convolution divides each channel by its own scale (S-12's class in the timing
    family): with the width equal to the channels it divided each column by a channel's scale."""
    from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation

    p = Perceptron(4, 6)
    p.output_channel_axis = 1
    scale = torch.tensor([0.5, 1.0, 2.0, 4.0])
    p.activation = TTFSActivation(T=8, activation_scale=scale)
    x = torch.rand(2, 4, 3, width) * 4.0
    with torch.no_grad():
        out = p.activation(x)
    for c in range(4):
        want = TTFSActivation(T=8, activation_scale=float(scale[c]))(x[:, c])
        torch.testing.assert_close(out[:, c], want)


def test_a_host_convolution_s_per_channel_scale_is_laid_out_per_element():
    """The deployed program's divisor table holds a host-run convolution's per-channel scale over its channel-first
    output per element (each channel's block), as its buffer holds it: one entry per channel failed unnamed at the
    boundary's division (the convolution seam at drawn gauges)."""
    from mimarsinan.mapping.ir import ComputeOp
    from mimarsinan.mapping.support.activation_scales import compute_node_output_scales

    program = pf.count_program("conv_seam", 0, gauges="drawn")
    scales = compute_node_output_scales(program.ir)
    encoding = program.perceptrons[0]
    (op,) = [n for n in program.ir.nodes if isinstance(n, ComputeOp)
             and getattr((n.params or {}).get("module"), "perceptron", None) is encoding]
    channel = torch.as_tensor(encoding.activation_scale).double()
    want = channel.repeat_interleave(int(np.prod(op.output_shape)) // channel.numel()).numpy()
    assert np.array_equal(np.asarray(scales[op.id]), want)


def test_a_perceptron_restored_before_its_convolution_mapper_keeps_its_axis():
    """[S-12] Reconstruction (a copy, a cache load) can restore a convolution mapper, which re-declares its perceptron
    channel-first, before the perceptron's own state: the declaration waits for the perceptron, whose activations
    carry their axis with them."""
    import copy

    _model, flow, _shape = pf.converted("conv_seam", 0)
    node = next(n for n in flow.get_mapper_repr()._exec_order
                if hasattr(n, "patch_gather") and getattr(n, "perceptron", None) is not None)
    perceptron = node.perceptron
    perceptron.activation = LIFActivation(T=hg.T, activation_scale=torch.tensor([1.0, 2.0]))
    assert perceptron.activation.channel_axis == 1
    perceptron.owner = [node]  # a back reference: the mapper's state is restored inside the perceptron's
    copied = copy.deepcopy(perceptron)
    assert copied.activation.channel_axis == 1 and copied.output_channel_axis == 1


def test_a_channel_axis_declared_after_the_activation_reaches_it():
    """[S-12] A perceptron whose channel-first axis is declared after its activation is set (a convolution mapper
    declares it on a built perceptron) stamps it on its spiking activations then."""
    p = Perceptron(4, 6)
    lif = LIFActivation(T=8, activation_scale=torch.tensor([0.5, 1.0, 2.0, 4.0]))
    p.base_activation = lif
    p.activation = lif
    assert lif.channel_axis == -1
    p.output_channel_axis = 1
    assert lif.channel_axis == 1
