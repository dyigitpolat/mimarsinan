"""[U-PFOLD] The forms whose fold one gauge per input cannot hold, as torch code, read by both walk families.

Each form is built from plain torch code by the converter, so every layout is one the converter builds:

- ``token_mixing``: two one-neuron producers ``a`` and ``b`` over the encoding, ``torch.cat([a, b], -1).transpose(1,
  2)`` read by a token-mixing Linear (the fourth verify's smallest form): every input channel of the reader holds one
  element of each producer, each position (a core) one producer; ``token_mixing_3`` reads it with three outputs (the
  positions unequal to the outputs);
- ``tree481``: the fourth verify's generated tree 481 (``L0(v0); L1(v1); L2(v2); L3(v3); cat2(v1,v2,v4); tr(v5);
  view(v6,2x13); L4(v7)``): the reader's rows mix the three producers per position;
- ``tree1301``: the verify's generated tree 1301, whose host-run encoding (per token) is read by a level stage of a
  re-timed program in a part of its outputs (the deployed boundary divides each slice by its own elements' gauges);
- ``shared_rows``: three tokens of two producers (the first and the last token one producer, the middle another), read
  token-wise: two positions hold one fold (one bank), the third another;
- ``conv_seam``: two convolutions of the encoding joined along the height, read by a 3x3 convolution with padding 1:
  the patches on the seam mix the two producers (``conv_seam_wide``: as many output channels as output positions;
  ``conv_seam_tall``: producers five rows high, so the patches inside one producer share one fold);
- ``conv_seam_square``: a seam of producers two rows and one row high, whose input holds as many elements as the
  reader's inputs (``in_channels x 3 x 3``): a stamp that decides by the vector's length reads it as one per input;
- ``mixed_token_mixing``: an absolute host value (a stem without activation) joined along the features with a core
  over it, transposed and read by a token-mixing Linear: a MIXED gather whose reader holds a fold per position.

The timing family is the cascaded identity program with the declared depth-balancing relays (the dead in-segment reads
turned off first, as Soft Core Mapping orders them: merge rule 6); the count family the
Soft Core Mapping sequence with the declared relays (``host_gather_forms.scm_sequence``), both LIF semantics; with
``wq`` the Weight Quantization step's projection (the count family's ``scm_sequence(wq=True)``, the timing family's
NAPQ and ``quantize_ir_graph`` with weight quantization)."""

from __future__ import annotations

import contextlib
import io
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn as nn

import unit.spiking.host_gather_forms as hg

from mimarsinan.chip_simulation.hybrid_run.output_decode import executor_output_values
from mimarsinan.mapping.support.gauge_placement import place_gauges, source_index
from mimarsinan.torch_mapping.converter import convert_torch_model

SAMPLES = 8
S_TIMING = 11


def _init(gen: torch.Generator, modules) -> None:
    with torch.no_grad():
        for m in modules:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                fan = int(np.prod(m.weight.shape[1:]))
                m.weight.copy_((torch.rand(m.weight.shape, generator=gen) * 1.6 - 0.5) / np.sqrt(fan))
                m.bias.copy_(torch.rand(m.bias.shape, generator=gen) * 0.2 - 0.05)


def _lin(i: int, o: int) -> nn.Module:
    return nn.Sequential(nn.Linear(i, o), nn.ReLU())


class TokenMixing(nn.Module):
    shape = (2, 2)

    def __init__(self, gen, width=2):
        super().__init__()
        self.e, self.a, self.b, self.r = _lin(2, 2), _lin(2, 1), _lin(2, 1), _lin(2, width)
        _init(gen, self.modules())

    def forward(self, x):
        e = self.e(x)
        return self.r(torch.cat([self.a(e), self.b(e)], dim=-1).transpose(1, 2)).flatten(1)


class Tree481(nn.Module):
    shape = (2, 4)

    def __init__(self, gen):
        super().__init__()
        self.l0, self.l1, self.l2, self.l3, self.l4 = _lin(4, 6), _lin(6, 3), _lin(3, 2), _lin(2, 4), _lin(13, 2)
        _init(gen, self.modules())

    def forward(self, x):
        v1 = self.l0(x)
        v2 = self.l1(v1)
        v4 = self.l3(self.l2(v2))
        v6 = torch.cat([v1, v2, v4], dim=2).transpose(1, 2)
        return self.l4(v6.reshape(-1, 2, 13)).flatten(1)


class Tree1301(nn.Module):
    """The verify's generated tree 1301 (``L0(v0); L1(v1); L2(v1); tr(v3); cat1(v4,v1); view(v5,3x5); L3(v6)``)."""

    shape = (3, 5)

    def __init__(self, gen):
        super().__init__()
        self.l0, self.l1, self.l2, self.l3 = _lin(5, 3), _lin(3, 3), _lin(3, 2), _lin(5, 4)
        _init(gen, self.modules())

    def forward(self, x):
        v1 = self.l0(x)
        self.l1(v1)
        v5 = torch.cat([self.l2(v1).transpose(1, 2), v1], dim=1)
        return self.l3(v5.reshape(-1, 3, 5)).flatten(1)


class SharedRows(nn.Module):
    shape = (3, 4)

    def __init__(self, gen):
        super().__init__()
        self.e, self.a, self.b, self.r = _lin(4, 3), _lin(3, 3), _lin(3, 3), _lin(3, 2)
        _init(gen, self.modules())

    def forward(self, x):
        e = self.e(x)
        a, b = self.a(e), self.b(e)
        return self.r(torch.cat([a[:, :1], b[:, 1:2], a[:, 2:]], dim=1)).flatten(1)


def _conv(i: int, o: int, k, p: int) -> nn.Module:
    return nn.Sequential(nn.Conv2d(i, o, k, padding=p), nn.ReLU())


class ConvSeam(nn.Module):
    shape = (2, 3, 3)

    def __init__(self, gen, out=2, kernels=(1, 1)):
        super().__init__()
        self.enc = _conv(2, 3, 1, 0)  # registered in the order the draws read them
        self.a, self.b = _conv(3, 2, kernels[0], 0), _conv(3, 2, kernels[1], 0)
        self.reader = _conv(2, out, 3, 1)
        _init(gen, self.modules())

    def forward(self, x):
        e = self.enc(x)
        return self.reader(torch.cat([self.a(e), self.b(e)], dim=2)).flatten(1)


class ConvSeamTall(ConvSeam):
    shape = (2, 5, 3)


class ConvSeamSquare(ConvSeam):
    def __init__(self, gen):
        super().__init__(gen, kernels=((2, 1), (3, 1)))


class MixedTokenMixing(nn.Module):
    shape = (2, 3)

    def __init__(self, gen):
        super().__init__()
        self.stem, self.c, self.r = nn.Linear(3, 2), _lin(2, 2), _lin(2, 2)
        _init(gen, self.modules())

    def forward(self, x):
        s = self.stem(x)
        return self.r(torch.cat([s, self.c(s)], dim=-1).transpose(1, 2)).flatten(1)


FORMS: Dict[str, Callable[[torch.Generator], nn.Module]] = {
    "token_mixing": TokenMixing,
    "token_mixing_3": lambda gen: TokenMixing(gen, width=3),
    "tree481": Tree481,
    "tree1301": Tree1301,
    "shared_rows": SharedRows,
    "conv_seam": ConvSeam,
    "conv_seam_wide": lambda gen: ConvSeam(gen, out=18),
    "conv_seam_tall": ConvSeamTall,
    "conv_seam_square": ConvSeamSquare,
    "mixed_token_mixing": MixedTokenMixing,
}
LINEAR_FORMS = ("token_mixing", "token_mixing_3", "tree481")
CONV_FORMS = ("conv_seam", "conv_seam_wide", "conv_seam_tall", "conv_seam_square")


def model_of(form: str, seed: int) -> nn.Module:
    return FORMS[form](torch.Generator().manual_seed(4_000 + seed)).eval()


def shape_of(form: str) -> tuple:
    return tuple(model_of(form, 0).shape)


def converted(form: str, seed: int, placement: str = "subsume"):
    model = model_of(form, seed)
    shape = tuple(model.shape)
    with torch.no_grad():
        n_out = int(model(torch.zeros(2, *shape)).shape[1])
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(model, shape, n_out, device="cpu", encoding_layer_placement=placement).eval()
    return model, flow, shape


def draw_gauges(flow, perceptrons, shape, seed: int, mode: str, scale: float = 1.0) -> list:
    """Each perceptron's gauge drawn around its calibrated range: one value (``one``), or one per output channel for
    every perceptron with more than one (``drawn``)."""
    rng = np.random.default_rng(6_000 + seed)
    maxima: dict = {}
    handles = []
    for i, p in enumerate(perceptrons):
        def hook(_m, _inp, out, idx=i):
            axis = 1 if int(getattr(perceptrons[idx], "output_channel_axis", -1)) == 1 else -1
            o = out.detach().movedim(axis, -1).reshape(-1, out.shape[axis]).abs().amax(0)
            maxima[idx] = o if idx not in maxima else torch.maximum(maxima[idx], o)
        handles.append(p.activation.register_forward_hook(hook))
    torch.manual_seed(11 + seed)
    with torch.no_grad():
        flow(scale * torch.rand(16, *shape))
    for h in handles:
        h.remove()
    out = []
    for i, p in enumerate(perceptrons):
        top = max(float(maxima[i].max()) if i in maxima else 1.0, 0.05)
        n = int(p.layer.weight.shape[0])
        g = top * rng.uniform(0.35, 1.3, size=n) if mode == "drawn" and n > 1 else np.array(top * rng.uniform(0.35, 1.3))
        out.append(torch.as_tensor(g, dtype=torch.float32).clone())
    return out


def _cores(ir) -> int:
    from mimarsinan.mapping.ir import NeuralCore

    return sum(int(isinstance(n, NeuralCore)) for n in ir.nodes)


def timing_read(form: str, seed: int, gauges: str = "one", wq: bool = False,
                prepare: Callable[[Any, list], None] | None = None) -> dict:
    """The timing family: the cascaded identity program with the declared relays, its walk and its deployed read
    (``prepare`` runs on the flow once its fold is stamped, before the mapping). [merge rule 6] The dead in-segment
    reads are turned off before the relay pass, the order Soft Core Mapping runs them in for a per-cycle cascade."""
    from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.latency.cycle_windows import silence_dead_segment_reads
    from mimarsinan.mapping.latency.depth_balancing import insert_depth_balancing_relays
    from mimarsinan.mapping.latency.ir import IRLatency
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
    from mimarsinan.models.spiking.hybrid.identity_flow import build_identity_spiking_flow
    from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward
    from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
    from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
    from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
        NormalizationAwarePerceptronQuantization)
    from mimarsinan.transformations.quantization_bounds import quantization_bounds

    model, flow, shape = converted(form, seed)
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    perceptrons = list(flow.get_perceptrons())
    drawn = draw_gauges(flow, perceptrons, shape, seed, gauges)
    for p, g in zip(perceptrons, drawn):
        p.set_activation_scale(torch.tensor(1.0) if getattr(p, "is_encoding_layer", False) else g)
    propagate_boundary_input_scales(flow, input_data_scale=1.0)
    for p in perceptrons:
        p.set_activation(TTFSActivation(T=S_TIMING, activation_scale=p.activation_scale,
                                        input_scale=p.input_activation_scale, bias=p.layer.bias,
                                        thresholding_mode="<=", encoding=getattr(p, "is_encoding_layer", False)))
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)
    if prepare is not None:
        prepare(flow, perceptrons)
    bits = hg.WQ_BITS if wq else 8
    _, q_max = quantization_bounds(bits)
    if wq:
        napq = NormalizationAwarePerceptronQuantization(bits, "cpu", rate=1.0, two_scale=False)
        for p in perceptrons:
            if not getattr(p, "is_encoding_layer", False):
                napq.transform(p)
    ir = IRMapping(q_max=q_max, firing_mode="TTFS", max_axons=2048, max_neurons=2048).map(repr_)
    silence_dead_segment_reads(ir)  # Soft Core Mapping's order: the dead reads off, then the relays
    insert_depth_balancing_relays(ir, thresholding_mode="<=", q_max=int(q_max))
    quantize_ir_graph(ir, bits, weight_quantization=wq)
    IRLatency(ir).calculate()
    executor = build_identity_spiking_flow(shape, ir, S_TIMING, None, "TTFS", "TTFS", thresholding_mode="<=",
                                           spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded").eval()
    flow = flow.double().eval()
    torch.manual_seed(7_700 + seed)
    x = torch.rand(SAMPLES, *shape, dtype=torch.float64)
    with torch.no_grad():
        walk = TTFSSegmentForward(flow.get_mapper_repr(), S_TIMING)(x).double().reshape(SAMPLES, -1)
        deployed = executor_output_values(executor, executor(x)).double().reshape(SAMPLES, -1)
    return {"walk": walk, "deployed": deployed, "flow": flow, "ir": ir, "perceptrons": perceptrons,
            "cores": _cores(ir), "x": x}


def count_program(form: str, seed: int, semantics: str = "streamed_lif", gauges: str = "one", wq: bool = False,
                  prepare: Callable[[Any, list], None] | None = None, scale: float = 2.0) -> hg.Program:
    """The count family's program: LIF nodes at the drawn gauges, the Soft Core Mapping sequence with the declared
    relays (``prepare`` runs on the flow once the gauges are set, before the sequence)."""
    from mimarsinan.models.nn.activations import LIFActivation

    _model, flow, shape = converted(form, seed)
    perceptrons = list(flow.get_perceptrons())
    drawn = draw_gauges(flow, perceptrons, shape, seed, gauges, scale=scale)
    for p, g in zip(perceptrons, drawn):
        p.set_activation_scale(g)
        lif = LIFActivation(T=hg.T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    if prepare is not None:
        prepare(flow, perceptrons)
    repr_, ir, hybrid = hg.scm_sequence(flow, perceptrons, wq=wq, retimed=hg.SEMANTICS[semantics][0], relays=True)
    thetas = [float(torch.as_tensor(p.activation_scale).max()) for p in perceptrons]
    return hg.Program(form, shape, flow, repr_, ir, hybrid, perceptrons, thetas, wq)


def count_read(program: hg.Program, semantics: str, seed: int, soma_law: Any = None) -> dict:
    """The count walk (in the program's host-kernel mode, inside the measurement plane) and the deployed read."""
    from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
    from mimarsinan.mapping.support.host_kernel_evaluation import twin_as_program
    from mimarsinan.models.nn.lif_kernels import measurement_plane
    from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver

    law = DEFAULT_SOMA_LAW if soma_law is None else soma_law
    retime = hg.SEMANTICS[semantics][0]

    def walk_of(prog, xs):
        with torch.no_grad(), measurement_plane():
            return SegmentForwardDriver(prog.repr_, hg.T, LifSegmentPolicy(retime=retime, soma_law=law))(xs)

    torch.manual_seed(8_800 + seed)
    x = 2.0 * torch.rand(SAMPLES, *program.shape)
    walk = twin_as_program(program.hybrid, walk_of, program, x).double().reshape(SAMPLES, -1)
    dflow = hg.deployed_flow(program, semantics)
    with torch.no_grad():
        deployed = executor_output_values(dflow, dflow(x)).double().reshape(SAMPLES, -1)
    return {"walk": walk, "deployed": deployed, "x": x}


def assert_equal_reads(read: dict) -> None:
    """Element by element within 1e-5 of max(1, |value|) (the verify's generator's rule)."""
    walk, deployed = read["walk"], read["deployed"]
    rel = (walk - deployed).abs() / deployed.abs().clamp(min=1.0)
    assert float(deployed.abs().max()) > 0.0, "the program reads a live output"
    assert float(rel.max()) <= 1e-5, (float(rel.max()), walk[:2], deployed[:2])


def fold_errors(ir, flow) -> dict:
    """Each reading core's fold against its sources' placed gauges (the m-5 check): per core, the weight on each
    axon over that axon's raw weight (``W_eff[axon, o] act_o / (W[o, slot] u_o)``) against the gauge of the element
    the axon reads (its source's producer gauge, read at its source index). Returns ``{cores checked, max relative
    error}`` over every core of a perceptron holding a stamp (relays excluded)."""
    held = {}
    for node in ir.nodes:
        scale = getattr(node, "activation_scale", None)
        if scale is None:
            module = (getattr(node, "params", None) or {}).get("module")
            module = getattr(module, "perceptron", module)
            scale = getattr(module, "activation_scale", None)
        if scale is not None:
            held[int(node.id)] = torch.as_tensor(scale).detach().double().reshape(-1).numpy()
    worst, checked = 0.0, 0
    order = flow.get_mapper_repr()._exec_order
    for p in flow.get_perceptrons():
        if getattr(p, "per_input_scales", None) is None or getattr(p, "is_encoding_layer", False):
            continue
        index = [getattr(n, "perceptron_index", None) for n in order if getattr(n, "perceptron", None) is p]
        cores = [n for n in ir.get_neural_cores() if index and index[0] is not None
                 and getattr(n, "perceptron_index", None) == index[0] and not getattr(n, "_depth_relay", False)]
        base = p.layer.weight.detach().double().reshape(p.layer.weight.shape[0], -1).t().numpy()
        act = torch.as_tensor(p.activation_scale).detach().double().reshape(-1).numpy()
        for core in cores:
            mat = core.get_core_matrix(ir)
            if mat is None:
                continue
            flat = list(core.input_sources.flatten())
            ids, idx = source_index(np.asarray(flat, dtype=object))
            want = place_gauges(ids, idx, lambda nid: held.get(nid, 1.0))
            rows = min(base.shape[0], mat.shape[0], len(flat))
            sl = getattr(core, "perceptron_output_slice", None) or (0, base.shape[1])
            cols = slice(int(sl[0]), int(sl[1]))
            w = base[:rows, cols]
            a = act[cols] if act.size > 1 else act
            ratio = np.asarray(mat[:rows, : w.shape[1]], dtype=np.float64) * a.reshape(1, -1) / np.where(
                w == 0, np.nan, w)
            with np.errstate(invalid="ignore"):
                got = np.nanmean(ratio, axis=1)
            ok = ~np.isnan(got) & (ids[:rows] >= 0)
            if ok.any():
                checked += 1
                err = np.abs(got[ok] - want[:rows][ok]) / np.maximum(np.abs(want[:rows][ok]), 1e-12)
                worst = max(worst, float(err.max()))
    return {"cores": checked, "max_rel": worst}
