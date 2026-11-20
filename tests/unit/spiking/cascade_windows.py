"""[U-TGAUGE5] Per-cycle cascades for the window rule's tests: programs written as torch code, converted by the
toolchain's converter, built in either walk family and read element by element by the walk and by the deployed
executor.

A program is built one of three ways (``how``):

- ``mapping``: Soft Core Mapping's own window calls (``SoftCoreMappingStep._apply_depth_balancing_relays`` before
  quantization, ``_require_cycle_windows`` on the final IR) under the plan's config as the pipeline folds it, so the
  relays are armed as the recipe arms them;
- ``off``: the same calls with the relays declared off (``lif_depth_balancing_relays = false``);
- ``bare``: no Soft Core Mapping call, as a program built outside the mapping (the property generators, fixtures).

The timing family is ``_build_pair``'s sequence (the encoding marked, the gauges set, the boundary scales propagated,
TTFS activations, per-source scales, the IR at 8 bits, latencies, the identity program on the cascaded schedule); the
count family is the Soft Core Mapping sequence of ``host_gather_forms`` with LIF nodes and one of its two semantics.
``prune`` runs IR pruning where Soft Core Mapping runs it (after quantization and latencies, before the assertion).
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.activation_semantics import fold_spiking_axes
from mimarsinan.chip_simulation.hybrid_run.output_decode import executor_output_values
from mimarsinan.config_schema.recipe_fold import fold_conversion_recipe
from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point
from spiking_points import declaring_fire_while_integrating, point_of_mode
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
from mimarsinan.mapping.support.gauge_placement import place_gauges, source_index
from mimarsinan.mapping.support.host_kernel_evaluation import declare_host_kernel_evaluation, twin_as_program
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.models.nn.lif_kernels import measurement_plane
from mimarsinan.models.spiking.hybrid.identity_flow import build_identity_spiking_flow
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import SoftCoreMappingStep
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.spiking.segment_forward import SegmentForwardDriver
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.transformations.quantization_bounds import quantization_bounds

import unit.spiking.host_gather_forms as hg

SAMPLES = 8
TIMING_BITS = 8
HOWS = ("mapping", "off", "bare")


# ---------------------------------------------------------------------------------------------------------------------
# programs as torch code: a list of instructions over a value list (Linear then ReLU modules, torch.cat joins)
# ---------------------------------------------------------------------------------------------------------------------
class Program(nn.Module):
    """``ops``: ``("lin", source, out_features)`` (a Linear then a ReLU, the converter's perceptron) or ``("cat", [sources])``
    (a join along the features); value 0 is the input, the last value the output. ``dead``: per Linear index, the input
    columns held at zero (an all-zero axon row of its core)."""

    def __init__(self, n_in: int, ops: Sequence[tuple], seed: int, dead: Optional[Dict[int, Sequence[int]]] = None,
                 kill: Optional[Dict[int, Sequence[int]]] = None):
        super().__init__()
        self.ops = [tuple(op) for op in ops]
        widths = [n_in]
        layers = []
        gen = torch.Generator().manual_seed(4_000 + seed)
        for op in self.ops:
            if op[0] == "lin":
                in_f, out_f = widths[op[1]], int(op[2])
                lin = nn.Linear(in_f, out_f)
                with torch.no_grad():
                    lin.weight.copy_((torch.rand(out_f, in_f, generator=gen) * 1.6 - 0.5) / np.sqrt(in_f))
                    lin.bias.copy_(torch.rand(out_f, generator=gen) * 0.2 - 0.05)
                layers.append(nn.Sequential(lin, nn.ReLU()))
                widths.append(out_f)
            else:
                widths.append(sum(widths[s] for s in op[1]))
        with torch.no_grad():
            for index, columns in (dead or {}).items():
                layers[index][0].weight[:, list(columns)] = 0.0
            for index, rows in (kill or {}).items():
                layers[index][0].weight[list(rows)] = 0.0
                layers[index][0].bias[list(rows)] = 0.0
        self.layers = nn.ModuleList(layers)
        self.n_in = n_in

    def forward(self, x):
        values = [x]
        k = 0
        for op in self.ops:
            if op[0] == "lin":
                values.append(self.layers[k](values[op[1]]))
                k += 1
            else:
                values.append(torch.cat([values[s] for s in op[1]], dim=1))
        return values[-1]


def _lin_ops(spec: Sequence[tuple]) -> List[tuple]:
    return [tuple(op) for op in spec]


#: The fourth verify's smallest case and its four minimal forms (q_smallest, q_gap), and the review's dead read (J-1):
#: ``name -> (input width, ops, dead columns per Linear)``. Value 1 is the encoding perceptron (subsumed on the host).
FORMS: Dict[str, Tuple[int, List[tuple], Dict[int, List[int]]]] = {
    # e; a over e; b over a; c over cat(a, b); output cat(b, c): a is read at two depths (c's gap after the alignment 0)
    "smallest": (2, _lin_ops([("lin", 0, 2), ("lin", 1, 1), ("lin", 2, 1), ("cat", [2, 3]), ("lin", 4, 1),
                              ("cat", [3, 5])]), {}),
    # the shallow producer is the segment's entry core, read at two depths
    "fanout_entry": (6, _lin_ops([("lin", 0, 5), ("lin", 1, 4), ("lin", 2, 4), ("cat", [2, 3]), ("lin", 4, 3),
                                  ("cat", [3, 5])]), {}),
    # the shallow producer is an inner core, read at two depths
    "fanout_inner": (6, _lin_ops([("lin", 0, 5), ("lin", 1, 4), ("lin", 2, 4), ("lin", 3, 4), ("cat", [3, 4]),
                                  ("lin", 5, 3), ("cat", [4, 6])]), {}),
    # an entry core with one reader across a gap: the alignment alone raises it (equal every way)
    "single_entry": (6, _lin_ops([("lin", 0, 5), ("lin", 1, 4), ("lin", 1, 4), ("lin", 3, 4), ("cat", [2, 4]),
                                  ("lin", 5, 3), ("cat", [4, 6])]), {}),
    # an inner core with one reader across a gap of three: its producer is read at two depths
    "single_inner": (6, _lin_ops([("lin", 0, 5), ("lin", 1, 4), ("lin", 2, 4), ("lin", 2, 4), ("lin", 4, 4),
                                  ("lin", 5, 4), ("cat", [3, 6]), ("lin", 7, 3), ("cat", [6, 8])]), {}),
    # J-1: A over e; B over A; Q over B; C over cat(A, Q) with every column of Q at zero; output cat(B, C)
    "dead_read": (4, _lin_ops([("lin", 0, 3), ("lin", 1, 2), ("lin", 2, 2), ("lin", 3, 2), ("cat", [2, 4]),
                               ("lin", 5, 2), ("cat", [3, 6])]), {4: [2, 3]}),
    # J-1 with a live unequal-depth join as well: C over cat(A, B, Q), Q's columns at zero
    "dead_read_join": (4, _lin_ops([("lin", 0, 3), ("lin", 1, 2), ("lin", 2, 2), ("lin", 3, 2), ("cat", [2, 3, 4]),
                                    ("lin", 5, 2), ("cat", [3, 6])]), {4: [4, 5]}),
    # [U-TGAUGE6] a chain read only through a dead read: A over e; B over A; Q1 over B; Q2 over Q1; C over cat(A, Q2)
    # with Q2's columns at zero; output cat(B, C): turning C's read off leaves Q2, then Q1, with no reader
    "dead_chain": (4, _lin_ops([("lin", 0, 3), ("lin", 1, 2), ("lin", 2, 2), ("lin", 3, 2), ("lin", 4, 2),
                                ("cat", [2, 5]), ("lin", 6, 2), ("cat", [3, 7])]), {5: [2, 3]}),
}


def make(form: str, seed: int, kill: Optional[Dict[int, Sequence[int]]] = None) -> Tuple[Program, tuple]:
    n_in, ops, dead = FORMS[form]
    return Program(n_in, ops, seed, dead=dead, kill=kill).eval(), (n_in,)


# ---------------------------------------------------------------------------------------------------------------------
# Soft Core Mapping's own window calls under a plan's config
# ---------------------------------------------------------------------------------------------------------------------
def plan_config(family: str, variant: str, *, relays: Optional[bool], steps: int) -> dict:
    """The plan's config as the pipeline folds it (axes, then the conversion recipe); ``relays`` False declares the
    relays off, True or None leaves the recipe's arming."""
    firing = "TTFS" if family == "ttfs" else "Default"
    cfg = {"spiking_family": family, "spiking_variant": variant, "weight_bits": TIMING_BITS, "thresholding_mode": "<=",
           "firing_mode": firing, "simulation_steps": int(steps),
           "spike_generation_mode": "TTFS" if family == "ttfs" else "Uniform"}
    if relays is False:
        cfg["lif_depth_balancing_relays"] = False
    cfg = declaring_fire_while_integrating(cfg)  # [composed head, rule 12] U-LIF2 W16: the form fires in its window
    fold_spiking_axes(cfg)
    fold_conversion_recipe(cfg, resolve_spiking_point(cfg), explicit_keys=set(cfg))
    return cfg


def scm_step(config: dict) -> SoftCoreMappingStep:
    step = SoftCoreMappingStep.__new__(SoftCoreMappingStep)
    step.pipeline = SimpleNamespace(config=config)
    return step


def scm_window_pass(step: SoftCoreMappingStep, ir) -> Tuple[bool, dict]:
    """Soft Core Mapping's window pass before quantization: ``(whether the relays ran, the pass's record)`` (the record
    of a tree whose pass returns the relay count alone holds that count)."""
    enabled, record = step._apply_depth_balancing_relays(ir, DeploymentPlan.resolve(step.pipeline.config))
    return bool(enabled), (dict(record) if isinstance(record, dict) else {"relay_cores_inserted": int(record)})


def scm_before_quantization(step: SoftCoreMappingStep, ir) -> bool:
    """Soft Core Mapping's window pass before quantization; returns whether the relays ran."""
    return scm_window_pass(step, ir)[0]


def scm_on_final_ir(step: SoftCoreMappingStep, ir, enabled: bool) -> None:
    """Soft Core Mapping's window assertion on the final IR (absent on a tree without the rule)."""
    require = getattr(step, "_require_cycle_windows", None)
    if require is not None:
        require(ir, DeploymentPlan.resolve(step.pipeline.config), enabled)


def neural_cores(ir) -> int:
    return sum(int(isinstance(n, NeuralCore)) for n in ir.nodes)


def relay_cores(ir) -> int:
    return sum(int(bool(getattr(n, "_depth_relay", False))) for n in ir.nodes)


@dataclass
class Read:
    walk: torch.Tensor
    deployed: torch.Tensor
    ir: object
    executor: object
    relays: int
    flow: object = None
    cores_mapped: int = 0  # the neural cores the IR mapping built, before any pass
    window: Optional[dict] = None  # the window pass's record (None where no mapping ran)

    def off(self) -> int:
        diff = (self.walk - self.deployed).abs()
        return int((diff > 1e-5 * self.deployed.abs().clamp(min=1.0)).sum())


def _gauges(flow, perceptrons, seed: int, x) -> List[torch.Tensor]:
    """Each perceptron's gauge near its calibrated range (seeded): one value, the third perceptron per channel."""
    maxima: Dict[int, float] = {}
    handles = []
    for i, p in enumerate(perceptrons):
        def hook(_m, _inp, out, idx=i):
            maxima[idx] = max(maxima.get(idx, 0.0), float(out.detach().abs().max()))
        handles.append(p.activation.register_forward_hook(hook))
    with torch.no_grad():
        flow(x)
    for h in handles:
        h.remove()
    rng = np.random.default_rng(3_000 + seed)
    out = []
    for i, p in enumerate(perceptrons):
        top = max(maxima.get(i, 1.0), 0.05)
        n = int(p.layer.weight.shape[0])
        g = top * rng.uniform(0.5, 1.2, size=n) if (i == 2 and n > 1) else np.array(top * rng.uniform(0.5, 1.2))
        out.append(torch.as_tensor(g, dtype=torch.float32).clone())
    return out


def _converted(form: str, seed: int, kill):
    model, shape = make(form, seed, kill)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(model, shape, int(model(torch.zeros(2, *shape)).shape[1]), device="cpu",
                                   encoding_layer_placement="subsume").eval()
    return flow, shape


def timing_activations(flow, gauges: Sequence, *, S: int = 11, perceptrons: Optional[Sequence] = None):
    """The timing family's model before mapping: ``flow``'s perceptrons (``perceptrons``, the flow's by default) at
    ``gauges`` (the encoding at the input's gauge as the calibration pins it), the boundary scales propagated, TTFS
    activations, per-source scales. Returns ``(mapper repr, perceptrons)``."""
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    perceptrons = list(flow.get_perceptrons() if perceptrons is None else perceptrons)
    for p, g in zip(perceptrons, gauges):
        p.set_activation_scale(torch.tensor(1.0) if getattr(p, "is_encoding_layer", False)
                               else torch.as_tensor(g, dtype=torch.float32).clone())
    propagate_boundary_input_scales(flow, input_data_scale=1.0)
    for p in perceptrons:
        p.set_activation(TTFSActivation(T=S, activation_scale=p.activation_scale, input_scale=p.input_activation_scale,
                                        bias=p.layer.bias, thresholding_mode="<=",
                                        encoding=getattr(p, "is_encoding_layer", False)))
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)
    return repr_, perceptrons


def timing_program(flow, shape, gauges: Sequence, seed: int, how: str, *, S: int = 11, prune: bool = False,
                   samples: Optional[torch.Tensor] = None, perceptrons: Optional[Sequence] = None) -> Read:
    """The timing family's cascaded program of ``flow`` (``perceptrons``, the flow's by default, at ``gauges``, the
    encoding at the input's gauge as the calibration pins it) built ``how``: ``_build_pair``'s sequence with Soft Core
    Mapping's window calls where the mapping makes them."""
    repr_, perceptrons = timing_activations(flow, gauges, S=S, perceptrons=perceptrons)
    _, q_max = quantization_bounds(TIMING_BITS)
    ir = IRMapping(q_max=q_max, firing_mode="TTFS", max_axons=2048, max_neurons=2048).map(repr_)
    before = neural_cores(ir)
    step = None if how == "bare" else scm_step(plan_config("ttfs", "cascaded", relays=(how == "mapping") or False,
                                                           steps=S))
    enabled, window = scm_window_pass(step, ir) if step is not None else (False, None)
    quantize_ir_graph(ir, TIMING_BITS, weight_quantization=False)
    IRLatency(ir).calculate()
    if prune:
        prune_ir_graph(ir, simulation_steps=S, spiking_mode="ttfs_cycle_based")
    if step is not None:
        scm_on_final_ir(step, ir, enabled)
    executor = build_identity_spiking_flow(shape, ir, S, None, "TTFS", "TTFS", thresholding_mode="<=",
                                           spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded").eval()
    flow = flow.double().eval()
    for p in perceptrons:  # the walk reads the float64 twin, as the pair's builder hands it over
        p.double()
    if samples is None:
        torch.manual_seed(7_700 + seed)
        samples = torch.rand(SAMPLES, *shape)
    x = samples.double()
    n = int(x.shape[0])
    with torch.no_grad():
        walk = TTFSSegmentForward(flow.get_mapper_repr(), S)(x).double().reshape(n, -1)
        deployed = executor_output_values(executor, executor(x)).double().reshape(n, -1)
    return Read(walk, deployed, ir, executor, relay_cores(ir), flow, before, window)


def count_program(flow, shape, gauges: Sequence, seed: int, how: str, *, semantics: str = "streamed_lif",
                  prune: bool = False, samples: Optional[torch.Tensor] = None,
                  perceptrons: Optional[Sequence] = None) -> Read:
    """The count family's program of ``flow`` (LIF nodes at ``gauges``, ``perceptrons`` the flow's by default) built
    ``how`` under one of the two LIF semantics: ``host_gather_forms``' Soft Core Mapping sequence with the mapping's
    window calls in their places."""
    perceptrons = list(flow.get_perceptrons() if perceptrons is None else perceptrons)
    mark_encoding_layers(flow.get_mapper_repr())
    for p, g in zip(perceptrons, gauges):
        p.set_activation_scale(torch.as_tensor(g, dtype=torch.float32).clone())
        lif = LIFActivation(T=hg.T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    repr_ = flow.get_mapper_repr()
    compute_per_source_scales(repr_, arm_wire_value_ops=True)
    propagate_boundary_input_scales(flow, input_data_scale=1.0)
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=256, max_neurons=256).map(repr_)
    before = neural_cores(ir)
    variant = "synchronized" if semantics == "windowed_lif" else "streamed"
    step = None if how == "bare" else scm_step(plan_config("lif", variant, relays=(how == "mapping") or False,
                                                           steps=hg.T))
    enabled, window = scm_window_pass(step, ir) if step is not None else (False, None)
    if prune:
        IRLatency(ir).calculate()
        ir = prune_ir_graph(ir, simulation_steps=hg.T, spiking_mode="lif")
    if step is not None:
        scm_on_final_ir(step, ir, enabled)
    declare_host_kernel_evaluation(ir, "per_sample")
    hybrid = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=hg.SEMANTICS[semantics][0])
    thetas = [float(torch.as_tensor(p.activation_scale).max()) for p in perceptrons]
    program = hg.Program("cascade", shape, flow, repr_, ir, hybrid, perceptrons, thetas, False)
    if samples is None:
        torch.manual_seed(8_800 + seed)
        samples = 2.0 * torch.rand(SAMPLES, *shape)
    x = samples.float()
    n = int(x.shape[0])

    def walk_of(prog, xs):
        with torch.no_grad(), measurement_plane():
            return SegmentForwardDriver(prog.repr_, hg.T, hg.SEMANTICS[semantics][2]())(xs)

    walk = twin_as_program(program.hybrid, walk_of, program, x).double().reshape(n, -1)
    dflow = hg.deployed_flow(program, semantics)
    with torch.no_grad():
        deployed = executor_output_values(dflow, dflow(x)).double().reshape(n, -1)
    return Read(walk, deployed, ir, dflow, relay_cores(ir), flow, before, window)


def timing(form: str, seed: int, how: str, *, S: int = 11, prune: bool = False,
           kill: Optional[Dict[int, Sequence[int]]] = None) -> Read:
    """The timing family's cascaded program of ``form`` built ``how``."""
    flow, shape = _converted(form, seed, kill)
    torch.manual_seed(11 + seed)
    gauges = _gauges(flow, list(flow.get_perceptrons()), seed, torch.rand(16, *shape))
    return timing_program(flow, shape, gauges, seed, how, S=S, prune=prune)


def count(form: str, seed: int, how: str, *, semantics: str = "streamed_lif", prune: bool = False,
          kill: Optional[Dict[int, Sequence[int]]] = None) -> Read:
    """The count family's program of ``form`` built ``how`` under one of the two LIF semantics."""
    flow, shape = _converted(form, seed, kill)
    torch.manual_seed(11 + seed)
    gauges = _gauges(flow, list(flow.get_perceptrons()), seed, 2.0 * torch.rand(16, *shape))
    return count_program(flow, shape, gauges, seed, how, semantics=semantics, prune=prune)


def fold_errors(read: Read) -> Tuple[int, float]:
    """[the fourth verify's m-5] Each reading core's fold against its sources' placed gauges: per input axon of a core of
    a perceptron (never a depth relay), the fold its effective weight applies (the IR weight times the core's
    threshold over the perceptron's unfolded weight, where both are nonzero) against the gauge its source's producer
    holds the element at (``place_gauges`` at the source index: a core at its threshold, a host op that runs a
    perceptron at that perceptron's, the raw input at one); an axon a plain host op feeds (its gauge is the op's
    emitted one, which the source walk places) is not read here. Returns ``(cores checked, max relative error)``."""
    ir, flow = read.ir, read.flow
    held: Dict[int, np.ndarray] = {}
    for node in ir.nodes:
        scale = getattr(node, "activation_scale", None)
        if scale is None:  # a host op that runs a perceptron (an encoding on the host) holds that perceptron's gauge
            module = (getattr(node, "params", None) or {}).get("module")
            module = getattr(module, "perceptron", module)
            scale = getattr(module, "activation_scale", None)
        if scale is not None:
            held[int(node.id)] = torch.as_tensor(scale).detach().double().reshape(-1).numpy()
    by_index = {getattr(node, "perceptron_index", None): node.perceptron
                for node in flow.get_mapper_repr()._exec_order if getattr(node, "perceptron", None) is not None}
    checked, worst = 0, 0.0
    for core in ir.get_neural_cores():
        p = by_index.get(getattr(core, "perceptron_index", None))
        if p is None or getattr(core, "_depth_relay", False) or getattr(p, "is_encoding_layer", False):
            continue
        mat = core.get_core_matrix(ir)
        base = p.layer.weight.detach().double().reshape(p.layer.weight.shape[0], -1).t().numpy()
        flat = list(core.input_sources.flatten())
        extra = flat[base.shape[0]:]  # the bias row's always-on source after the inputs
        if mat is None or len(flat) < base.shape[0] or not all(getattr(s, "node_id", None) == -3 for s in extra):
            continue
        flat = flat[: base.shape[0]]
        ids, idx = source_index(np.asarray(flat, dtype=object))
        want = place_gauges(ids, idx, lambda nid: held.get(nid, 1.0))
        theta = torch.as_tensor(p.activation_scale).detach().double().reshape(-1).numpy()
        cols = np.asarray(getattr(core, "perceptron_output_slice", None) or (0, base.shape[1]))
        part = base[:, int(cols[0]):int(cols[1])] if cols.size == 2 else base
        theta_part = theta if theta.size == 1 else theta[int(cols[0]):int(cols[1])]
        mat = np.asarray(mat, dtype=np.float64)[: len(flat), : part.shape[1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = mat * theta_part.reshape(1, -1) / np.where(part == 0, np.nan, part)
            got = np.nanmean(ratio, axis=1)
        ok = ~np.isnan(got) & np.isin(ids, list(held))
        if not ok.any():
            continue
        checked += 1
        worst = max(worst, float(np.max(np.abs(got[ok] - want[ok]) / np.maximum(np.abs(want[ok]), 1e-12))))
    return checked, worst


FAMILIES: Dict[str, Callable[..., Read]] = {"timing": timing, "count": count}
