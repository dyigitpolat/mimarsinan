"""The encoding-skip program: the first ResNet block at minimum size, the seam of the C5 T-LIF parity refusal.

``s = relu(stem(x))`` is the encoding perceptron, subsumed on the host; ``z = relu(s + fc2(relu(fc1(s))))`` puts fc1 on a
LIF core and fc2, the residual add and its relu on the host; ``relu(fc3(z))`` is the LIF core that reads the join. The
add's ``ScaleNormalizingWrapper`` decodes the stem source at the stem's ``kappa_fold`` (theta_enc), while the IR records
the host-run stem's buffer raw (``kappa_buf = 1``), so the join is where the two currencies meet. The program is built
by the Soft Core Mapping sequence (per-source scales with the wire value ops armed, boundary scales, the identity
program), optionally weight-quantized as the Weight Quantization step and ``quantize_ir_graph`` quantize it, and read
on both twins: the torch NF walk (``SegmentForwardDriver``) and the deployed executors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.autograd import ChipInputQuantizer
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.core.nf_scm_parity import _group_record_by_perceptron
from mimarsinan.spiking.lif_utils import arm_integer_membrane_lattice
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)

T = 8
D_IN, D = 8, 6
#: theta of fc1 and fc3 (fc2 is a host op); theta_enc, the stem's, is the parameter of every test.
THETAS = (0.5, 1.0)
#: the C5 first-core values (stem theta, first-core theta) for the weight-quantized program: fc1's integer threshold
#: lands at 1, so exact ties on the membrane lattice are common.
WQ_THETA_ENC, WQ_THETAS, WQ_BITS = 2.289466619491577, (0.5244447588920593, 1.0), 4

#: The LIF semantics of one program, ``name -> (re-timed level stages, flow kwargs, NF policy)``: the two a deployment
#: selects (streamed; windowed, the per-hop re-timed lifsync) and the synchronized count executor the X3a screen runs.
SEMANTICS: Dict[str, Tuple[bool, dict, Callable[[], LifSegmentPolicy]]] = {
    "streamed_lif": (False, dict(spiking_mode="lif", cycle_accurate_lif_forward=True, pass_transfer=VERBATIM),
                     lambda: LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW)),
    "windowed_lif": (True, dict(spiking_mode="lif", cycle_accurate_lif_forward=True, pass_transfer=COLLAPSE),
                     lambda: LifSegmentPolicy(retime=True, soma_law=DEFAULT_SOMA_LAW)),
    "synchronized_lif": (False, dict(spiking_mode="lif", cycle_accurate_lif_forward=True,
                                     lif_execution_synchronized=True, pass_transfer=COLLAPSE),
                         lambda: LifSegmentPolicy(synchronized=True, soma_law=DEFAULT_SOMA_LAW)),
}
DEPLOYED_SEMANTICS = ("streamed_lif", "windowed_lif")


class EncodingSkipBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(D_IN, D)
        self.r0 = nn.ReLU()
        self.fc1 = nn.Linear(D, D)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(D, D)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(D, D)
        self.r3 = nn.ReLU()

    def forward(self, x):
        s = self.r0(self.stem(x.flatten(1)))
        z = self.r2(s + self.fc2(self.r1(self.fc1(s))))
        return self.r3(self.fc3(z))


class NFModel(nn.Module):
    """The torch twin as the pipeline holds it after LIF adaptation: ``forward`` IS the NF segment walk."""

    def __init__(self, program: "Program", policy: LifSegmentPolicy):
        super().__init__()
        self.repr_ = program.repr_
        self._perceptrons = nn.ModuleList(program.perceptrons)
        self.policy = policy

    def get_perceptrons(self):
        return list(self._perceptrons)

    def forward(self, x):
        return SegmentForwardDriver(self.repr_, T, self.policy)(x)


@dataclass
class Program:
    repr_: Any
    ir: Any
    hybrid: Any
    perceptrons: List[Any]
    theta_out: float

    def join(self) -> ComputeOp:
        """The residual add: the one host op whose wrapper reads more than one producer."""
        joins = [n for n in self.ir.nodes if isinstance(n, ComputeOp)
                 and isinstance((n.params or {}).get("module"), ScaleNormalizingWrapper)
                 and len({int(s.node_id) for s in n.input_sources.flatten()}) > 1]
        assert len(joins) == 1, [n.name for n in joins]
        return joins[0]

    def stem(self) -> ComputeOp:
        return next(n for n in self.ir.nodes if isinstance(n, ComputeOp)
                    and getattr((n.params or {}).get("module"), "is_encoding_layer", False))

    def core_perceptron_indices(self) -> Dict[int, int]:
        return {int(n.id): int(n.perceptron_index) for n in self.ir.nodes
                if isinstance(n, NeuralCore) and getattr(n, "perceptron_index", None) is not None}


def build_program(theta_enc: float, *, retimed: bool = False, thetas=THETAS,
                  weight_bits: Optional[int] = None, stem_entry_grid: bool = False) -> Program:
    """``stem_entry_grid``: the stem's input passes the window's entry quantizer (the 1/T entry grid its samples sit
    on), declared before the projection, so its membrane lives on the ``s T`` lattice the projection then stamps
    (U-POST G4: a host-side stem reading a float input carries no lattice)."""
    torch.manual_seed(0)
    net = EncodingSkipBlock()
    with torch.no_grad():  # a live cascade: both cores' counts span the window
        net.stem.weight.mul_(2.0); net.stem.bias.fill_(0.3)
        net.fc1.weight.mul_(2.0); net.fc1.bias.fill_(0.2)
        net.fc2.weight.mul_(2.0); net.fc2.bias.fill_(0.1)
        net.fc3.weight.mul_(2.0); net.fc3.bias.fill_(0.2)
    flow = convert_torch_model(net, (D_IN,), D, device="cpu", encoding_layer_placement="subsume").eval()
    perceptrons = list(flow.get_perceptrons())
    for p, theta in zip(perceptrons, (theta_enc, *thetas)):
        p.set_activation_scale(torch.tensor(float(theta)))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    if stem_entry_grid:
        perceptrons[0].append_input_wire_op(ChipInputQuantizer(T=T, activation_scale=perceptrons[0].input_activation_scale))
    repr_ = flow.get_mapper_repr()
    compute_per_source_scales(repr_, arm_wire_value_ops=True)
    propagate_boundary_input_scales(flow, input_data_scale=1.0)
    q_max = 127.0
    if weight_bits is not None:
        napq = NormalizationAwarePerceptronQuantization(weight_bits, "cpu", rate=1.0, two_scale=False)
        for p in perceptrons:
            napq.transform(p)
        q_max = float(2 ** (weight_bits - 1) - 1)
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=q_max, firing_mode="Default", max_axons=32, max_neurons=32).map(repr_)
    if weight_bits is not None:
        quantize_ir_graph(ir, weight_bits, weight_quantization=True)
        arm_integer_membrane_lattice(flow)
    hybrid = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=retimed)
    return Program(repr_, ir, hybrid, perceptrons, float(perceptrons[-1].activation_scale))


def samples(n: int = 16) -> torch.Tensor:
    torch.manual_seed(1)
    return 2.0 * torch.rand(n, D_IN)


def deployed_flow(program: Program, semantics: str, **extra) -> SpikingHybridCoreFlow:
    return SpikingHybridCoreFlow((D_IN,), program.hybrid, simulation_length=T, thresholding_mode="<=",
                                 **SEMANTICS[semantics][1], **extra).eval()


def nf_window_counts(program: Program, x: torch.Tensor, policy) -> Tuple[torch.Tensor, Dict[int, np.ndarray]]:
    """The NF output and every core perceptron's window counts (the decoded train mean back on the count grid)."""
    rec: dict = {}
    with torch.no_grad():
        out = SegmentForwardDriver(program.repr_, T, policy)(x, node_value_recorder=rec)
    counts = {i: np.rint(rec[id(p)].double().numpy() / float(p.activation_scale) * T)
              for i, p in enumerate(program.perceptrons) if id(p) in rec}
    return out, counts


def per_perceptron_counts(records, hybrid) -> Dict[int, np.ndarray]:
    per = [_group_record_by_perceptron(r, hybrid, values_of=lambda c: c.output_spike_count[: c.n_out_used])
           for r in records]
    return {pi: np.stack([s[pi] for s in per]).astype(np.float64) for pi in per[0]}


def scm_window_counts(program: Program, x: torch.Tensor, flow: SpikingHybridCoreFlow):
    """The deployed output, every core perceptron's window counts, and the per-sample run records."""
    records = []
    with torch.no_grad():
        for i in range(x.shape[0]):
            records.append(flow.forward_with_recording(x[i:i + 1], sample_index=i)[1])
        out = flow(x)
    return out, per_perceptron_counts(records, program.hybrid), records


def count_mismatches(nf: Dict[int, np.ndarray], scm: Dict[int, np.ndarray]) -> Dict[int, int]:
    """Per perceptron, the window counts that differ (rows compared as sorted multisets, the gate's convention)."""
    assert set(scm) <= set(nf), (sorted(nf), sorted(scm))
    return {pi: int((np.sort(nf[pi], 1) != np.sort(scm[pi], 1)).sum()) for pi in sorted(scm)}
