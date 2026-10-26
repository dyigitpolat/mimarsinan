"""[E6] Programs of the forms the integrated units build, each built once per test session and checked by W15:
N2's operand banks (a two-head attention block, softmax and ReLU forms, multi-pass on 64 x 64 cores), the lanes'
fused adds and pool lanes (a residual block net on 256 x 32 cores), N7's pruned readers beside a whole-step offset
grid (verify2's W1 arms), WI-5g's typed cores (4-bit and 2-bit types, fractional weight scale and bias at the law's
resolution), WI-5f's neuron split before coalescing and W02's position multiplexing (pack), W03's fit pins (a
weight-resident chain the planner pins), the closure's per-axon sign rows, depth-balancing relays and the columnar
axon sources of an identity program (Swin W9). Each builder returns ``(program, ir_graph, declaration)``."""

from __future__ import annotations

import contextlib
import functools
import io
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch

from mimarsinan.certification.program_check import CheckDeclaration
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.depth_balancing import insert_depth_balancing_relays
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.mapping.platform.platform_constraints import PER_AXON, SIGN_GRANULARITY_FIELD

from unit.certification.program_check import programs as P
from unit.mapping.residency.executed import vehicles as V

Built = Tuple[Any, Any, CheckDeclaration]
VALUE = dict(semantics="value_integer", window=1, pass_transfer="collapse",
             accumulation={"value_census": "float64", "int64_golden": "int64"}, upload={"value_census": "float32"})
OPERAND_GEOMETRY = dict(axons=64, neurons=64, count=3, passes=128)
LANE_CORES = {"max_axons": 256, "max_neurons": 32, "count": 999}
CHAIN_CORES = ({"max_axons": 8, "max_neurons": 8, "count": 2},)
MUX_CORES = [{"max_axons": 256, "max_neurons": 64, "count": 4}]
SPLIT_GEOMETRY = dict(axons=16, neurons=8, count=4000)


def value_decl(cores=()) -> CheckDeclaration:
    return CheckDeclaration(core_types=tuple(dict(c) for c in cores), **VALUE)


def _quiet(build: Callable[[], Any]) -> Any:
    with contextlib.redirect_stdout(io.StringIO()):
        return build()


@functools.lru_cache(maxsize=None)
def operand_program(form: str):
    from unit.chip_simulation.value_run.operand_programs import attention_program

    return _quiet(lambda: attention_program(form))


def operand(form: str = "softmax") -> Built:
    program = operand_program(form)
    geometry = OPERAND_GEOMETRY
    cores = [{"max_axons": geometry["axons"], "max_neurons": geometry["neurons"], "count": geometry["count"]}]
    return program.packed_mapping(**geometry), program.ir, value_decl(cores)


def operand_identity(form: str = "softmax") -> Built:
    program = operand_program(form)
    return program.identity_mapping(), program.ir, value_decl()


@functools.lru_cache(maxsize=None)
def lane_program(typed: bool = False):
    from unit.chip_simulation.value_run.integer_programs import residual_block_net
    from unit.mapping import test_core_types as C
    from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform

    extra = {**C.typed(C.LAST_STAGE), **C.GRIDS} if typed else {}
    return _quiet(lambda: build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10,
                                             platform=lane_platform(**extra), hardware_bias=True))


def lanes() -> Built:
    program = lane_program(False)
    geometry = dict(axons=LANE_CORES["max_axons"], neurons=LANE_CORES["max_neurons"], count=LANE_CORES["count"])
    return program.packed_mapping(**geometry), program.ir, value_decl([LANE_CORES])


def typed() -> Built:
    program = lane_program(True)
    widths = (4, 2)
    mapping = program.packed_mapping(axons=LANE_CORES["max_axons"], neurons=LANE_CORES["max_neurons"],
                                     count=LANE_CORES["count"], widths=widths)
    return mapping, program.ir, value_decl([dict(LANE_CORES, weight_bits=w) for w in widths])


@functools.lru_cache(maxsize=None)
def pruned_program(arm: str, prune: bool = True, mult_bits: int = 8, seed: int = 3, k: int = 2):
    """verify2's W1 arm (``test_pruned_offset_readers._build``): ``arm`` beside an offset grid of ``k`` whole steps."""
    from unit.mapping.lanes import test_pruned_offset_readers as W1

    with tempfile.TemporaryDirectory(prefix="e6_w1_") as directory:
        return _quiet(lambda: W1._build(Path(directory), arm, seed, k, mult_bits=mult_bits, prune=prune))


def pruned(arm: str = "lane_direct", prune: bool = True, mult_bits: int = 8, seed: int = 3, k: int = 2) -> Built:
    program = pruned_program(arm, prune, mult_bits, seed, k)
    cores = [{"max_axons": 512, "max_neurons": 64, "count": 999}]
    return program.packed_mapping(axons=512, neurons=64), program.ir, value_decl(cores)


@functools.lru_cache(maxsize=None)
def split_program():
    from unit.mapping.packing.test_neuron_split_before_coalescing import SHAPE, _wide_cnn
    from unit.chip_simulation.value_run.integer_programs import build_integer_program

    return _quiet(lambda: build_integer_program(_wide_cnn(), SHAPE, num_classes=5, max_axons=16, max_neurons=8,
                                                allow_coalescing=True))


def neuron_split() -> Built:
    program = split_program()
    geometry = SPLIT_GEOMETRY
    mapping = program.packed_mapping(**geometry, split=True, coalescing=True)
    cores = [{"max_axons": geometry["axons"], "max_neurons": geometry["neurons"], "count": geometry["count"]}]
    return mapping, program.ir, value_decl(cores)


@functools.lru_cache(maxsize=None)
def mux_program():
    from unit.mapping.position_multiplexing import vehicles as M

    return _quiet(M.integer_resnet)


def multiplexed() -> Built:
    from unit.mapping.position_multiplexing import vehicles as M

    program = mux_program()
    return _quiet(lambda: M.build(program.ir, MUX_CORES, True)), program.ir, value_decl(MUX_CORES)


#: [fix round F1] The exactness review's sched_mux row: 64 x 16 cores with a bias, 64 of them.
WINDOWED_MUX_CORE = {"max_axons": 64, "max_neurons": 16, "count": 64, "has_bias": True}


@functools.lru_cache(maxsize=None)
def windowed_mux_program():
    """[fix round F1] The pack unit's LIF conv chain on integer codes (``round(5 w)``, bias 1, threshold 6: the integer
    mode of the pass-1 review) at 64 x 16, position-multiplexed by the pipeline's builder under windowed LIF (per-hop
    retiming: every pass executes as its retimed level units). Returns ``(program, ir, the semantics run)``."""
    import copy

    from mimarsinan.certification.randomized import executors as ex
    from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
    from unit.mapping.position_multiplexing import vehicles as M

    rep = M.lif_conv_chain(seed=0, widths=(6, 6))
    for perceptron in rep.get_perceptrons():
        with torch.no_grad():
            perceptron.layer.weight.copy_(torch.round(perceptron.layer.weight * 5))
            if perceptron.layer.bias is not None:
                perceptron.layer.bias.fill_(1.0)
            perceptron.base_activation.activation_scale.fill_(6.0)
    ir = M.lif_ir(rep, WINDOWED_MUX_CORE["max_axons"], WINDOWED_MUX_CORE["max_neurons"])
    run = ex.RUNS[ex.WINDOWED_LIF]
    constraints = dict(allow_scheduling=True, max_schedule_passes=M.PASSES, allow_neuron_splitting=True,
                       position_multiplexing=True, allow_coalescing=False, cores=[dict(WINDOWED_MUX_CORE)])
    program = _quiet(lambda: build_hybrid_mapping_for_pipeline(copy.deepcopy(ir), constraints,
                                                               pipeline_config=dict(run.config)))
    return program, ir, run


def windowed_multiplexed() -> Built:
    from unit.mapping.position_multiplexing import vehicles as M

    program, ir, run = windowed_mux_program()
    return program, ir, CheckDeclaration(semantics="windowed_lif", window=M.T, pass_transfer=run.transfer,
                                         core_types=(dict(WINDOWED_MUX_CORE),))


def fit_chain() -> Built:
    program, ir = P.resident_chain()
    return program, ir, P.decl("synchronized_lif", core_types=CHAIN_CORES)


def per_axon(seed: int = 0, count: int = 6) -> Built:
    """G5's U1 fuzz vehicle on per-axon-sign cores of twice the vehicle's axons (the closure's CL-SG split gives a
    mixed-sign row two physical rows)."""
    ir = P.integer_ir(V.random_rep(seed))
    cores = [{"max_axons": 2 * V.MAX_AXONS, "max_neurons": V.MAX_NEURONS, "count": count,
              SIGN_GRANULARITY_FIELD: PER_AXON}]
    program = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores,
                                             strategy=MappingStrategy.resolve(V.capabilities(None)),
                                             max_schedule_passes=8)
    return program, ir, P.decl("streamed_lif", core_types=tuple(cores))


def relay_ir(*, unbounded: bool = False, shape: str = "skew3"):
    """``shape``'s IR with depth-balancing relays (``SoftCoreMappingStep``'s order: relays before quantization, then
    the integer lattice and the IR latency pass), at the vehicle's per-core limits or (``unbounded``) at none."""
    limits = {} if unbounded else dict(max_axons=V.MAX_AXONS, max_neurons=V.MAX_NEURONS)
    ir = IRMapping(q_max=127.0, firing_mode="Default", **limits).map(V.branchy_repr(shape))
    insert_depth_balancing_relays(ir, thresholding_mode="<=")
    for node in ir.get_neural_cores():
        node.parameter_scale = torch.tensor(0.0)
    for bank in ir.weight_banks.values():
        bank.parameter_scale = torch.tensor(0.0)
    quantize_ir_graph(ir, 8, weight_quantization=True)
    IRLatency(ir).calculate()
    return ir


def relays(count: int = 2, semantics: str = "streamed_lif") -> Built:
    ir = relay_ir()
    program = V.build(ir, count, None, retimed_level_stages=semantics == "windowed_lif")
    return program, ir, P.decl(semantics, count=count)


def columnar() -> Built:
    """The identity build of G2's integer MLP as a deployment: every soft core's axon sources columnar (W9)."""
    program = P.integer_value()
    return build_identity_hybrid_mapping(ir_graph=program.ir), program.ir, value_decl()


BUILDERS: Dict[str, Callable[[], Built]] = {
    "operand_softmax": lambda: operand("softmax"), "operand_relu": lambda: operand("relu"),
    "operand_identity": lambda: operand_identity("softmax"), "lanes": lanes, "typed": typed,
    "pruned_lane_direct": lambda: pruned("lane_direct"), "pruned_linear_reader": lambda: pruned("linear_reader"),
    "pruned_entry_host": lambda: pruned("entry_host"),
    "pruned_entry_host_plain": lambda: pruned("entry_host_plain"), "neuron_split": neuron_split,
    "multiplexed": multiplexed, "windowed_multiplexed": windowed_multiplexed, "fit_chain": fit_chain, "per_axon": per_axon, "relays": relays, "columnar": columnar,
}
