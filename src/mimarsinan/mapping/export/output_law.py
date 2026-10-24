"""The identity output law (WI-5d): an output core writes the exact integer ``M0 acc + B``, never a requantized code."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Set

import numpy as np
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.export.law_grids import CoreGrid
from mimarsinan.mapping.ir import IRSource, NeuralCore
from mimarsinan.mapping.ir.core_law import CoreLaw
from mimarsinan.mapping.mappers.leading_dim import Ensure2DMapper, MergeLeadingDimsMapper, SplitLeadingDimMapper
from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
from mimarsinan.mapping.mappers.structural import (
    ConcatMapper,
    EinopsRearrangeMapper,
    PermuteMapper,
    ReshapeMapper,
    StackMapper,
    SubscriptMapper,
)
from mimarsinan.models.nn.activations.integer_law import (
    FP64_EXACT_BOUND,
    IntegerLawRangeError,
    accumulator_width,
    grid_levels,
    law_magnitude_bound,
    round_half_away,
)

#: Mappers that only move values (no IR node, no host op): an output read through them is still an output.
WIRING_MAPPERS = (ReshapeMapper, EinopsRearrangeMapper, StackMapper, ConcatMapper, SubscriptMapper, PermuteMapper,
                  MergeLeadingDimsMapper, SplitLeadingDimMapper, Ensure2DMapper)


class OutputLawError(ToolchainRefusal, ValueError):
    """An identity output law on an output that another core or host op also reads, or whose model and program
    outputs disagree."""


def identity_output_multiplier(mult_bits: int) -> int:
    """M0 of the identity law: the top bit of the ``mult_bits`` multiplier, the bias resolution of a solved law."""
    return 1 << (int(mult_bits) - 1)


def unsaturated_adc_exponent(need: int, levels: int) -> int:
    """The smallest t with ``levels 2^t >= need``: the solver's ADC step where no output saturates."""
    return max(0, ((int(need) + int(levels) - 1) // int(levels) - 1).bit_length())


def solve_identity_law(*, core: str, theta: float, bias: np.ndarray, acc_bounds: Sequence[int], grid: CoreGrid,
                       spec) -> CoreLaw:
    """``z = M0 f_adc(acc) + B`` per column with shift 0 and no clamp, for ``(Δin acc + b) / θ = z Δout``:
    ``M0 = 2^(mult_bits - 1)``, ``B = rnd(M0 b / Δin)``, ``Δout = Δin / (θ M0)``. A narrower ADC takes the finest
    step spanning each column's |acc| bound (nothing saturates an identity law)."""
    columns = int(np.asarray(bias).size)
    m0 = identity_output_multiplier(spec.mult_bits)
    scaled = np.ldexp(np.asarray(bias, dtype=np.float64) / grid.input_step, int(spec.mult_bits) - 1)
    b = round_half_away(torch.as_tensor(scaled)).numpy().astype(np.int64)
    bounds = [int(v) for v in acc_bounds]
    width = accumulator_width(max(bounds, default=0))
    if spec.adc_bits >= width:
        adc_levels, adc_shift = grid_levels(width), [0] * columns
    else:
        adc_levels = grid_levels(spec.adc_bits)
        adc_shift = [unsaturated_adc_exponent(bound, adc_levels) for bound in bounds]
    bound = law_magnitude_bound(bounds, m0, b, adc_shift)
    if bound >= FP64_EXACT_BOUND:
        raise IntegerLawRangeError(
            f"core {core!r}: the identity output law's |z| reaches {bound} >= 2^53 (accumulator width {width} "
            f"bits, mult_bits {spec.mult_bits}); fp64 execution of its law is not exact"
        )
    full = np.full(columns, 1)
    return CoreLaw(
        input_step=full * grid.input_step, m0=full * m0, shift=full * 0, bias=b, lo=full * -bound, hi=full * bound,
        output_step=full * (grid.input_step / (theta * m0)), adc_shift=np.asarray(adc_shift, dtype=np.int64),
        adc_levels=full * adc_levels,
    )


def _readers(ir_graph) -> Dict[int, List[str]]:
    """{node id: names of the IR nodes that read it}."""
    readers: Dict[int, List[str]] = {}
    for node in ir_graph.nodes:
        for source in np.asarray(node.input_sources).flatten():
            if isinstance(source, IRSource) and source.node_id >= 0:
                readers.setdefault(int(source.node_id), []).append(str(node.name))
    return readers


def ir_output_perceptrons(ir_graph, names: Mapping[int, str]) -> Set[int]:
    """Perceptron indices of the cores that write program outputs, refused by name where another node reads one."""
    cores = {int(n.id): n for n in ir_graph.nodes if isinstance(n, NeuralCore)}
    readers = _readers(ir_graph)
    out: Set[int] = set()
    for source in np.asarray(ir_graph.output_sources).flatten():
        core = cores.get(int(source.node_id)) if isinstance(source, IRSource) else None
        if core is None or core.perceptron_index is None:
            continue
        index = int(core.perceptron_index)
        if readers.get(int(core.id)):
            raise OutputLawError(
                f"perceptron {names.get(index, core.name)!r} writes program outputs and is read by "
                f"{sorted(set(readers[int(core.id)]))[:3]}: an identity output law has no grid its readers read"
            )
        out.add(index)
    return out


def model_output_perceptrons(model_repr) -> List[Any]:
    """Perceptrons whose every consumer, through wiring mappers only, is the model's output."""
    consumers = model_repr.consumer_map()
    order, _, _ = model_repr._ensure_exec_graph()
    out = []
    for node in order:
        perceptron = getattr(node, "perceptron", None)
        if perceptron is None:
            continue
        frontier, wiring_only = list(consumers.get(id(node), [])), True
        while frontier and wiring_only:
            reader = frontier.pop()
            wiring_only = isinstance(reader, WIRING_MAPPERS)
            frontier.extend(consumers.get(id(reader), []))
        if wiring_only:
            out.append(perceptron)
    return out


def identity_output_indices(ir_graph, model_repr, quantizers: Mapping[int, Any]) -> Set[int]:
    """The perceptron indices whose cores take the identity law: the program's outputs, which must be the model's
    (the Readout Law step made those quantizers inert)."""
    fx = {id(p): name for name, p in perceptron_fx_names(model_repr).items()}
    names = {index: fx.get(id(entry[1]), str(entry[1].name)) for index, entry in quantizers.items()}
    program = ir_output_perceptrons(ir_graph, names)
    model = {index for index, entry in quantizers.items()
             if any(entry[1] is p for p in model_output_perceptrons(model_repr))}
    if not program:
        raise OutputLawError("readout_law declares output_law 'identity' but no core writes a program output (a host "
                             "op computes the outputs)")
    if program != model:
        raise OutputLawError(
            f"the program's output perceptrons {sorted(names.get(i, i) for i in program)} are not the model's "
            f"{sorted(names.get(i, i) for i in model)}"
        )
    return program
