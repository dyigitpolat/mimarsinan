"""The always-on bias row's drive on the integer value path: a code value of the grid its reader reads.

A parameter-encoded bias (cores without a bias register) rides an always-on axon row of the shared weight grid. On the
integer value path (a declared ``readout_law`` on value-domain cores) the network keeps its own units, so the row's
input value was the raw 1.0, a code of the reader's input grid only by accident; where a folded bias is large in those
units it set the shared grid and rounded most weights to 0 (W38: ResNet-50 at 8 bits, max|b| up to 21.9 max|w|). The
row now carries a code value of the grid its reader's input lies on (``reader_grids.reader_input_grids``):

* a reader of host-sourced axons reads its armed entry grid, whose top value is ``code_max step - offset``;
* a reader of core outputs reads the grid the law solver assigns it: the producers read together by any one core share
  the widest range of their union (``law_grids.assign_core_grids``, ``lane_grids.assign_lane_grids``), so the reader's
  grid is that union's grid and its top value ``levels (range / levels) - offset``;
* a reader whose grid is not determinable here (a reader mixing host and core inputs, an unarmed grid, groups of
  different levels joined by one reader: each a form the law solver refuses) is refused by name at Weight
  Quantization entry wherever a nonzero bias rides its row; without one its row carries no value and keeps 1.0, printed.

[U-BRANGE2] The code is the drive site's rule (``drive_rule``): the code whose row grid departs least from the bias,
with the rows the deployment gives it (``RowBudget``). The value is stamped on the perceptron at Weight Quantization
entry (``stamp_bias_row_drives``), before any projection, so every projection of the run (rungs, probes, recovery) codes
the row on one lattice; the IR's cores carry it from their perceptron, one source (``install_bias_row_drives`` refuses
by name a core already carrying another drive); the readout law's bias term multiplies the row codes by it
(``law_rules.core_bias_and_data_weights``), and each solver refuses by name a core whose drive is no code value of the
input grid it assigned that core (``refuse_drives_off_their_grid``).

Where the bias is not on a row of the weight grid (an on-chip register, a law-resolution bias, the register route), no
integer law reads the row, or the cores are not value-domain cores (the spiking wires, whose normalized full scale is 1),
no drive is stamped and every program is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.support.drive_rule import (
    DriveGrid,
    DriveRowBudgetRefusal,
    RowBudget,
    least_error_drive_and_rows,
)
from mimarsinan.mapping.support.reader_grids import (
    _perceptron_nodes,
    entry_drive_grid,
    entry_grid_top,
    reader_input_full_scales,
    reader_input_grids,
)
from mimarsinan.transformations.quantization_bounds import layer_weight_bits
from mimarsinan.transformations.perceptron.bias_drive import bias_row_drive, set_bias_row_drive
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

__all__ = [
    "BiasRowDriveRefusal", "BiasRowDriveStamps", "bias_rows_take_a_drive", "clear_bias_row_drives",
    "entry_drive_grid", "entry_grid_top", "install_bias_row_drives", "model_bias_row_drives", "perceptron_bias_row_drives",
    "reader_input_full_scales", "reader_input_grids", "refuse_bias_row_drives", "refuse_drives_off_their_grid", "stamp_bias_row_drives",
]


class BiasRowDriveRefusal(ToolchainRefusal, ValueError):
    """A pass that reads an always-on axon at 1.0 met a row driven at another value, a core's row drive is no code value
    of the input grid its law solver assigned it, or a core carries another drive than its perceptron's stamp."""


def bias_rows_take_a_drive(config, *, on_chip_bias: bool, law_bias: bool, register_bias: bool,
                           value_domain: bool) -> bool:
    """Whether the run's always-on bias rows carry a drive: a declared integer readout law on value-domain cores, and
    the bias on an always-on row of the weight grid (no on-chip register, no law-resolution bias, not the register
    route)."""
    return (config.get("readout_law") is not None and bool(value_domain)
            and not (on_chip_bias or law_bias or register_bias))


@dataclass
class BiasRowDriveStamps:
    """What :func:`stamp_bias_row_drives` stamped: ``{perceptron name: drive}`` of the drives other than 1.0, the
    perceptrons whose reader grid is not determinable (their rows stay at 1.0), and ``{perceptron name: rows}`` of the
    biases the drive site reads on more than one always-on row (the row cost: one axon per row on every core)."""

    drives: Dict[str, float] = field(default_factory=dict)
    undetermined: Tuple[str, ...] = ()
    rows: Dict[str, int] = field(default_factory=dict)


def _name(perceptron) -> str:
    return str(getattr(perceptron, "name", id(perceptron)))


def _rides_a_row(perceptron) -> bool:
    """Whether a nonzero effective bias rides the perceptron's always-on row (a zero bias carries no row value)."""
    bias = PerceptronTransformer(positions=True).get_effective_bias(perceptron)  # [U-PFOLD] every position's
    return bias is not None and bool(torch.any(torch.as_tensor(bias) != 0))


def stamp_bias_row_drives(model, *, absorbed: FrozenSet[int] = frozenset(), weight_bits: Optional[int] = None,
                          fractional: bool = False, budget: Optional[RowBudget] = None) -> BiasRowDriveStamps:
    """Stamp every core-mapped perceptron's row drive, a code value of its reader's input grid
    (:func:`reader_input_grids`): [U-BRANGE2] given the projection's ``weight_bits``, scale rule and row ``budget``
    (one row unless declared), the code whose row grid departs least from the bias (``drive_rule``), else the top
    code value; encoding layers (run on the host) and undeterminable readers keep 1.0. A perceptron whose bias needs
    more rows than the budget is refused by name."""
    budget = RowBudget() if budget is None else budget
    model_repr = model.get_mapper_repr()
    why: Dict[int, str] = {}
    grids = reader_input_grids(model_repr, absorbed=absorbed, why=why)
    stamps = BiasRowDriveStamps()
    undetermined = []
    transformer = PerceptronTransformer(positions=True)  # [U-PFOLD r2, m-3] every weight row and position's bias
    for node in _perceptron_nodes(model_repr):
        perceptron = node.perceptron
        if getattr(perceptron, "is_encoding_layer", False):
            set_bias_row_drive(perceptron, 1.0)
            continue
        grid = grids.get(id(node))
        drive = None if grid is None else grid.top
        if grid is not None and weight_bits is not None:
            bias = transformer.get_effective_bias(perceptron)
            if bias is not None:
                q_max = (1 << (layer_weight_bits(perceptron, int(weight_bits)) - 1)) - 1
                try:
                    drive, rows = least_error_drive_and_rows(transformer.get_effective_weight(perceptron),
                                                             torch.as_tensor(bias), grid, q_max,
                                                             fractional=fractional, budget=budget)
                except DriveRowBudgetRefusal as refused:
                    raise DriveRowBudgetRefusal(f"perceptron {_name(perceptron)!r}: {refused}") from None
                if rows > 1:
                    stamps.rows[_name(perceptron)] = int(rows)
        if drive is None and _rides_a_row(perceptron):  # [U-LMP] never the raw 1.0 gauge silently
            raise BiasRowDriveRefusal(
                f"perceptron {_name(perceptron)!r}: its always-on bias row has no drive on the integer value path: "
                f"{why.get(id(node), 'its reader grid is not determinable')}")
        set_bias_row_drive(perceptron, 1.0 if drive is None else drive)
        if drive is None:
            undetermined.append(_name(perceptron))
        elif drive != 1.0:
            stamps.drives[_name(perceptron)] = drive
    stamps.undetermined = tuple(undetermined)
    return stamps


def clear_bias_row_drives(model) -> None:
    """Remove every perceptron's row drive (the route takes none)."""
    for perceptron in model.get_perceptrons():
        set_bias_row_drive(perceptron, 1.0)


def perceptron_bias_row_drives(model) -> Dict[str, float]:
    """``{perceptron name: drive}`` of the model's stamped perceptrons (drives other than 1.0)."""
    return {_name(p): bias_row_drive(p) for p in model.get_perceptrons() if bias_row_drive(p) != 1.0}


def model_bias_row_drives(model) -> Dict[int, float]:
    """``{perceptron index: drive}`` of the model's stamped perceptrons (drives other than 1.0)."""
    out: Dict[int, float] = {}
    for node in _perceptron_nodes(model.get_mapper_repr()):
        index = getattr(node, "perceptron_index", None)
        drive = bias_row_drive(node.perceptron)
        if index is not None and drive != 1.0:
            out[int(index)] = drive
    return out


def install_bias_row_drives(ir_graph, model) -> int:
    """Stamp each IR core with its perceptron's row drive (idempotent); return how many carry one other than 1.0.
    [U-BRANGE2] One source: a core that already carries a drive other than its perceptron's stamp is refused by name
    (a code value of the grid is not enough: the drive is its perceptron's)."""
    drives = model_bias_row_drives(model)
    stamped = 0
    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        drive = drives.get(node.perceptron_index, 1.0) if node.perceptron_index is not None else 1.0
        carried = node.__dict__.get("bias_row_drive")
        if carried is not None and float(carried) != drive:
            raise BiasRowDriveRefusal(
                f"core {node.name!r}: its always-on rows carry the drive {float(carried)!r}, not its perceptron's "
                f"stamp {drive!r}: a row's drive has one source, the perceptron Weight Quantization stamped")
        set_bias_row_drive(node, drive)
        stamped += int(drive != 1.0)
    return stamped


def _has_always_on_rows(core: NeuralCore) -> bool:
    return any(source.is_always_on() for source in core.input_sources.flatten())


def reads_a_core(sources: Iterable, nodes_by_id: Mapping[int, object]) -> bool:
    """Whether any of a core's data-axon ``sources`` is a core output (else its data axons are host-sourced)."""
    return any(source.node_id >= 0 and isinstance(nodes_by_id.get(int(source.node_id)), NeuralCore)
               for source in sources)


def refuse_drives_off_their_grid(ir_graph, core_fed_grid: Callable[[NeuralCore], Optional[DriveGrid]]) -> int:
    """Refuse by name a core whose always-on rows carry a drive that is no positive code value of the input grid its
    law solver assigned it: ``core_fed_grid(core)`` for a reader of core outputs (None for a reader of host-sourced
    axons, which reads its entry grid). [U-BRANGE2] any code of that grid is a value its DAC writes. Returns how many
    cores were checked."""
    checked = 0
    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore) or bias_row_drive(node) == 1.0 or not _has_always_on_rows(node):
            continue
        grid = core_fed_grid(node)
        if grid is None:
            grid = entry_drive_grid(getattr(node, "boundary_grid", None))
        drive = bias_row_drive(node)
        if grid is None or not grid.holds(drive):
            raise BiasRowDriveRefusal(
                f"core {node.name!r}: its always-on rows are driven at {drive!r}, which is no code value of the input "
                f"grid its readout law reads ({grid!r}): the drive stamped at Weight Quantization is not a code of the "
                "grid the deployed core reads")
        checked += 1
    return checked


def refuse_bias_row_drives(ir_graph, what: str) -> None:
    """Refuse by name a pass that reads always-on axons at 1.0 on an IR whose cores carry another row drive
    (``install_bias_row_drives`` stamps them right after mapping)."""
    drives = [bias_row_drive(n) for n in ir_graph.nodes if isinstance(n, NeuralCore) and bias_row_drive(n) != 1.0]
    if drives:
        raise BiasRowDriveRefusal(
            f"{what} reads an always-on axon at 1.0, but {len(drives)} core(s) carry their bias rows at the drive "
            f"rule's codes of their reader's input grid (drives {min(drives):.6g} to {max(drives):.6g}): this "
            "combination is not built on this tree")
