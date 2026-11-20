"""[CH-C1] Readout Law: install and range the STE readout quantizer on every perceptron output."""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List

import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.export.law_solver import ReadoutLawDeclarationError
from mimarsinan.mapping.mappers.operand import operand_perceptrons
from mimarsinan.mapping.mappers.operand.consumer_groups import consumer_group_partition
from mimarsinan.mapping.support.operand_lowering import PER_CONSUMER_GROUP, operand_declaration
from mimarsinan.mapping.support.range_rule import RangeCensus, RangeSite, least_squares_pooled, least_squares_ranges
from mimarsinan.mapping.support.reader_grids import read_together
from mimarsinan.models.nn.activations.integer_law import grid_levels
from mimarsinan.models.nn.activations.readout_quantizer import ReadoutQuantizer
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_readout_state
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    CALIBRATION_BATCHES,
    grid_site,
    zero_magnitude_condition,
)
from mimarsinan.mapping.lanes.model_lanes import install_planned_lanes, lane_absorbed, lane_clipping_census, lane_site
from mimarsinan.tuning.orchestration.genuine_probe import iter_val_batches


class ReadoutLawCalibrationError(ToolchainRefusal, RuntimeError):
    """A range site whose calibration values are all zero: its grid has no range to search."""


def install_readout_quantizers(perceptrons: list, activation_bits: int) -> List[ReadoutQuantizer]:
    """One inert quantizer per perceptron, in its scaler slot (no tuner rebuilds it)."""
    quantizers = []
    for perceptron in perceptrons:
        if not isinstance(perceptron.scaler, nn.Identity):
            raise ReadoutLawDeclarationError(
                f"perceptron {perceptron.name!r} already carries a scaler; the readout quantizer needs the slot"
            )
        quantizer = ReadoutQuantizer(activation_bits)
        perceptron.set_scaler(quantizer)
        quantizers.append(quantizer)
    return quantizers


def _bank_perceptrons(model) -> list:
    """N2's bank perceptrons of a converted model (none for a model without a mapper graph)."""
    graph = getattr(model, "get_mapper_repr", None)
    return list(operand_perceptrons(graph())) if callable(graph) else []


def install_consumer_groups(model, perceptrons: list, quantizers: list, platform) -> int:
    """Under ``readout_ranges: per_consumer_group``, give every perceptron an operand product reads one range per
    group of the columns its readers read whole; groups an operand bank writes as payload take ``payload_bits``.
    Only perceptrons an operand product reads are grouped, so without an operand core type (a projections-only
    twin of an operand deployment) nothing is grouped and the law is G2's per-perceptron law."""
    if platform.get("readout_ranges") != PER_CONSUMER_GROUP:
        return 0
    declaration = operand_declaration(platform)
    bits = int(platform["activation_bits"])
    payload_bits = declaration.payload_bits if declaration is not None else bits
    partition = consumer_group_partition(model.get_mapper_repr())
    grouped = 0
    for perceptron, quantizer in zip(perceptrons, quantizers):
        groups = partition.get(id(perceptron))
        if groups is None:
            continue
        axis = int(getattr(perceptron, "output_channel_axis", -1))
        quantizer.set_groups(groups.channels, tuple(grid_levels(payload_bits if pay else bits)
                                                    for pay in groups.payload), axis)
        grouped += 1
    return grouped


def readout_site(quantizer) -> RangeSite:
    """[U-BRANGE F-4] The range site of a per-perceptron readout quantizer: its input, on its own signed grid."""
    if isinstance(quantizer, ReadoutQuantizer):
        return RangeSite(quantizer, grid_levels(quantizer.activation_bits))
    return grid_site(quantizer)  # a bank's payload grid (a value-grid quantizer)


def _group_magnitudes(quantizer: ReadoutQuantizer, group: int):
    """The magnitudes of one consumer group's columns of the quantizer's input."""
    def magnitudes(y: torch.Tensor) -> torch.Tensor:
        groups = torch.as_tensor(quantizer.channel_groups, device=y.device)
        return y.index_select(quantizer.channel_axis, torch.nonzero(groups == group).reshape(-1)).abs()
    return magnitudes


def group_censuses(model, quantizers: list, batches) -> List[List[RangeCensus]]:
    """Per grouped quantizer, each group's census by the boundary rule on the group's columns and its own grid."""
    sites = [[RangeSite(q, int(levels), _group_magnitudes(q, g)) for g, levels in enumerate(q.group_levels)]
             for q in quantizers]
    flat = least_squares_ranges(model, [site for group in sites for site in group], batches)
    out, start = [], 0
    for group in sites:
        out.append(list(flat[start:start + len(group)]))
        start += len(group)
    return out


def measure_group_ranges(model, quantizers: list, batches) -> List[List[float]]:
    """Per grouped quantizer, each group's range by the boundary rule on the group's columns and its own grid."""
    return [[census.scale for census in group] for group in group_censuses(model, quantizers, batches)]


def _refuse_zero_sites(named: list, batches: int, what: str) -> None:
    """[U-BRANGE2] Refuse by name the sites whose calibration values are all zero (``named``: ``(name, census)``)."""
    zero = [(name, census) for name, census in named if census.scale == 0.0]
    if zero:
        raise ReadoutLawCalibrationError(
            f"Readout Law refuses {what} " + zero_magnitude_condition([n for n, _ in zero], [c for _, c in zero],
                                                                      batches)
            + ": a range site needs calibration values")


def _unions(model, keys: list, absorbed: FrozenSet[int]) -> List[List[int]]:
    """[U-BRANGE2, the review's m-3] The site indices of every union of keys one reader reads together
    (``reader_grids.read_together``: the law solvers' own union), among the ``(id(quantizer), group)`` keys given."""
    graph = getattr(model, "get_mapper_repr", None)
    if not callable(graph):  # a model without a mapper graph: no reader joins grids
        return []
    position = {key: i for i, key in enumerate(keys)}
    return [[position[(id(q), g)] for q, g in union if (id(q), g) in position]
            for union in read_together(graph(), absorbed=absorbed)]


def calibrate_readout_quantizers(model, perceptrons: list, quantizers: list, batches, *, platform,
                                 absorbed: FrozenSet[int] = frozenset()) -> List[float]:
    """Every readout range by the boundary rule (G2's: per perceptron, the declared ``range_value`` if fixed), per
    consumer group where declared; then the payload grids of re-snapped operand banks. [U-BRANGE F-4] The boundary
    rule is ``range_rule``'s, the entry grids' own: the least squared error of each site's own grid, at any magnitude
    (no floor); [U-BRANGE2] the grids one reader reads together are one site, their values pooled; a site whose
    calibration values are all zero is refused by name with the measured condition. Returns the per-perceptron
    range."""
    batches = list(batches)
    install_consumer_groups(model, perceptrons, quantizers, platform)
    fixed = (platform.get("readout_law") or {}).get("range_value")
    banks = [op for op in _bank_perceptrons(model) if op.payload_quantizer is not None]
    payload = [op.payload_quantizer for op in banks]
    names = {id(q): str(p.name) for p, q in zip(perceptrons, quantizers)}
    named: list = []  # (key, site, label) per range site
    for quantizer in (quantizers if fixed is None else []):
        if quantizer.channel_groups is None:
            named.append(((id(quantizer), 0), readout_site(quantizer), names[id(quantizer)]))
            continue
        named += [((id(quantizer), group), RangeSite(quantizer, int(levels), _group_magnitudes(quantizer, group)),
                   f"{names[id(quantizer)]} group {group}") for group, levels in enumerate(quantizer.group_levels)]
    named += [((id(op.payload_quantizer), 0), readout_site(op.payload_quantizer), f"{op.name} payload") for op in banks]
    keys, sites, labels = [n[0] for n in named], [n[1] for n in named], [n[2] for n in named]
    ranges = least_squares_pooled(model, sites, _unions(model, keys, absorbed), batches) if sites else []
    census = dict(zip(keys, ranges))
    _refuse_zero_sites(list(zip(labels, ranges)), len(batches), "the range site(s)")
    for quantizer in quantizers:
        if fixed is not None:
            quantizer.calibrate(float(fixed))
        elif quantizer.channel_groups is not None:
            quantizer.calibrate_groups([census[(id(quantizer), g)].scale for g in range(len(quantizer.group_levels))])
        else:
            quantizer.calibrate(census[(id(quantizer), 0)].scale)
    for grid in payload:
        grid.calibrate(census[(id(grid), 0)].scale)
    if fixed is not None:
        return [float(fixed)] * len(quantizers)
    return [min(census[(id(q), g)].scale for g in range(len(q.group_levels) if q.channel_groups is not None else 1))
            for q in quantizers]


def clipping_census(model, perceptrons: list, quantizers: list, batches: Iterable[torch.Tensor]) -> Dict[str, dict]:
    """Per perceptron: its range and the fraction of its outputs beyond it on the calibration batches."""
    totals = {id(q): [0, 0] for q in quantizers}
    handles = []
    for quantizer in quantizers:
        def hook(module, args, q=quantizer):
            y = args[0].detach()
            if float(q.range) > 0.0:  # an inert quantizer (an identity output) clips nothing
                totals[id(q)][0] += int((y.abs() > float(q.range)).sum())
            totals[id(q)][1] += int(y.numel())
        handles.append(quantizer.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for x in batches:
                model(x)
    finally:
        for handle in handles:
            handle.remove()
    return {
        str(p.name): {"range": float(q.range), "clipped": totals[id(q)][0], "outputs": totals[id(q)][1],
                      "clipped_fraction": totals[id(q)][0] / max(totals[id(q)][1], 1)}
        for p, q in zip(perceptrons, quantizers)
    }


class ReadoutLawStep(PipelineStep):
    """The declared readout law's trained forward: each perceptron output is quantized onto its grid
    (calibrated or the declared fixed range) through an STE; Soft Core Mapping stamps the integer law."""

    REQUIRES = ("model", "platform_constraints_resolved")
    PROMISES = ("readout_law_report",)
    UPDATES = ("model",)
    CLEARS = ()

    @classmethod
    def applies_to(cls, plan):
        return plan.config.get("readout_law") is not None

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)
        self.trainer = None

    def validate(self):
        assert self.trainer is not None  # process() ran first
        return self.trainer.test()

    @staticmethod
    def _calibrated(model, perceptrons, quantizers, batches, lanes, absorbed) -> List[float]:
        """WI-5e, with executed lanes only (N2's calibration covers the rest): each lane's range at its end (after
        its clamp or fused add); pool lanes by FX name; [U-BRANGE2] the grids one reader reads together pooled."""
        names = [str(p.name) for p in perceptrons] + [str(name) for name, _ in lanes]
        keys = [(id(q), 0) for q in quantizers]
        measured = least_squares_pooled(model, [lane_site(q) for q in quantizers], _unions(model, keys, absorbed),
                                        batches)
        _refuse_zero_sites(list(zip(names, measured)), len(batches), "the range(s) of perceptron(s) and lane(s)")
        return [census.scale for census in measured]

    def process(self):
        model = self.get_entry("model")
        platform = self.get_entry("platform_constraints_resolved")
        readout_law = platform.get("readout_law")
        if readout_law is None:
            raise ReadoutLawDeclarationError("readout_law is not on the resolved platform surface")
        bits = platform.get("activation_bits")
        if bits is None:
            raise ReadoutLawDeclarationError(
                "readout_law needs activation_bits: the output requantization writes the activation (DAC) grid"
            )
        if not DeploymentPlan.of(self.pipeline).weight_quantization:
            raise ReadoutLawDeclarationError("readout_law needs weight quantization: the law reads integer weights")
        self.trainer = trainer = make_basic_trainer(self.pipeline, model)
        device = self.pipeline.config["device"]
        batches = [x.to(device) for x, _ in iter_val_batches(trainer, CALIBRATION_BATCHES)]
        perceptrons = list(model.get_perceptrons()) + _bank_perceptrons(model)
        quantizers = install_readout_quantizers(perceptrons, int(bits))
        lanes = install_planned_lanes(model, platform, int(bits))  # WI-5e: None without executed lanes
        every = quantizers + [q for _, q in lanes or ()]
        fixed = readout_law.get("range_value")
        absorbed = lane_absorbed(model, self.pipeline.config)
        if lanes is None:  # [N2] per consumer group where declared, then the payload grids of re-snapped banks
            calibrate_readout_quantizers(model, perceptrons, quantizers, batches, platform=platform, absorbed=absorbed)
        else:  # WI-5e: each lane's range at its end, pool lanes by FX name
            ranges = [float(fixed)] * len(every) if fixed is not None else self._calibrated(
                model, perceptrons, every, batches, lanes, absorbed)
            for quantizer, value in zip(every, ranges):
                quantizer.calibrate(value)
        declared = declare_readout_state(model, perceptrons, quantizers, platform, lanes or ())  # {} undeclared
        census = clipping_census(model, perceptrons, quantizers, batches) if lanes is None else lane_clipping_census(
            model, [(str(p.name), q) for p, q in zip(perceptrons, quantizers)] + list(lanes), batches)
        print(
            f"[ReadoutLawStep] {len(quantizers)} readout quantizer(s) at {bits} bits, range "
            f"{'fixed' if fixed is not None else 'calibrated'}; clipped fraction max "
            f"{max((c['clipped_fraction'] for c in census.values()), default=0.0):.4f}"
        )
        self.add_entry("readout_law_report", {
            "activation_bits": int(bits), "range_fixed": fixed is not None, "perceptrons": census, **declared,
        }, "basic")
        self.update_entry("model", model, "torch_model")
