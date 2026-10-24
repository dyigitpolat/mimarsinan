"""[CH-C1] Readout Law: install and range the STE readout quantizer on every perceptron output."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.export.law_solver import ReadoutLawDeclarationError
from mimarsinan.mapping.mappers.operand import operand_perceptrons
from mimarsinan.mapping.mappers.operand.consumer_groups import consumer_group_partition
from mimarsinan.mapping.support.operand_lowering import PER_CONSUMER_GROUP, operand_declaration
from mimarsinan.mapping.support.tensor_stats import safe_quantile
from mimarsinan.models.nn.activations.integer_law import grid_levels
from mimarsinan.models.nn.activations.readout_quantizer import ReadoutQuantizer
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_readout_state
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    CALIBRATION_BATCHES,
    CALIBRATION_QUANTILE,
    MIN_BOUNDARY_SCALE,
    measure_seam_quantiles,
)
from mimarsinan.mapping.lanes.model_lanes import install_planned_lanes, lane_clipping_census, measure_lane_ranges
from mimarsinan.tuning.orchestration.genuine_probe import iter_val_batches


class ReadoutLawCalibrationError(ToolchainRefusal, RuntimeError):
    """A perceptron the calibration batches never reach: a silent floor range would saturate its outputs."""


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


def measure_group_quantiles(model, quantizers: list, batches) -> List[List[float]]:
    """Per grouped quantizer, the largest per-batch quantile of each group's input magnitude (the boundary rule)."""
    maxima = {id(q): [0.0] * len(q.group_levels) for q in quantizers}
    handles = []
    for quantizer in quantizers:
        def hook(module, args, q=quantizer):
            y = args[0].detach().abs()
            groups = torch.as_tensor(q.channel_groups, device=y.device)
            for g in range(len(q.group_levels)):
                columns = torch.nonzero(groups == g).reshape(-1)
                value = safe_quantile(y.index_select(q.channel_axis, columns), CALIBRATION_QUANTILE).item()
                maxima[id(q)][g] = max(maxima[id(q)][g], value)
        handles.append(quantizer.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for x in batches:
                model(x)
    finally:
        for handle in handles:
            handle.remove()
    return [maxima[id(q)] for q in quantizers]


def calibrate_readout_quantizers(model, perceptrons: list, quantizers: list, batches, *, platform) -> List[float]:
    """Every readout range by the boundary rule (G2's: per perceptron, the declared ``range_value`` if fixed), per
    consumer group where declared; then the payload grids of re-snapped operand banks. Returns the per-perceptron
    measurement (0.0: never reached) for the caller's refusal."""
    batches = list(batches)
    install_consumer_groups(model, perceptrons, quantizers, platform)
    fixed = (platform.get("readout_law") or {}).get("range_value")
    grouped = [q for q in quantizers if q.channel_groups is not None]
    plain = [q for q in quantizers if q.channel_groups is None]
    payload = [op.payload_quantizer for op in _bank_perceptrons(model) if op.payload_quantizer is not None]
    seams = (payload if fixed is not None else plain + payload)
    measured = dict(zip(map(id, seams), measure_seam_quantiles(model, seams, batches) if seams else []))
    groups = dict(zip(map(id, grouped), measure_group_quantiles(model, grouped, batches)
                      if grouped and fixed is None else []))
    for quantizer in quantizers:
        if fixed is not None:
            quantizer.calibrate(float(fixed))
        elif quantizer.channel_groups is not None:
            quantizer.calibrate_groups([max(v, MIN_BOUNDARY_SCALE) for v in groups[id(quantizer)]])
        else:
            quantizer.calibrate(max(measured[id(quantizer)], MIN_BOUNDARY_SCALE))
    for grid in payload:
        grid.calibrate(max(measured[id(grid)], MIN_BOUNDARY_SCALE))
    if fixed is not None:
        return [float(fixed)] * len(quantizers)
    return [measured[id(q)] if id(q) in measured else min(groups[id(q)]) for q in quantizers]


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
    def _calibrated(model, perceptrons, quantizers, batches, lanes) -> List[float]:
        """WI-5e, with executed lanes only (N2's calibration covers the rest): each lane's range at its end (after
        its clamp or fused add); pool lanes by FX name."""
        names = [p.name for p in perceptrons] + [name for name, _ in lanes]
        measured = measure_lane_ranges(model, quantizers, batches, CALIBRATION_QUANTILE)
        unseen = [names[i] for i, value in enumerate(measured) if value == 0.0]
        if unseen:
            raise ReadoutLawCalibrationError(
                f"Readout Law calibration never reached perceptron(s) {unseen}: a silent floor range would "
                f"saturate their outputs"
            )
        return [max(value, MIN_BOUNDARY_SCALE) for value in measured]

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
        if lanes is None:  # [N2] per consumer group where declared, then the payload grids of re-snapped banks
            measured = calibrate_readout_quantizers(model, perceptrons, quantizers, batches, platform=platform)
            unseen = [perceptrons[i].name for i, value in enumerate(measured) if value == 0.0]
            if fixed is None and unseen:
                raise ReadoutLawCalibrationError(
                    f"Readout Law calibration never reached perceptron(s) {unseen}: a silent floor range would "
                    f"saturate their outputs"
                )
        else:  # WI-5e: each lane's range at its end, pool lanes by FX name
            ranges = [float(fixed)] * len(every) if fixed is not None else self._calibrated(
                model, perceptrons, every, batches, lanes)
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
