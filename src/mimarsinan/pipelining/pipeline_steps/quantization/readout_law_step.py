"""[CH-C1] Readout Law: install and range the STE readout quantizer on every perceptron output."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.export.law_solver import ReadoutLawDeclarationError
from mimarsinan.models.nn.activations.readout_quantizer import ReadoutQuantizer
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    CALIBRATION_BATCHES,
    MIN_BOUNDARY_SCALE,
    measure_seam_quantiles,
)
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


def clipping_census(model, perceptrons: list, quantizers: list, batches: Iterable[torch.Tensor]) -> Dict[str, dict]:
    """Per perceptron: its range and the fraction of its outputs beyond it on the calibration batches."""
    totals = {id(q): [0, 0] for q in quantizers}
    handles = []
    for quantizer in quantizers:
        def hook(module, args, q=quantizer):
            y = args[0].detach()
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
    def _calibrated(model, perceptrons, quantizers, batches) -> List[float]:
        measured = measure_seam_quantiles(model, quantizers, batches)
        unseen = [perceptrons[i].name for i, value in enumerate(measured) if value == 0.0]
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
        perceptrons = list(model.get_perceptrons())
        quantizers = install_readout_quantizers(perceptrons, int(bits))
        fixed = readout_law.get("range_value")
        ranges = [float(fixed)] * len(quantizers) if fixed is not None else self._calibrated(
            model, perceptrons, quantizers, batches)
        for quantizer, value in zip(quantizers, ranges):
            quantizer.calibrate(value)
        census = clipping_census(model, perceptrons, quantizers, batches)
        print(
            f"[ReadoutLawStep] {len(quantizers)} readout quantizer(s) at {bits} bits, range "
            f"{'fixed' if fixed is not None else 'calibrated'}; clipped fraction max "
            f"{max((c['clipped_fraction'] for c in census.values()), default=0.0):.4f}"
        )
        self.add_entry("readout_law_report", {
            "activation_bits": int(bits), "range_fixed": fixed is not None, "perceptrons": census,
        }, "basic")
        self.update_entry("model", model, "torch_model")
