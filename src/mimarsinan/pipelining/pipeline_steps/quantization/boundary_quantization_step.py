"""[mvm AQ] Boundary Quantization: calibrate + install value-grid quantizers at segment entries."""

from __future__ import annotations

from typing import Iterable

import torch

from mimarsinan.mapping.platform.packaging_contract import packaging_contract_for
from mimarsinan.mapping.support.tensor_stats import safe_quantile
from mimarsinan.models.nn.activations.value_quantizer import ValueGridQuantizer
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
from mimarsinan.tuning.orchestration.genuine_probe import iter_val_batches

CALIBRATION_BATCHES = 2
CALIBRATION_QUANTILE = 0.999
MIN_BOUNDARY_SCALE = 1e-3


def install_boundary_quantizers(
    entries: list, activation_bits: int
) -> list[ValueGridQuantizer]:
    """Append an INERT quantizer (scale 0.0 = identity) per entry. Each owns
    its own grid buffer — the event domain's ``input_activation_scale`` keeps
    its single wire-currency meaning. Returns them aligned with ``entries``."""
    quantizers = []
    for perceptron in entries:
        quantizer = ValueGridQuantizer(activation_bits)  # inert until calibrated
        perceptron.append_input_wire_op(quantizer)
        quantizers.append(quantizer)
    return quantizers


def calibrate_boundary_scales(
    model: torch.nn.Module,
    entries: list,
    quantizers: list,
    batches: Iterable[torch.Tensor],
    quantile: float = CALIBRATION_QUANTILE,
) -> None:
    """Measure the input-magnitude quantile AT the quantizer seam itself and
    write the scales. Hooking the installed module (not the perceptron) makes
    the measurement fire on every execution path — the conv mappers invoke
    ``input_activation`` functionally, bypassing ``Perceptron.__call__``."""
    maxima = {id(q): 0.0 for q in quantizers}
    handles = []
    for quantizer in quantizers:
        def hook(module, args, q=quantizer):
            value = safe_quantile(args[0].detach().abs(), quantile).item()
            maxima[id(q)] = max(maxima[id(q)], value)
        handles.append(quantizer.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for x in batches:
                model(x)
    finally:
        for handle in handles:
            handle.remove()
    unseen = [i for i, q in enumerate(quantizers) if maxima[id(q)] == 0.0]
    if unseen:
        raise RuntimeError(
            f"Boundary Quantization calibration never reached entry "
            f"quantizer(s) {unseen} ({[entries[i].name for i in unseen]}): "
            f"every host→chip boundary must be exercised by the calibration "
            f"batches — a silent floor scale would saturate the boundary."
        )
    for quantizer in quantizers:
        quantizer.calibrate(max(maxima[id(quantizer)], MIN_BOUNDARY_SCALE))


class BoundaryQuantizationStep(PipelineStep):
    """Value-domain activation quantization, armed by platform ``activation_bits``.

    Calibrates each segment-entry perceptron's input range on validation
    batches and installs the ``ValueGridQuantizer`` — so the model forward,
    WQ QAT, the R-edge gate, and the value executor share one boundary grid.
    """

    REQUIRES = ("model",)
    PROMISES = ()
    UPDATES = ("model",)
    CLEARS = ()

    @classmethod
    def applies_to(cls, plan):
        return packaging_contract_for(plan).boundary_is_gridded

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)
        self.trainer = None

    def validate(self):
        assert self.trainer is not None  # process() ran first
        return self.trainer.test()

    def process(self):
        model = self.get_entry("model")
        bits = int(self.pipeline.config["activation_bits"])
        entries = list(segment_entry_perceptrons(model.get_mapper_repr()))
        if not entries:
            raise RuntimeError(
                "Boundary Quantization found no segment-entry perceptrons; "
                "an armed activation_bits platform needs at least one "
                "host→chip boundary to quantize."
            )
        self.trainer = trainer = make_basic_trainer(self.pipeline, model)
        device = self.pipeline.config["device"]
        quantizers = install_boundary_quantizers(entries, bits)
        calibrate_boundary_scales(
            model, entries, quantizers,
            (x.to(device) for x, _ in iter_val_batches(trainer, CALIBRATION_BATCHES)),
        )
        print(
            f"[BoundaryQuantizationStep] installed {bits}-bit value-grid "
            f"quantizers on {len(entries)} segment-entry perceptron(s); "
            f"scales calibrated at q={CALIBRATION_QUANTILE} over "
            f"{CALIBRATION_BATCHES} validation batch(es)."
        )
        self.update_entry("model", model, "torch_model")
