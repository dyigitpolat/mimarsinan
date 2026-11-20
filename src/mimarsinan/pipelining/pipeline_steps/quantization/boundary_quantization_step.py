"""[mvm AQ] Boundary Quantization: calibrate + install value-grid quantizers at segment entries."""

from __future__ import annotations

from typing import Iterable, List

import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mappers.operand import operand_entry_perceptrons
from mimarsinan.mapping.lanes.model_lanes import lane_absorbed
from mimarsinan.mapping.platform.packaging_contract import packaging_contract_for
from mimarsinan.mapping.support.range_rule import (
    RANGE_CANDIDATES,
    RANGE_OCTAVES,
    RANGE_RULE,
    RangeCensus,
    RangeSite,
    least_squares_ranges,
)
from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid, ValueGridQuantizer, value_grid_levels
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import declare_entry_state
from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
from mimarsinan.tuning.orchestration.genuine_probe import iter_val_batches

CALIBRATION_BATCHES = 2
#: [U-BRANGE F-2] The step's report: the range rule and the clamp census of every boundary (the record's
#: ``boundary_ranges``).
BOUNDARY_REPORT = "boundary_quantization_report"


class BoundaryQuantizationRefusal(ToolchainRefusal, RuntimeError):
    """[U-BRANGE2] Boundary Quantization cannot grid the model's boundaries as declared: no host-to-chip boundary, or a
    boundary whose calibration values are all zero (its grid has no range to search)."""


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


def grid_site(quantizer: ValueGridQuantizer) -> RangeSite:
    """The range site of a value-grid quantizer: its input, on its own symmetric grid of its bits."""
    return RangeSite(quantizer, value_grid_levels(quantizer.activation_bits))


def zero_magnitude_condition(names, ranges: List[RangeCensus], batches: int) -> str:
    """[U-BRANGE2] The measured condition of a site whose calibration values are all zero, as a refusal states it."""
    return (f"{list(names)}: every magnitude the {batches} calibration batch(es) send through each is 0 (largest "
            f"{max((r.maximum for r in ranges), default=0.0):.6g}, over {sum(r.entries for r in ranges)} entries), so "
            "its grid has no range to search")


def calibrate_boundary_scales(
    model: torch.nn.Module,
    entries: list,
    quantizers: list,
    batches: Iterable[torch.Tensor],
) -> List[RangeCensus]:
    """[U-BRANGE, U-BRANGE2] Each boundary's scale is the scale of least squared error of its own grid
    (``quantize_to_value_grid`` at the quantizer's bits) over the calibration batches, the rounding and the clamp both
    counted, over the whole interval up to the calibration maximum (``range_rule``): it clamps where the error minimum
    says so, at any bit width and any magnitude (no floor). Measured AT the quantizer seam itself. A boundary whose
    calibration values are all zero is refused by name. Returns each boundary's census."""
    batches = list(batches)
    ranges = least_squares_ranges(model, [grid_site(q) for q in quantizers], batches)
    unseen = [i for i, census in enumerate(ranges) if census.scale == 0.0]
    if unseen:
        raise BoundaryQuantizationRefusal(
            f"Boundary Quantization refuses entry quantizer(s) {unseen} "
            + zero_magnitude_condition([entries[i].name for i in unseen], [ranges[i] for i in unseen], len(batches))
            + ": every host-to-chip boundary must carry calibration values")
    for quantizer, census in zip(quantizers, ranges):
        quantizer.calibrate(census.scale)
    return ranges


_CENSUS_FIELDS = ("scale", "entries", "clamped", "clamped_fraction", "clamped_share", "largest_over_scale",
                  "squared_error")


def deployed_census(model, quantizers: list, batches: list, indices) -> dict:
    """[U-BRANGE2 (e)] The clamp census of each indexed boundary's deployed grid (a declared one: its own snap, offset,
    rounding and codes) on the calibration batches' float seams, every quantizer inert during the pass as during
    calibration: the grid's squared error, the entries outside its value range and their share of the squared
    magnitude, and the largest value over the range end it passes. The quantizers are restored after the pass."""
    indices = sorted(set(indices))
    if not indices:
        return {}
    grids = {i: quantizers[i].grid for i in indices}
    acc = {i: {"entries": 0, "clamped": 0, "energy": 0.0, "clamp_energy": 0.0, "error": 0.0, "largest": 0.0}
           for i in indices}
    saved = [(q.__dict__.get("_declared"), float(q.scale)) for q in quantizers]

    def hook_for(i):
        def hook(module, args):
            x = args[0].detach().to(torch.float64)
            grid, a = grids[i], acc[i]
            low, high = grid.value_range
            out = (x < low) | (x > high)
            square = x * x
            a["entries"] += x.numel()
            a["clamped"] += int(out.sum())
            a["energy"] += float(square.sum())
            a["clamp_energy"] += float(square[out].sum())
            a["error"] += float(((grid.snap(x) - x) ** 2).sum())
            over = [float(x.max()) / high] if high > 0 and x.numel() else []
            over += [float(x.min()) / low] if low < 0 and x.numel() else []
            a["largest"] = max([a["largest"]] + over)
        return hook

    handles = [quantizers[i].register_forward_pre_hook(hook_for(i)) for i in indices]
    try:
        for q in quantizers:
            q.__dict__.pop("_declared", None)
            q.scale.fill_(0.0)
        with torch.no_grad():
            for x in batches:
                model(x)
    finally:
        for handle in handles:
            handle.remove()
        for q, (declared, scale) in zip(quantizers, saved):
            q.scale.fill_(scale)
            if declared is not None:
                q.__dict__["_declared"] = declared
    return {i: {"scale": float(grids[i].scale), "entries": a["entries"], "clamped": a["clamped"],
                "clamped_fraction": a["clamped"] / a["entries"] if a["entries"] else 0.0,
                "clamped_share": a["clamp_energy"] / a["energy"] if a["energy"] > 0 else 0.0,
                "largest_over_scale": a["largest"], "squared_error": a["error"]} for i, a in acc.items()}


def boundary_census(entries: list, quantizers: list, ranges: List[RangeCensus], deployed=None) -> List[dict]:
    """[U-BRANGE F-2] Per boundary: its name, the deployed scale, and the clamp census of the grid it deploys: the
    rule's census where the calibrated grid is deployed; [U-BRANGE2 (e)] a declared grid's own census
    (``deployed_census``) where one replaced it, with the rule's calibration kept under ``calibration``."""
    rows = []
    deployed = {} if deployed is None else deployed
    for index, (entry, quantizer, census) in enumerate(zip(entries, quantizers, ranges)):
        declared = isinstance(quantizer.grid, DeclaredBoundaryGrid)
        row = {"index": index, "name": str(entry.name), "deployed_scale": float(quantizer.grid.scale),
               "declared": declared, **census.to_dict()}
        if declared and index in deployed:
            row["calibration"] = {k: row[k] for k in _CENSUS_FIELDS}
            row.update(deployed[index])
        rows.append(row)
    return rows


def _census_line(row: dict) -> str:
    return (f"[BoundaryQuantizationStep] boundary {row['index']} {row['name']}: scale {row['scale']:.6g} (searched "
            f"{row['low_span']:.6g} to {row['maximum']:.6g}); clamped {row['clamped']} of {row['entries']} "
            f"({row['clamped_fraction']:.4%}), share of squared magnitude {row['clamped_share']:.4%}, largest value "
            f"over scale {row['largest_over_scale']:.3f}" + (" (a declared grid replaces it)" if row["declared"] else ""))


class BoundaryQuantizationStep(PipelineStep):
    """Value-domain activation quantization, armed by platform ``activation_bits``.

    Calibrates each segment-entry perceptron's input range on validation
    batches (the least squared error of its own grid) and installs the ``ValueGridQuantizer`` — so the model forward,
    WQ QAT, the R-edge gate, and the value executor share one boundary grid. Reports each boundary's clamp census.
    """

    REQUIRES = ("model",)
    PROMISES = (BOUNDARY_REPORT,)
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
        absorbed = lane_absorbed(model, self.pipeline.config)  # WI-5e: a perceptron reading a lane is no entry
        # [N2] a bank whose streamed operand is host-fed (softmax's P) takes its own entry grid too.
        entries = list(segment_entry_perceptrons(model.get_mapper_repr(), absorbed)) + list(
            operand_entry_perceptrons(model.get_mapper_repr()))
        if not entries:
            raise BoundaryQuantizationRefusal(
                "Boundary Quantization found no segment-entry perceptrons; "
                "an armed activation_bits platform needs at least one "
                "host→chip boundary to quantize."
            )
        self.trainer = trainer = make_basic_trainer(self.pipeline, model)
        device = self.pipeline.config["device"]
        quantizers = install_boundary_quantizers(entries, bits)
        batches = [x.to(device) for x, _ in iter_val_batches(trainer, CALIBRATION_BATCHES)]
        ranges = calibrate_boundary_scales(model, entries, quantizers, batches)
        declare_entry_state(model, entries, quantizers, self.pipeline.config)  # WI-5a: none when undeclared
        declared = [i for i, q in enumerate(quantizers) if isinstance(q.grid, DeclaredBoundaryGrid)]
        rows = boundary_census(entries, quantizers, ranges, deployed_census(model, quantizers, batches, declared))
        print(
            f"[BoundaryQuantizationStep] installed {bits}-bit value-grid "
            f"quantizers on {len(entries)} segment-entry perceptron(s); "
            f"scales calibrated by the {RANGE_RULE}, over "
            f"{CALIBRATION_BATCHES} validation batch(es); clamp census: clamped fraction max "
            f"{max(r['clamped_fraction'] for r in rows):.4f}, clamped share of squared magnitude max "
            f"{max(r['clamped_share'] for r in rows):.4f}, largest value over scale max "
            f"{max(r['largest_over_scale'] for r in rows):.3f}."
        )
        for row in rows:
            print(_census_line(row))
        self.add_entry(BOUNDARY_REPORT, {
            "activation_bits": bits, "range_rule": RANGE_RULE, "calibration_batches": CALIBRATION_BATCHES,
            "candidates": RANGE_CANDIDATES, "octaves": RANGE_OCTAVES, "boundaries": rows,
        }, "basic")
        self.update_entry("model", model, "torch_model")
