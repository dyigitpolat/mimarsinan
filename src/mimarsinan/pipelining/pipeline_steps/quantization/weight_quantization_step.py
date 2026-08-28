import math
from typing import Iterable, cast

import torch

from mimarsinan.advisories.graph_common import name_of
from mimarsinan.advisories.rules_graph_scale import (
    BIAS_DOMINANCE_LEVEL_FLOOR,
    bias_dominance_ratio_limit,
    worst_bias_grid_dominance,
)
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.support.bias_rows import bias_row_bound
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)
from mimarsinan.chip_simulation.spiking_semantics import is_lif, requires_ttfs_firing
from mimarsinan.mapping.support.bias_compensation import (
    LIF_HALF_STEP_FLAG,
    apply_lif_half_step_bias_compensation,
)
from mimarsinan.tuning.orchestration.adaptation_manager import sync_exact_qat_active
from mimarsinan.tuning.orchestration.lif_exact_qat import model_trained_lif_exact
from mimarsinan.tuning.orchestration.ttfs_exact_qat import model_trained_ttfsq_exact
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    resolve_wq_two_scale_projection as resolve_wq_two_scale_projection,
    resolve_wq_weight_only_grid,
)
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.layers import FrozenStatsNormalization
from mimarsinan.transformations.perceptron.bias_canonicalization import (
    canonicalize_starved_bias_outliers,
)
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
    NormalizationAwarePerceptronQuantizationTuner,
)

import torch.nn as nn

_CANONICALIZATION_BATCHES = 4


class BiasGridDominanceError(ValueError):
    """A shared weight-quantization grid the bias would set; raised at WQ entry."""


class BiasRowSplitEventSerialError(ValueError):
    """Splitting a bias into k rows under an event-serial saturating soma."""


def refuse_split_under_event_serial_soma(model_repr, bits: int, soma_law) -> None:
    """Refuse a k>1 split on a soma that evaluates threshold per ARRIVING EVENT
    with a fixed-width membrane register.

    MEASURED, not conjectured: k=1 is bit-exact at every soma point, and k>1 is
    bit-exact both with an unbounded membrane and under per_cycle firing — but
    under per_event WITH a register the NF adds the bias as ONE number while the
    deployed twin delivers it as k separately-thresholded event contributions,
    and NF<->SCM raster exactness (atol=0) breaks. The mismatch count is
    invariant to the register's width and signedness, so this is the event-serial
    discipline itself, not a rail magnitude. Refused HERE, before a training
    budget is spent, rather than at the parity gate an hour later.
    """
    if not (soma_law.is_per_event and soma_law.saturates):
        return
    q_max = float(2 ** (int(bits) - 1) - 1)
    transformer = PerceptronTransformer()
    worst_rows, worst_name = 1, None
    for perceptron in model_repr.get_perceptrons():
        weight = transformer.get_effective_weight(perceptron)
        bias = transformer.get_effective_bias(perceptron)
        w_max = float(weight.abs().max()) if weight.numel() else 0.0
        b_max = float(bias.abs().max()) if bias.numel() else 0.0
        if w_max <= 0.0:
            continue
        weight_scale = max(1.0, math.floor(q_max / w_max))
        rows = bias_row_bound(b_max, weight_scale, q_max)
        if rows > worst_rows:
            worst_rows, worst_name = rows, name_of(perceptron)
    if worst_rows <= 1:
        return
    raise BiasRowSplitEventSerialError(
        f"WeightQuantizationStep: bias_row_splitting is active and {worst_name!r} "
        f"needs {worst_rows} always-on rows, but this soma is "
        f"{soma_law.point_tag()!r} — per-event thresholding with a fixed-width "
        f"membrane register. There the split is NOT a value-preserving "
        f"re-encoding: the training twin adds the bias as one number while the "
        f"chip delivers it as {worst_rows} separately-thresholded event "
        f"contributions, and NF<->SCM raster exactness (atol=0) breaks. "
        f"Measured: k=1 exact at every soma point; k>1 exact under per_cycle and "
        f"with an unbounded membrane; k>1 refused only here. Either deploy this "
        f"vehicle at firing_granularity='per_cycle' / membrane_bits=0, or bring "
        f"a vehicle whose computed bound is 1 row (max|b| <= max|w| * q_max / "
        f"s_w). Modelling the split in the event-serial twin is the open work."
    )


def refuse_bias_dominated_grid(model_repr, bits: int, *, weight_only_grid: bool) -> None:
    """Refuse a shared grid a dominant bias would starve, on the FINAL tensors.

    ``rule_bias_grid_dominance`` predicts this at Torch Mapping, but the ratio
    grows through conversion (theta normalization, DFQ bias corrections), so
    the advisory's clean read does not certify the WQ entry — this is the same
    predicate, at the point of no return.
    """
    if weight_only_grid:
        return
    worst = worst_bias_grid_dominance(model_repr, bits)
    if worst is None:
        return
    ratio, name = worst
    limit = bias_dominance_ratio_limit(bits)
    raise BiasGridDominanceError(
        f"WeightQuantizationStep: {name!r} enters weight quantization with "
        f"max|effective bias| / max|effective weight| = {ratio:.1f} at "
        f"{bits} bits (limit q_max/{BIAS_DOMINANCE_LEVEL_FLOOR:.0f} = "
        f"{limit:.1f}). The shared per-perceptron grid is scaled by "
        f"max(|w|,|b|), so the bias would set it and the largest weight would "
        f"retain under {BIAS_DOMINANCE_LEVEL_FLOOR:.0f} levels (measured: 3 "
        f"levels, 99.6% zeros, 0.9314 -> 0.6117). Remedy: set "
        f"bias_row_splitting='auto' so the bias rides k always-on rows and the "
        f"weight grid comes from max|w| alone; on a platform with an on-chip "
        f"bias register, wq_two_scale_projection does the same."
    )


class WeightQuantizationStep(TunerPipelineStep):
    REQUIRES = ("model", "adaptation_manager")
    UPDATES = ("model", "adaptation_manager")

    @classmethod
    def applies_to(cls, plan):
        return plan.weight_quantization

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")
        self._freeze_exact_qat_theta(model)
        self._apply_lif_half_step_entry_fold(model)
        self._canonicalize_starved_bias_outliers(model)
        compute_per_source_scales(
            model.get_mapper_repr(),
            arm_wire_value_ops=not requires_ttfs_firing(
                str(DeploymentPlan.of(self.pipeline).spiking_mode)
            ),
        )
        for perceptron in model.get_perceptrons():
            if not isinstance(perceptron.normalization, nn.Identity):
                for param in perceptron.normalization.parameters():
                    param.requires_grad = False
                perceptron.normalization = FrozenStatsNormalization(
                    perceptron.normalization
                )
        bits = self.pipeline.config["weight_bits"]
        print(f"Quantizing to {bits} bits")
        splitting = DeploymentPlan.of(self.pipeline).bias_row_splitting
        weight_only = resolve_wq_weight_only_grid(self.pipeline.config)
        if bool(self.pipeline.config.get("wq_two_scale_projection", False)) and not weight_only:
            print(
                "[WeightQuantizationStep] wq_two_scale_projection requested but "
                "the platform has no on-chip bias register (param-encoded bias "
                "rows share the weight grid); using the shared-grid projection. "
                "bias_row_splitting buys the weight-only grid on this platform."
            )
        if splitting.mode != "off" and not splitting.active:
            print(
                f"[WeightQuantizationStep] bias_row_splitting={splitting.mode!r} "
                "is inert: every declared core carries an on-chip bias register, "
                "so no always-on row exists to split."
            )
        if weight_only:
            delivery = (
                f"bias split across always-on rows"
                f"{'' if splitting.rows_override is None else f' (floor {splitting.rows_override})'}"
                if splitting.active else "bias on its own on-chip grid"
            )
            print(f"[WeightQuantizationStep] weight-only grid from max|w| alone; "
                  f"{delivery} (integer-ratio-snapped).")
        refuse_bias_dominated_grid(
            model.get_mapper_repr(), int(bits), weight_only_grid=weight_only,
        )
        if splitting.active:
            refuse_split_under_event_serial_soma(
                model.get_mapper_repr(), int(bits),
                SomaLaw.resolve(self.pipeline.config),
            )
        self.run_tuner(
            NormalizationAwarePerceptronQuantizationTuner,
            model,
            adaptation_manager,
            quantization_bits=bits,
            two_scale_projection=weight_only,
            bias_rows_floor=splitting.rows_override,
        )

    def _freeze_exact_qat_theta(self, model) -> None:
        """[R1/P-L6] the WQ stage trains weights on a FIXED scale lattice: the
        effective-weight fold and the NAPQ projection share theta stamped at WQ
        entry (compute_per_source_scales), so ANY exact-QAT's in-loop theta
        (lif/ttfsq/sync) freezes here — a theta trained under the WQ endpoint
        drifts the lattice out from under the projection (torch<->deployed
        0.9688 vs 1.0000; leaving it trainable floods the projection with
        degenerate-channel bias routing and OOMs the graph)."""
        cfg = self.pipeline.config
        promotes = (
            model_trained_lif_exact(model)
            or model_trained_ttfsq_exact(model)
            or (sync_exact_qat_active(cfg) and bool(cfg.get("sync_exact_qat_theta", False)))
        )
        if not promotes:
            return
        frozen = 0
        for perceptron in model.get_perceptrons():
            theta = perceptron.activation_scale
            if torch.is_tensor(theta) and theta.requires_grad:
                theta.requires_grad_(False)
                frozen += 1
        if frozen:
            print(
                f"[WeightQuantizationStep] exact-QAT: in-loop theta frozen on "
                f"{frozen} perceptron(s) (the WQ scale lattice is fixed at entry)."
            )

    def _apply_lif_half_step_entry_fold(self, model) -> None:
        """Fold the LIF theta/(2T) half-step as a TRAINABLE entry bias BEFORE the
        weight-quantization QAT, so the QAT reconciles the shifted operating point
        and the float NF stays bit-exact with the quantized deployed sim. Idempotent per perceptron."""
        plan = DeploymentPlan.of(self.pipeline)
        if not is_lif(plan.spiking_mode) or not plan.activation_quantization:
            return
        if model_trained_lif_exact(model):
            # [P-L6] fold-once: the exact-QAT AQ install already folded and
            # TRAINED the half-step; a second bake would displace the
            # converged operating point (measured -2.06 pp).
            for perceptron in model.get_perceptrons():
                if getattr(perceptron, "is_encoding_layer", False):
                    continue
                assert getattr(perceptron, LIF_HALF_STEP_FLAG, False), (
                    "lif_exact_qat marker present but the half-step fold flag is "
                    f"missing on {getattr(perceptron, 'name', '<unnamed>')!r}; "
                    "the exact-QAT install owns the fold (P-L6)."
                )
            print(
                "[WeightQuantizationStep] LIF exact-QAT: half-step fold owned by "
                "the AQ install; WQ-entry fold skipped (marker-asserted)."
            )
            return
        if not bool(self.pipeline.config.get("lif_half_step_bias", False)):
            return
        folded = apply_lif_half_step_bias_compensation(
            model, int(self.pipeline.config["simulation_steps"]),
        )
        print(
            f"[WeightQuantizationStep] LIF half-step head-start folded on "
            f"{folded} perceptrons before the QAT (theta/(2T), T="
            f"{int(self.pipeline.config['simulation_steps'])})."
        )

    def _canonicalize_starved_bias_outliers(self, model) -> None:
        """Guarded empirical bias canonicalization at the QAT entry: outlier bias
        mass the provable OFF-clip cannot reach (empirically constant channels) is
        shrunk to its observed saturation slack and VERIFIED (decision agreement
        on the calibration batches; restored on any flip)."""
        trainer = make_basic_trainer(self.pipeline, model)
        device = self.pipeline.config["device"]
        val_batches = cast(
            "Iterable[tuple[torch.Tensor, torch.Tensor]]",
            trainer.iter_validation_batches(_CANONICALIZATION_BATCHES),
        )
        batches = [x.to(device) for x, _ in val_batches]
        report = canonicalize_starved_bias_outliers(
            model, batches, bits=int(self.pipeline.config["weight_bits"]),
        )
        if any(report.values()):
            self.pipeline.reporter.report(
                "wq_bias_canonicalization", dict(report),
            )
