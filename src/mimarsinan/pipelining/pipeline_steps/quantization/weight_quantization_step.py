import math
from typing import Iterable, cast

import torch

from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics
from mimarsinan.advisories.graph_common import name_of
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.support.bias_rows import bias_row_bound
from mimarsinan.pipelining.pipeline_steps.quantization.bias_row_drive_route import stamp_and_report_drives
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
from mimarsinan.mapping.support.core_types import install_layer_weight_bits, refuse_unbuilt_domain
from mimarsinan.mapping.support.grid_declarations import projection_grids
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    resolve_bias_mode,
    resolve_wq_two_scale_projection as resolve_wq_two_scale_projection,
    resolve_wq_weight_only_grid,
)
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.pipelining.pipeline_steps.quantization.readout_aware_training import readout_aware_endpoint
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.layers import FrozenStatsNormalization
from mimarsinan.transformations.perceptron.bias_canonicalization import (
    canonicalize_starved_bias_outliers,
)
from mimarsinan.transformations.perceptron.input_shift import refuse_unrecorded_shift_bakes
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
    NormalizationAwarePerceptronQuantizationTuner,
)

import torch.nn as nn

_CANONICALIZATION_BATCHES = 4


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
        # A shift bake the projection cannot re-derive (no recorded shift) is refused before any budget is spent.
        refuse_unrecorded_shift_bakes(model.get_perceptrons())
        # [U-ADC] declared readout-aware training: its scope judged now (refused by name), its loop opened around the
        # endpoint recovery; None when undeclared.
        endpoint_forward = readout_aware_endpoint(self.pipeline)
        grids = projection_grids(
            self.pipeline.config, two_scale=bool(self.pipeline.config.get("wq_two_scale_projection", False)),
            on_chip_bias=lambda: resolve_bias_mode(self.pipeline.config) == "on_chip",
        )
        # [WI-5g] core types with their own weight width: each layer quantizes on its type's grid.
        refuse_unbuilt_domain(self.pipeline.config, value_domain=is_mvm_core_semantics(DeploymentPlan.of(self.pipeline).core_semantics),
                              weight_quantization=True)
        install_layer_weight_bits(model, self.pipeline.config)
        stamp_and_report_drives(model, self.pipeline.config)  # [bias-row gauge] before any projection
        self._freeze_exact_qat_theta(model)
        self._apply_lif_half_step_entry_fold(model)
        if not grids.law_bias:  # a float bias has no grid a bias outlier could starve
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
        # The two routes of the weight-only grid, named: an on-chip bias register (the bias on the weight lattice
        # over its own span) or bias rows (param-encoded bias-row splitting: the r-multiple grid). They exclude
        # each other (one bias delivery per platform).
        register_bias = resolve_wq_two_scale_projection(self.pipeline.config)
        assert not (register_bias and splitting.active), "an on-chip bias register and bias rows on one platform"
        if bool(self.pipeline.config.get("wq_two_scale_projection", False)) and not weight_only:
            print(
                "[WeightQuantizationStep] wq_two_scale_projection requested but "
                "the platform has no on-chip bias register (param-encoded bias "
                "rows share the weight grid); using the shared-grid projection."
            )
        if splitting.mode != "off" and not splitting.active:
            print(
                f"[WeightQuantizationStep] bias_row_splitting={splitting.mode!r} "
                "is inert: every declared core carries an on-chip bias register, "
                "so no always-on row exists to split."
            )
        if splitting.active:
            floor = '' if splitting.rows_override is None else f' (floor {splitting.rows_override})'
            print(f"[WeightQuantizationStep] weight-only grid from max|w| alone; "
                  f"bias split across always-on rows{floor} (integer-ratio-snapped).")
        elif register_bias:
            print("[WeightQuantizationStep] two-scale projection: weight grid "
                  "from max|w| alone; bias on the weight lattice in its on-chip "
                  "register (span ±q_max r).")
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
            fractional_scale=grids.fractional_scale,
            float_bias=grids.law_bias,
            bias_rows_floor=splitting.rows_override,
            register_bias=register_bias,
            endpoint_forward=endpoint_forward,
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
