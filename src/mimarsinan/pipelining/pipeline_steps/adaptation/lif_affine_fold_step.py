"""LIF per-channel affine fold pipeline step (C4, pre-weight-quantization)."""

from __future__ import annotations

from typing import Iterable, cast

import torch

from mimarsinan.chip_simulation.spiking_semantics import is_lif
from mimarsinan.mapping.support.bias_compensation import (
    apply_lif_half_step_bias_compensation,
)
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.trainer_pipeline_step import TrainerPipelineStep
from mimarsinan.tuning.lif_affine_fold import (
    apply_lif_affine_fold,
    evaluate_crater_premise,
)

_CALIBRATION_BATCHES = 4

# The Novena affine repair overfits the 5-level grid at S=4 (0.9010 -> 0.8845);
# the memo gates it at S >= 8 (lif_deployment_exactness.md §4 C4).
_NOVENA_MIN_STEPS = 8


class LIFAffineFoldStep(TrainerPipelineStep):
    # [R2b] aq_reference_read is the premise teacher (a plain float): AQ
    # preconditioning applies to every LIF plan, so the assembly DAG check
    # enforces AQ-before-fold on every admitted cell.
    REQUIRES = ("model", "aq_reference_read")
    UPDATES = ("model",)

    @classmethod
    def applies_to(cls, plan):
        # [R2c] Novena cells are admitted regardless of the AQ flag — the C4
        # affine is the only sanctioned repair for the V7 zero-reset discard
        # and the t0_02-class fp cells never scheduled it (ledger §2D). The
        # fold math holds without AQ grids: the estimator fits deployed
        # counts/T against the clamp(z/theta,0,1) envelope and lands in
        # PerceptronTransformer effective (wire) params — no AQ staircase
        # appears anywhere in its currency.
        return (
            is_lif(plan.spiking_mode)
            and bool(plan.config.get("lif_affine_fold", False))
            and (
                plan.activation_quantization
                or str(plan.config.get("firing_mode", "Default")) == "Novena"
            )
        )

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        model = self.get_entry("model")
        aq_reference = float(self.get_entry("aq_reference_read"))
        config = self.pipeline.config
        simulation_steps = int(config["simulation_steps"])
        firing_mode = str(config.get("firing_mode", "Default"))

        if firing_mode == "Novena" and simulation_steps < _NOVENA_MIN_STEPS:
            print(
                f"[LIFAffineFoldStep] Novena affine repair is gated at S >= "
                f"{_NOVENA_MIN_STEPS} (overfits the coarse grid); "
                f"S={simulation_steps} — skipping the fold."
            )
            self.update_entry("model", model, "torch_model")
            return

        self.trainer = make_basic_trainer(self.pipeline, model)
        batches = cast(
            "Iterable[tuple[torch.Tensor, torch.Tensor]]",
            self.trainer.iter_validation_batches(_CALIBRATION_BATCHES),
        )
        device = config["device"]
        pairs = [(x.to(device), y.to(device)) for x, y in batches]
        cal_x = torch.cat([x for x, _ in pairs], dim=0)
        cal_y = torch.cat([y for _, y in pairs], dim=0)

        verdict = evaluate_crater_premise(model, cal_x, cal_y, aq_reference)
        premise_witness = {
            "deployed_read": verdict.deployed_read,
            "aq_reference": verdict.reference_read,
            "premise_se": verdict.standard_error,
        }
        if not verdict.holds:
            # [R2a] a premise-skip is a TRUE no-op: every model mutation
            # (half-step entry fold included) lives behind this gate.
            print(
                f"[LIFAffineFoldStep] deployed calibration read "
                f"{verdict.deployed_read:.4f} >= AQ reference "
                f"{verdict.reference_read:.4f} - SE {verdict.standard_error:.4f}: "
                "no crater; skipping — the model leaves this step untouched."
            )
            self.pipeline.reporter.report(
                "lif_affine_fold",
                {"folded": 0, "skipped_premise": True, **premise_witness},
            )
            self.update_entry("model", model, "torch_model")
            return

        # A destructive fold must never survive: snapshot for keep-best
        # rollback (measured collapse 0.9386 -> 0.3847 on a true-crater cell).
        pre_fold_state = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

        # The estimator is fitted on the nearest (half-step) chain; the fold is
        # idempotent, so the WeightQuantizationStep's own bake stays a no-op.
        if bool(config.get("lif_half_step_bias", False)):
            folded = apply_lif_half_step_bias_compensation(model, simulation_steps)
            if folded:
                print(
                    f"[LIFAffineFoldStep] LIF half-step entry fold applied on "
                    f"{folded} perceptrons ahead of the affine calibration."
                )
        # Effective weights must be wire-domain (rate inputs) for the fold
        # currency; idempotent in activation_scales.
        compute_per_source_scales(model.get_mapper_repr())

        report = apply_lif_affine_fold(model, cal_x, simulation_steps)

        with torch.no_grad():
            post_read = float(
                (model(cal_x).argmax(dim=-1) == cal_y).float().mean().item()
            )
        if post_read < verdict.deployed_read:
            model.load_state_dict(pre_fold_state)
            print(
                f"[LIFAffineFoldStep] fold ROLLED BACK: post-fold calibration "
                f"read {post_read:.4f} < entry {verdict.deployed_read:.4f} "
                "(the estimator's regime does not hold on this crater)."
            )
            self.pipeline.reporter.report(
                "lif_affine_fold",
                {"folded": 0, "rolled_back": True,
                 "post_fold_read": post_read, **premise_witness},
            )
            self.update_entry("model", model, "torch_model")
            return
        report["post_fold_read"] = post_read
        print(
            f"[LIFAffineFoldStep] affine folds: {report['consumer_folds']} "
            f"consumer, {report['readout_folds']} readout; skipped: "
            f"{report['skipped'] or 'none'}"
        )
        self.pipeline.reporter.report(
            "lif_affine_fold",
            {
                "folded": report["folded"],
                "consumer_folds": report["consumer_folds"],
                "readout_folds": report["readout_folds"],
                "skipped": dict(report["skipped"]),
                **premise_witness,
            },
        )
        self.update_entry("model", model, "torch_model")
