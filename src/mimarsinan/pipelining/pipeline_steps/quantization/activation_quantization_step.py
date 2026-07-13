from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.orchestration.lif_exact_qat import lif_exact_qat_kd_active
from mimarsinan.tuning.tuners.activation_quantization_tuner import ActivationQuantizationTuner


class ActivationQuantizationStep(TunerPipelineStep):
    REQUIRES = ("model", "adaptation_manager")
    PROMISES = ("aq_reference_read",)
    UPDATES = ("model", "adaptation_manager")

    # The staged AQ install (sync's conversion endpoint) carries
    # the measured rung-2 training-draw variance.
    DRAW_SELECTED = True

    @classmethod
    def applies_to(cls, plan):
        return plan.requires_activation_quantization_preconditioning

    def __init__(self, pipeline):
        requires = self.REQUIRES
        # [lif_exact_qat_program §8] the exact-QAT KD lever distils to the
        # Reference Teacher Snapshot; the dependency is per-plan (same gate as
        # the snapshot step, so the promise always precedes the requirement).
        if lif_exact_qat_kd_active(pipeline.config):
            requires = requires + ("reference_teacher_model",)
        super().__init__(requires, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")
        kd_teacher = (
            self.get_entry("reference_teacher_model")
            if "reference_teacher_model" in self.requires
            else None
        )
        self.run_tuner(
            ActivationQuantizationTuner,
            model,
            adaptation_manager,
            target_tq=self.pipeline.config["target_tq"],
            kd_teacher=kd_teacher,
        )
        # [R2b] the post-AQ, PRE-adaptation staircase reference for the LIF
        # affine-fold premise (lossless_refinement_ledger.md §2D): a plain
        # float in the VALIDATION domain, because the premise differences it
        # against a validation-batch calibration read and test reads are never
        # differenced against eval-subset reads (ledger conventions).
        assert self.tuner is not None, "run_tuner must set the tuner"
        self.add_entry("aq_reference_read", float(self.tuner.validate()))
