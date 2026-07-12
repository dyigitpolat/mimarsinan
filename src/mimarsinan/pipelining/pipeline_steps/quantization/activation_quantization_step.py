from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
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
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")
        self.run_tuner(
            ActivationQuantizationTuner,
            model,
            adaptation_manager,
            target_tq=self.pipeline.config["target_tq"],
        )
        # [R2b] the post-AQ, PRE-adaptation staircase reference for the LIF
        # affine-fold premise (lossless_refinement_ledger.md §2D): a plain
        # float in the VALIDATION domain, because the premise differences it
        # against a validation-batch calibration read and test reads are never
        # differenced against eval-subset reads (ledger conventions).
        assert self.tuner is not None, "run_tuner must set the tuner"
        self.add_entry("aq_reference_read", float(self.tuner.validate()))
