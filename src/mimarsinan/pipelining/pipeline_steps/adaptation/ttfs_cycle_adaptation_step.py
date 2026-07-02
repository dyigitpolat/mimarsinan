"""TTFS-Cycle Fine-Tuning pipeline step."""

from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.spiking.scale_aware_boundaries import calibrate_scale_aware_boundaries
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import TTFSCycleAdaptationTuner


class TTFSCycleAdaptationStep(TunerPipelineStep):
    # Class-level STATIC contract (the V5 DAG-validation lower bound). The
    # ``activation_scales`` requirement is INSTANCE-SPECIFIC (opt-in flag) and is
    # therefore added in ``__init__``, not declared here — it is also always
    # produced earlier by the unconditional Activation Analysis step, so the
    # static contract remains a sound conservative bound for assembly-time
    # validation.
    REQUIRES = ("model", "adaptation_manager")
    UPDATES = ("model", "adaptation_manager")

    @classmethod
    def applies_to(cls, plan):
        # Cascaded-only: it installs the genuine ceil segment driver
        # (TTFSCycleActivation). The synchronized cell now keeps the floor NF from
        # ActivationQuantizationStep (train ttfs_quantized floor recovery, deploy
        # the mode-derived ceil kernel) instead of swapping to the ceil staircase.
        return plan.is_cascaded_ttfs

    def __init__(self, pipeline):
        requires = list(self.REQUIRES)
        self._scale_aware_boundaries = bool(
            pipeline.config.get("ttfs_scale_aware_boundaries", False)
        )
        # Opt-in: scale-aware boundaries need the Activation-Analysis theta_out.
        # Adding it to ``requires`` only when the flag is on keeps the flag-off
        # path byte-identical (and never depends on Activation Analysis output).
        if self._scale_aware_boundaries:
            requires = requires + ["activation_scales"]
        super().__init__(requires, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        model = self.get_entry("model")
        if self._scale_aware_boundaries:
            calibrate_scale_aware_boundaries(
                model, self.get_entry("activation_scales")
            )
        self.run_tuner(
            TTFSCycleAdaptationTuner,
            model,
            self.get_entry("adaptation_manager"),
        )
