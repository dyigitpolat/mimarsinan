"""Activation Adaptation: gradual non-ReLU → ReLU replacement."""

from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.pipelining.pipeline_steps.activation_utils import has_non_relu_activations
from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.mapping.mappers.base import resolve_activation_type
from mimarsinan.tuning.tuners.activation_adaptation_tuner import ActivationAdaptationTuner


class ActivationAdaptationStep(TunerPipelineStep):
    @classmethod
    def applies_to(cls, plan):
        return not plan.is_lif_style

    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")

        if has_non_relu_activations(model):
            self.tuner = ActivationAdaptationTuner(
                self.pipeline,
                model=model,
                target_accuracy=self.pipeline.get_target_metric(),
                lr=self.pipeline.config["lr"],
                adaptation_manager=adaptation_manager,
            )
            result = self.tuner.run()
            print(
                "[ActivationAdaptationStep] Gradual ReLU adaptation done; "
                f"accuracy: {result:.4f}"
            )
        else:
            print(
                "[ActivationAdaptationStep] All activations are ReLU-compatible — "
                "no adaptation needed."
            )

        act_types = {}
        for p in model.get_perceptrons():
            t = resolve_activation_type(p)
            base = (t or "").split(" + ")[0].strip()
            act_types[base] = act_types.get(base, 0) + 1
        if act_types:
            summary = ", ".join(f"{k}: {v}" for k, v in sorted(act_types.items()))
            print(f"[ActivationAdaptationStep] Activation types (as seen by IR): {summary}")

        self._commit_tuner_entries(model, adaptation_manager)
