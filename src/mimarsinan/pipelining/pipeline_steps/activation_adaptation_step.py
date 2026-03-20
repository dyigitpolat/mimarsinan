"""Activation Adaptation: gradual non-ReLU → ReLU replacement.

Always runs after Activation Analysis. Uses ActivationAdaptationTuner
(SmartSmoothAdaptation) to progressively blend non-ReLU chip-targeted
activations (e.g. GELU, LeakyReLU) toward ReLU. When all activations are
already ReLU-compatible, this step is a no-op.

Does not apply activation_scales or set clamp_rate -- those are the
responsibility of downstream steps (Clamp Adaptation, etc.).
"""

from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.pipelining.pipeline_steps.activation_utils import (
    has_non_relu_activations,
)
from mimarsinan.tuning.tuners.activation_adaptation_tuner import (
    ActivationAdaptationTuner,
)
from mimarsinan.mapping.mappers.base import resolve_activation_type


class ActivationAdaptationStep(PipelineStep):
    """Gradual ReLU adaptation via SmartSmoothAdaptation.

    If any chip-targeted perceptron has a non-ReLU base (GELU, LeakyReLU),
    uses ActivationAdaptationTuner to gradually blend activations toward
    ReLU. When all are already ReLU-compatible, this is a no-op.
    """

    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.tuner = None

    def validate(self):
        if self.tuner is not None:
            # Return the metric measured right after the ReLU commit, cached in
            # run(). Avoid eager evaluation of tuner.validate(): using
            # getattr(..., self.tuner.validate()) would still advance the
            # iterator and emit a noisy minibatch metric even when the cached
            # metric exists.
            if hasattr(self.tuner, "_committed_metric"):
                return self.tuner._committed_metric
            return self.tuner.validate()
        return self.pipeline.get_target_metric()

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")

        if has_non_relu_activations(model):
            self.tuner = ActivationAdaptationTuner(
                self.pipeline,
                model=model,
                target_accuracy=self.pipeline.get_target_metric(),
                lr=self.pipeline.config["lr"] * 1e-3,
                adaptation_manager=adaptation_manager,
            )
            self.tuner.run()
            print(
                "[ActivationAdaptationStep] Gradual ReLU adaptation done; "
                f"accuracy: {self.tuner._committed_metric:.4f}"
            )
        else:
            print(
                "[ActivationAdaptationStep] All activations are ReLU-compatible — "
                "no adaptation needed."
            )

        # Diagnostic: verify activation types as the mapper/IR will see them.
        act_types = {}
        for p in model.get_perceptrons():
            t = resolve_activation_type(p)
            base = (t or "").split(" + ")[0].strip()
            act_types[base] = act_types.get(base, 0) + 1
        if act_types:
            summary = ", ".join(f"{k}: {v}" for k, v in sorted(act_types.items()))
            print(f"[ActivationAdaptationStep] Activation types (as seen by IR): {summary}")

        self.update_entry("adaptation_manager", adaptation_manager, "pickle")
        self.update_entry("model", model, "torch_model")
