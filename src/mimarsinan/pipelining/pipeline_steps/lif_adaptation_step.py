"""LIF Adaptation pipeline step.

Runs only in LIF deployment mode.  Swaps every Perceptron's
``base_activation`` from the post-ActivationAdaptation ReLU-like module to
:class:`LIFActivation`, then runs a knowledge-distillation recovery loop
(:class:`LIFAdaptationTuner`) that uses the pre-swap model as teacher.

After this step the student model's forward is cycle-accurate
integrate-and-fire at the perceptron level, so:

- Weight Quantization fine-tunes weights that are already aware of the LIF
  quantization (T + 1 output levels), not the continuous ReLU output they
  used to see.  Gradients flow through SpikingJelly's ATan surrogate, so
  the existing ``PerceptronTransformTrainer`` pattern keeps working.
- Normalization Fusion, Soft-Core Mapping, Hard-Core Mapping, nevresim and
  Lava all consume a model whose outputs already match what the cycle-based
  simulator produces — no post-hoc threshold tuning (the old CoreFlowTuning
  step) is needed.

``AdaptationManager.lif_active`` is set to ``True`` inside the tuner so
subsequent ``update_activation`` calls drop the clamp/shift/quantize
decorators that would otherwise double-apply LIF's intrinsic range-and-level
semantics.
"""

from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner


class LIFAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.tuner = None

    def validate(self):
        if self.tuner is not None:
            return self.tuner.validate()
        return self.pipeline.get_target_metric()

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")

        self.tuner = LIFAdaptationTuner(
            self.pipeline,
            model=model,
            target_accuracy=self.pipeline.get_target_metric(),
            lr=self.pipeline.config["lr"],
            adaptation_manager=adaptation_manager,
        )
        self.tuner.run()

        self.update_entry("adaptation_manager", adaptation_manager, "pickle")
        self.update_entry("model", model, "torch_model")
