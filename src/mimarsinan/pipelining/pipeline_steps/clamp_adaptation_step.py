from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner


class ClampAdaptationStep(PipelineStep):
    """Introduces activation clamping (ClampDecorator) with recovery training.

    Always uses ClampTuner — ClampAdaptationStep is only added to the pipeline
    when activation_quantization=True or spiking_mode in ("ttfs",
    "ttfs_quantized").  In both cases, applying hard clamping (clamp_rate=1.0)
    without recovery training degrades accuracy by up to ~28% because the model
    was never trained with clamped activations.  ClampTuner uses
    SmartSmoothAdaptation to ramp clamp_rate from 0 → 1 with training at each
    step, recovering accuracy throughout.

    The previously-present "fast path" (no tuner when all activations were
    already ReLU-compatible) has been removed.  After ActivationAdaptationStep
    commits non-ReLU bases to LeakyGradReLU, has_non_relu_activations() returns
    False — but the model still needs recovery training to learn to work within
    the clamped range.  The fast path was therefore never safe when
    ClampAdaptationStep is in the pipeline.
    """

    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager", "activation_scales"]
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
        model = self.get_entry('model')
        adaptation_manager = self.get_entry("adaptation_manager")

        self.tuner = ClampTuner(
            self.pipeline,
            model=model,
            target_accuracy=self.pipeline.get_target_metric(),
            lr=self.pipeline.config['lr'] * 1e-3,
            adaptation_manager=adaptation_manager,
            activation_scales=self.get_entry("activation_scales"),
        )
        self.tuner.run()

        self.update_entry("adaptation_manager", adaptation_manager, 'pickle')
        self.update_entry("model", model, 'torch_model')
