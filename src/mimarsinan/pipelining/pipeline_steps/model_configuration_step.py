from mimarsinan.mapping.coalescing import CANONICAL_KEY, normalize_coalescing_config
from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.pipelining.model_registry import ModelRegistry


class ModelConfigurationStep(PipelineStep):
    """
    Resolve model configuration and platform constraints from pipeline config.

    Used when ``search_mode == "fixed"`` (no architecture search).
    Takes model_config directly from ``pipeline.config['model_config']``.

    Emits ``platform_constraints_resolved`` and a default
    ``scaled_simulation_length`` so that downstream mapping / simulation
    steps always have them.
    """

    def __init__(self, pipeline):
        requires = []
        promises = [
            "model_config",
            "model_builder",
            "platform_constraints_resolved",
            "scaled_simulation_length",
        ]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def process(self):
        model_type = self.pipeline.config['model_type']
        builder_cls = ModelRegistry.get_builder_cls(model_type)
        builder = builder_cls(
            self.pipeline.config['device'],
            self.pipeline.config['input_shape'],
            self.pipeline.config['num_classes'],
            self.pipeline.config,
        )

        model_config = self.pipeline.config['model_config']

        self.add_entry("model_builder", builder, 'pickle')
        self.add_entry("model_config", model_config)

        # --- Emit resolved platform constraints ---
        cores_config = self.pipeline.config.get("cores")
        if cores_config is None:
            cores_config = [{"max_axons": 256, "max_neurons": 256, "count": 1000}]

        # Propagate has_bias to every core type
        global_has_bias = self.pipeline.config.get("platform_constraints", {}).get("has_bias", True)
        for ct in cores_config:
            ct.setdefault("has_bias", global_has_bias)

        pcfg = {
            "cores": cores_config,
            "allow_neuron_splitting": bool(self.pipeline.config.get("allow_neuron_splitting", False)),
            "allow_scheduling": bool(self.pipeline.config.get("allow_scheduling", False)),
        }
        if CANONICAL_KEY in self.pipeline.config:
            pcfg[CANONICAL_KEY] = bool(self.pipeline.config[CANONICAL_KEY])
        else:
            pcfg[CANONICAL_KEY] = False
        normalize_coalescing_config(pcfg)
        self.add_entry("platform_constraints_resolved", pcfg)

        # --- Emit default simulation length ---
        sim_steps = int(round(self.pipeline.config.get("simulation_steps", 32)))
        self.add_entry("scaled_simulation_length", sim_steps)
