from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.search.small_step_evaluator import SmallStepEvaluator
from mimarsinan.search.mlp_mixer_configuration_sampler import MLP_Mixer_ConfigurationSampler
from mimarsinan.search.optimizers.sampler_optimizer import SamplerOptimizer
from mimarsinan.search.problems.evaluator_problem import EvaluatorProblem
from mimarsinan.search.results import ObjectiveSpec
from mimarsinan.models.builders import PerceptronMixerBuilder
from mimarsinan.models.builders import SimpleMLPBuilder
from mimarsinan.models.builders import SimpleConvBuilder
from mimarsinan.models.builders import VGG16Builder

class ModelConfigurationStep(PipelineStep):
    """
    Resolve model configuration and platform constraints from pipeline config.

    In ``"user"`` configuration_mode the model config is taken directly from
    ``pipeline.config['model_config']``.  In ``"nas"`` mode a lightweight
    sampler-based search is performed (for the full NSGA-II / Kedi joint
    search, use ``ArchitectureSearchStep`` instead).

    This step also emits ``platform_constraints_resolved`` and a default
    ``scaled_simulation_length`` so that downstream mapping / simulation
    steps always have them, regardless of which training or spiking-mode
    steps are included in the pipeline.
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
        builders = {
            "mlp_mixer": PerceptronMixerBuilder(
                self.pipeline.config['device'],
                self.pipeline.config['input_shape'], 
                self.pipeline.config['num_classes'], 
                self.pipeline.config['max_axons'], 
                self.pipeline.config['max_neurons'],
                self.pipeline.config),
            "simple_mlp": SimpleMLPBuilder(
                self.pipeline.config['device'],
                self.pipeline.config['input_shape'],
                self.pipeline.config['num_classes'],
                self.pipeline.config['max_axons'],
                self.pipeline.config['max_neurons'],
                self.pipeline.config
            ),
            "simple_conv": SimpleConvBuilder(
                self.pipeline.config['device'],
                self.pipeline.config['input_shape'],
                self.pipeline.config['num_classes'],
                self.pipeline.config['max_axons'],
                self.pipeline.config['max_neurons'],
                self.pipeline.config
            ),
            "vgg16": VGG16Builder(
                self.pipeline.config['device'],
                self.pipeline.config['input_shape'],
                self.pipeline.config['num_classes'],
                self.pipeline.config['max_axons'],
                self.pipeline.config['max_neurons'],
                self.pipeline.config
            )
        }
        builder = builders[self.pipeline.config['model_type']]
        
        configuration_mode = self.pipeline.config['configuration_mode']

        if configuration_mode == "nas":
            if self.pipeline.config["model_type"] != "mlp_mixer":
                raise NotImplementedError(
                    f"NAS configuration_mode only implemented for model_type='mlp_mixer' "
                    f"(got {self.pipeline.config['model_type']})"
                )

            sampler = MLP_Mixer_ConfigurationSampler()
            evaluator = SmallStepEvaluator(
                self.pipeline.data_provider_factory,
                self.pipeline.loss,
                self.pipeline.config["lr"],
                self.pipeline.config["device"],
                builders["mlp_mixer"],
            )

            problem = EvaluatorProblem(
                evaluator=evaluator,
                objective=ObjectiveSpec(name="accuracy", goal="max"),
            )

            optimizer = SamplerOptimizer(
                sampler=sampler,
                cycles=int(self.pipeline.config["nas_cycles"]),
                batch_size=int(self.pipeline.config["nas_batch_size"]),
                workers=int(self.pipeline.config["nas_workers"]),
            )

            result = optimizer.optimize(problem)
            model_config = result.best.configuration
        elif configuration_mode == "user":
            model_config = self.pipeline.config['model_config']
        else:
            raise ValueError("Invalid configuration mode: " + configuration_mode)

        self.add_entry("model_builder", builder, 'pickle')
        self.add_entry("model_config", model_config)

        # --- Emit resolved platform constraints ---
        cores_config = self.pipeline.config.get("cores")
        if cores_config is None:
            cores_config = [
                {
                    "max_axons": int(self.pipeline.config["max_axons"]),
                    "max_neurons": int(self.pipeline.config["max_neurons"]),
                    "count": 1000,  # generous default
                }
            ]
        self.add_entry("platform_constraints_resolved", {
            "cores": cores_config,
            "max_axons": int(self.pipeline.config.get("max_axons", 256)),
            "max_neurons": int(self.pipeline.config.get("max_neurons", 256)),
        })

        # --- Emit default simulation length ---
        # CoreFlowTuningStep will override this if present in the pipeline;
        # otherwise the raw config value is used (e.g. for TTFS pipelines).
        sim_steps = int(round(self.pipeline.config.get("simulation_steps", 32)))
        self.add_entry("scaled_simulation_length", sim_steps)