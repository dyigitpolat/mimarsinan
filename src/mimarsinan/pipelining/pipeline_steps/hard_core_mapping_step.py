from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.mapping.softcore_mapping import HardCoreMapping, HardCore
from mimarsinan.visualization.hardcore_visualization import HardCoreMappingVisualizer

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.models.spiking_core_flow import SpikingCoreFlow

class HardCoreMappingStep(PipelineStep):

    def __init__(self, pipeline):
        requires = ["tuned_soft_core_mapping", "model"]
        promises = ["hard_core_mapping"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.preprocessor = None

    def validate(self):
        return self.pipeline.get_target_metric()

    def process(self):
        self.preprocessor = self.get_entry("model").get_preprocessor()
        
        soft_core_mapping = self.get_entry('tuned_soft_core_mapping')

        # support heterogeneous hardware cores
        available_hardware_cores = []
        for core_type in self.pipeline.config['cores']:
            for _ in range(core_type['count']):
                hard_core = HardCore(core_type['max_axons'], core_type['max_neurons'])
                available_hardware_cores.append(hard_core)
                

        hard_core_mapping = HardCoreMapping(available_hardware_cores)
        
        hard_core_mapping.map(soft_core_mapping)
        print("Hard Core Mapping done.")

        HardCoreMappingVisualizer(hard_core_mapping).visualize(
            self.pipeline.working_directory + "/hardcore_mapping.png"
        )
        print("Hard Core Mapping visualized.")

        print("Hard Core Mapping Test:", BasicTrainer(
            SpikingCoreFlow(
                self.pipeline.config["input_shape"], 
                hard_core_mapping, 
                self.pipeline.config["simulation_steps"], self.preprocessor,
                self.pipeline.config["firing_mode"],
                self.pipeline.config["spike_generation_mode"],
                self.pipeline.config["thresholding_mode"]), 
            self.pipeline.config["device"], 
            DataLoaderFactory(self.pipeline.data_provider_factory), None).test())

        self.add_entry("hard_core_mapping", hard_core_mapping, 'pickle')