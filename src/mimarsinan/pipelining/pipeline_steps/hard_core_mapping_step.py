from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.mapping.softcore_mapping import HardCoreMapping
from mimarsinan.visualization.hardcore_visualization import HardCoreMappingVisualizer

class HardCoreMappingStep(PipelineStep):

    def __init__(self, pipeline):
        requires = ["soft_core_mapping"]
        promises = ["hard_core_mapping"]
        clears = []
        super().__init__(requires, promises, clears, pipeline)

    def process(self):
        soft_core_mapping = self.pipeline.cache['soft_core_mapping']
        axons_per_core = self.pipeline.config['max_axons']
        neurons_per_core = self.pipeline.config['max_neurons']

        hard_core_mapping = HardCoreMapping(
            axons_per_core, neurons_per_core)
        
        hard_core_mapping.map(soft_core_mapping)

        HardCoreMappingVisualizer(hard_core_mapping).visualize(
            self.pipeline.working_directory + "/hardcore_mapping.png"
        )

        self.pipeline.cache.add("hard_core_mapping", hard_core_mapping, 'pickle')