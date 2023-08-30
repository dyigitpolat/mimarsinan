from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.mapping.softcore_mapping import HardCoreMapping
from mimarsinan.visualization.hardcore_visualization import HardCoreMappingVisualizer

class HardCoreMappingStep(PipelineStep):

    def __init__(self, pipeline):
        requires = ["tuned_soft_core_mapping"]
        promises = ["hard_core_mapping"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def process(self):
        soft_core_mapping = self.get_entry('tuned_soft_core_mapping')
        axons_per_core = self.pipeline.config['max_axons']
        neurons_per_core = self.pipeline.config['max_neurons']

        hard_core_mapping = HardCoreMapping(
            axons_per_core, neurons_per_core)
        
        hard_core_mapping.map(soft_core_mapping)

        HardCoreMappingVisualizer(hard_core_mapping).visualize(
            self.pipeline.working_directory + "/hardcore_mapping.png"
        )

        self.add_entry("hard_core_mapping", hard_core_mapping, 'pickle')