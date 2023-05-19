from mimarsinan.transformations.chip_quantization import ChipQuantization
from mimarsinan.mapping.softcore_mapping import HardCoreMapping
from mimarsinan.visualization.hardcore_visualization import HardCoreMappingVisualizer

class HardCoreMapper:
    def __init__(self, pipeline, soft_core_mapping):
        self.model = pipeline.model
        self.soft_core_mapping = soft_core_mapping
        self.axons_per_core = pipeline.max_axons
        self.neurons_per_core = pipeline.max_neurons
        self.wd = pipeline.working_directory

    def run(self):
        ChipQuantization(bits = 4).calculate_core_thresholds(
            self.soft_core_mapping.cores)

        hard_core_mapping = HardCoreMapping(
            self.axons_per_core, self.neurons_per_core)
        
        hard_core_mapping.map(self.soft_core_mapping)

        HardCoreMappingVisualizer(hard_core_mapping).visualize(
            self.wd + "/hardcore_mapping.png"
        )

        return hard_core_mapping
    