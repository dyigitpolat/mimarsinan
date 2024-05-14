from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.chip_simulation.simulation_runner import SimulationRunner

import torch.nn as nn

class SimulationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["hard_core_mapping", "scaled_simulation_length", "model"]
        promises = []
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.accuracy = None

    def validate(self):
        return self.accuracy

    def process(self):
        preprocessor = nn.Sequential(
            self.get_entry("model").get_preprocessor(),
            self.get_entry("model").in_act)

        runner = SimulationRunner(
            self.pipeline,
            self.get_entry('hard_core_mapping'),
            self.get_entry('scaled_simulation_length'),
            preprocessor)
        
        self.accuracy = runner.run()
        print("Simulation accuracy: ", self.accuracy)