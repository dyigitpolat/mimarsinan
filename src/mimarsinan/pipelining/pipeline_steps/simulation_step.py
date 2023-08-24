from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.chip_simulation.simulation_runner import SimulationRunner

class SimulationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["hard_core_mapping", "scaled_simulation_length"]
        promises = []
        clears = []
        super().__init__(requires, promises, clears, pipeline)

        self.accuracy = None

    def validate(self):
        return self.accuracy

    def process(self):
        runner = SimulationRunner(
            self.pipeline,
            self.get_entry('hard_core_mapping'),
            self.get_entry('scaled_simulation_length'))
        
        self.accuracy = runner.run()
        print("Simulation accuracy: ", self.accuracy)