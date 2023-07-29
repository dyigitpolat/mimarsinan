from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.chip_simulation.simulation_runner import SimulationRunner

class SimulationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["tuned_hard_core_mapping"]
        promises = []
        clears = []
        super().__init__(requires, promises, clears, pipeline)

        self.accuracy = None

    def validate(self):
        return self.accuracy

    def process(self):
        runner = SimulationRunner(
            self.pipeline,
            self.pipeline.cache['tuned_hard_core_mapping'])
        
        self.accuracy = runner.run()
        print("Simulation accuracy: ", self.accuracy)