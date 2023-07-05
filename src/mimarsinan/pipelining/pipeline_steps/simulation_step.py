from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.chip_simulation.simulation_runner import SimulationRunner

class SimulationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["tuned_hard_core_mapping", "cf_threshold_scale", "tuned_cf_accuracy"]
        promises = ["simulation_accuracy"]
        clears = []
        super().__init__(requires, promises, clears, pipeline)

    def process(self):
        runner = SimulationRunner(
            self.pipeline,
            self.pipeline.cache['tuned_hard_core_mapping'],
            self.pipeline.cache['cf_threshold_scale'])
        
        sim_accuracy = runner.run()
        print("Simulation accuracy: ", sim_accuracy)

        assert sim_accuracy >= self.pipeline.cache['tuned_cf_accuracy'] * 0.9,\
            "Simulation step failed to retain validation accuracy."

        self.pipeline.cache.add("simulation_accuracy", sim_accuracy)