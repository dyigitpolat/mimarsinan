"""Optional Loihi-target simulation step.

Added to the pipeline when ``enable_loihi_simulation`` is true in
``deployment_parameters``.  Runs only for LIF mode (see
``LavaLoihiRunner`` for the rationale — TTFS does not map onto Loihi's
LIF dynamics).

Runs *after* :class:`SimulationStep` so users still get the nevresim
cycle-accurate reference; this step then verifies the same mapping under
Loihi-compatible LIF dynamics (du = dv = 0, subtractive reset).
"""

from mimarsinan.chip_simulation.lava_loihi_runner import LavaLoihiRunner
from mimarsinan.pipelining.pipeline_step import PipelineStep


class LoihiSimulationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["hard_core_mapping", "scaled_simulation_length"]
        promises = []
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.accuracy = None

    def validate(self):
        return self.accuracy

    def process(self):
        runner = LavaLoihiRunner(
            self.pipeline,
            self.get_entry("hard_core_mapping"),
            self.get_entry("scaled_simulation_length"),
        )
        self.accuracy = runner.run()
        print("Loihi simulation accuracy: ", self.accuracy)
