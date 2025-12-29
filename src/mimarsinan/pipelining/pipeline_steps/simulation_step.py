from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.chip_simulation.simulation_runner import SimulationRunner

import torch.nn as nn
import torch

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow
from mimarsinan.mapping.hybrid_hardcore_mapping import HybridHardCoreMapping

class SimulationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["hard_core_mapping", "scaled_simulation_length", "model", "ir_graph"]
        promises = []
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.accuracy = None

    def validate(self):
        return self.accuracy

    def process(self):
        model = self.get_entry("model")
        hard_core_mapping = self.get_entry("hard_core_mapping")
        sim_len = int(self.get_entry("scaled_simulation_length"))
        ir_graph = self.get_entry("ir_graph")

        preprocessor = nn.Sequential(model.get_preprocessor(), model.in_act)

        has_compute_ops = ir_graph is not None and len(ir_graph.get_compute_ops()) > 0
        if has_compute_ops:
            # Prefer hybrid runtime if available (deployable: neural segments map to chip cores).
            if isinstance(hard_core_mapping, HybridHardCoreMapping):
                flow = SpikingHybridCoreFlow(
                    self.pipeline.config["input_shape"],
                    hard_core_mapping,
                    sim_len,
                    preprocessor,
                    self.pipeline.config["firing_mode"],
                    self.pipeline.config["spike_generation_mode"],
                    self.pipeline.config["thresholding_mode"],
                )
                self.accuracy = BasicTrainer(
                    flow,
                    self.pipeline.config["device"],
                    DataLoaderFactory(self.pipeline.data_provider_factory),
                    None,
                ).test()
                print("Simulation accuracy (Hybrid runtime): ", self.accuracy)
                return

            # Fallback: unified IR python simulation path (spiking + sync barriers for compute ops).
            if hard_core_mapping is None:
                flow = SpikingUnifiedCoreFlow(
                    self.pipeline.config["input_shape"],
                    ir_graph,
                    sim_len,
                    preprocessor,
                    self.pipeline.config["firing_mode"],
                    self.pipeline.config["spike_generation_mode"],
                    self.pipeline.config["thresholding_mode"],
                )
                self.accuracy = BasicTrainer(
                    flow,
                    self.pipeline.config["device"],
                    DataLoaderFactory(self.pipeline.data_provider_factory),
                    None,
                ).test()
                print("Simulation accuracy (Unified IR): ", self.accuracy)
                return

        # Default chip simulation path (neural-only).
        runner = SimulationRunner(self.pipeline, hard_core_mapping, sim_len, preprocessor)
        self.accuracy = runner.run()
        print("Simulation accuracy: ", self.accuracy)