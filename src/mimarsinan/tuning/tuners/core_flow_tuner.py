"""
CoreFlowTuner: Threshold tuning for unified IRGraph-based workflow.

This tuner adjusts thresholds of NeuralCore nodes in an IRGraph to optimize
spiking simulation accuracy by minimizing the difference between stable 
(deterministic) spike rates and regular event-based spike rates.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Dict

import torch.nn as nn

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow, StableSpikingUnifiedCoreFlow
from mimarsinan.mapping.ir import NeuralCore


@dataclass
class CoreFlowTuningResult:
    tuned_ir_graph: object
    scaled_simulation_length: int
    best_validation_accuracy: float


class CoreFlowTuner:
    """
    Threshold tuner that minimizes the difference between stable spiking 
    network spike rates and regular event-based network spike rates.
    
    The algorithm:
    1. Run StableSpikingUnifiedCoreFlow to get target spike rates per core
    2. Run SpikingUnifiedCoreFlow to get current spike rates per core
    3. Adjust thresholds to minimize the difference
    4. Repeat for a few tuning cycles
    """

    def __init__(self, pipeline, ir_graph, preprocessor: nn.Module):
        self.pipeline = pipeline
        self.device = pipeline.config["device"]
        self.data_loader_factory = DataLoaderFactory(pipeline.data_provider_factory)
        self.report_function = pipeline.reporter.report

        self.input_shape = pipeline.config["input_shape"]
        self.simulation_steps = int(round(pipeline.config["simulation_steps"]))

        self.firing_mode = pipeline.config["firing_mode"]
        self.spike_mode = pipeline.config["spike_generation_mode"]
        self.thresholding_mode = pipeline.config["thresholding_mode"]
        self.spiking_mode = pipeline.config.get("spiking_mode", "rate")

        self.preprocessor = preprocessor
        self.ir_graph = copy.deepcopy(ir_graph)

        self.accuracy = None

    def _make_stable_flow(self) -> StableSpikingUnifiedCoreFlow:
        return StableSpikingUnifiedCoreFlow(
            self.input_shape,
            self.ir_graph,
            self.simulation_steps,
            self.preprocessor,
            self.firing_mode,
            self.thresholding_mode,
            spiking_mode=self.spiking_mode,
        )

    def _make_spiking_flow(self) -> SpikingUnifiedCoreFlow:
        return SpikingUnifiedCoreFlow(
            self.input_shape,
            self.ir_graph,
            self.simulation_steps,
            self.preprocessor,
            self.firing_mode,
            self.spike_mode,
            self.thresholding_mode,
            spiking_mode=self.spiking_mode,
        )

    def _validate(self, flow) -> float:
        trainer = BasicTrainer(flow.to(self.device), self.device, self.data_loader_factory, None)
        trainer.report_function = self.report_function
        return trainer.validate()
    
    def _test(self, flow) -> float:
        trainer = BasicTrainer(flow.to(self.device), self.device, self.data_loader_factory, None)
        trainer.report_function = self.report_function
        return trainer.test()

    def run(
        self,
        *,
        threshold_scales=None,
        simulation_lengths=None,
        coordinate_descent_passes: int = 1,
    ) -> CoreFlowTuningResult:
        """
        Run threshold tuning using spike rate matching algorithm.
        """
        # Get neural cores from the graph
        cores = [n for n in self.ir_graph.nodes if isinstance(n, NeuralCore)]
        if not cores:
            print("  CoreFlowTuner: No neural cores to tune")
            return CoreFlowTuningResult(
                tuned_ir_graph=self.ir_graph,
                scaled_simulation_length=self.simulation_steps,
                best_validation_accuracy=self._validate(self._make_spiking_flow()),
            )

        # Build latency groups for propagation
        latency_groups: Dict[int, list[int]] = {}
        for idx, core in enumerate(cores):
            lat = core.latency if core.latency is not None else 0
            if lat not in latency_groups:
                latency_groups[lat] = []
            latency_groups[lat].append(idx)

        max_latency = max(latency_groups.keys()) if latency_groups else 1

        # Step 1: Get stable spike rates as targets
        stable_flow = self._make_stable_flow().to(self.device)
        stable_acc = self._validate(stable_flow)
        print(f"  Stable SpikingUnifiedCoreFlow Accuracy: {stable_acc}")
        
        # Spike rates are collected during validation forward passes
        stable_spike_rates = stable_flow.get_core_spike_rates()
        print(f"  Stable spike rates (first 5): {stable_spike_rates[:min(5, len(stable_spike_rates))]}")

        # Step 2: Get current spiking flow
        spiking_flow = self._make_spiking_flow().to(self.device)
        print(f"  Initial SpikingUnifiedCoreFlow Accuracy: {self._validate(spiking_flow)}")

        # Step 3: Tune thresholds to match stable spike rates
        thresholds = [float(core.threshold) for core in cores]
        best_thresholds = thresholds.copy()
        max_acc = 0.0
        
        tuning_cycles = 5
        lr = 1.0

        print("  Tuning thresholds...")
        for cycle in range(tuning_cycles):
            # Get current spike rates
            current_spike_rates = spiking_flow.get_core_spike_rates()
            
            acc = self._validate(spiking_flow)
            print(f"  Tuning cycle {cycle+1}/{tuning_cycles}, acc: {acc}")
            
            if acc > max_acc:
                max_acc = acc
                best_thresholds = [float(core.threshold) for core in cores]

            # Compute perturbations for each core
            perturbations = []
            for core_idx, core in enumerate(cores):
                current_rate = current_spike_rates[core_idx] + 0.01
                target_rate = (random.uniform(0.99, 1.02) * stable_spike_rates[core_idx] + 0.01) * 1.05
                
                perturbation = target_rate - current_rate

                # Scale perturbation based on latency position and tuning progress
                core_lat = core.latency if core.latency is not None else 0
                if max_latency > 1 and tuning_cycles > 1:
                    t = cycle / (tuning_cycles - 1)
                    multiplier = (1 - core_lat / (max_latency)) * (1 - t) + (core_lat / max_latency) * t
                else:
                    multiplier = 1.0
                perturbation *= multiplier

                perturbations.append(perturbation)

            # Average perturbation per latency group
            avg_perturbation_per_group = {}
            for lat, indices in latency_groups.items():
                total = sum(perturbations[i] for i in indices)
                avg_perturbation_per_group[lat] = total / len(indices)

            # Propagate perturbations from later layers
            for core_idx, core in enumerate(cores):
                core_lat = core.latency if core.latency is not None else 0
                if core_lat + 1 in avg_perturbation_per_group:
                    perturbations[core_idx] += avg_perturbation_per_group[core_lat + 1] * 0.5

            # Apply perturbations to thresholds
            for core_idx, core in enumerate(cores):
                pert = max(-0.90, min(1.10, perturbations[core_idx]))
                thresholds[core_idx] = thresholds[core_idx] * (1 - pert * lr)
                core.threshold = max(1.0, round(thresholds[core_idx]))

            # Refresh thresholds in the flow
            spiking_flow.refresh_thresholds()
            
            lr *= math.pow(0.8, 1 / tuning_cycles)

        # Restore best thresholds
        for core_idx, core in enumerate(cores):
            core.threshold = max(1.0, round(best_thresholds[core_idx]))

        # Step 4: Final quantization and test
        spiking_flow = self._make_spiking_flow().to(self.device)
        self.accuracy = self._test(spiking_flow)
        print(f"  Final SpikingUnifiedCoreFlow Accuracy: {self.accuracy}")

        return CoreFlowTuningResult(
            tuned_ir_graph=self.ir_graph,
            scaled_simulation_length=self.simulation_steps,
            best_validation_accuracy=self.accuracy,
        )

    def validate(self):
        return self.accuracy

    def print_colorful(self, color, text):
        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "purple": "\033[95m",
            "cyan": "\033[96m",
            "gray": "\033[90m"
        }
        if color not in colors:
            raise ValueError(f"Invalid color: {color}. Choose from: {', '.join(colors.keys())}")
        print(f"{colors[color]}{text}\033[0m", end="")
