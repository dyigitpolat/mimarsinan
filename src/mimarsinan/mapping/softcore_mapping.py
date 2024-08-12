from mimarsinan.code_generation.cpp_chip_model import *

import torch
import numpy as np

def is_off(idx): return idx == -1
def is_input(idx): return idx == -2
def is_always_on(idx): return idx == -3

class SoftCore:
    def __init__(self, core_matrix, axon_sources, id, activation_scale = torch.tensor(1.0), parameter_scale = torch.tensor(1.0), input_activation_scale = torch.tensor(1.0)):
        self.core_matrix = core_matrix
        self.axon_sources = axon_sources

        self.id = id
        self.input_activation_scale = input_activation_scale
        self.activation_scale = activation_scale
        self.parameter_scale = parameter_scale
        self.threshold = 1.0

        self.latency = None

    def get_input_count(self):
        return len(self.axon_sources)
    
    def get_output_count(self):
        return self.core_matrix.shape[-1]
    
class HardCore:
    def __init__(self, axons_per_core, neurons_per_core):
        self.axons_per_core = axons_per_core
        self.neurons_per_core = neurons_per_core

        self.core_matrix = np.zeros((axons_per_core, neurons_per_core))
        self.axon_sources = []

        self.available_axons = axons_per_core
        self.available_neurons = neurons_per_core

        self.input_activation_scale = None
        self.activation_scale = None
        self.parameter_scale = None
        self.threshold = None
        self.latency = None
    
        self.unusable_space = 0

    def get_input_count(self):
        return self.axons_per_core
    
    def get_output_count(self):
        return self.neurons_per_core

    def add_softcore(self, softcore):
        assert self.available_axons >= softcore.get_input_count() 
        assert self.available_neurons >= softcore.get_output_count()            

        axon_offset = self.axons_per_core - self.available_axons
        neuron_offset = self.neurons_per_core - self.available_neurons
        
        self.core_matrix[
            axon_offset : axon_offset+softcore.get_input_count(), 
            neuron_offset : neuron_offset+softcore.get_output_count()] \
                = softcore.core_matrix
        
        self.axon_sources.extend(softcore.axon_sources)

        self.available_axons -= softcore.get_input_count()
        self.available_neurons -= softcore.get_output_count()

        if self.threshold is None:
            self.threshold = softcore.threshold

        if self.input_activation_scale is None:
            self.input_activation_scale = softcore.input_activation_scale

        if self.activation_scale is None:
            self.activation_scale = softcore.activation_scale

        if self.parameter_scale is None:
            self.parameter_scale = softcore.parameter_scale
        
        if self.latency is None:
            self.latency = softcore.latency

        self.unusable_space += \
            (neuron_offset * softcore.get_input_count()) + \
            (axon_offset * softcore.get_output_count())

class HardCoreMapping:
    def __init__(self, chip_cores):
        self.unused_cores = chip_cores

        self.cores = []
        self.output_sources = []
        self.neuron_mapping = {}

        self.unusable_space = 0

    def merge_softcore_into(self, hardcore, softcore):
        prev_output_count = hardcore.neurons_per_core - hardcore.available_neurons
        hardcore.add_softcore(softcore)
        
        for soft_neuron_idx in range(softcore.get_output_count()):
            target_core_idx = len(self.cores) - 1
            target_neuron_idx = prev_output_count + soft_neuron_idx
            self.neuron_mapping[(softcore.id, soft_neuron_idx)] = \
                (target_core_idx, target_neuron_idx)
                
    def map(self, softcore_mapping):
        def is_mapping_possible(core, hardcore):
            tolerance = 0.2
            if hardcore.threshold is not None:
                threshold_diff = abs(core.threshold - hardcore.threshold)
                diff_rate = threshold_diff / (hardcore.threshold + 1)
            else:
                diff_rate = 0.0

            return \
                diff_rate <= tolerance and \
                core.get_input_count() <= hardcore.available_axons and \
                core.get_output_count() <= hardcore.available_neurons and \
                (core.latency == hardcore.latency or hardcore.latency is None)

        def update_unusable_space(hardcore):
            available_axons = hardcore.available_axons
            available_neurons = hardcore.available_neurons
            wasted_space = \
                (hardcore.neurons_per_core * available_axons) + \
                ((hardcore.axons_per_core - available_axons) * available_neurons)
            self.unusable_space += wasted_space + hardcore.unusable_space   

        def pick_suitable_hardcore(softcore, hardcores):
            for hardcore in hardcores:
                if is_mapping_possible(softcore, hardcore):
                    return hardcore
            return None 
        
        def pick_best_softcore(unmapped_cores):
            unmapped_cores.sort(key=lambda core: core.get_input_count(), reverse=True)
            core_a = unmapped_cores[0]
            unmapped_cores.sort(key=lambda core: core.get_output_count(), reverse=True)
            core_b = unmapped_cores[0]
            
            core = None
            if core_a is None and core_b is None: core = None
            elif core_a is None: core = core_b
            elif core_b is None: core = core_a
            else:
                if core_a.get_input_count() > core_b.get_output_count():
                    core = core_a
                else:
                    core = core_b

            return core
        
        unmapped_cores = [core for core in softcore_mapping.cores]
        while len(unmapped_cores) > 0:
            core = pick_best_softcore(unmapped_cores)
            hardcore = pick_suitable_hardcore(core, self.cores)

            if hardcore is None:
                assert len(self.unused_cores) > 0, "No more hard cores available"
                new_hardcore = pick_suitable_hardcore(core, self.unused_cores)
                if new_hardcore is None:
                    raise Exception("No more hard cores available")
                self.cores.append(new_hardcore)
                self.unused_cores.remove(new_hardcore)
                hardcore = self.cores[-1]
                update_unusable_space(self.cores[-1])

            self.merge_softcore_into(hardcore, core)
            unmapped_cores.remove(core)

        def remap_sources(sources):
            for source in sources:
                if source.is_off_ or source.is_input_ or source.is_always_on_: continue
                source.core_, source.neuron_ = \
                    self.neuron_mapping[(source.core_, source.neuron_)]
            
        for hardcore in self.cores:
            remap_sources(hardcore.axon_sources)
                
        self.output_sources = np.array(softcore_mapping.output_sources)
        remap_sources(self.output_sources)

        for hardcore in self.cores:
            axon_count = len(hardcore.axon_sources)
            for _ in range(hardcore.axons_per_core - axon_count):
                hardcore.axon_sources.append(
                    SpikeSource(-1, 0, is_input=False, is_off=True))
                    