from mimarsinan.code_generation.cpp_chip_model import *

import numpy as np

def is_off(idx): return idx == -1
def is_input(idx): return idx == -2

class SoftCore:
    def __init__(self, core_matrix, axon_sources, id):
        self.core_matrix = core_matrix
        self.axon_sources = axon_sources

        self.id = id
        self.threshold = 1.0

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

        self.threshold = None
    
        self.unusable_space = 0

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

        self.unusable_space += \
            (neuron_offset * softcore.get_input_count()) + \
            (axon_offset * softcore.get_output_count())

class HardCoreMapping:
    def __init__(self, axons_per_core, neurons_per_core):
        self.axons_per_core = axons_per_core
        self.neurons_per_core = neurons_per_core

        self.hardcores = []
        self.output_sources = []
        self.neuron_mapping = {}

        self.unusable_space = 0

    def merge_softcore_into(self, hardcore, softcore):
        prev_output_count = hardcore.neurons_per_core - hardcore.available_neurons
        hardcore.add_softcore(softcore)
        
        for soft_neuron_idx in range(softcore.get_output_count()):
            target_core_idx = len(self.hardcores) - 1
            target_neuron_idx = prev_output_count + soft_neuron_idx
            self.neuron_mapping[(softcore.id, soft_neuron_idx)] = \
                (target_core_idx, target_neuron_idx)
                
    def map(self, softcores_list, output_sources):
        def is_mapping_possible(core, hardcore):
            tolerance = 0.1
            if hardcore.threshold is not None:
                threshold_diff = abs(core.threshold - hardcore.threshold)
                diff_rate = threshold_diff / hardcore.threshold
            else:
                diff_rate = 0.0

            return \
                diff_rate <= tolerance and \
                core.get_input_count() <= hardcore.available_axons and \
                core.get_output_count() <= hardcore.available_neurons
        
        def get_first_core_that_can_map(cores):
            for core in cores:
                if is_mapping_possible(core, self.hardcores[-1]):
                    return core
            return None
        
        def get_core_with_most_inputs_that_can_map(cores):
            cores.sort(key=lambda core: core.get_input_count(), reverse=True)
            return get_first_core_that_can_map(cores)
        
        def get_core_with_most_outputs_that_can_map(cores):
            cores.sort(key=lambda core: core.get_output_count(), reverse=True)
            return get_first_core_that_can_map(cores)
        
        def verify_softcores(softcores):
            for core in softcores:
                if core.get_input_count() > self.axons_per_core:
                    raise Exception("Too many inputs for a core")
                if core.get_output_count() > self.neurons_per_core:
                    raise Exception("Too many outputs for a core")

        def pick_best_core(unmapped_cores):
            core_a = get_core_with_most_inputs_that_can_map(unmapped_cores)
            core_b = get_core_with_most_outputs_that_can_map(unmapped_cores)
            
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

        def update_unusable_space(hardcore):
            available_axons = hardcore.available_axons
            available_neurons = hardcore.available_neurons
            wasted_space = \
                (self.neurons_per_core * available_axons) + \
                ((self.axons_per_core - available_axons) * available_neurons)
            self.unusable_space += wasted_space + hardcore.unusable_space    
        
        unmapped_cores = [core for core in softcores_list]
        verify_softcores(unmapped_cores)

        self.hardcores.append(HardCore(self.axons_per_core, self.neurons_per_core))
        while len(unmapped_cores) > 0:
            core = pick_best_core(unmapped_cores)
            
            if core is None:
                update_unusable_space(self.hardcores[-1])
                self.hardcores.append(HardCore(self.axons_per_core, self.neurons_per_core))
            else:
                self.merge_softcore_into(self.hardcores[-1], core)
                unmapped_cores.remove(core)

        def remap_sources(sources):
            for source in sources:
                if source.is_off_ or source.is_input_: continue
                source.core_, source.neuron_ = \
                    self.neuron_mapping[(source.core_, source.neuron_)]
            
        for hardcore in self.hardcores:
            remap_sources(hardcore.axon_sources)
                
        remap_sources(output_sources)

        for hardcore in self.hardcores:
            axon_count = len(hardcore.axon_sources)
            for _ in range(self.axons_per_core - axon_count):
                hardcore.axon_sources.append(
                    SpikeSource(0, 0, is_input=False, is_off=True))
                
            

        
        
        
    


    