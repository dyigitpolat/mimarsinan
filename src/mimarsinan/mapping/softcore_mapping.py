import numpy as np

def is_off(idx): return idx == -1
def is_input(idx): return idx == -2

class SoftCore:
    def __init__(self, core_matrix, axon_sources):
        self.core_matrix = core_matrix
        self.axon_sources = axon_sources

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

    def add_softcore(self, softcore):
        assert self.available_axons >= softcore.get_input_count() 
        assert self.available_neurons >= softcore.get_output_count()            

        self.core_matrix[:softcore.get_input_count(), :softcore.get_output_count()] = softcore.core_matrix
        self.axon_sources.extend(softcore.axon_sources)

        self.available_axons -= softcore.get_input_count()
        self.available_neurons -= softcore.get_output_count()
    

class HardCoreMapping:
    def __init__(self, axons_per_core, neurons_per_core):
        self.axons_per_core = axons_per_core
        self.neurons_per_core = neurons_per_core

        self.hardcores = []

    def map(self, softcores_list):
        
        def is_mapping_possible(core, hardcore):
            return \
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
        
        unmapped_cores = [core for core in softcores_list]
        verify_softcores(unmapped_cores)

        self.hardcores.append(HardCore(self.axons_per_core, self.neurons_per_core))
        while len(unmapped_cores) > 0:
            core_a = get_core_with_most_inputs_that_can_map(unmapped_cores)
            core_b = get_core_with_most_outputs_that_can_map(unmapped_cores)

            if core_a.get_input_count() > core_b.get_output_count():
                core = core_a
            else:
                core = core_b
        
            if core is None:
                self.hardcores.append(HardCore(self.axons_per_core, self.neurons_per_core))
            else:
                self.hardcores[-1].add_softcore(core)
                unmapped_cores.remove(core)
                
            

        
        
        
    


    