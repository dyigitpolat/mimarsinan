from mimarsinan.mapping.chip_latency import *

import torch.nn as nn
import torch

class SpikingCoreFlow(nn.Module):
    def __init__(self, input_shape, core_mapping, simulation_length, preprocessor, firing_mode, spike_mode, thresholding_mode):
        super(SpikingCoreFlow, self).__init__()
        self.input_shape = input_shape

        self.preprocessor = preprocessor

        self.core_mapping = core_mapping
        self.cores = core_mapping.cores
        self.output_sources = core_mapping.output_sources
        self.core_params = nn.ParameterList(
            [nn.Parameter(torch.tensor(
                self.cores[core].core_matrix.transpose(), dtype=torch.float32, requires_grad=False))\
                    for core in range(len(self.cores))]
        )
        
        self.thresholds = nn.ParameterList(
            [nn.Parameter(torch.tensor(
                self.cores[core].threshold, dtype=torch.float32), requires_grad=False)\
                    for core in range(len(self.cores))])
        
        self.firing_mode = firing_mode
        assert firing_mode in ["Default", "Novena"]

        self.spike_mode = spike_mode
        assert spike_mode in ["Stochastic", "Deterministic", "FrontLoaded", "Uniform"]

        self.thresholding_mode = thresholding_mode
        assert thresholding_mode in ["<", "<="]

        self.latency = ChipLatency(core_mapping).calculate()
        self.cycles = self.latency + simulation_length
        self.simulation_length = simulation_length

        # Stats
        self.core_avgs = None
        self.total_spikes = 0

    def set_simulation_length(self, simulation_length):
        self.cycles = self.latency + simulation_length
        self.simulation_length = simulation_length
        
    def update_cores(self):
        for idx, core in enumerate(self.cores):
            core.core_matrix[:,:] = \
                self.core_params[idx].detach().numpy().transpose()
            
    def refresh_thresholds(self):
        self.thresholds = nn.ParameterList(
            [nn.Parameter(torch.tensor(
                self.cores[core].threshold, dtype=torch.float32), requires_grad=False)\
                    for core in range(len(self.cores))])
    
    def get_signal(
            self, x, buffers, core, neuron, is_input, is_off, is_always_on, cycle):

        if is_input:
            return x[:, neuron]
        
        if is_off:
            return torch.zeros_like(x[:, 0])
        
        if is_always_on:
            return torch.ones_like(x[:, 0])
        
        return buffers[core][:, neuron]
    
    def get_signal_tensor(self, x, buffers, sources, cycle):
        signal_tensor = torch.empty(x.shape[0], len(sources), device=x.device)
        for idx, spike_source in enumerate(sources):
            signal_tensor[:, idx] = self.get_signal(
                x, 
                buffers,
                spike_source.core_, 
                spike_source.neuron_, 
                spike_source.is_input_, 
                spike_source.is_off_, 
                spike_source.is_always_on_,
                cycle)
        
        return signal_tensor

    def update_stats(self, buffers, batch_size, cycle):
        for i in range(len(self.cores)):
            if (self.cores[i].latency is not None) and cycle >= self.cores[i].latency and cycle < self.simulation_length + self.cores[i].latency:
                self.core_avgs[i] += torch.sum(buffers[i]).item() / (self.simulation_length * batch_size * self.cores[i].get_output_count())

    def get_core_spike_rates(self):
        return self.core_avgs
    
    def get_total_spikes(self):
        return self.total_spikes
    
    def to_lif_spikes(self, tensor, membrane_potentials, threshold):
        membrane_potentials += tensor
        spikes = (membrane_potentials > threshold).float()
        membrane_potentials[spikes == 1] -= threshold
        return spikes

    def to_stochastic_spikes(self, tensor):
        return (torch.rand(tensor.shape, device=tensor.device) < tensor).float()
    
    def to_front_loaded_spikes(self, tensor, cycle):
        return torch.round(tensor * self.simulation_length) > cycle

    def to_deterministic_spikes(self, tensor, threshold = 0.5):
        return (tensor > threshold).float()

    def to_uniform_spikes(self, tensor, cycle):
        T = self.simulation_length
        
        # Compute N for all elements in the tensor at once
        N = torch.round(tensor * T).to(torch.long)
        
        # Create a mask for edge cases
        mask = (N != 0) & (N != T) & (cycle < T)
        
        # Compute spacing for all elements
        spacing = T / N.float()
        
        # Compute the result for non-edge cases
        result = mask & (torch.floor(cycle / spacing) < N) & (torch.floor(cycle % spacing) == 0)
        
        # Handle edge cases
        result = result.float()
        result[N == T] = 1.0
        
        return result
    
    def to_spikes(self, tensor, cycle):
        if self.spike_mode == "Stochastic":
            return self.to_stochastic_spikes(tensor)
        elif self.spike_mode == "Deterministic":
            return self.to_deterministic_spikes(tensor)
        elif self.spike_mode == "FrontLoaded":
            return self.to_front_loaded_spikes(tensor, cycle)
        elif self.spike_mode == "Uniform":
            return self.to_uniform_spikes(tensor, cycle)
        else:
            raise ValueError("Invalid spike mode: " + self.spike_mode)

    def forward(self, x):
        print("thresholds: ", [t.item() for t in self.thresholds])
        
        x = self.preprocessor(x)
        
        x = x.view(x.shape[0], -1)

        self.core_avgs = [0.0] * len(self.cores)


        # thresholding selection
        ops = { "<": lambda x, y: x < y, "<=": lambda x, y: x <= y }

        buffers = []
        input_signals = []
        membrane_potentials = []
        input_membrane_potentials = torch.zeros(
            x.shape[0], x.shape[1], device=x.device)
        for core in self.cores:
            buffers.append(torch.zeros(x.shape[0], core.get_output_count(), device=x.device))
            input_signals.append(
                torch.zeros(x.shape[0], core.get_input_count(), device=x.device))
            membrane_potentials.append(
                torch.zeros(x.shape[0], core.get_output_count(), device=x.device))
        
        output_signals = torch.zeros(x.shape[0], len(self.output_sources), device=x.device)

        # print(f"threshold:", self.thresholds[10].item())
        for cycle in range(self.cycles):
            input_spikes = self.to_spikes(x, cycle)
            for core_idx in range(len(self.cores)):
                input_signals[core_idx] = self.get_signal_tensor(
                    input_spikes, 
                    buffers, self.cores[core_idx].axon_sources,
                    cycle)

            for core_idx in range(len(self.cores)):
                if (self.cores[core_idx].latency is not None) and cycle >= self.cores[core_idx].latency and cycle < self.simulation_length + self.cores[core_idx].latency:
                    memb = membrane_potentials[core_idx]

                    memb += \
                        torch.matmul(
                            self.core_params[core_idx], 
                            input_signals[core_idx].T).T
                    

                    #buffers[core_idx] = (memb > self.thresholds[core_idx]).float()
                    buffers[core_idx] = (ops[self.thresholding_mode](self.thresholds[core_idx], memb)).float()

                    # if(core_idx == 10):
                    #     print(f"({memb[core_idx].flatten()[24]} : {buffers[core_idx].flatten()[24]})", end=" ")

                    # novena reset
                    if self.firing_mode == "Novena":
                        memb[ops[self.thresholding_mode](self.thresholds[core_idx], memb)] = 0.0

                    # subtract reset
                    if self.firing_mode == "Default":
                        memb[ops[self.thresholding_mode](self.thresholds[core_idx], memb)] -= self.thresholds[core_idx]
        
            self.update_stats(buffers, x.shape[0], cycle)
            output_signals += self.get_signal_tensor(input_spikes, buffers, self.output_sources, cycle)
        
        # print()
        self.total_spikes = torch.sum(output_signals).item()
        print("total spikes: ", self.total_spikes)
        return output_signals