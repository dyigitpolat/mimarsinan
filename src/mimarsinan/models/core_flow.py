from mimarsinan.mapping.chip_latency import *

import torch.nn as nn
import torch

class CoreFlow(nn.Module):
    def __init__(self, input_shape, core_mapping):
        super(CoreFlow, self).__init__()
        self.input_shape = input_shape

        self.core_mapping = core_mapping
        self.cores = core_mapping.cores
        self.output_sources = core_mapping.output_sources
        self.core_params = nn.ParameterList(
            [nn.Parameter(torch.tensor(
                self.cores[core].core_matrix.transpose(), dtype=torch.float32))\
                    for core in range(len(self.cores))]
        )

        self.activation = nn.ReLU()
        self.cycles = ChipLatency(core_mapping).calculate()

        # Stats
        self.core_avgs = [None] * len(self.cores)
        self.chip_avg = None

    def update_cores(self):
        for idx, core in enumerate(self.cores):
            core.core_matrix[:,:] = \
                self.core_params[idx].detach().numpy().transpose()
    
    def get_signal(
            self, x, buffers, core, neuron, is_input, is_off, is_always_on):
        
        if is_input:
            return x[:, neuron]
        
        if is_off:
            return torch.zeros_like(x[:, 0])
        
        if is_always_on:
            return torch.ones_like(x[:, 0])
        
        return buffers[core][:, neuron]
    
    def get_signal_tensor(self, x, buffers, sources):
        signal_tensor = torch.empty(x.shape[0], len(sources), device=x.device)
        for idx, spike_source in enumerate(sources):
            signal_tensor[:, idx] = self.get_signal(
                x, 
                buffers,
                spike_source.core_, 
                spike_source.neuron_, 
                spike_source.is_input_, 
                spike_source.is_off_, 
                spike_source.is_always_on_)
        
        return signal_tensor
    
    def set_activation(self, activation):
        self.activation = activation

    def update_stats(self, buffers):
        for i in range(len(self.cores)):
            if self.core_avgs[i] is None:
                self.core_avgs[i] = torch.mean(buffers[i]).item()
            else:
                self.core_avgs[i] = \
                    0.9 * self.core_avgs[i] + 0.1 * torch.mean(buffers[i]).item()
        
        if self.chip_avg is None:
            self.chip_avg = torch.mean(torch.cat(buffers)).item()
        else:
            self.chip_avg = \
                0.9 * self.chip_avg + 0.1 * torch.mean(torch.cat(buffers)).item()
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)

        buffers = []
        input_signals = []
        for core in self.cores:
            buffers.append(torch.zeros(x.shape[0], core.get_output_count()))
            input_signals.append(
                torch.zeros(x.shape[0], core.get_input_count()))
        
        for _ in range(self.cycles):
            for core_idx in range(len(self.cores)):
                input_signals[core_idx] = self.get_signal_tensor(
                    x, buffers, self.cores[core_idx].axon_sources)

            for core_idx in range(len(self.cores)):
                buffers[core_idx] = self.activation(
                    torch.matmul(
                        self.core_params[core_idx], 
                        input_signals[core_idx].T).T)
        
        self.update_stats(buffers)
        output_signals = self.get_signal_tensor(x, buffers, self.output_sources)
        return output_signals