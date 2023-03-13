import torch.nn as nn
import torch

class HardCoreFlow(nn.Module):
    def __init__(self, input_shape, hard_core_mapping):
        super(HardCoreFlow, self).__init__()
        self.input_shape = input_shape

        self.cores = hard_core_mapping.hardcores
        self.output_count = hard_core_mapping.neurons_per_core
        self.output_sources = hard_core_mapping.output_sources
        self.core_params = nn.ParameterList(
            [nn.Parameter(torch.tensor(self.cores[core].core_matrix.transpose(), dtype=torch.float32)) for core in range(len(self.cores))]
        )

    def update_cores(self):
        for core in self.cores:
            core.core_matrix[:,:] = self.core_params[core].detach().numpy().transpose()
    
    def get_signal(self, x, buffers, core, neuron, is_input, is_off, is_always_on):
        if is_input:
            return x[:, neuron]
        
        if is_off:
            return torch.zeros_like(x[:, 0])
        
        if is_always_on:
            return torch.ones_like(x[:, 0])
        
        return buffers[core][:, neuron]
    
    def get_signal_tensor(self, x, buffers, sources):
        signals = []
        for spike_source in sources:
            signals.append( self.get_signal(
                x, 
                buffers,
                spike_source.core_, 
                spike_source.neuron_, 
                spike_source.is_input_, 
                spike_source.is_off_, 
                spike_source.is_always_on_))
        
        return torch.stack(signals).transpose(0, 1)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)

        buffers = []
        for core in range(len(self.cores)):
            buffers.append(torch.zeros(x.shape[0], self.output_count))
        
        chip_delay = len(self.cores)
        for _ in range(chip_delay):
            for core in range(len(self.cores)):
                input_signals = self.get_signal_tensor(
                    x, buffers, self.cores[core].axon_sources)

                buffers[core] = torch.matmul(
                    self.core_params[core], input_signals.T).T
                buffers[core] = nn.ReLU()(buffers[core])
        
        output_signals = self.get_signal_tensor(x, buffers, self.output_sources)
        return output_signals

        



        





