class ChipDelay:
    def __init__(self, mapping):
        self.mapping = mapping
        self.memo = {}

    def __get_non_zero_axon_sources(self, core, neuron_idx):
        non_zero_axon_sources = []
        for axon_idx, w in enumerate(core.core_matrix[:, neuron_idx]):
            if abs(w) > 0:
                non_zero_axon_sources.append(core.axon_sources[axon_idx])
        
        return non_zero_axon_sources

    def __is_direct_signal(self, source):
        return source.core_ < 0

    def get_delay_for(self, source):
        source_core_idx = source.core_

        if source_core_idx in self.memo: return self.memo[source_core_idx]
        
        if self.__is_direct_signal(source):
            self.memo[source_core_idx] = 0
            return 0
        
        current_core = self.mapping.cores[source.core_]
        non_zero_axon_sources = self.__get_non_zero_axon_sources(
            current_core, source.neuron_)

        result = 1 + max([
            self.get_delay_for(source) for source in non_zero_axon_sources])

        self.memo[source_core_idx] = result
        return result
        
    def calculate(self):
        self.memo = {}
        return max([
            self.get_delay_for(source) for source in self.mapping.output_sources])