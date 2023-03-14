class ChipDelay:
    def __init__(self, mapping):
        self.mapping = mapping
        self.memo = {}

    def get_delay_for(self, source):
        if source.core_ in self.memo: return self.memo[source.core_]
        if source.is_input_: 
            self.memo[source.core_] = 0
            return 0
        if source.is_off_: 
            self.memo[source.core_] = 0
            return 0
        if source.is_always_on_: 
            self.memo[source.core_] = 0
            return 0

        axon_sources = self.mapping.cores[source.core_].axon_sources

        result = 1 + max([
            self.get_delay_for(source) for source in axon_sources])

        self.memo[source.core_] = result
        return result
        
    def calculate(self):
        return max([
            self.get_delay_for(source) for source in self.mapping.output_sources])