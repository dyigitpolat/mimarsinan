class ChipDelay:
    def __init__(self, mapping):
        self.mapping = mapping
        self.memo = {}
        self.visited = set()

    def get_delay_for(self, source_core_idx, depth=0):
        if source_core_idx in self.memo: return self.memo[source_core_idx]
        if source_core_idx in self.visited: return 0
        
        if source_core_idx < 0:
            self.memo[source_core_idx] = 0
            return 0
        
        assert source_core_idx >= 0
        axon_sources = self.mapping.cores[source_core_idx].axon_sources

        source_cores = set()
        for source in axon_sources:
            source_cores.add(source.core_)
        
        self.visited.add(source_core_idx)
        result = 1 + max([
            self.get_delay_for(core, depth+1) for core in source_cores])
        self.visited.remove(source_core_idx)

        self.memo[source_core_idx] = result
        return result
        
    def calculate(self):
        self.memo = {}
        self.visited = set()
        print("Calculating chip delay...")
        print("Total cores:", len(self.mapping.cores))
        return max([
            self.get_delay_for(source.core_) for source in self.mapping.output_sources])