class PipelineStep:
    def __init__(self, requires, promises, clears, pipeline):
        self.requires = requires
        self.promises = promises
        self.clears = clears
        self.pipeline = pipeline

    def run(self):
        assert all([requirement in self.pipeline.cache for requirement in self.requires])
        self.process()
        for entry in self.clears:
            self.pipeline.cache.remove(entry)
        assert all([promise in self.pipeline.cache for promise in self.promises])
        assert all([entry not in self.pipeline.cache for entry in self.clears])

    def process(self):
        raise NotImplementedError
    
    def validate(self):
        raise NotImplementedError
    
    def get_entry(self, key):
        assert key in self.requires, f"You cannot retrieve a non-required entry ({key}) from the cache."
        return self.pipeline.cache[key]
    
    def add_entry(self, key, object, load_store_strategy = "basic"):
        assert key in self.promises, f"You cannot add a non-promised entry ({key}) to the cache."
        self.pipeline.cache.add(key, object, load_store_strategy)

