class PipelineStep:
    def __init__(self, requires, promises, clears, pipeline):
        self.requires = requires
        self.promises = promises
        self.clears = clears
        self.pipeline = pipeline

        self._accessed_entries = set()

    def run(self):
        assert all([requirement in self.pipeline.cache for requirement in self.requires]), \
            f"Pipeline error: Some requirements are not found in the cache."

        self._accessed_entries = set()
        self.process()
        for entry in self.clears:
            self.pipeline.cache.remove(entry)

        assert all([requirement in self._accessed_entries for requirement in self.requires]), \
            f"Pipeline error: Some required entries ({set(self.requires) - self._accessed_entries}) were not accessed."
        assert all([promise in self.pipeline.cache for promise in self.promises]), \
            f"Pipeline error: Some promised entries were not added."
        assert all([entry not in self.pipeline.cache for entry in self.clears]), \
            f"Pipeline error: Some cleared entries were not removed."

    def process(self):
        raise NotImplementedError
    
    def validate(self):
        raise NotImplementedError
    
    def get_entry(self, key):
        assert key in self.requires, f"A non-required entry ({key}) cannot be retrieved from the cache."
        self._accessed_entries.add(key)
        return self.pipeline.cache[key]
    
    def add_entry(self, key, object, load_store_strategy = "basic"):
        assert key in self.promises, f"A non-promised entry ({key}) cannot be added to the cache."
        self.pipeline.cache.add(key, object, load_store_strategy)

