class PipelineStep:
    def __init__(self, requires, promises, updates, clears, pipeline):
        self.name = self.__class__.__name__
        self.requires = requires
        self.promises = promises
        self.updates = updates
        self.clears = clears
        self.pipeline = pipeline

        self._accessed_entries = set()

    def run(self):
        self._accessed_entries = set()
        self._updated_entries = set()
        self.process()
        assert all([requirement in self._accessed_entries for requirement in self.requires]), \
            f"Pipeline error: Some required entries ({set(self.requires) - self._accessed_entries}) were not accessed."
        assert all([entry in self._updated_entries for entry in self.updates]), \
            f"Pipeline error: Some to-be-updated entries ({set(self.updates) - self._updated_entries}) were not updated."
        
    def process(self):
        raise NotImplementedError
    
    def validate(self):
        raise NotImplementedError
    
    def get_entry(self, key):
        assert key in self.requires, f"A non-required entry ({key}) cannot be retrieved from the cache."
        self._accessed_entries.add(key)
        return self.pipeline.get_entry(self, key)
    
    def add_entry(self, key, object, load_store_strategy = "basic"):
        assert key in self.promises, f"A non-promised entry ({key}) cannot be added to the cache."
        self.pipeline.add_entry(self, key, object, load_store_strategy)

    def update_entry(self, key, object, load_store_strategy = "basic"):
        assert key in self.updates, f"The \"{key}\" could not be updated in the cache, update contract missing."
        self._updated_entries.add(key)
        self.pipeline.update_entry(self, key, object, load_store_strategy)

