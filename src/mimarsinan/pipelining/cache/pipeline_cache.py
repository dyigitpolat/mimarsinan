from mimarsinan.pipelining.cache.load_store_strategies import *

import json
import os

class PipelineCache:
    LOAD_STORE_STRATEGIES = {
        "basic": BasicLoadStoreStrategy,
        "torch_model": TorchModelLoadStoreStrategy
    }

    def __init__(self):
        self.cache = {}

    def add(self, name, object, load_store_strategy = "basic"):
        self.cache[name] = (object, load_store_strategy)
    
    def get(self, name):
        if name not in self.cache:
            return None
        
        return self.cache[name][0]
    
    def remove(self, name):
        if name in self.cache:
            del self.cache[name]

    def store(self, cache_directory):
        if not os.path.exists(cache_directory):
            os.makedirs(cache_directory)

        metadata = {}
        for name, (_, load_store_strategy) in self.cache.items():
            filename = f"{name}.json"
            metadata[name] = (load_store_strategy, filename)
    
        with open(f"{cache_directory}/metadata.json", "w") as f:
            json.dump(metadata, f)
        
        for name, (object, load_store_strategy) in self.cache.items():
            strategy = self.LOAD_STORE_STRATEGIES[load_store_strategy](filename)
            strategy.store(cache_directory, object)
    
    def load(self, cache_directory):
        self.cache = {}

        with open(f"{cache_directory}/metadata.json", "r") as f:
            metadata = json.load(f)
        
        for name, (load_store_strategy, filename) in metadata.items():
            strategy = self.LOAD_STORE_STRATEGIES[load_store_strategy](filename)
            self.cache[name] = (strategy.load(cache_directory), load_store_strategy)