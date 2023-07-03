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

        if os.path.exists(f"{cache_directory}/metadata.json"):
            with open(f"{cache_directory}/metadata.json", "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        for name, (_, load_store_strategy) in self.cache.items():
            filename = f"{name}"
            metadata[name] = (load_store_strategy, filename)
    
        with open(f"{cache_directory}/metadata.json", "w") as f:
            json.dump(metadata, f)
        
        for name, (object, load_store_strategy) in self.cache.items():
            filename = f"{name}"
            strategy = self.LOAD_STORE_STRATEGIES[load_store_strategy](filename)
            strategy.store(cache_directory, object)
    
    def load(self, cache_directory):
        self.cache = {}

        if os.path.exists(f"{cache_directory}/metadata.json"):
            with open(f"{cache_directory}/metadata.json", "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        for name, (load_store_strategy, filename) in metadata.items():
            filename = f"{name}"
            strategy = self.LOAD_STORE_STRATEGIES[load_store_strategy](filename)
            self.cache[name] = (strategy.load(cache_directory), load_store_strategy)

    def keys(self):
        return self.cache.keys()

    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, name):
        return self.get(name)
    
    def __setitem__(self, name, object):
        self.add(name, object)
    
    def __delitem__(self, name):
        self.remove(name)
    
    def __contains__(self, name):
        return name in self.cache
    
    def __iter__(self):
        return iter(self.cache)
