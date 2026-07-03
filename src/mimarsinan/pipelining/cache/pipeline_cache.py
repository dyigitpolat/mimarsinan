from mimarsinan.pipelining.cache.load_store_strategies import *

import json
import os

from typing import Any

class PipelineCache:
    LOAD_STORE_STRATEGIES = {
        "basic": BasicLoadStoreStrategy,
        "torch_model": TorchModelLoadStoreStrategy,
        "pickle": PickleLoadStoreStrategy
    }

    def __init__(self):
        self.cache = {}
        # Entries modified since the last store(); only these are re-serialized.
        self._dirty: set[str] = set()

    def add(self, name, object, load_store_strategy = "basic"):
        self.cache[name] = (object, load_store_strategy)
        self._dirty.add(name)

    @staticmethod
    def _filename_for(name):
        # Keys may contain '/'; on-disk filenames must stay flat.
        return name.replace("/", "%2F")

    def get(self, name) -> Any:
        if name not in self.cache:
            return None

        return self.cache[name][0]

    def remove(self, name):
        if name in self.cache:
            del self.cache[name]
        self._dirty.discard(name)

    def store(self, cache_directory):
        if not os.path.exists(cache_directory):
            os.makedirs(cache_directory)

        if os.path.exists(f"{cache_directory}/metadata.json"):
            with open(f"{cache_directory}/metadata.json", "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        for name, (_, load_store_strategy) in self.cache.items():
            metadata[name] = (load_store_strategy, self._filename_for(name))

        for name in list(self._dirty):
            if name not in self.cache:
                continue
            object, load_store_strategy = self.cache[name]
            strategy = self.LOAD_STORE_STRATEGIES[load_store_strategy](self._filename_for(name))
            strategy.store(cache_directory, object)

        with open(f"{cache_directory}/metadata.json", "w") as f:
            json.dump(metadata, f)

        self._dirty.clear()

    def load(self, cache_directory):
        self.cache = {}
        self._dirty.clear()

        if os.path.exists(f"{cache_directory}/metadata.json"):
            with open(f"{cache_directory}/metadata.json", "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        for name, (load_store_strategy, _) in metadata.items():
            strategy = self.LOAD_STORE_STRATEGIES[load_store_strategy](self._filename_for(name))
            self.cache[name] = (strategy.load(cache_directory), load_store_strategy)

    def keys(self):
        return self.cache.keys()

    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, name) -> Any:
        return self.get(name)
    
    def __setitem__(self, name, object):
        existing = self.cache.get(name)
        strategy = existing[1] if existing is not None else "basic"
        self.add(name, object, strategy)
    
    def __delitem__(self, name):
        self.remove(name)
    
    def __contains__(self, name):
        return name in self.cache
    
    def offload_torch_models_to_cpu(self):
        """Move all cached torch models to CPU to free GPU memory."""
        for name, (obj, strategy) in self.cache.items():
            if strategy == "torch_model" and hasattr(obj, 'cpu'):
                obj.cpu()

    def __iter__(self):
        return iter(self.cache)
