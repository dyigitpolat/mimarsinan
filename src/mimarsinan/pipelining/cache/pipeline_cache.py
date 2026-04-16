from mimarsinan.pipelining.cache.load_store_strategies import *

import json
import os

class PipelineCache:
    LOAD_STORE_STRATEGIES = {
        "basic": BasicLoadStoreStrategy,
        "torch_model": TorchModelLoadStoreStrategy,
        "pickle": PickleLoadStoreStrategy
    }

    def __init__(self):
        self.cache = {}
        # Track which entries have been added/updated since the last successful
        # ``store(...)``. Only dirty entries get re-written, which avoids
        # redundantly re-pickling large upstream ``.pt`` files (20+ GiB each
        # in the deployment pipeline) at every subsequent step boundary.
        self._dirty: set[str] = set()

    def add(self, name, object, load_store_strategy = "basic"):
        self.cache[name] = (object, load_store_strategy)
        self._dirty.add(name)

    def get(self, name):
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
            filename = f"{name}"
            metadata[name] = (load_store_strategy, filename)

        # Only re-serialize entries modified since the last store(). Loaded-
        # from-disk entries stay on disk untouched.
        for name in list(self._dirty):
            if name not in self.cache:
                continue
            object, load_store_strategy = self.cache[name]
            filename = f"{name}"
            strategy = self.LOAD_STORE_STRATEGIES[load_store_strategy](filename)
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
    
    def offload_torch_models_to_cpu(self):
        """Move all cached torch models to CPU to free GPU memory."""
        for name, (obj, strategy) in self.cache.items():
            if strategy == "torch_model" and hasattr(obj, 'cpu'):
                obj.cpu()

    def __iter__(self):
        return iter(self.cache)
