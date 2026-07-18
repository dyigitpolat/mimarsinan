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

        def _write(path):
            with open(path, "w") as f:
                json.dump(metadata, f)

        write_atomically(f"{cache_directory}/metadata.json", _write)

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
            try:
                loaded = strategy.load(cache_directory)
            except ENTRY_LOAD_FAILURES as exc:
                raise CorruptCacheEntryError(
                    name, cache_directory, load_store_strategy, exc,
                ) from exc
            self.cache[name] = (loaded, load_store_strategy)

    @classmethod
    def quarantine_entry(cls, cache_directory, name):
        """Move a corrupt entry's payload to '<payload>.corrupt' and drop it from
        metadata so the producing step re-runs from its predecessor; returns the
        quarantined path (None when no payload file existed)."""
        meta_path = f"{cache_directory}/metadata.json"
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        if name not in metadata:
            raise KeyError(f"cache entry {name!r} not present in {meta_path}")
        strategy_name, filename = metadata.pop(name)
        payload = cls.LOAD_STORE_STRATEGIES[strategy_name](filename).path(cache_directory)
        quarantined = None
        if os.path.exists(payload):
            quarantined = f"{payload}.corrupt"
            os.replace(payload, quarantined)

        def _write(path):
            with open(path, "w") as f:
                json.dump(metadata, f)

        write_atomically(meta_path, _write)
        return quarantined

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
