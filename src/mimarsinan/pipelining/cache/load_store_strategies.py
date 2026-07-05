import torch
import json
import logging
import pickle

from mimarsinan.transformations.pruning.committed_masks import (
    commit_model_pruning,
    verify_model_pruning,
)

logger = logging.getLogger(__name__)

class LoadStoreStrategy:
    def __init__(self, filename):
        self.filename = filename

    def load(self, cache_directory):
        raise NotImplementedError

    def store(self, cache_directory, object):
        raise NotImplementedError
    
class BasicLoadStoreStrategy(LoadStoreStrategy):
    def __init__(self, filename):
        super().__init__(filename)

    def load(self, cache_directory):
        with open(f"{cache_directory}/{self.filename}.json", "r") as f:
            return json.load(f)

    def store(self, cache_directory, object):
        with open(f"{cache_directory}/{self.filename}.json", "w") as f:
            json.dump(object, f)

class TorchModelLoadStoreStrategy(LoadStoreStrategy):
    def __init__(self, filename):
        super().__init__(filename)

    def load(self, cache_directory):
        (object, device) = torch.load(f"{cache_directory}/{self.filename}.pt", map_location=torch.device('cpu'), weights_only=False)
        object._cached_original_device = device
        if isinstance(object, torch.nn.Module):
            # Round-trip half of the prune-parity contract: an artifact whose raw
            # params violate its committed masks must fail loud, never silently
            # reproduce different metrics than the live run (theory §5g-v (i)).
            verify_model_pruning(object, where=f"cache-load:{self.filename}")
        return object  # stay on CPU; consumers move to device as needed

    def store(self, cache_directory, object):
        if hasattr(object, "device"):
            device = object.device
        else:
            p = next(object.parameters(), None)
            device = p.device if p is not None else torch.device("cpu")

        if isinstance(object, torch.nn.Module):
            # The store boundary is an enforcement point like any hooked forward:
            # masks must hold in committed raw params before the artifact is written.
            commit_model_pruning(object)
            verify_model_pruning(object, where=f"cache-store:{self.filename}")

        object.cpu()
        torch.save((object, device), f"{cache_directory}/{self.filename}.pt")
        # If the recorded device is no longer visible (narrower CUDA_VISIBLE_DEVICES) fall back to CPU rather than crash mid-save.
        try:
            object.to(device)
        except RuntimeError:
            logger.warning(
                "could not restore %s to %s after save; leaving on CPU",
                self.filename, device, exc_info=True,
            )

class PickleLoadStoreStrategy(LoadStoreStrategy):
    def __init__(self, filename):
        super().__init__(filename)

    def load(self, cache_directory):
        with open(f"{cache_directory}/{self.filename}.pickle", "rb") as f:
            return pickle.load(f)
        
    def store(self, cache_directory, object):
        with open(f"{cache_directory}/{self.filename}.pickle", "wb") as f:
            pickle.dump(object, f)
