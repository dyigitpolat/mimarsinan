import torch
import json
import pickle

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
        return object  # stay on CPU; consumers move to device as needed

    def store(self, cache_directory, object):
        if hasattr(object, "device"):
            device = object.device
        else:
            # Native nn.Module: infer device from first parameter.
            p = next(object.parameters(), None)
            device = p.device if p is not None else torch.device("cpu")

        object.cpu()
        torch.save((object, device), f"{cache_directory}/{self.filename}.pt")
        # Restore original placement. If the originally-recorded device is no
        # longer visible (e.g. the process was launched with a narrower
        # CUDA_VISIBLE_DEVICES than when the object was first loaded), fall
        # back to CPU so cache persistence does not crash mid-save.
        try:
            object.to(device)
        except Exception:
            pass

class PickleLoadStoreStrategy(LoadStoreStrategy):
    def __init__(self, filename):
        super().__init__(filename)

    def load(self, cache_directory):
        with open(f"{cache_directory}/{self.filename}.pickle", "rb") as f:
            return pickle.load(f)
        
    def store(self, cache_directory, object):
        with open(f"{cache_directory}/{self.filename}.pickle", "wb") as f:
            pickle.dump(object, f)
