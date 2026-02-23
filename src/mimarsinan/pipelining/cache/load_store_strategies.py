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
        return object.to(device)

    def store(self, cache_directory, object):
        # assert hasattr(object, "device"), \
        #     "only models with 'device' attribute can be stored"
        
        device = object.device
        object.cpu()
        torch.save((object, device), f"{cache_directory}/{self.filename}.pt")

class PickleLoadStoreStrategy(LoadStoreStrategy):
    def __init__(self, filename):
        super().__init__(filename)

    def load(self, cache_directory):
        with open(f"{cache_directory}/{self.filename}.pickle", "rb") as f:
            return pickle.load(f)
        
    def store(self, cache_directory, object):
        with open(f"{cache_directory}/{self.filename}.pickle", "wb") as f:
            pickle.dump(object, f)
