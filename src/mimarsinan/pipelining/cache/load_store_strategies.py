import torch
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
        with open(f"{cache_directory}/{self.filename}", "r") as f:
            return pickle.load(f)

    def store(self, cache_directory, object):
        with open(f"{cache_directory}/{self.filename}", "w") as f:
            pickle.dump(object, f)

class TorchModelLoadStoreStrategy(LoadStoreStrategy):
    def __init__(self, filename):
        super().__init__(filename)

    def load(self, cache_directory):
        return torch.load(f"{cache_directory}/{self.filename}")

    def store(self, cache_directory, object):
        torch.save(object, f"{cache_directory}/{self.filename}")