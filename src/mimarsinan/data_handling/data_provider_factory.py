from mimarsinan.data_handling.data_provider import DataProvider

class DataProviderFactory:
    def create(self) -> DataProvider:
        raise NotImplementedError

class BasicDataProviderFactory(DataProviderFactory):
    _provider_registry = {}

    def __init__(self, name: str, datasets_path: str):
        self._name = name
        self._datasets_path = datasets_path
        
        if self._name not in self._provider_registry:
             raise ValueError(f"Data provider '{self._name}' not registered. Available providers: {list(self._provider_registry.keys())}")

    @classmethod
    def register(cls, name: str):
        def decorator(provider_cls):
            cls._provider_registry[name] = provider_cls
            return provider_cls
        return decorator

    def create(self) -> DataProvider:
        if self._name not in self._provider_registry:
            raise ValueError(f"Data provider '{self._name}' not registered.")
        return self._provider_registry[self._name](self._datasets_path)
