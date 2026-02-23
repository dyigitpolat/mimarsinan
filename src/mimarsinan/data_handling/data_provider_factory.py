from mimarsinan.data_handling.data_provider import DataProvider

class DataProviderFactory:
    def create(self) -> DataProvider:
        raise NotImplementedError

class BasicDataProviderFactory(DataProviderFactory):
    _provider_registry = {}

    def __init__(self, name: str, datasets_path: str, *, seed: int | None = 0, cache: bool = True):
        self._name = name
        self._datasets_path = datasets_path
        self._seed = seed
        self._cache = bool(cache)
        self._cached_provider: DataProvider | None = None
        
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

        # Many code paths (trainers/evaluators/pipeline config) call create() repeatedly.
        # For NAS, it's critical that train/val splits remain stable across evaluations.
        if self._cache and self._cached_provider is not None:
            return self._cached_provider

        provider_cls = self._provider_registry[self._name]
        try:
            provider = provider_cls(self._datasets_path, seed=self._seed)
        except TypeError:
            # Backward compatibility: allow providers that don't yet accept seed.
            provider = provider_cls(self._datasets_path)

        if self._cache:
            self._cached_provider = provider

        return provider
