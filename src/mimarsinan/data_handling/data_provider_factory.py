import importlib

from mimarsinan.data_handling.data_provider import DataProvider


def _ensure_providers_registered() -> None:
    """Import the providers package so its @register decorators populate the registry."""
    importlib.import_module("mimarsinan.data_handling.data_providers")


class DataProviderFactory:
    def create(self) -> DataProvider:
        raise NotImplementedError

class BasicDataProviderFactory(DataProviderFactory):
    _provider_registry = {}

    def __init__(self, name: str, datasets_path: str, *, seed: int | None = 0, cache: bool = True, preprocessing=None, batch_size=None):
        self._name = name
        self._datasets_path = datasets_path
        self._seed = seed
        self._cache = bool(cache)
        self._preprocessing = preprocessing
        self._batch_size = batch_size
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

        # Cache so train/val splits stay stable across repeated create() calls (NAS determinism).
        if self._cache and self._cached_provider is not None:
            return self._cached_provider

        provider_cls = self._provider_registry[self._name]
        try:
            provider = provider_cls(
                self._datasets_path,
                seed=self._seed,
                preprocessing=self._preprocessing,
                batch_size=self._batch_size,
            )
        except TypeError:
            try:
                provider = provider_cls(
                    self._datasets_path,
                    seed=self._seed,
                    preprocessing=self._preprocessing,
                )
            except TypeError:
                try:
                    provider = provider_cls(self._datasets_path, seed=self._seed)
                except TypeError:
                    provider = provider_cls(self._datasets_path)

        if self._cache:
            self._cached_provider = provider

        return provider

    @classmethod
    def list_registered(cls) -> list[dict]:
        """Registered provider ids, labels, and static capabilities (for GUI).

        Excludes ``input_shape`` / ``num_classes`` (those need instantiation;
        use :meth:`get_metadata`).
        """
        _ensure_providers_registered()
        result = []
        for name in sorted(cls._provider_registry.keys()):
            provider_cls = cls._provider_registry[name]
            label = getattr(provider_cls, "DISPLAY_LABEL", None)
            if label is None:
                label = name.replace("_DataProvider", "").replace("_", " ")
            result.append({
                "id": name,
                "label": label,
                "supports_preprocessing": bool(getattr(provider_cls, "SUPPORTS_PREPROCESSING", True)),
            })
        return result

    @classmethod
    def get_metadata(
        cls,
        name: str,
        datasets_path: str = "./datasets",
        *,
        preprocessing=None,
    ) -> dict:
        """Instantiate ``name`` with ``preprocessing`` and report its shape / classes.

        Returns id/label/input_shape/num_classes/supports_preprocessing;
        unknown providers raise ValueError and instantiation errors propagate.
        """
        _ensure_providers_registered()
        if name not in cls._provider_registry:
            raise ValueError(f"Data provider {name!r} not registered.")
        provider_cls = cls._provider_registry[name]

        factory = cls(
            name, datasets_path,
            seed=0, cache=False, preprocessing=preprocessing,
        )
        provider = factory.create()
        shape = tuple(int(d) for d in provider.get_input_shape())

        label = getattr(provider_cls, "DISPLAY_LABEL", None)
        if label is None:
            label = name.replace("_DataProvider", "").replace("_", " ")

        return {
            "id": name,
            "label": label,
            "input_shape": list(shape),
            "num_classes": int(provider.get_output_shape()),
            "supports_preprocessing": bool(getattr(provider_cls, "SUPPORTS_PREPROCESSING", True)),
        }
