from mimarsinan.data_handling.data_provider import DataProvider

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

        # Many code paths (trainers/evaluators/pipeline config) call create() repeatedly.
        # For NAS, it's critical that train/val splits remain stable across evaluations.
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
        """Return list of registered provider ids, labels, and static capabilities (for GUI).

        ``input_shape`` / ``num_classes`` are NOT included here because they
        require instantiating the provider, which may load a non-trivial
        dataset from disk.  Use :meth:`get_metadata` per-provider for that.
        """
        import mimarsinan.data_handling.data_providers  # noqa: F401 - populate registry
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

        Returns ``{"id", "label", "input_shape", "num_classes",
        "supports_preprocessing"}``.  Raises :class:`ValueError` for unknown
        providers; instantiation errors (e.g. ImageNet root missing) propagate
        so the caller can report them.
        """
        import mimarsinan.data_handling.data_providers  # noqa: F401 - populate registry
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
