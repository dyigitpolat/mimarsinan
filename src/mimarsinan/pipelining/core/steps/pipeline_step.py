class PipelineStep:
    REQUIRES: tuple[str, ...] = ()
    PROMISES: tuple[str, ...] = ()
    UPDATES: tuple[str, ...] = ()
    CLEARS: tuple[str, ...] = ()

    @classmethod
    def applies_to(cls, plan) -> bool:
        """Whether this step belongs in the pipeline for the resolved ``plan`` (base: always)."""
        return True

    @classmethod
    def declared_contract(cls) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Class-level data contract ``(requires, promises, updates, clears)`` for assembly-time DAG checks.

        The instance contract may extend this for opt-in cases; the DAG check uses
        the always-present static contract as the conservative lower bound.
        """
        return (cls.REQUIRES, cls.PROMISES, cls.UPDATES, cls.CLEARS)

    def __init__(self, requires, promises, updates, clears, pipeline):
        self.name = self.__class__.__name__
        self.requires = list(requires)
        self.promises = list(promises)
        self.updates = list(updates)
        self.clears = list(clears)
        self.pipeline = pipeline

        self._accessed_entries = set()

    def run(self):
        self._accessed_entries = set()
        self._updated_entries = set()
        self.process()
        assert all([requirement in self._accessed_entries for requirement in self.requires]), \
            f"Pipeline error: Some required entries ({set(self.requires) - self._accessed_entries}) were not accessed."
        assert all([entry in self._updated_entries for entry in self.updates]), \
            f"Pipeline error: Some to-be-updated entries ({set(self.updates) - self._updated_entries}) were not updated."
        
    def process(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def pipeline_metric(self):
        """Definitive metric for pipeline progression; the ONLY place allowed to call ``trainer.test()``.

        Resolves to the tuner's trainer, else the step's own trainer, else ``self.validate()``.
        """
        tuner = getattr(self, "tuner", None)
        if tuner is not None:
            trainer = getattr(tuner, "trainer", None)
            if trainer is not None and hasattr(trainer, "test"):
                return trainer.test()
        trainer = getattr(self, "trainer", None)
        if trainer is not None and hasattr(trainer, "test"):
            return trainer.test()
        return self.validate()

    def cleanup(self):
        """Release resources acquired during process() (closes tuner/trainer DataLoader workers).

        Called by the pipeline in a finally block; subclasses may override.
        """
        tuner = getattr(self, "tuner", None)
        if tuner is not None and hasattr(tuner, "close"):
            tuner.close()
            return
        trainer = getattr(self, "trainer", None)
        if trainer is not None and hasattr(trainer, "close"):
            trainer.close()

    def get_entry(self, key):
        assert key in self.requires, f"A non-required entry ({key}) cannot be retrieved from the cache."
        self._accessed_entries.add(key)
        return self.pipeline.get_entry(self, key)
    
    def add_entry(self, key, object, load_store_strategy = "basic"):
        assert key in self.promises, f"A non-promised entry ({key}) cannot be added to the cache."
        self.pipeline.add_entry(self, key, object, load_store_strategy)

    def update_entry(self, key, object, load_store_strategy = "basic"):
        assert key in self.updates, f"The \"{key}\" could not be updated in the cache, update contract missing."
        self._updated_entries.add(key)
        self.pipeline.update_entry(self, key, object, load_store_strategy)

