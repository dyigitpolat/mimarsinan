class PipelineStep:
    def __init__(self, requires, promises, updates, clears, pipeline):
        self.name = self.__class__.__name__
        self.requires = requires
        self.promises = promises
        self.updates = updates
        self.clears = clears
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
        """Definitive metric for pipeline progression — used by ``Pipeline``
        to set ``__target_metric`` after each step.

        Test-set isolation rule: this is the **one** place in the pipeline
        that is allowed to call ``trainer.test()``. Tuner internals must
        NEVER call ``test()`` (any validation / rollback / safety-net
        decision must use ``validate()`` or ``validate_n_batches``).

        Resolution order:
        1. If the step exposes a ``tuner`` with a live ``trainer``, call
           ``trainer.test()`` on the tuner's trainer.
        2. Else if the step exposes its own ``trainer``, call
           ``trainer.test()``.
        3. Else fall back to ``self.validate()`` (tuners that don't own a
           PyTorch model report their own final metric via ``validate()``).
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
        """Release resources acquired during process() (e.g. DataLoader workers).

        Called by the pipeline after validate(), in a finally block so it runs
        even if later pipeline logic fails.

        Auto-discovers tuners (via ``self.tuner``) and trainers (via
        ``self.trainer``) and closes their DataLoader workers to prevent
        multiprocessing cleanup errors at process exit.
        Subclasses may override for additional cleanup.
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

