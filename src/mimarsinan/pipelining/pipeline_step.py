class PipelineStep:
    # Phase B3 skip-list: subclasses that do not produce a meaningful test
    # metric (pure configuration / model-building / mapping steps whose
    # ``validate()`` just forwards the running target) should set
    # ``skip_from_floor_check = True``.  The pipeline engine then skips
    # both floor assertions for that step AND leaves ``previous_metric``
    # untouched so the next real step is still compared to the last
    # observed baseline instead of a bogus 0.0.
    skip_from_floor_check: bool = False

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
        """Definitive metric for pipeline progression.

        Called by ``Pipeline._run_step`` to set ``__target_metric`` and to
        enforce the step-level hard-floor assertion. This is the ONLY place
        in the framework allowed to hit the test set -- tuners run entirely
        on validation. Running ``trainer.test()`` here, after the tuner has
        already finalised the model, keeps the test labels out of any
        training-time decision (rate commits, rollback gates, LR probes).

        Resolution order:
        1. A tuner attached to the step is preferred -- its ``trainer``
           already has loaders warm. We run its ``trainer.test()`` here.
        2. The step's own ``trainer`` (for steps that don't use a tuner).
        3. The step's ``validate()`` (for pass-through / mapping steps
           that never touch a trainer -- these steps must ensure
           ``validate()`` returns a metric on the same scale as
           ``trainer.test()`` or simply defers to ``pipeline.get_target_metric()``).
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

