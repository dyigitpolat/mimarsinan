import gc
import sys
import time as _time

import torch

from mimarsinan.tuning.orchestration import run_ledger
from mimarsinan.pipelining.core.accuracy_budget import (
    AccuracyBudget,
    PretrainEnvelopeError,
)
from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache
from mimarsinan.common.file_utils import prepare_containing_directory
from mimarsinan.common.diagnostics import cuda_guard, phase_profiler
from mimarsinan.pipelining.core.engine.pipeline_resource_debug import log_resource_snapshot
from mimarsinan.pipelining.core.engine import pipeline_resume
from mimarsinan.pipelining.core.engine.step_instrumentation import (
    assert_metric_retention,
    emit_step_profile,
)


class Pipeline:
    def __init__(self, working_directory) -> None:
        self.steps = []
        self.cache = PipelineCache()
        self.key_translations = {}

        self.working_directory = working_directory
        prepare_containing_directory(self.working_directory)

        self.tolerance = 0.95
        self.step_tolerances = {}

        self.load_cache()
        if '__target_metric' not in self.cache.keys():
            self.set_target_metric(0.0)

        self.accuracy_budget = AccuracyBudget(budget_total=0.0)

        self.post_step_hooks: list = []
        self.pre_step_hooks: list = []

    def add_pipeline_step(self, name, pipeline_step):
        pipeline_step.name = name
        self.steps.append((name, pipeline_step))
        self.verify()

    def set_up_requirements(self):
        latest_promises = {}
        for _, step in self.steps:
            for promise in step.promises:
                latest_promises[promise] = step.name

            self.key_translations[step.name] = {}
            for requirement in step.requires:
                self.key_translations[step.name][requirement] = self._create_real_key(latest_promises[requirement], requirement)

            for entry in step.updates:
                latest_promises[entry] = step.name

    def verify(self):
        mock_cache = set()
        for name, step in self.steps:
            for requirement in step.requires:
                assert requirement in mock_cache, f"Pipeline step '{name}' requires '{requirement}'"
            
            for promise in step.promises:
                mock_cache.add(promise)

            for entry in step.clears:
                if entry in mock_cache:
                    mock_cache.remove(entry)

    def _reset_run_scoped_ledgers(self):
        """A FRESH full run must not inherit a previous attempt's run-scoped
        state from a reused cache directory (a stale exhausted endpoint-step
        ledger silently demotes armed floors to the patience geometry);
        ``run_from`` keeps them — resumption continues the same run."""
        run_ledger.reset(self.cache)

    def run(self, *, stop_step: str | None = None):
        self.set_up_requirements()
        self.verify()
        self._reset_run_scoped_ledgers()
        for name, step in self.steps:
            self._run_step(name, step)
            if stop_step is not None and name == stop_step:
                break

    def get_resolved_start_step(self, step_name: str) -> str:
        """Return the step name the pipeline would actually start from (e.g. when dependencies are missing)."""
        assert step_name in [name for name, _ in self.steps], f"Step '{step_name}' does not exist in pipeline"
        self.set_up_requirements()
        self.verify()
        starting_step_idx = pipeline_resume.find_starting_step_idx(self, step_name)
        return self.steps[starting_step_idx][0]

    def run_from(self, step_name, *, stop_step: str | None = None):
        assert step_name in [name for name, _ in self.steps], f"Step '{step_name}' does not exist in pipeline"
        
        self.set_up_requirements()
        self.verify()
        starting_step_idx = pipeline_resume.find_starting_step_idx(self, step_name)
        
        for idx, (name, step) in enumerate(self.steps):
            if idx < starting_step_idx:
                continue
            self._run_step(name, step)
            if stop_step is not None and name == stop_step:
                break

    def save_cache(self):
        self.cache.store(self.working_directory)

    def load_cache(self):
        self.cache.load(self.working_directory)

    def get_entry(self, client_step, key):
        real_key = self._translate_key(client_step.name, key)
        return self.cache.get(real_key)
    
    def add_entry(self, client_step, key, object, load_store_strategy = "basic"):
        real_key = self._create_real_key(client_step.name, key)
        self.cache.add(real_key, object, load_store_strategy)

    def update_entry(self, client_step, key, object, load_store_strategy = "basic"):
        old_real_key = self._translate_key(client_step.name, key)
        new_real_key = self._create_real_key(client_step.name, key)

        self.cache.remove(old_real_key)
        self.cache.add(new_real_key, object, load_store_strategy)

    def remove_entry(self, client_step, key):
        real_key = self._translate_key(client_step.name, key)
        self.cache.remove(real_key)

    def get_target_metric(self):
        return self.cache['__target_metric']

    def set_target_metric(self, target_metric):
        self.cache.add('__target_metric', target_metric)

    def _step_tolerance(self, step_name: str) -> float:
        """Retention factor for one step: per-step override or the global one."""
        return float(self.step_tolerances.get(step_name, self.tolerance))

    def register_post_step_hook(self, hook):
        self.post_step_hooks.append(hook)

    def register_pre_step_hook(self, hook):
        self.pre_step_hooks.append(hook)

    def _release_gpu_memory(self):
        """Release unreferenced GPU memory between pipeline steps."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _create_real_key(self, client_step_name, key):
        return client_step_name + '.' + key

    def _translate_key(self, client_step_name, key):
        return self.key_translations[client_step_name][key]

    def _run_step(self, name, step):
        _t0_step = _time.time()
        print(f"Running '{name}'...")
        log_resource_snapshot(f"pre:{name}")

        for hook in self.pre_step_hooks:
            hook(name, step)

        previous_metric = self.get_target_metric()
        self._assert_requirements_present(step)
        step.pipeline_previous_metric = previous_metric

        try:
            self._execute_step(name, step)
            self._record_step_metric(step)
            self._persist_step_outputs(name, step)
            self._assert_step_contract(step)
            self._assert_metric_retention(step, previous_metric)

            for hook in self.post_step_hooks:
                hook(name, step)
        finally:
            step.cleanup()
            self._release_gpu_memory()
            log_resource_snapshot(f"post:{name}")
            emit_step_profile(
                self, name, step,
                wall_s=_time.time() - _t0_step,
                final_metric=self.get_target_metric(),
                previous_metric=previous_metric,
            )

    def _assert_requirements_present(self, step):
        assert all(
            self._translate_key(step.name, requirement) in self.cache
            for requirement in step.requires
        ), "Pipeline error: Some requirements are not found in the cache."

    def _execute_step(self, name, step):
        cuda_debug = bool(getattr(self, "cuda_debug", False))
        try:
            with cuda_guard(name, enabled=cuda_debug):
                step.run()
        except Exception:
            if cuda_debug:
                req_keys = [self._translate_key(step.name, r) for r in step.requires]
                prod_keys = [self._create_real_key(step.name, p) for p in step.promises]
                print(
                    f"[Pipeline] Step '{name}' failed. "
                    f"requires={req_keys} promises={prod_keys}. "
                    f"Resume with: python run.py --headless <config.json> "
                    f"--resume-from \"{name}\" --debug",
                    file=sys.stderr,
                )
            raise

    def _record_step_metric(self, step):
        step_metric = step.pipeline_metric()
        self.set_target_metric(step_metric)
        seeded_before = self.accuracy_budget.seeded()
        self.accuracy_budget.observe(step_metric)
        if not seeded_before and self.accuracy_budget.seeded():
            self._assert_pretrain_envelope(step.name, step_metric)
        self.accuracy_budget.warn_if_over_budget(step.name)

    def _pretrain_envelope_chance_multiple(self) -> float:
        config = getattr(self, "config", None)
        if not isinstance(config, dict):
            return 0.0
        return float(config.get("pretrain_floor_chance_multiple", 5.0))

    def _assert_pretrain_envelope(self, step_name: str, metric: float) -> None:
        """Absolute floor on the FIRST seeded metric (classification only):
        every relative gate downstream is blind to a chance-level backbone."""
        config = getattr(self, "config", None)
        num_classes = config.get("num_classes") if isinstance(config, dict) else None
        if not num_classes or int(num_classes) <= 1:
            return
        multiple = self._pretrain_envelope_chance_multiple()
        if multiple <= 0.0:
            return
        floor = multiple / float(num_classes)
        if float(metric) < floor:
            raise PretrainEnvelopeError(
                step_name, float(metric), floor, int(num_classes), multiple
            )

    def _persist_step_outputs(self, name, step):
        with phase_profiler(f"Pipeline::{name}", "save_cache"):
            self.save_cache()
        with phase_profiler(f"Pipeline::{name}", "offload_torch_models_to_cpu"):
            self.cache.offload_torch_models_to_cpu()
        for entry in step.clears:
            self.cache.remove(self._create_real_key(step.name, entry))

    def _assert_step_contract(self, step):
        assert all(
            self._create_real_key(step.name, promise) in self.cache
            for promise in step.promises
        ), "Pipeline error: Some promised entries were not added."
        assert all(
            self._create_real_key(step.name, entry) not in self.cache
            for entry in step.clears
        ), "Pipeline error: Some cleared entries were not removed."
        assert all(
            self._translate_key(step.name, entry) not in self.cache
            for entry in step.updates
        ), "Pipeline error: Old values of some updated entries are still in the cache."
        assert all(
            self._create_real_key(step.name, entry) in self.cache
            for entry in step.updates
        ), "Pipeline error: New values of some updated entries are not found in the cache."

    def _assert_metric_retention(self, step, previous_metric):
        assert_metric_retention(
            self, step, previous_metric, self._step_tolerance(step.name)
        )
