import gc
import sys
import time as _time

import torch

from mimarsinan.pipelining.core.accuracy_budget import AccuracyBudget
from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache
from mimarsinan.common.file_utils import prepare_containing_directory
from mimarsinan.common.diagnostics import cuda_guard, phase_profiler
from mimarsinan.pipelining.core.engine.pipeline_resource_debug import log_resource_snapshot
from mimarsinan.pipelining.core.engine import pipeline_resume


class Pipeline:
    def __init__(self, working_directory) -> None:
        self.steps = []
        self.cache = PipelineCache()
        self.key_translations = {} # key_translations[step_name][virtual_entry_key] : real_entry_key

        self.working_directory = working_directory
        prepare_containing_directory(self.working_directory)

        self.tolerance = 0.95

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

    def run(self, *, stop_step: str | None = None):
        self.set_up_requirements()
        self.verify()
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

        assert all([self._translate_key(step.name, requirement) in self.cache for requirement in step.requires]), \
            f"Pipeline error: Some requirements are not found in the cache."

        step.pipeline_previous_metric = previous_metric

        cuda_debug = bool(getattr(self, "cuda_debug", False))
        try:
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
            step_metric = step.pipeline_metric()
            self.set_target_metric(step_metric)
            self.accuracy_budget.observe(step_metric)
            self.accuracy_budget.warn_if_over_budget(step.name)
            with phase_profiler(f"Pipeline::{name}", "save_cache"):
                self.save_cache()
            with phase_profiler(f"Pipeline::{name}", "offload_torch_models_to_cpu"):
                self.cache.offload_torch_models_to_cpu()

            for entry in step.clears:
                self.cache.remove(self._create_real_key(step.name, entry))

            assert all([self._create_real_key(step.name, promise) in self.cache for promise in step.promises]), \
                f"Pipeline error: Some promised entries were not added."
            assert all([self._create_real_key(step.name, entry) not in self.cache for entry in step.clears]), \
                f"Pipeline error: Some cleared entries were not removed."

            assert all([self._translate_key(step.name, entry) not in self.cache for entry in step.updates]), \
                f"Pipeline error: Old values of some updated entries are still in the cache."
            assert all([self._create_real_key(step.name, entry) in self.cache for entry in step.updates]), \
                f"Pipeline error: New values of some updated entries are not found in the cache."

            assert self.get_target_metric() >= previous_metric * self.tolerance, \
                f"[{step.name}] step failed to retain performance within tolerable limits: {self.get_target_metric()} < ({previous_metric} * {self.tolerance}) = {previous_metric * self.tolerance}"

            for hook in self.post_step_hooks:
                hook(name, step)
        finally:
            step.cleanup()
            self._release_gpu_memory()
            log_resource_snapshot(f"post:{name}")
            dt = _time.time() - _t0_step
            final_metric = self.get_target_metric()
            delta = final_metric - previous_metric
            print(
                f"[PROFILE] step='{name}' wall={dt:7.2f}s "
                f"metric={final_metric:.4f} "
                f"Δ={delta:+.4f} (prev={previous_metric:.4f})"
            )
