from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache

from mimarsinan.common.file_utils import prepare_containing_directory

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

        self.post_step_hook = None
        self.pre_step_hook = None

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

    def run_from(self, step_name, *, stop_step: str | None = None):
        assert step_name in [name for name, _ in self.steps], f"Step '{step_name}' does not exist in pipeline"
        
        self.set_up_requirements()
        self.verify()
        starting_step_idx = self._find_starting_step_idx(step_name)
        
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
        self.post_step_hook = hook

    def register_pre_step_hook(self, hook):
        self.pre_step_hook = hook

    def _create_real_key(self, client_step_name, key):
        return client_step_name + '.' + key

    def _translate_key(self, client_step_name, key):
        return self.key_translations[client_step_name][key]

    def _run_step(self, name, step):
        print(f"Running '{name}'...")

        if self.pre_step_hook is not None:
            self.pre_step_hook(step)

        previous_metric = self.get_target_metric()

        assert all([self._translate_key(step.name, requirement) in self.cache for requirement in step.requires]), \
            f"Pipeline error: Some requirements are not found in the cache."

        step.run()
        self.set_target_metric(step.validate())
        self.save_cache()

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

        if self.post_step_hook is not None:
            self.post_step_hook(step)

    def _find_starting_step_idx(self, step_name):
        requirements = self._get_all_requirements(step_name)
        missing_requirements = requirements - set(self.cache.keys())

        if len(missing_requirements) > 0:
            print(f"Cannot start from '{step_name}' because of missing requirements: {missing_requirements}")
            starting_step_idx = self._find_latest_possible_step_idx(missing_requirements, step_name)
        else:
            starting_step_idx = self._get_step_idx(step_name)

        return starting_step_idx
    
    def _get_all_requirements(self, step_name):
        requirements = set()
        for name, step in self.steps:
            for requirement in step.requires:
                requirements.add(self._translate_key(step.name, requirement))
            
            if name == step_name:
                break

            for entry in step.updates:
                real_entry = self._translate_key(step.name, entry)
                if real_entry in requirements:
                    requirements.remove(real_entry)

            for entry in step.clears:
                real_entry = self._create_real_key(step.name, entry)
                if real_entry in requirements:
                    requirements.remove(real_entry)
        
        return requirements

    def _find_latest_possible_step_idx(self, missing_requirements, step_name):
        print(f"Finding the earliest step that can be started from...")

        starting_step_idx = None

        begin_idx = self._get_step_idx(step_name)
        for idx in range(begin_idx - 1, -1, -1):
            name = self.steps[idx][0]
            step = self.steps[idx][1]

            for promise in step.promises:
                real_promise = self._create_real_key(step.name, promise)
                if real_promise in missing_requirements:
                    missing_requirements.remove(real_promise)

            for entry in step.updates:
                real_entry = self._create_real_key(step.name, entry)
                if real_entry in missing_requirements:
                    missing_requirements.remove(real_entry)
            
            if len(missing_requirements) == 0:
                starting_step_idx = idx
                print(f"Starting from '{name}'")
                break

        assert starting_step_idx is not None
        return starting_step_idx
    
    def _get_step_idx(self, step_name):
        for idx, (name, _) in enumerate(self.steps):
            if name == step_name:
                return idx
        return None