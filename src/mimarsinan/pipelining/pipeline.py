from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache

from mimarsinan.common.file_utils import prepare_containing_directory

class Pipeline:
    def __init__(self, working_directory) -> None:
        self.steps = []
        self.cache = PipelineCache()

        self.working_directory = working_directory
        prepare_containing_directory(self.working_directory)

        self.tolerance = 0.95

        self.load_cache()
        self.set_target_metric(0.0)

    def add_pipeline_step(self, name, pipeline_step):
        self.steps.append((name, pipeline_step))
        self.verify()

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

    def run(self):
        self.verify()
        for name, step in self.steps:
            self._run_step(name, step)

    def run_from(self, step_name):
        assert step_name in [name for name, _ in self.steps], f"Step '{step_name}' does not exist in pipeline"
        
        self.verify()
        starting_step_idx = self._find_starting_step_idx(step_name)
        
        for idx, (name, step) in enumerate(self.steps):
            if idx < starting_step_idx:
                continue
            self._run_step(name, step)

    def save_cache(self):
        self.cache.store(self.working_directory)

    def load_cache(self):
        self.cache.load(self.working_directory)

    def get_target_metric(self):
        return self.cache['__target_metric']
    
    def set_target_metric(self, target_metric):
        self.cache.add('__target_metric', target_metric)

    def _run_step(self, name, step):
        print(f"Running '{name}'...")

        previous_metric = self.get_target_metric()

        step.run()
        self.set_target_metric(step.validate())
        self.save_cache()

        assert self.get_target_metric() >= previous_metric * self.tolerance, \
            f"[{name}] step failed to retain performance within tolerable limits: {self.get_target_metric()} < ({previous_metric} * {self.tolerance}) = {previous_metric * self.tolerance}"


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
                requirements.add(requirement)
            
            if name == step_name:
                break

            for entry in step.clears:
                if entry in requirements:
                    requirements.remove(entry)
        
        return requirements

    def _find_latest_possible_step_idx(self, missing_requirements, step_name):
        print(f"Finding the earliest step that can be started from...")

        starting_step_idx = None

        begin_idx = self._get_step_idx(step_name)
        for idx in range(begin_idx - 1, -1, -1):
            name = self.steps[idx][0]
            step = self.steps[idx][1]

            for promise in step.promises:
                if promise in missing_requirements:
                    missing_requirements.remove(promise)
            
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