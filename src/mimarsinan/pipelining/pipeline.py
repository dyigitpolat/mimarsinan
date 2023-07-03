from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache

from mimarsinan.common.file_utils import prepare_containing_directory

class Pipeline:
    def __init__(self, working_directory) -> None:
        self.steps = []
        self.cache = PipelineCache()

        self.working_directory = working_directory
        prepare_containing_directory(self.working_directory)

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

    def _run_step(self, name, step):
        print(f"Running '{name}'...")
        step.run()
        self.save_cache()

    def _find_starting_step_idx(self, step_name):
        requirements = set()
        for name, step in self.steps:
            for requirement in step.requires:
                requirements.add(requirement)
            
            if name == step_name:
                break

            for entry in step.clears:
                if entry in requirements:
                    requirements.remove(entry)

        missing_requirements = requirements - set(self.cache.keys())
        if len(missing_requirements) > 0:
            print(f"Cannot start from '{step_name}' because of missing requirements: {missing_requirements}")
            print(f"Finding the earliest step that can be started from...")

            starting_step_idx = None
            for idx, (name, step) in enumerate(self.steps):
                for promise in step.promises:
                    if promise in missing_requirements:
                        missing_requirements.remove(promise)
                
                if len(missing_requirements) == 0:
                    starting_step_idx = idx
                    print(f"Starting from '{name}'")
                    break

            assert starting_step_idx is not None
        else:
            for idx, (name, step) in enumerate(self.steps):
                if name == step_name:
                    starting_step_idx = idx
                    break

        return starting_step_idx