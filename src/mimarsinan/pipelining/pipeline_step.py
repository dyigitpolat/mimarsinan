class PipelineStep:
    def __init__(self, requires, promises, clears, pipeline):
        self.requires = requires
        self.promises = promises
        self.clears = clears
        self.pipeline = pipeline

    def run(self):
        assert all([requirement in self.pipeline.cache for requirement in self.requires])
        self.process()
        assert all([promise in self.pipeline.cache for promise in self.promises])
        assert all([entry not in self.pipeline.cache for entry in self.clears])

    def process(self):
        raise NotImplementedError
    
    def validate(self):
        raise NotImplementedError

