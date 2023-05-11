class NormalizationFuser:
    def __init__(self, pipeline):
        self.model = pipeline.model

    def run(self):
        self.model.fuse_normalization()