from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.mapping.mapping_utils import SoftCoreMapping

class SoftCoreMappingStep(PipelineStep):

    def __init__(self, pipeline):
        requires = ["wq_model"]
        promises = ["soft_core_mapping"]
        clears = []
        super().__init__(requires, promises, clears, pipeline)

    def process(self):
        model = self.pipeline.cache['wq_model']
        soft_core_mapping = SoftCoreMapping()
        soft_core_mapping.map(model.get_mapper_repr())
        
        self.pipeline.cache.add("soft_core_mapping", soft_core_mapping, 'pickle')