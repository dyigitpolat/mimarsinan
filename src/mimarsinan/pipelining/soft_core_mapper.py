from mimarsinan.mapping.mapping_utils import SoftCoreMapping

class SoftCoreMapper:
    def __init__(self, pipeline):
        self.model = pipeline.model

    def run(self):
        soft_core_mapping = SoftCoreMapping()
        soft_core_mapping.map(self.model.get_mapper_repr())
        return soft_core_mapping