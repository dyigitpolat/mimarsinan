class SequentialTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, param):
        for transform in self.transforms:
            param = transform(param)

        return param
    
    def __call__(self, param):
        return self.transform(param)