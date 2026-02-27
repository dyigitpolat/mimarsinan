"""Builder that wraps a user-provided model factory callable."""


class TorchCustomBuilder:
    """Produces a native ``nn.Module`` from a user-supplied factory function.

    The factory is called as ``model_factory(configuration)`` and must return
    an ``nn.Module``.  This lets users plug arbitrary PyTorch models into
    the deployment pipeline.
    """

    def __init__(
        self,
        device,
        input_shape,
        num_classes,
        max_axons,
        max_neurons,
        pipeline_config,
        model_factory=None,
    ):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config
        self.model_factory = model_factory

    def build(self, configuration):
        if self.model_factory is None:
            raise ValueError(
                "TorchCustomBuilder requires a model_factory callable "
                "in pipeline_config['model_factory']."
            )
        return self.model_factory(configuration)
