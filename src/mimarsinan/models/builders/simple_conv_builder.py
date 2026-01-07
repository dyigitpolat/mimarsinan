from __future__ import annotations

from mimarsinan.models.preprocessing.input_cq import InputCQ
from mimarsinan.models.simple_conv import SimpleConvMapper
from mimarsinan.models.supermodel import Supermodel


class SimpleConvBuilder:
    def __init__(self, device, input_shape, num_classes, max_axons, max_neurons, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        cfg = dict(configuration or {})

        # First conv layer params
        conv_out_channels = int(cfg.get("conv_out_channels", 3))
        conv_kernel_size = cfg.get("conv_kernel_size", 3)
        conv_stride = cfg.get("conv_stride", 4)
        conv_padding = cfg.get("conv_padding", 1)
        conv_dilation = cfg.get("conv_dilation", 1)
        conv_bias = bool(cfg.get("conv_bias", True))
        
        # Second conv layer params
        conv2_out_channels = cfg.get("conv2_out_channels", None)  # defaults to conv_out_channels
        conv2_kernel_size = cfg.get("conv2_kernel_size", 3)
        conv2_stride = cfg.get("conv2_stride", 1)
        conv2_padding = cfg.get("conv2_padding", 1)
        conv2_dilation = cfg.get("conv2_dilation", 1)
        
        use_batchnorm = bool(cfg.get("use_batchnorm", True))
        use_pool = bool(cfg.get("use_pool", False))
        pool_kernel_size = cfg.get("pool_kernel_size", 2)
        pool_stride = cfg.get("pool_stride", 2)
        pool_padding = cfg.get("pool_padding", 0)
        
        # FC hidden layer size
        fc_hidden_features = cfg.get("fc_hidden_features", None)

        preprocessor = InputCQ(self.pipeline_config["target_tq"])
        perceptron_flow = SimpleConvMapper(
            self.device,
            self.input_shape,
            self.num_classes,
            conv_out_channels=conv_out_channels,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            conv_padding=conv_padding,
            conv_dilation=conv_dilation,
            conv_bias=conv_bias,
            conv2_out_channels=conv2_out_channels,
            conv2_kernel_size=conv2_kernel_size,
            conv2_stride=conv2_stride,
            conv2_padding=conv2_padding,
            conv2_dilation=conv2_dilation,
            use_pool=use_pool,
            pool_kernel_size=pool_kernel_size,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            use_batchnorm=use_batchnorm,
            fc_hidden_features=fc_hidden_features,
            max_axons=self.max_axons,
            max_neurons=self.max_neurons,
            name="simple_conv",
        )

        supermodel = Supermodel(
            self.device,
            self.input_shape,
            self.num_classes,
            preprocessor,
            perceptron_flow,
            self.pipeline_config["target_tq"],
        )

        allow_axon_tiling = bool(self.pipeline_config.get("allow_axon_tiling", False))
        if not allow_axon_tiling:
            for perceptron in supermodel.get_perceptrons():
                in_axons = perceptron.layer.weight.shape[1]
                if in_axons > self.max_axons - 1:
                    raise ValueError(
                        f"not enough axons ({in_axons} > {self.max_axons - 1})"
                    )

        return supermodel


