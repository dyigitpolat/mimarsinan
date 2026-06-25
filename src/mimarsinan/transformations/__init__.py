"""Model transformations: weight quantization, normalization fusion, perceptron transforms, pruning."""

from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.weight_quantization import TensorQuantization
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.transformation_utils import transform_np_array
from mimarsinan.transformations.pruning import compute_pruning_masks, apply_pruning_masks
from mimarsinan.transformations.pruning.magnitude import (
    ChannelPruningResult,
    kept_output_channels,
    prune_perceptron_chain,
)
from mimarsinan.transformations.activation_scale_policy import (
    ActivationScalePolicy,
    make_activation_scale_policy,
    DEFAULT_ACTIVATION_SCALE_POLICY,
)

