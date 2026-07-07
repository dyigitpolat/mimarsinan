from mimarsinan.models.nn.layers import MaxValueScaler, LeakyGradReLU, norm_affine_params

import torch.nn as nn
import torch


def effective_preactivation_bias(perceptron):
    """Additive constant of ``normalization(layer(x))`` under frozen stats —
    the bias the deployed chip charges per cycle. ``None`` when there is none.
    Differentiable through ``layer.bias`` and the norm's affine params."""
    bias = perceptron.layer.bias
    normalization = perceptron.normalization
    if isinstance(normalization, nn.Identity):
        return bias
    u, beta, mean = norm_affine_params(normalization)
    if bias is None:
        bias = torch.zeros_like(mean)
    return (bias - mean) * u + beta


def activation_channel_axis(perceptron, t: torch.Tensor) -> int:
    """Resolve the output-channel axis of ``t``, an activation (or decoded-value)
    tensor of ``perceptron``, from the owner-declared ``output_channel_axis``.
    Fails loud when the declaration is missing or inconsistent with the tensor."""
    axis = getattr(perceptron, "output_channel_axis", None)
    if axis is None:
        raise ValueError(
            f"{type(perceptron).__name__} "
            f"{getattr(perceptron, 'name', '<unnamed>')!r} does not declare "
            "output_channel_axis; the activation channel axis is ambiguous. "
            "The owner constructing the perceptron must declare its output "
            "layout (nn.Linear layouts are channels-last; conv mappers are "
            "channels-first, dim 1)."
        )
    resolved = int(axis) % t.dim()
    out_channels = int(perceptron.layer.weight.shape[0])
    if int(t.shape[resolved]) != out_channels:
        raise ValueError(
            f"{type(perceptron).__name__} "
            f"{getattr(perceptron, 'name', '<unnamed>')!r} declares channel "
            f"axis {axis} but tensor shape {tuple(t.shape)} has size "
            f"{int(t.shape[resolved])} there, not the layer's "
            f"{out_channels} output channels."
        )
    return resolved


ACTIVATION_REGISTRY = {
    "ReLU": LeakyGradReLU,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
}


def make_activation(name=None):
    """Create an activation module from a string name. Default: LeakyGradReLU."""
    if name is None or name not in ACTIVATION_REGISTRY:
        return LeakyGradReLU()
    return ACTIVATION_REGISTRY[name]()


class Perceptron(nn.Module):
    def __init__(
        self,
        output_channels, input_features, bias=True,
        normalization: nn.Module = nn.Identity(),
        base_activation_name=None,
        name="Perceptron"):

        super(Perceptron, self).__init__()
        self.name = name

        self.input_features = input_features
        self.output_channels = output_channels

        self.layer = nn.Linear(
            input_features, output_channels, bias=bias)

        self.normalization = normalization

        self.scaler = nn.Identity()
        self.base_scaler = MaxValueScaler()
        self.scale_factor = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.input_activation = nn.Identity()

        self.base_activation_name = base_activation_name or "ReLU"
        self.base_activation = make_activation(self.base_activation_name)
        self.activation = make_activation(self.base_activation_name)

        self.regularization = nn.Identity()

        self.parameter_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.input_activation_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.activation_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # Non-persistent buffer: moves with .to()/offload, absent from
        # state_dict (cache/golden/parity layouts unchanged).
        self.register_buffer("per_input_scales", None, persistent=False)

        # nn.Linear puts output channels on the LAST dim; conv mappers driving
        # this perceptron through F.conv override this to 1 (channels-first).
        self.output_channel_axis = -1

        self.is_encoding_layer = False

    def __setstate__(self, state):
        super().__setstate__(state)
        # Caches saved before the layout declaration carry the nn.Linear
        # channels-last layout; conv mappers re-stamp channels-first on load.
        self.__dict__.setdefault("output_channel_axis", -1)
        # Caches saved before the buffer registration carry per_input_scales
        # as a plain attr that .to()/offload cannot move; migrate it.
        if "per_input_scales" not in self._buffers:
            self._buffers["per_input_scales"] = self.__dict__.pop(
                "per_input_scales", None,
            )

    def set_parameter_scale(self, new_scale):
        if isinstance(new_scale, float):
            new_scale = torch.tensor(new_scale)
        self.parameter_scale.data = new_scale.data
    
    def set_activation_scale(self, new_scale):
        if isinstance(new_scale, float):
            new_scale = torch.tensor(new_scale)
        self.activation_scale.data = new_scale.data

    def set_input_activation_scale(self, new_scale):
        if isinstance(new_scale, float):
            new_scale = torch.tensor(new_scale)
        self.input_activation_scale.data = new_scale.data

    def append_input_wire_op(self, module):
        """Append a wire op (e.g. an STE input quantizer) after ``input_activation``."""
        if isinstance(self.input_activation, nn.Identity):
            self.input_activation = module
        else:
            self.input_activation = nn.Sequential(self.input_activation, module)

    def set_scale_factor(self, new_scale):
        if isinstance(new_scale, float):
            new_scale = torch.tensor(new_scale)
        self.scale_factor.data = new_scale.data
        
    def effective_preactivation_bias(self):
        return effective_preactivation_bias(self)

    def set_activation(self, activation):
        self.activation = activation

    def set_regularization(self, regularizer):
        self.regularization = regularizer

    def set_scaler(self, scaler):
        self.scaler = scaler

    def forward(self, x):
        if not isinstance(self.input_activation, nn.Identity):
            x = self.input_activation(x)

        out = self.layer(x)
        if not isinstance(self.normalization, nn.Identity):
            out = self.normalization(out)
        if not isinstance(self.scaler, nn.Identity):
            out = self.scaler(out)

        out = self.activation(out)

        if self.training and not isinstance(self.regularization, nn.Identity):
            out = self.regularization(out)

        return out

    def forward_spiking(self, x):
        """Encoding-layer spiking forward — return ``(T, B, ...)`` spike train.

        Mirrors :meth:`forward` up to the activation, then emits the wrapped
        ``LIFActivation``'s cycle-by-cycle spike train; raises if none is reachable.
        """
        from mimarsinan.spiking.lif_utils import unwrap_lif_activation

        lif = unwrap_lif_activation(self.activation)
        if lif is None:
            raise ValueError(
                "Perceptron.forward_spiking requires self.activation to wrap "
                "a LIFActivation; got " + type(self.activation).__name__
            )
        if not isinstance(self.input_activation, nn.Identity):
            x = self.input_activation(x)
        out = self.layer(x)
        if not isinstance(self.normalization, nn.Identity):
            out = self.normalization(out)
        if not isinstance(self.scaler, nn.Identity):
            out = self.scaler(out)
        return lif.forward_spiking(out)
