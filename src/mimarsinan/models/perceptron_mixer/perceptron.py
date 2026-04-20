from mimarsinan.models.layers import MaxValueScaler, LeakyGradReLU


import torch.nn as nn
import torch


# Canonical mapping from string names to activation constructors.
# "ReLU" maps to LeakyGradReLU (ReLU forward, leaky gradient backward) —
# the default pipeline activation for SNN training.
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
        normalization=nn.Identity(),
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
        # Log-space learnable clamp ceiling (Phase B2).  ``ClampTuner`` flips
        # ``requires_grad`` on while it's running so gradient descent can
        # refine the ceiling seeded from the p99 activation scale; at the
        # end of the tuner the learnt value is written back into
        # ``activation_scale`` and ``requires_grad`` is turned off again.
        # Keeping the parameter in log-space guarantees positivity under any
        # optimiser update.  Stored as a buffer of shape () so parameter
        # counts remain one scalar per perceptron.
        self.log_clamp_ceiling = nn.Parameter(
            torch.log(self.activation_scale.detach().clone()),
            requires_grad=False,
        )

        self.per_input_scales = None

        # LSQ weight quantizer (Phase C1).  ``None`` until
        # ``NormalizationAwarePerceptronQuantization.transform`` installs one;
        # at that point it becomes a proper child module so its
        # ``log_scale`` parameter is picked up by ``model.parameters()`` and
        # therefore by ``BasicTrainer``'s optimizer -- STE gradients flow
        # from the (quantized) forward pass through ``log_scale`` and into
        # the underlying FP weights in a single backward call.
        self.weight_quantizer = None

        # True for the first perceptron of each neural segment (host-side ComputeOp in IR).
        self.is_encoding_layer = False

    def set_parameter_scale(self, new_scale):
        if isinstance(new_scale, float):
            new_scale = torch.tensor(new_scale)
        self.parameter_scale.data = new_scale.data
    
    def set_activation_scale(self, new_scale):
        if isinstance(new_scale, float):
            new_scale = torch.tensor(new_scale)
        self.activation_scale.data = new_scale.data
        # Keep the learnable clamp ceiling in sync with the static scale.
        # Writing directly to .data avoids disturbing any gradient hooks
        # tuners might have attached; the log-space storage guarantees the
        # value stays positive under subsequent optimiser updates.
        with torch.no_grad():
            log_val = torch.log(new_scale.detach().to(self.log_clamp_ceiling.device))
            self.log_clamp_ceiling.data.copy_(log_val.reshape(self.log_clamp_ceiling.shape))

    def effective_clamp_ceiling(self):
        """Return ``exp(log_clamp_ceiling)`` -- the learnable clamp ceiling
        decorators and analysis tools should consult for Phase B2.  This
        view is live (tracks gradients) when ``log_clamp_ceiling.requires_grad``
        is True."""
        return torch.exp(self.log_clamp_ceiling)

    def set_input_activation_scale(self, new_scale):
        if isinstance(new_scale, float):
            new_scale = torch.tensor(new_scale)
        self.input_activation_scale.data = new_scale.data

    def set_scale_factor(self, new_scale):
        if isinstance(new_scale, float):
            new_scale = torch.tensor(new_scale)
        self.scale_factor.data = new_scale.data
        
    def set_activation(self, activation):
        self.activation = activation

    def set_regularization(self, regularizer):
        self.regularization = regularizer

    def set_scaler(self, scaler):
        self.scaler = scaler

    def set_weight_quantizer(self, quantizer):
        """Attach an LSQ weight quantizer as a child module (Phase C1).

        Stored via ``nn.Module.__setattr__`` semantics so the quantizer's
        parameters are included in ``self.parameters()`` and therefore
        reach the trainer's optimizer automatically.
        """
        self.weight_quantizer = quantizer

    def forward(self, x):
        # input_activation and scaler are nn.Identity throughout training/tuning
        # (only set to something else during soft-core-mapping, post-tuning).
        # Skip their __call__ + hook machinery when that's the case.
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
