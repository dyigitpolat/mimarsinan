import torch
import torch.nn as nn

from mimarsinan.transformations.pruning.committed_masks import (
    commit_perceptron_pruning,
)

_DEGENERATE_U_EPS = 1e-6
"""Below this |u| = |gamma/sigma| a channel is structurally dead: its raw
parameter is unobservable through the normalization, so effective->raw
division is numerically meaningless."""

_RAW_WRITE_AMPLIFICATION_BOUND = 1e3
"""Max tolerated |raw write| / |effective delta| before the bias inversion
routes the delta through the normalization affine instead of dividing by u."""


class PerceptronTransformer:
    def __init__(self):
        pass

    def _commit_raw_write(self, perceptron, param, new_value, what: str):
        """Write an effective->raw inversion result: re-commit the perceptron's
        prune masks (pruned rows stay exactly zero — the degenerate-``u``
        inversion must not poison them) and fail loud on non-finite values."""
        with torch.no_grad():
            param.data[:] = new_value
        commit_perceptron_pruning(perceptron)
        if not torch.isfinite(param.data).all():
            raise RuntimeError(
                f"PerceptronTransformer wrote non-finite raw {what} into "
                f"perceptron {getattr(perceptron, 'name', '<unnamed>')!r} "
                "(degenerate normalization fold factor u = gamma/sigma on a "
                "live row?)."
            )

    def get_effective_parameters(self, perceptron):
        return self.get_effective_weight(perceptron), self.get_effective_bias(perceptron)
    
    def apply_effective_parameter_transform(self, perceptron, parameter_transform):
        self.apply_effective_weight_transform(perceptron, parameter_transform)
        self.apply_effective_bias_transform(perceptron, parameter_transform)

    def _get_input_scale(self, perceptron):
        """Return broadcastable input scale: per-channel tensor or scalar 1.0."""
        pis = getattr(perceptron, 'per_input_scales', None)
        if pis is not None:
            pis = pis.to(perceptron.layer.weight.data.device)
            extra = perceptron.layer.weight.data.dim() - 2  # 0 for FC, 2 for Conv2D
            return pis.view(1, -1, *([1] * extra))
        return 1.0

    def _activation_scale_for_weight(self, perceptron):
        """``activation_scale`` broadcastable against the weight ``[out, in, ...]``.

        A per-output-channel theta (``ttfs_theta_cotrain``: 1-D, len == out_features)
        must divide per OUTPUT channel, so view it as ``[out, 1, ...]``; a scalar (or
        1-element) activation_scale passes through unchanged.
        """
        a = perceptron.activation_scale
        w = perceptron.layer.weight.data
        if torch.is_tensor(a) and a.dim() >= 1 and a.numel() == w.shape[0]:
            return a.reshape(w.shape[0], *([1] * (w.dim() - 1))).to(w.device)
        return a

    def get_effective_weight(self, perceptron):
        scale = self._get_input_scale(perceptron)
        act = self._activation_scale_for_weight(perceptron)
        if isinstance(perceptron.normalization, nn.Identity):
            return scale * perceptron.layer.weight.data / act
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            return scale * (perceptron.layer.weight.data * u.unsqueeze(-1)) / act
        
    def get_effective_bias(self, perceptron):
        if perceptron.layer.bias is None:
            layer_bias = torch.zeros(perceptron.layer.weight.shape[0])
        else:
            layer_bias = perceptron.layer.bias.data

        if isinstance(perceptron.normalization, nn.Identity):
            return layer_bias / perceptron.activation_scale
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            return ((layer_bias - mean) * u + beta) / perceptron.activation_scale
        
    def apply_effective_weight_transform(self, perceptron, weight_transform):
        effective_weight = self.get_effective_weight(perceptron)
        scale = self._get_input_scale(perceptron)
        act = self._activation_scale_for_weight(perceptron)

        if isinstance(perceptron.normalization, nn.Identity):
            new_weight = (weight_transform(effective_weight) * act) / scale
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            raw = perceptron.layer.weight.data
            dead = (u.abs() < _DEGENERATE_U_EPS).unsqueeze(-1)
            u_bc = u.unsqueeze(-1)
            while u_bc.dim() < raw.dim():
                u_bc = u_bc.unsqueeze(-1)
                dead = dead.unsqueeze(-1)
            safe_u = torch.where(dead, torch.ones_like(u_bc), u_bc)
            inverted = ((weight_transform(effective_weight) * act) / scale) / safe_u
            # Dead channels keep their raw weights: their effective weight is ~0 and stays so.
            new_weight = torch.where(dead, raw, inverted)
        self._commit_raw_write(
            perceptron, perceptron.layer.weight, new_weight, "weight",
        )

    def apply_effective_bias_transform(self, perceptron, bias_transform):
        if perceptron.layer.bias is None:
            return
        effective_bias = self.get_effective_bias(perceptron)
        target = bias_transform(effective_bias)
        act = perceptron.activation_scale

        if isinstance(perceptron.normalization, nn.Identity):
            self._commit_raw_write(
                perceptron, perceptron.layer.bias, target * act, "bias",
            )
            return

        u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
        raw = perceptron.layer.bias.data
        structurally_dead = u.abs() < _DEGENERATE_U_EPS
        safe_u = torch.where(structurally_dead, torch.ones_like(u), u)
        inverted = ((target * act - beta) / safe_u) + mean
        delta_effective = (target - effective_bias) * act
        amplified = (
            (inverted - raw).abs()
            > _RAW_WRITE_AMPLIFICATION_BOUND * delta_effective.abs()
        ) & (delta_effective.abs() > 0)
        degenerate = structurally_dead | amplified
        if bool(degenerate.any()):
            self._realize_effective_bias_through_normalization(
                perceptron, degenerate, target, act, raw, u, mean
            )
        new_bias = torch.where(degenerate, raw, inverted)
        self._commit_raw_write(
            perceptron, perceptron.layer.bias, new_bias, "bias",
        )

    def _realize_effective_bias_through_normalization(
        self, perceptron, degenerate, target, act, raw, u, mean
    ):
        """On degenerate channels the raw bias is unobservable through the fold
        factor u, so write the requested effective bias EXACTLY into the
        normalization affine (beta_new = target*act - (raw-mean)*u; raw unchanged)."""
        norm_bias = perceptron.normalization.bias
        new_beta = target * act - (raw - mean) * u
        with torch.no_grad():
            norm_bias.data = torch.where(degenerate, new_beta, norm_bias.data)
        print(
            f"[PerceptronTransformer] {getattr(perceptron, 'name', '<unnamed>')}: "
            f"routed {int(degenerate.sum())} degenerate-channel bias delta(s) "
            "through normalization beta (total inversion)"
        )

    def _get_u_beta_mean(self, bn_layer):
        from mimarsinan.models.nn.layers import norm_affine_params

        u, beta, mean = norm_affine_params(bn_layer)
        return u.detach(), beta.detach(), mean.detach()
