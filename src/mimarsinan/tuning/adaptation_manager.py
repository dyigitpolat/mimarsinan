from mimarsinan.models.layers import *
from mimarsinan.tuning.shift_calculation import calculate_activation_shift

import torch.nn as nn


# Names of the scalar rate fields that ``AdaptationManager`` exposes and for
# which per-perceptron overrides are supported.  Keeping them listed here
# makes ``get_rate`` / ``set_per_perceptron_rate`` reject typos (e.g.
# "clamp-rate") with a clear error rather than silently creating a dead
# override.
_KNOWN_RATE_FIELDS: frozenset = frozenset({
    "activation_adaptation_rate",
    "clamp_rate",
    "shift_rate",
    "quantization_rate",
    "pruning_rate",
})
"""The live rate dimensions driven by pipeline tuners.

Phase D3 removed ``noise_rate`` (no live tuner writes it -- ``NoiseTuner``
was never instantiated by any pipeline step) and ``scale_rate`` (the
``ScaleDecorator`` path was deleted as dead code).  Reintroducing either
name here without a corresponding tuner resurrects dead schedulers --
add the tuner first, then the rate."""


class AdaptationManager(nn.Module):
    def __init__(self):
        super(AdaptationManager, self).__init__()

        # Scalar "global" rates -- applied to every perceptron that does not
        # have a per-perceptron override registered.  Keeping the scalar API
        # means every existing tuner continues to work unchanged; the new
        # per-perceptron dispatch layers on top without breaking anything.
        self.activation_adaptation_rate = 0.0
        self.clamp_rate = 0.0
        self.shift_rate = 0.0
        self.quantization_rate = 0.0
        self.pruning_rate = 0.0

        # Per-perceptron overrides: {rate_name: {perceptron_name: float}}.
        # ``get_rate`` looks up the override first and falls back to the
        # scalar field if none is registered.  The override dict lives in
        # plain Python (not an ``nn.Module`` container) because it holds
        # scalars, not trainable tensors.
        self._per_perceptron_rates: dict[str, dict[str, float]] = {
            name: {} for name in _KNOWN_RATE_FIELDS
        }

    # ------------------------------------------------------------------
    # Per-perceptron rate overrides (Phase B1)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_perceptron_name(perceptron) -> str | None:
        """Return a stable identifier for a perceptron, or None if it has no
        usable name.  Falling back to ``None`` keeps ``get_rate`` safe for
        code paths that construct perceptrons on the fly in tests or
        benchmarks without attaching a name."""
        name = getattr(perceptron, "name", None)
        if isinstance(name, str) and name:
            return name
        return None

    def _check_rate_name(self, rate_name: str) -> None:
        if rate_name not in _KNOWN_RATE_FIELDS:
            raise ValueError(
                f"unknown rate {rate_name!r}; known rates are "
                f"{sorted(_KNOWN_RATE_FIELDS)}"
            )

    def get_rate(self, rate_name: str, perceptron) -> float:
        """Return the effective rate ``rate_name`` for the given perceptron.

        If a per-perceptron override is registered for this perceptron's
        ``name`` it wins; otherwise the scalar field ``self.<rate_name>`` is
        used.  Unknown rate names raise ``ValueError`` so typos surface
        immediately rather than silently returning 0.
        """
        self._check_rate_name(rate_name)
        name = self._resolve_perceptron_name(perceptron)
        if name is not None:
            override = self._per_perceptron_rates[rate_name].get(name)
            if override is not None:
                return float(override)
        return float(getattr(self, rate_name))

    def set_per_perceptron_rate(
        self, rate_name: str, perceptron_name: str, value: float
    ) -> None:
        """Register a per-perceptron override for ``rate_name``.  Passing
        ``None`` for the value removes the override (subsequent
        ``get_rate`` calls return the scalar again)."""
        self._check_rate_name(rate_name)
        if not isinstance(perceptron_name, str) or not perceptron_name:
            raise ValueError(
                f"perceptron_name must be a non-empty string; got "
                f"{perceptron_name!r}"
            )
        if value is None:
            self._per_perceptron_rates[rate_name].pop(perceptron_name, None)
            return
        self._per_perceptron_rates[rate_name][perceptron_name] = float(value)

    def clear_per_perceptron_overrides(self, rate_name: str | None = None) -> None:
        """Drop every per-perceptron override for ``rate_name`` (or, when
        ``rate_name`` is None, for every rate)."""
        if rate_name is None:
            for k in self._per_perceptron_rates:
                self._per_perceptron_rates[k].clear()
            return
        self._check_rate_name(rate_name)
        self._per_perceptron_rates[rate_name].clear()

    def set_per_perceptron_schedule(
        self,
        rate_name: str,
        t: float,
        sensitivities: dict[str, float],
    ) -> None:
        """Distribute a global progress ``t`` across perceptrons using
        ``sensitivities``.

        Design:
          * ``t`` is a scalar in ``[0, 1]`` that plays the role the scalar
            field used to play -- it's the "global amount of
            transformation" we are currently willing to apply.
          * ``sensitivities`` maps ``perceptron_name -> sensitivity``
            (larger = more sensitive = should receive less transformation
            at a given ``t``).  The absolute scale of sensitivities does
            not matter -- only their ratios are used.
          * For each perceptron the per-perceptron rate is set to
            ``clip(t * (s_max / s_i), 0, 1)``.  This makes the least
            sensitive perceptron reach 1.0 at the smallest ``t``, and the
            most sensitive one saturates last.
          * ``t = 0`` forces every rate to 0, ``t = 1`` forces every rate
            to 1 regardless of sensitivity so the schedule always lands at
            "fully transformed" at the end of the run.

        This function is a *default* schedule; tuners are free to call
        ``set_per_perceptron_rate`` directly to implement fancier policies.
        """
        self._check_rate_name(rate_name)
        t = float(t)
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"t must be in [0, 1]; got {t}")
        if not sensitivities:
            self._per_perceptron_rates[rate_name].clear()
            return

        if t <= 0.0:
            # Force every perceptron to 0 so the override wins against any
            # stale scalar the caller may have left in place.
            for name in sensitivities:
                self._per_perceptron_rates[rate_name][name] = 0.0
            return

        if t >= 1.0:
            for name in sensitivities:
                self._per_perceptron_rates[rate_name][name] = 1.0
            return

        sens_vals = [float(s) for s in sensitivities.values()]
        # Guard against non-positive sensitivities: treat them as the
        # smallest positive sensitivity so ratios remain finite.
        positive = [s for s in sens_vals if s > 0]
        if not positive:
            fallback = 1.0
        else:
            fallback = min(positive)
        safe_sens = {
            name: float(s) if float(s) > 0 else fallback
            for name, s in sensitivities.items()
        }
        s_max = max(safe_sens.values())
        for name, s in safe_sens.items():
            raw = t * (s_max / s)
            self._per_perceptron_rates[rate_name][name] = max(0.0, min(1.0, raw))

    def update_activation(self, pipeline_config, perceptron):
        use_ttfs = pipeline_config.get("spiking_mode", "rate") in ("ttfs", "ttfs_quantized")

        # Resolve every rate once so the decorator factories below can work
        # off the per-perceptron effective values. ``get_rate`` returns the
        # scalar field if no per-perceptron override is registered, so the
        # behaviour is identical to the previous scalar API when tuners
        # haven't opted into per-layer schedules yet.
        aa_rate = self.get_rate("activation_adaptation_rate", perceptron)
        clamp_rate = self.get_rate("clamp_rate", perceptron)
        quant_rate = self.get_rate("quantization_rate", perceptron)
        shift_rate = self.get_rate("shift_rate", perceptron)

        decorators = []
        if aa_rate > 0:
            decorators.append(
                self.get_rate_adjusted_activation_replacement_decorator(
                    perceptron, aa_rate
                )
            )
        # Each rate-adjusted decorator short-circuits internally at rate==0 but
        # still costs a Python call per forward. update_activation is invoked
        # whenever a rate changes, so we can safely omit inactive decorators.
        if clamp_rate != 0.0:
            decorators.append(
                self.get_rate_adjusted_clamp_decorator(perceptron, clamp_rate)
            )
        if quant_rate != 0.0:
            decorators.append(
                self.get_rate_adjusted_quantization_decorator(
                    pipeline_config, perceptron, quant_rate, shift_rate
                )
            )
        if not use_ttfs and shift_rate != 0.0:
            # At shift_rate==0 the amount is a zero tensor; ShiftDecorator would
            # still execute torch.sub(x, 0) every forward. Omit entirely.
            decorators.append(
                self.get_shift_decorator(pipeline_config, perceptron, shift_rate)
            )

        perceptron.set_activation(
            TransformedActivation(perceptron.base_activation, decorators))

        # Regularization is left at its default (``nn.Identity`` -- set in
        # ``Perceptron.__init__``).  Phase D3 deleted the only caller that
        # installed anything non-trivial here (the stochastic-noise branch
        # driven by the now-removed noise tuner), so there is nothing to
        # wire up -- rebuilding the activation chain above is the full
        # contract of ``update_activation``.


    def get_rate_adjusted_activation_replacement_decorator(self, perceptron, rate=None):
        """Gradually blend the base activation toward LeakyGradReLU (chip ReLU)."""
        from mimarsinan.models.activations import LeakyGradReLU
        if rate is None:
            rate = self.get_rate("activation_adaptation_rate", perceptron)
        return RateAdjustedDecorator(
            rate,
            ActivationReplacementDecorator(LeakyGradReLU()),
            MixAdjustmentStrategy())

    def get_rate_adjusted_clamp_decorator(self, perceptron, rate=None):
        if rate is None:
            rate = self.get_rate("clamp_rate", perceptron)
        # Use the learnable variant so ``ClampTuner`` can drive
        # ``perceptron.log_clamp_ceiling`` with gradient descent (Phase B2).
        # When the parameter's ``requires_grad`` is False the behaviour is
        # identical to the old static ``ClampDecorator`` because
        # ``effective_clamp_ceiling()`` is just ``exp(frozen_value)``.
        return RateAdjustedDecorator(
            rate,
            LearnableClampDecorator(perceptron, torch.tensor(0.0)),
            MixAdjustmentStrategy())

    def get_shift_decorator(self, pipeline_config, perceptron, rate=None):
        if rate is None:
            rate = self.get_rate("shift_rate", perceptron)
        shift_amount = calculate_activation_shift(pipeline_config["target_tq"], perceptron.activation_scale) * rate
        return ShiftDecorator(shift_amount)

    def get_rate_adjusted_quantization_decorator(
        self, pipeline_config, perceptron, quant_rate=None, shift_rate=None
    ):
        if quant_rate is None:
            quant_rate = self.get_rate("quantization_rate", perceptron)
        if shift_rate is None:
            shift_rate = self.get_rate("shift_rate", perceptron)

        # For TTFS: shift the other way — use shift_back = -shift so ReLU sees (x + shift) → staircase(ReLU(x + shift)).
        # For rate: shift_back = -shift * shift_rate to undo the outer shift.
        use_ttfs = pipeline_config.get("spiking_mode", "rate") in ("ttfs", "ttfs_quantized")
        shift = calculate_activation_shift(
            pipeline_config["target_tq"], perceptron.activation_scale
        )
        if use_ttfs:
            shift_back_amount = -shift  # ReLU gets (x - (-shift)) = (x + shift)
        else:
            shift_back_amount = -shift * shift_rate

        # ``quant_rate`` is a binary gate here: any non-zero value installs
        # the hard ``StaircaseFunction``-backed ``QuantizeDecorator``
        # (bit-exact with the post-mapping numeric path).  Cycle-by-cycle
        # rollout is driven externally by ``ActivationQuantizationTuner``
        # via per-perceptron overrides -- the decorator itself has no
        # smooth-annealing mode.  ``update_activation`` short-circuits the
        # ``quant_rate == 0`` case before it ever calls this factory.
        target_tq = torch.tensor(pipeline_config["target_tq"])
        return NestedDecoration([
            ShiftDecorator(shift_back_amount),
            QuantizeDecorator(target_tq, perceptron.activation_scale),
        ])
