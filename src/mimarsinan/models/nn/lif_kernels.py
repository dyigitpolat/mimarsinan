"""Shared LIF integrate-and-fire step (unified and hybrid core flows)."""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any

import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.models.spiking.wire_semantics import lif_window_count

_MEASUREMENT_PLANE = contextvars.ContextVar("lif_measurement_plane", default=False)


@contextlib.contextmanager
def measurement_plane():
    """The chip-lattice MEASUREMENT plane: inside, armed LIF nodes snap
    membranes to the integer-chip lattice so exact ties are decided by
    exact values (parity twins, certificates). Outside — training AND
    tuning-plane telemetry — the continuous membrane and the fused kernels
    are untouched: snapped telemetry deterministically steered keep-best /
    floor decisions onto different trajectories (t0_01 fresh 0.9771 vs
    0.9799, n8d 2026-08-10)."""
    token = _MEASUREMENT_PLANE.set(True)
    try:
        yield
    finally:
        _MEASUREMENT_PLANE.reset(token)


def in_measurement_plane() -> bool:
    return bool(_MEASUREMENT_PLANE.get())

_THRESHOLD_OPS = {"<": torch.lt, "<=": torch.le}


class MembraneRailTouchedError(ToolchainRefusal, ValueError):
    """A membrane reached a rail of a register whose law says it cannot.

    ``membrane_arithmetic='saturating_signed'`` claims to hold exactly the
    number the unbounded accumulator holds, which is true only strictly inside
    the register's interval. The deployment carries a static no-saturation
    bound proving the rails are unreachable, so touching one is a broken proof,
    not this law's physics — refused everywhere (torch, nevresim, RTL), never
    clamped, because a clamp reports a number the contract denies. [U-DOOR] A
    refusal by name (``ToolchainRefusal``): the program is refused, never read.
    """


def enforce_membrane_rails(
    memb: torch.Tensor, membrane_bounds: tuple[float, float]
) -> None:
    """Refuse a membrane that reached either rail of its declared register.

    Armed only where the law asserts the rails are unreachable
    (``SomaLaw.asserts_no_saturation``); a register whose saturation IS the
    modelled substrate never calls this. Deliberately conservative: a value
    that merely EQUALS a rail without having been clamped there also refuses,
    because from the membrane alone the two are indistinguishable and the
    honest answer is to widen the register rather than to guess.
    """
    low, high = membrane_bounds
    if not bool(torch.any((memb <= low) | (memb >= high))):
        return
    raise MembraneRailTouchedError(
        f"the membrane reached a rail of its declared register "
        f"[{low}, {high}]: this law holds the same number the unbounded "
        f"accumulator holds only strictly inside the interval, and the "
        f"deployment's no-saturation bound said the rails were unreachable. "
        f"Refused, never clamped — a clamp would report a number the "
        f"deployment's own contract denies. Widen membrane_bits, lower the "
        f"fan-in, or raise theta."
    )


#: [U-DOOR] The integer chip lattice's scale in chip units (a half-unit quantum covers the half-step init): the ONE
#: lattice a weight-quantized program's membrane lives on, whichever executor snaps onto it (the per-cycle policy, the
#: event-serial fold, and the NF's lattice node at ``2 * theta_int`` in its normalized units).
CHIP_LATTICE_SCALE = 2.0


class PerNeuronThresholdRefusal(ToolchainRefusal, ValueError):
    """[U-PCH] Per-neuron weight scales (one threshold per neuron) where the program, a consumer or the target holds
    one threshold per core or per perceptron, refused by name."""


class NeuronLattice:
    """[U-PCH] One membrane lattice per output neuron: ``scales[c]`` along the membrane's ``axis`` (its perceptron's
    declared ``output_channel_axis``), for a perceptron projected per neuron (``theta_c = s_c``): the NF's normalized
    membrane of neuron ``c`` lives on the ``1 / scales[c]`` grid."""

    def __init__(self, scales: torch.Tensor, axis: int) -> None:
        self.scales = torch.as_tensor(scales).detach().to(device="cpu", dtype=torch.float64).reshape(-1).clone()
        self.axis = int(axis)

    def times(self, factor: float) -> "NeuronLattice":
        return NeuronLattice(self.scales * float(factor), self.axis)

    def on(self, memb: torch.Tensor) -> torch.Tensor:
        """The scales shaped to broadcast against ``memb`` (its neurons along ``axis``), on its device and dtype."""
        axis = self.axis % memb.dim()
        if int(memb.shape[axis]) != self.scales.numel():
            raise PerNeuronThresholdRefusal(
                f"a per-neuron membrane lattice of {self.scales.numel()} neurons read on a membrane of shape "
                f"{tuple(memb.shape)} along axis {self.axis}: the membrane's neurons are not its perceptron's")
        shape = [1] * memb.dim()
        shape[axis] = self.scales.numel()
        return self.scales.to(device=memb.device, dtype=memb.dtype).reshape(shape)


def neuron_lattice(scale, axis: int) -> "NeuronLattice | None":
    """The per-neuron integer lattice of a per-neuron scale when every entry is a positive integer, else None."""
    values = torch.as_tensor(scale).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if values.numel() == 0 or bool((values <= 0).any()) or bool(((values - values.round()).abs() > 1e-6).any()):
        return None
    return NeuronLattice(values.round(), axis)


def integral_lattice(scale, axis: int = -1) -> "int | NeuronLattice | None":
    """[U-PCH2] The integer lattice a weight scale puts a membrane on, the one reading every arming takes: the integer
    of one grid per perceptron, a ``NeuronLattice`` of one grid per output neuron (along ``axis``), None for a scale
    off the positive integers. [U-PCH3] A perceptron's weight scale is declared per output neuron: one held entry is
    every neuron's (one grid, whatever its shape), and more entries are one per neuron (``NeuronLattice.on``
    refuses any other count by name); its size never selects another layout."""
    if torch.is_tensor(scale) and scale.numel() > 1:
        return neuron_lattice(scale, axis)
    value = float(scale.reshape(-1)[0].item() if torch.is_tensor(scale) else scale)
    return round(value) if value > 0 and abs(value - round(value)) <= 1e-6 else None


def membrane_lattice_scale(theta_int) -> "float | NeuronLattice | None":
    """The lattice an armed node snaps on: ``CHIP_LATTICE_SCALE * theta_int`` (per neuron for a ``NeuronLattice``),
    None to clear."""
    if isinstance(theta_int, NeuronLattice):
        return theta_int.times(CHIP_LATTICE_SCALE)
    return CHIP_LATTICE_SCALE * float(theta_int) if theta_int else None


def lattice_scalar(lattice_scale, consumer: str) -> float:
    """An armed lattice for a consumer that reads one threshold per perceptron; a per-neuron one is refused by name."""
    if isinstance(lattice_scale, NeuronLattice):
        raise PerNeuronThresholdRefusal(
            f"{consumer} reads one threshold per perceptron, but the perceptron's membrane lattice is per neuron "
            f"(weight_scale_granularity 'per_neuron'): declare weight_scale_granularity 'per_perceptron'")
    return float(lattice_scale)


def snap_membrane_to_lattice(memb: torch.Tensor, lattice_scale) -> None:
    """Project the membrane onto its exact arithmetic lattice, in place.

    On the integer chip every true membrane value is a multiple of the
    lattice quantum (half a chip unit covers the half-step init); float
    summation-order noise (~1e-7) pushed values ACROSS threshold ties —
    which the integer-theta lattice makes common (the 2026-08-09 NF↔SCM
    13/788 catch: a true tie fired on one twin and not the other). This is
    an EXACT projection, not a tolerance: noise ≪ half a quantum. A
    ``NeuronLattice`` [U-PCH] snaps each neuron on its own quantum."""
    scale = lattice_scale.on(memb) if isinstance(lattice_scale, NeuronLattice) else float(lattice_scale)
    memb.mul_(scale).round_().div_(scale)


def lif_fire_and_reset(
    memb: torch.Tensor,
    threshold: torch.Tensor,
    *,
    thresholding_mode: str,
    firing_mode: str,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Threshold ``memb``, return spike tensor, apply Novena/Default reset in-place.

    Masked arithmetic instead of boolean fancy indexing: ``memb[fired]`` lowers
    through ``nonzero`` and forces a host-device sync per call (the dominant
    cycle-loop wall cost); ``fired * threshold`` is bit-exact for finite
    thresholds since ``fired`` is exactly 0 or 1.
    """
    fired = _THRESHOLD_OPS[thresholding_mode](threshold, memb)
    if firing_mode == "Novena":
        memb.masked_fill_(fired, 0.0)
    elif firing_mode == "Default":
        fired_typed = fired.to(memb.dtype)
        memb.sub_(fired_typed * threshold)
        # The reset conversion doubles as the output when dtypes agree (the
        # hybrid cycle loop's hot path) — same values, one fewer kernel.
        if output_dtype == memb.dtype or (output_dtype is None and memb.dtype == torch.float32):
            return fired_typed
    if output_dtype is not None:
        return fired.to(output_dtype)
    return fired.float()


@contextlib.contextmanager
def window_counts(lif: Any):
    """[U-LIF2 W10] The training walk's count branch: evaluate ``lif`` (an ``LIFActivation``, when it is one) by its
    window count for the duration. A synchronized LIF hop that integrates then emits is read as ``lif_window_count`` of
    its window charge, never through the IF kernel (whose float accumulation order decides ties differently under its
    torch and compiled forms)."""
    if lif is None:
        yield
        return
    previous = lif._window_count_mode
    lif._window_count_mode = True
    try:
        yield
    finally:
        lif._window_count_mode = previous


def window_count_forward(lif: Any, x: torch.Tensor) -> torch.Tensor:
    """[U-LIF2 W10] The hop's window count times ``theta / T``: a value held for the window's T cycles charges
    ``T x / theta`` on the precharge ``v0`` (the membrane guard), and the measurement plane's chip lattice decides its
    ties as the IF kernel's snap does. Its gradient is the straight-through clamp of the charge to the window."""
    safe_scale = lif._safe_scale(x)
    charge = x / safe_scale * lif.T + float(getattr(lif, "membrane_init", 0.0))
    decided = charge.detach().clone()
    lattice = getattr(lif.if_node, "_snapped_scale", lambda: None)()
    if lattice is not None:
        snap_membrane_to_lattice(decided, float(lattice))
    count = lif_window_count(decided, 1.0, lif.T, compare_mode=lif.thresholding_mode)
    surrogate = charge.clamp(0.0, float(lif.T))
    return (surrogate + (count - surrogate).detach()) / lif.T * safe_scale
