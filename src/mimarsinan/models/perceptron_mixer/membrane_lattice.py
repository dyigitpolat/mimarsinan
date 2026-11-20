"""[U-POST G2, G4; U-POST2] The integer membrane lattice of a perceptron: its recorded kind, the lattice derived from
the persisted parameters, the arming, and the read plane of an integer program.

A perceptron's membrane lives on the integer chip lattice only where its weight projection put it there. The
projection records the lattice's KIND on the perceptron (``membrane_lattice_kind``: a core; an encoder on the window's
1/T entry grid; none, as on a host-side encoder reading a float input) and nothing else. The lattice itself is DERIVED
at every arming from the kind and the perceptron's persisted ``parameter_scale``: theta = s on a core, s T on an encoder
on the entry grid, none on a fractional scale; [U-PCH] the scale is read through the one lattice reading
(``integral_lattice``: one grid per perceptron, or a ``NeuronLattice`` per output neuron where the scale holds one entry
per neuron). The LIF node's armed lattice is written by the arming alone, never by the
projection, so it is never a cache a restore can leave behind: every state-dictionary restore returns the scale, and
the next arming reads it (the default 1.0 of an unprojected perceptron arms nothing, since no kind is recorded).

An integer program (every LIF perceptron on cores carries a lattice) has its published reads inside the measurement
plane, the parity gate's read: an exact threshold tie is decided by its exact value on every evaluator.

Low in the import graph on purpose (the pipeline step base reads it at module scope): the LIF node is found by the
structural walk of ``spiking.lif_utils.unwrap_lif_activation`` (a transformed activation's base, a blend's LIF node).
"""

from __future__ import annotations

import contextlib

from mimarsinan.models.nn.activations import ChipInputQuantizer, LIFActivation
from mimarsinan.models.nn.layers import TransformedActivation
from mimarsinan.models.nn.lif_kernels import NeuronLattice, integral_lattice, measurement_plane
from mimarsinan.models.perceptron_mixer.perceptron import placed_on_cores

__all__ = [
    "CORE_LATTICE",
    "ENTRY_GRID_LATTICE",
    "arm_membrane_lattices",
    "input_on_entry_grid",
    "integer_membrane_program",
    "integer_program_read",
    "lif_node",
    "membrane_lattice_theta",
    "record_membrane_lattice_kind",
]

#: A core's membrane integrates spike-driven charges on the 1/s grid: lattice theta = s.
CORE_LATTICE = "core"
#: An encoding layer whose input passes the window's 1/T entry grid integrates on the 1/(s T) grid: theta = s T.
ENTRY_GRID_LATTICE = "entry_grid_encoder"


def lif_node(activation) -> LIFActivation | None:
    """The LIF node an activation wraps (a transformed activation's base, a blend's ``lif_activation``), or None."""
    for _ in range(8):
        if activation is None or isinstance(activation, LIFActivation):
            return activation
        attr = "base_activation" if isinstance(activation, TransformedActivation) else "lif_activation"
        activation = getattr(activation, attr, None)
    return None


def input_on_entry_grid(perceptron, simulation_steps: int) -> bool:
    """Whether the perceptron's input passes the window's 1/T entry grid (an entry quantizer at ``T`` on its wire)."""
    wire = getattr(perceptron, "input_activation", None)
    return wire is not None and any(
        isinstance(m, ChipInputQuantizer) and int(m.T) == int(simulation_steps) for m in wire.modules()
    )


def record_membrane_lattice_kind(perceptron) -> str | None:
    """The weight projection's record: the kind of lattice its projection puts the perceptron's membrane on (none
    without a LIF node, and on an encoding layer whose input is not on the entry grid), written on the perceptron.
    Arms nothing. Returns the kind."""
    lif = lif_node(getattr(perceptron, "activation", None))
    if lif is None:
        kind = None
    elif getattr(perceptron, "is_encoding_layer", False):
        kind = ENTRY_GRID_LATTICE if input_on_entry_grid(perceptron, lif.T) else None
    else:
        kind = CORE_LATTICE
    perceptron.membrane_lattice_kind = kind
    return kind


def membrane_lattice_theta(perceptron) -> float | NeuronLattice | None:
    """The perceptron's membrane lattice in theta units, derived from its recorded kind and its persisted
    ``parameter_scale`` (read through ``integral_lattice``: one grid per perceptron, or one per output neuron along
    the perceptron's ``output_channel_axis``); None where the membrane lives on no lattice (no kind, no LIF node, a
    fractional scale)."""
    kind = getattr(perceptron, "membrane_lattice_kind", None)
    lif = lif_node(getattr(perceptron, "activation", None))
    scale = getattr(perceptron, "parameter_scale", None)
    if kind is None or lif is None or scale is None:
        return None
    lattice = integral_lattice(scale, int(getattr(perceptron, "output_channel_axis", -1)))
    factor = float(lif.T) if kind == ENTRY_GRID_LATTICE else 1.0
    if isinstance(lattice, NeuronLattice):
        return lattice.times(factor) if factor != 1.0 else lattice
    return None if lattice is None else float(lattice) * factor


def arm_membrane_lattices(model) -> int:
    """Arm every LIF perceptron's membrane snap at the lattice derived from its persisted parameters (a perceptron with
    none is disarmed). Idempotent; returns the perceptrons armed."""
    armed = 0
    for perceptron in model.get_perceptrons():
        lif = lif_node(getattr(perceptron, "activation", None))
        if lif is None:
            continue
        theta = membrane_lattice_theta(perceptron)
        lif.set_membrane_lattice(theta)
        armed += theta is not None
    return armed


def integer_membrane_program(model) -> bool:
    """Whether ``model`` is an integer program: it has LIF perceptrons on cores, and every one of them carries a lattice
    (host-placed perceptrons run in float and are not asked)."""
    perceptrons = getattr(model, "get_perceptrons", None)
    if perceptrons is None:
        return False
    cores = [p for p in perceptrons() if placed_on_cores(p) and lif_node(getattr(p, "activation", None)) is not None]
    return bool(cores) and all(membrane_lattice_theta(p) is not None for p in cores)


@contextlib.contextmanager
def integer_program_read(model):
    """The plane of a published read. An integer program reads inside the measurement plane with its derived lattices
    armed (the parity gate's read, the deployed program's tie rule); any other model reads as it is, and the tuning
    plane is untouched. Every LIF node's arming is restored after the read. Yields whether the plane was entered."""
    if not integer_membrane_program(model):
        yield False
        return
    nodes = [lif for lif in (lif_node(getattr(p, "activation", None)) for p in model.get_perceptrons())
             if lif is not None]
    saved = [(lif, lif.if_node.lattice_scale) for lif in nodes]
    arm_membrane_lattices(model)
    try:
        with measurement_plane():
            yield True
    finally:
        for lif, scale in saved:
            lif.if_node.lattice_scale = scale
