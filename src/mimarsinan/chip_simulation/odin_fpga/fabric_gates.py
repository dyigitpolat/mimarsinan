"""WHICH GATES a segment meets: the fabric its declared envelope IDENTIFIES.

A row signed once by ``SPI_SYN_SIGN`` is the vendored crossbar, whose gates are
its exporter's and whose product is a programmable image. A synapse cell that
signs itself is a generated core, which has no vendored image to emit — so its
gates run without one. The COUNT CURRENCY runs on both: it is what a segment
boundary carries, not what a core is, and no crossbar width lifts it.
"""

from __future__ import annotations

from mimarsinan.chip_simulation.odin_fpga.chip_selection import chip_config_for
from mimarsinan.chip_simulation.soma_axes import PER_SYNAPSE_SIGN
from mimarsinan.mapping.export.odin.exporter import export_odin
from mimarsinan.mapping.export.odin_gen.feasibility import gate_variant_segment


def gate_segment(hcm, *, soma_law, weight_bits, effective_max_axons,
                  membrane_init, weight_sign_granularity, cycles):
    """Run the gates of the fabric the declared envelope IDENTIFIES.

    A row signed once by ``SPI_SYN_SIGN`` is the vendored crossbar, whose gates
    are its exporter's; a synapse cell that signs itself is a generated core,
    which has no vendored image to emit — so its gates run without one and the
    count currency, which neither fabric lifts, runs in both.
    """
    if str(weight_sign_granularity) == PER_SYNAPSE_SIGN:
        chip = chip_config_for(
            weight_bits=int(weight_bits),
            weight_sign_granularity=str(weight_sign_granularity),
            effective_max_axons=int(effective_max_axons),
            membrane_bits=int(soma_law.membrane_bits))
        return None, gate_variant_segment(
            hcm, spec=chip.core_spec, membrane_init=int(membrane_init),
            cycles=int(cycles))
    return export_odin(
        hcm, soma_law=soma_law, weight_bits=int(weight_bits),
        weight_sign_granularity=str(weight_sign_granularity),
        effective_max_axons=int(effective_max_axons),
        membrane_init=int(membrane_init)), None


def require_programmable_image(export, *, stage: str, granularity: str) -> None:
    """A live transport programs a VENDORED image; a generated fabric has none."""
    if export is not None:
        return
    raise ValueError(
        f"stage {stage!r} is mapped against a GENERATED fabric "
        f"({granularity!r}), which has no vendored image for a transport to "
        f"program. Its gates ran and its deployment path is the frozen bundle "
        f"(chip_simulation/odin_hacc); wire a variant transport before asking "
        f"this backend to run one live")
