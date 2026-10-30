"""The STOCK spec is a vendored passthrough: generation must reproduce the tree."""

from __future__ import annotations

from typing import Dict, Tuple

from mimarsinan.chip_simulation.soma_axes import (
    MEMBRANE_BITS_KEY,
    PER_AXON_SIGN,
    PER_EVENT_FIRING,
    WEIGHT_SIGN_GRANULARITY_KEY,
)
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.export.odin_gen.spec import CoreSpec
from mimarsinan.mapping.export.odin_gen.templates import HW_VENDOR_ROOT
from mimarsinan.mapping.platform.imc_platforms import IMCPlatform

#: The name of the backend's stock-core declaration below.
STOCK_PLATFORM_NAME = "odin_stock_fabric"

_STOCK_PROVENANCE = (
    'frenkel2019odin: "ODIN is based on a single 256-neuron 64k-synapse crossbar '
    'neurosynaptic core ... each synapse occupies 4 bits" (Trans. BioCAS 13(1), '
    "pp.145-158; upstream doc/README.md Sec.1 and Sec.3.2), transcribed from the "
    "RTL at ChFrenkel/ODIN @ 1781931. The DECLARED max_axons is the LOGICAL twin "
    "of the quoted 256 physical rows: weight_sign_granularity='per_axon' spends "
    "an excitatory/inhibitory row pair per logical slot (SPI_SYN_SIGN, "
    "src/synaptic_core.v:128), so 128 logical x 2 = the 256 quoted rows and "
    "128 x 256 x 2 = the quoted 64k synapses."
)

#: [U-H] The FPGA backend's declaration of the vendored stock core: logical slots, the RTL's 8-bit membrane. Held
#: here, not in the IMC registry, which on the release head is its curated literature and synthetic table; the
#: release head's chip description of ODIN for mapping and cost is the target bundle ``targets/odin.json`` (geometry
#: ``odin_stock_core``: 256 physical rows, per-axon sign on the core type), and a test pins both to one physical core.
STOCK_FABRIC = IMCPlatform(
    name=STOCK_PLATFORM_NAME,
    cores=({"max_axons": 128, "max_neurons": 256, "count": 1, "has_bias": False},),
    weight_bits=4,
    provenance=_STOCK_PROVENANCE,
    capabilities={
        WEIGHT_SIGN_GRANULARITY_KEY: PER_AXON_SIGN,
        MEMBRANE_BITS_KEY: 8,
        # No inter-core partial-sum transfer and no re-thresholding of a
        # partial sum: a per-event soma law makes both count-changing.
        "allow_coalescing": False,
        "allow_neuron_splitting": False,
        "allow_scheduling": False,
    },
    # Population 1 is the degenerate single-core case a curated table quarantines,
    # and the geometry is a conditioned LOGICAL model of the quoted crossbar.
    claim_eligibility="quarantined",
).validate()

VENDOR_SRC = HW_VENDOR_ROOT / "src"


class StockPassthroughError(ValueError):
    """Generating the stock spec did not reproduce the vendored tree."""


def vendored_file_set() -> Tuple[Tuple[str, bytes], ...]:
    """The vendored RTL as ``(relative path, BYTES)``, in a stable order.

    Bytes, not text: the upstream headers are latin-1 and byte identity is the
    claim being made. Read from ``hw/vendor/odin`` and never rewritten -- the
    OUT-OF-THE-BOX path uses these files directly, so a passthrough that
    produced an equal copy of something else would still be the wrong thing to
    deploy.
    """
    if not VENDOR_SRC.is_dir():
        raise StockPassthroughError(
            f"no vendored ODIN tree at {VENDOR_SRC}; the stock core is a "
            f"vendored artifact, not a generated one.")
    return tuple(
        (str(path.relative_to(VENDOR_SRC)), path.read_bytes())
        for path in sorted(VENDOR_SRC.rglob("*.v"))
    )


def stock_core_spec() -> CoreSpec:
    """The stock ODIN core, PROJECTED from its platform declaration (``STOCK_FABRIC``)."""
    platform = STOCK_FABRIC
    capabilities: Dict[str, object] = dict(platform.capabilities or {})
    law = SomaLaw.resolve({
        "spiking_family": "lif", "spiking_variant": "streamed",
        "firing_mode": "Novena", "thresholding_mode": "<=",
        "firing_granularity": PER_EVENT_FIRING,
        MEMBRANE_BITS_KEY: capabilities[MEMBRANE_BITS_KEY],
    })
    return CoreSpec.project(
        platform.cores[0], soma_law=law, weight_bits=platform.weight_bits,
        weight_sign_granularity=str(capabilities[WEIGHT_SIGN_GRANULARITY_KEY]),
    )


def is_stock_spec(spec: CoreSpec) -> bool:
    """Whether this spec IS the vendored stock core rather than a variant."""
    return spec == stock_core_spec()


def assert_stock_passthrough(files: Tuple[Tuple[str, bytes], ...]) -> None:
    """Refuse anything but a byte-identical reproduction of the vendored tree."""
    expected = vendored_file_set()
    got = dict(files)
    want = dict(expected)
    if sorted(got) != sorted(want):
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        raise StockPassthroughError(
            f"the stock spec must generate exactly the vendored file set; "
            f"missing={missing} extra={extra}")
    differing = [name for name in sorted(want) if got[name] != want[name]]
    if differing:
        raise StockPassthroughError(
            f"the stock spec generated files that differ from the vendored "
            f"tree: {', '.join(differing)}. The out-of-the-box path deploys the "
            f"vendored core itself, so a generator that rewrites it would make "
            f"'stock ODIN' mean two different things.")
