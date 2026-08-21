"""The STOCK spec is a vendored passthrough: generation must reproduce the tree."""

from __future__ import annotations

from typing import Dict, Tuple

from mimarsinan.chip_simulation.soma_axes import (
    MEMBRANE_BITS_KEY,
    PER_EVENT_FIRING,
    WEIGHT_SIGN_GRANULARITY_KEY,
)
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.export.odin_gen.spec import CoreSpec
from mimarsinan.mapping.export.odin_gen.templates import HW_VENDOR_ROOT
from mimarsinan.mapping.platform.imc_platforms import get_imc_platform

#: The registered platform whose declaration the stock spec projects.
STOCK_PLATFORM_NAME = "odin_stock_core"

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
    """The stock ODIN core, PROJECTED from its registered platform declaration."""
    platform = get_imc_platform(STOCK_PLATFORM_NAME)
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
