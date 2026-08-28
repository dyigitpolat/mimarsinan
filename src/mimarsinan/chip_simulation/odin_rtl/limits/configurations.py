"""The five configurations the compile-limits study measures, and their memories.

One `SynthesisTarget` per configuration, all through the SAME P5.5a yosys driver
and the SAME `stat`-census method: the stock core, the three cosim-proven
GENERATED variants, and the Vitis kernel wrapper around one core. The stock row
is not re-measured here -- it is READ from `hw/fpga/synth_resources.json`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from mimarsinan.chip_simulation.odin_rtl.synthesis import (
    SynthesisTarget,
    vendored_target,
)
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    HW_ROOT,
    kernel_sources,
    materialize_sources,
    overlay_sources,
    vendor_sources,
)
from mimarsinan.mapping.export.odin_gen.generate import generate_core
from mimarsinan.mapping.export.odin_gen.render import CORE_MODULE
from mimarsinan.mapping.export.odin_gen.variants import PROVEN_VARIANTS, NamedVariant

STOCK_KEY = "stock_a256n256_vendored"
WRAPPER_TOP = "odin_fpga_kernel_top"
WRAPPER_RTL = HW_ROOT / "fpga" / "kernel" / "odin_fpga_kernel_top.v"

#: NOTHING IS SHRUNK ANY MORE. The wrapper holds no program RAM to shrink: the
#: op stream is streamed through an elastic FIFO whose depth buys latency
#: tolerance and not storage, so both wrapper memories below are synthesized at
#: the depth the kernel actually SHIPS with. The first capture depth IS the
#: wrapper's shipped `CAP_WORDS`; the second is twice it, which makes the
#: per-capture-word cost a MEASUREMENT rather than an inference.
WRAPPER_FIFO_WORDS = 1024
WRAPPER_CAP_WORDS: Tuple[int, ...] = (16384, 32768)

_PARAM_DEFAULT = "parameter {name}\\s*=\\s*(?:NC\\s*\\*\\s*)?(\\d+)"


@dataclass(frozen=True)
class Configuration:
    """One measured point: what was synthesized, and the memories it declares."""

    key: str
    label: str
    kind: str
    target: SynthesisTarget
    geometry: Dict[str, Any]
    memories: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def memory_bits(self) -> int:
        return sum(int(memory["bits"]) for memory in self.memories)

    def as_record(self) -> Dict[str, Any]:
        return {
            "key": self.key, "label": self.label, "kind": self.kind,
            "top": self.target.top,
            "parameters": dict(sorted(self.target.parameters.items())),
            "geometry": dict(self.geometry),
            "memories": [dict(memory) for memory in self.memories],
            "memory_bits": self.memory_bits,
        }


def wrapper_shipped_depths() -> Dict[str, int]:
    """The `FIFO_WORDS`/`CAP_WORDS` the wrapper SHIPS with, read from its RTL."""
    text = WRAPPER_RTL.read_text(encoding="utf-8")
    depths: Dict[str, int] = {}
    for name in ("FIFO_WORDS", "CAP_WORDS"):
        match = re.search(_PARAM_DEFAULT.format(name=name), text)
        if match is None:
            raise ValueError(
                f"no default for `{name}` in {WRAPPER_RTL}; the shipped depth is "
                f"the RTL's, and this study will not assume one")
        depths[name] = int(match.group(1))
    return depths


def stock_configuration() -> Configuration:
    """The vendored 256x256 core: the P5.5a row, reused rather than re-measured."""
    return Configuration(
        key=STOCK_KEY,
        label="stock ODIN core, 256 axons x 256 neurons (vendored + BRAM overlay)",
        kind="vendored",
        target=vendored_target(overlay=True),
        geometry={
            "max_axons": 256, "max_neurons": 256, "weight_bits": 4,
            "membrane_bits": 8, "law": "stock ODIN LIF (SPI-programmed)",
        },
        memories=(
            {"array": "SRAM_256x128_wrapper (neuron)", "words": 256, "width": 128,
             "bits": 256 * 128},
            {"array": "SRAM_8192x32_wrapper (synapse)", "words": 8192, "width": 32,
             "bits": 8192 * 32},
        ),
    )


def variant_configuration(variant: NamedVariant) -> Configuration:
    """One GENERATED variant, materialised through the generator's own writer."""
    core = generate_core(variant.spec)
    sources = materialize_sources(core.files, key=variant.spec.spec_key())
    spec = variant.spec
    return Configuration(
        key=variant.name,
        label=(f"generated core, {spec.max_axons} axons x {spec.max_neurons} "
               f"neurons, {spec.membrane_bits}-bit "
               f"{'signed' if spec.membrane_signed else 'unsigned'} membrane, "
               f"{'per-event' if spec.per_event else 'sync-fire'} law"),
        kind="generated",
        target=SynthesisTarget(
            label=variant.name, top=CORE_MODULE, sources=tuple(sources)),
        geometry={
            "max_axons": spec.max_axons, "max_neurons": spec.max_neurons,
            "weight_bits": spec.weight_bits, "membrane_bits": spec.membrane_bits,
            "membrane_signed": spec.membrane_signed, "per_event": spec.per_event,
            "spec_key": spec.spec_key(), "proven_by": variant.proven_by,
        },
        memories=variant.memory_shapes(),
    )


def wrapper_configuration(cap_words: int) -> Configuration:
    """The Vitis wrapper around ONE stock core, at a stated capture depth."""
    shipped = wrapper_shipped_depths()
    return Configuration(
        key=f"wrapper_nc1_fifo{WRAPPER_FIFO_WORDS}_cap{cap_words}",
        label=(f"kernel wrapper `{WRAPPER_TOP}` at NC=1 "
               f"(FIFO_WORDS={WRAPPER_FIFO_WORDS}, CAP_WORDS={cap_words}) "
               f"-- sequencer + AXI streaming DMA + capture + one stock core"),
        kind="wrapper",
        target=SynthesisTarget(
            label=f"{WRAPPER_TOP} NC=1 cap={cap_words}", top=WRAPPER_TOP,
            sources=tuple(overlay_sources() + kernel_sources() + vendor_sources()),
            parameters={"FIFO_WORDS": WRAPPER_FIFO_WORDS, "CAP_WORDS": cap_words},
        ),
        geometry={
            "n_cores": 1, "fifo_words": WRAPPER_FIFO_WORDS, "cap_words": cap_words,
            "shipped_fifo_words": shipped["FIFO_WORDS"],
            "shipped_cap_words": shipped["CAP_WORDS"],
        },
        memories=(
            {"array": "fifo_ram", "words": WRAPPER_FIFO_WORDS, "width": 32,
             "bits": WRAPPER_FIFO_WORDS * 32},
            {"array": "cap_ram", "words": cap_words, "width": 32,
             "bits": cap_words * 32},
        ),
    )


def configurations() -> Tuple[Configuration, ...]:
    """Every measured point of the study, stock first."""
    return (
        (stock_configuration(),)
        + tuple(variant_configuration(v) for v in PROVEN_VARIANTS)
        + tuple(wrapper_configuration(cap) for cap in WRAPPER_CAP_WORDS)
    )


def configuration_named(key: str) -> Configuration:
    """The configuration of ``key``, refusing one the study does not carry."""
    for configuration in configurations():
        if configuration.key == key:
            return configuration
    raise KeyError(
        f"{key!r} is not a studied configuration; the study carries "
        f"{', '.join(c.key for c in configurations())}")
