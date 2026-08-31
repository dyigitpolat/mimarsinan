"""Named CHIP CONFIGURATIONS: which fabric a bitstream is built from, and what it holds.

A chip configuration is the FABRIC axis, exactly as ``scripts/hacc/cards.sh`` is
the card axis: one name selects the core the kernel instantiates, the RTL source
set the build compiles, and the capacities the host declares. Nothing
chip-shaped is written down anywhere else -- the geometry is READ from the
``CoreSpec`` the cosimulation gates prove, and the capture depth is READ from
``kernel_registers``, so a configuration cannot claim a geometry no gate ran or
a capacity no bitstream was built at.

CROSS-LANGUAGE CONTRACT -- ``scripts/hacc/chips.sh`` is the shell copy of the
table below (the packaged build and cache scripts ship without ``src/``), and
``scripts/hacc/package/host/odin_board_driver.py`` is the host copy.
``scripts/hacc/make_package.py`` REFUSES to package when either has drifted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.chip_simulation.odin_fpga.kernel_registers import (
    SHIPPED_CAPTURE_EVENTS,
    SHIPPED_KERNEL_CORES,
    KernelCapacity,
)
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    HW_ROOT,
    design_sources,
    kernel_sources,
)
from mimarsinan.mapping.export.odin_gen.fabric import kernel_filename
from mimarsinan.mapping.export.odin_gen.passthrough import stock_core_spec
from mimarsinan.mapping.export.odin_gen.render import core_filename
from mimarsinan.mapping.export.odin_gen.spec import CoreSpec
from mimarsinan.mapping.export.odin_gen.variants import (
    WIDE_CHIP_VARIANT,
    variant_named,
)

STOCK_CHIP = "odin_stock_256x256"
WIDE_CHIP = "odin_wide_1024x256_mb16"

CORE_KIND_VENDORED = "vendored"
CORE_KIND_GENERATED = "generated"

KERNEL_DIR = HW_ROOT / "fpga" / "kernel"

#: Where a GENERATED configuration's emitted RTL is COMMITTED. It is committed
#: for the same reason `kernel.xml` is frozen into the package: the cluster build
#: runs from a package that ships no ``src/`` and cannot expand a template.
#: ``scripts/hacc/gen_chip_rtl.py`` writes these bytes and gates them.
CHIP_RTL_ROOT = HW_ROOT / "gen" / "chips"

#: The wrapper every configuration shares: the Vitis top, the AXI4 engine and
#: the AER bridge are the SAME files for both fabrics, which is what keeps
#: `kernel.xml`, the argument table and the host driver one implementation.
SHARED_KERNEL_FILES: Tuple[str, ...] = (
    "odin_aer_bridge.v", "odin_fpga_kernel_top.v")


class ChipConfigError(ValueError):
    """A chip configuration this build does not know."""


@dataclass(frozen=True)
class ChipConfig:
    """One fabric the build can produce, and what the host may declare of it."""

    name: str
    label: str
    core_kind: str
    core_spec: CoreSpec
    variant: Optional[str]
    cores: int = SHIPPED_KERNEL_CORES
    capture_events: int = SHIPPED_CAPTURE_EVENTS

    @property
    def is_stock(self) -> bool:
        return self.core_kind == CORE_KIND_VENDORED

    @property
    def max_axons(self) -> int:
        return int(self.core_spec.max_axons)

    @property
    def max_neurons(self) -> int:
        return int(self.core_spec.max_neurons)

    @property
    def physical_axon_rows(self) -> int:
        """The CROSSBAR's row count: one logical slot costs a per-axon-signed
        substrate an excitatory/inhibitory PAIR and a per-synapse one a row."""
        return self.max_axons * int(self.core_spec.physical_row_factor)

    @property
    def effective_max_axons(self) -> int:
        """The LOGICAL fan-in one core offers the mapper.

        The declared core type is already logical (the stock platform declares
        128 slots on its 256 physical rows), so the only deduction here is the
        always-on row the param-encoded bias occupies at the tail of the
        canonical slot order.
        """
        return self.max_axons - 1

    @property
    def weight_range(self) -> Tuple[int, int]:
        """The representable weight set of ONE synapse cell on this fabric."""
        bits = int(self.core_spec.weight_bits)
        if int(self.core_spec.physical_row_factor) == 1:
            return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        # An unsigned magnitude signed once per physical row is SYMMETRIC.
        return (-((1 << (bits - 1)) - 1), (1 << (bits - 1)) - 1)

    @property
    def membrane_bits(self) -> int:
        """The width this fabric declares its membrane register at.

        A CLAIM, named here because more than the geometry reads it: theta's
        ceiling, the bundle's fabric-match block, and the chip's COUNT CURRENCY
        (``models/spiking/serial/refusals.count_ceiling``) are all this number.
        """
        return int(self.core_spec.membrane_bits)

    @property
    def theta_ceiling(self) -> int:
        """theta shares the membrane register, so it shares its top value."""
        return (1 << self.membrane_bits) - 1

    def generated_filenames(self) -> Tuple[str, ...]:
        """The RTL this configuration EMITS; empty for the vendored fabric."""
        if self.is_stock:
            return ()
        return (core_filename(self.core_spec), kernel_filename())

    @property
    def committed_rtl_root(self) -> Path:
        """Where this configuration's emitted RTL is committed."""
        if self.is_stock:
            raise ChipConfigError(
                f"{self.name} is the VENDORED fabric: its kernel is the "
                f"committed hw/fpga/kernel tree and nothing is generated for it.")
        return CHIP_RTL_ROOT / self.name

    def rtl_sources(self, generated_root: Optional[Path] = None) -> Tuple[Path, ...]:
        """The compile order this configuration's fabric is built from.

        The stock fabric is the committed kernel tree plus the vendored design;
        a generated one is the shared wrapper plus the two files it emits, taken
        from ``generated_root`` (its committed location by default). A generated
        fabric compiles NEITHER the stock kernel body nor the vendored tree --
        the two declare the same module and are never built together.
        """
        if self.is_stock:
            return tuple(design_sources(overlay=True) + kernel_sources())
        root = Path(generated_root) if generated_root is not None \
            else self.committed_rtl_root
        return tuple(
            [KERNEL_DIR / name for name in SHARED_KERNEL_FILES]
            + [root / name for name in self.generated_filenames()])

    def kernel_capacity(self) -> KernelCapacity:
        """What a host may declare of a bitstream built from this configuration."""
        return KernelCapacity(
            cores=self.cores, capture_events=self.capture_events,
            chip=self.name, neurons_per_core=self.max_neurons,
            axon_slots_per_core=self.max_axons,
        )

    def bundle_claims(self) -> Dict[str, Any]:
        """The ``chip_config`` fields of a deployment bundle THIS fabric implies.

        A sealed bundle already declares the fabric-distinguishing half of its
        `chip_config` block (`odin_hacc/freeze.py`): the weight width, the sign
        granularity, the membrane width inside the soma law and the effective
        fan-in. This is the same subset read off the configuration, so a bundle
        and a bitstream can be compared in ONE equality instead of four
        hand-written ones -- and a bundle mapped against the stock geometry is
        detectably not runnable on a wide fabric, and vice versa.
        """
        return {
            "weight_bits": int(self.core_spec.weight_bits),
            "weight_sign_granularity": str(
                self.core_spec.weight_sign_granularity),
            "effective_max_axons": self.effective_max_axons,
            "membrane_bits": self.membrane_bits,
        }

    @staticmethod
    def claims_of_bundle(chip_config: Mapping[str, Any]) -> Dict[str, Any]:
        """The same subset, read out of a sealed bundle's own block."""
        return {
            "weight_bits": int(chip_config["weight_bits"]),
            "weight_sign_granularity": str(
                chip_config["weight_sign_granularity"]),
            "effective_max_axons": int(chip_config["effective_max_axons"]),
            "membrane_bits": int(chip_config["soma_law"]["membrane_bits"]),
        }

    def as_dict(self) -> Dict[str, Any]:
        """The chip identity a deployment bundle carries beside its semantics."""
        low, high = self.weight_range
        return {
            "chip": self.name,
            "label": self.label,
            "core_kind": self.core_kind,
            "variant": self.variant,
            "spec_key": self.core_spec.spec_key(),
            "cores": int(self.cores),
            "capture_events": int(self.capture_events),
            "max_axons": self.max_axons,
            "max_neurons": self.max_neurons,
            "physical_axon_rows": self.physical_axon_rows,
            "effective_max_axons": self.effective_max_axons,
            "weight_bits": int(self.core_spec.weight_bits),
            "weight_sign_granularity": str(self.core_spec.weight_sign_granularity),
            "weight_range": [low, high],
            "membrane_bits": self.membrane_bits,
            "theta_ceiling": self.theta_ceiling,
        }


def _stock_config() -> ChipConfig:
    return ChipConfig(
        name=STOCK_CHIP,
        label=("stock ODIN core, 128 logical slots on 256 physical rows x 256 "
               "neurons, SPI-programmed (the vendored tree; the only fabric "
               "that has been placed and routed)"),
        core_kind=CORE_KIND_VENDORED,
        core_spec=stock_core_spec(),
        variant=None,
    )


def _wide_config() -> ChipConfig:
    variant = variant_named(WIDE_CHIP_VARIANT)
    spec = variant.spec
    return ChipConfig(
        name=WIDE_CHIP,
        label=(f"generated core, {spec.max_axons} axons x {spec.max_neurons} "
               f"neurons, {spec.weight_bits}-bit signed synapse cell, "
               f"{spec.membrane_bits}-bit unsigned membrane, event-serial law "
               f"(a 784-line raster maps whole onto one core)"),
        core_kind=CORE_KIND_GENERATED,
        core_spec=spec,
        variant=variant.name,
    )


def chip_configs() -> Tuple[ChipConfig, ...]:
    """Every configuration the build knows, the stock fabric first."""
    return (_stock_config(), _wide_config())


def chip_config_named(name: str) -> ChipConfig:
    """The configuration of ``name``, refusing one the build does not carry."""
    for config in chip_configs():
        if config.name == name:
            return config
    raise ChipConfigError(
        f"{name!r} is not a chip configuration this build knows; it carries "
        f"{', '.join(c.name for c in chip_configs())}. Add a profile to "
        f"chip_configs.py and scripts/hacc/chips.sh rather than special-casing "
        f"a fabric anywhere else.")


def chip_config_names() -> Tuple[str, ...]:
    return tuple(config.name for config in chip_configs())
