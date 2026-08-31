"""The chip-configuration table: its two copies agree, and the stock one is FROZEN.

A chip configuration is the FABRIC axis of the ODIN FPGA build. Three claims are
gated here:

  * the SHELL copy (``scripts/hacc/chips.sh``, which the packaged build reads
    because the package ships no ``src/``) says exactly what the Python SSOT
    says;
  * the STOCK configuration's RTL identity has not moved -- its source set is
    the same 24 files it always was, and the digest ``make_package.rtl_digest``
    computes over them is byte-stable, which is what keeps an already-routed
    bitstream findable in the chip cache;
  * the generated configuration's committed RTL is byte-identical to what the
    template emits today.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from mimarsinan.chip_simulation.odin_fpga.chip_configs import (
    STOCK_CHIP,
    WIDE_CHIP,
    ChipConfig,
    ChipConfigError,
    chip_config_named,
    chip_config_names,
    chip_configs,
)
from mimarsinan.chip_simulation.odin_fpga.kernel_registers import (
    SHIPPED_AXON_SLOTS_PER_CORE,
    SHIPPED_CAPTURE_EVENTS,
    SHIPPED_CHIP_CONFIG,
    SHIPPED_KERNEL_CORES,
    SHIPPED_NEURONS_PER_CORE,
    KernelCapacity,
)
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    REPO_ROOT,
    design_sources,
    kernel_sources,
)
from mimarsinan.mapping.export.odin_gen.fabric import (
    FabricError,
    generate_fabric,
    kernel_filename,
)

CHIPS_SH = REPO_ROOT / "scripts" / "hacc" / "chips.sh"

#: The digest `scripts/hacc/make_package.rtl_digest` computes over the STOCK
#: fabric's source set, recorded here as a constant so a change to any file in
#: that set is a RED TEST rather than a silently re-keyed chip cache. The routed
#: U55C artifact is keyed on this number; moving it orphans that artifact.
STOCK_RTL_DIGEST = "e40079099dad0db04dc7b09614b01ca55fa7792c844a03197eece4bc3c9d094d"

STOCK_RTL_FILE_COUNT = 24

_PROFILE_LINE = re.compile(r"^([a-z_]+)=(.*)$")


def _shell_profile(chip: str) -> dict:
    """One profile out of the shell table, through bash itself."""
    script = (
        f'source "{CHIPS_SH}"\n'
        f'odin_chip_profile {chip}\n'
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True,
        cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    fields = {}
    for line in result.stdout.splitlines():
        match = _PROFILE_LINE.match(line)
        assert match is not None, line
        fields[match.group(1)] = match.group(2)
    return fields


def _shell_sources(chip: str) -> list:
    script = f'source "{CHIPS_SH}"\nodin_chip_sources {chip}\n'
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True,
        cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line]


def _digest_of(paths) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(Path(path).relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(hashlib.sha256(
            Path(path).read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


class TestTheStockFabricIsFrozen:
    """The chip axis must not have moved the RTL identity of the routed fabric."""

    def test_the_stock_source_set_is_the_same_twenty_four_files(self):
        stock = chip_config_named(STOCK_CHIP)
        assert len(stock.rtl_sources()) == STOCK_RTL_FILE_COUNT
        assert set(stock.rtl_sources()) == set(
            design_sources(overlay=True) + kernel_sources())

    def test_the_stock_rtl_digest_is_byte_stable(self):
        assert _digest_of(chip_config_named(STOCK_CHIP).rtl_sources()) == \
            STOCK_RTL_DIGEST, (
                "the STOCK chip configuration's RTL digest moved. That digest is "
                "the `rtl_sha256` every published chip-cache entry was keyed on, "
                "so a place-and-routed xclbin just became unfindable. If the "
                "change is deliberate, re-adopt the artifact "
                "(scripts/chip_cache.sh adopt) and update this constant IN THE "
                "SAME commit that explains why.")

    def test_the_stock_fabric_generates_nothing(self):
        stock = chip_config_named(STOCK_CHIP)
        assert stock.generated_filenames() == ()
        with pytest.raises(ChipConfigError, match="VENDORED"):
            _ = stock.committed_rtl_root
        with pytest.raises(FabricError, match="vendored passthrough"):
            generate_fabric(stock.core_spec)

    def test_the_shipped_capacity_defaults_are_the_stock_configuration(self):
        """`kernel_registers` stays the SSOT for what a bitstream was built
        with; the chip table only names which fabric those numbers describe."""
        stock = chip_config_named(STOCK_CHIP)
        assert SHIPPED_CHIP_CONFIG == stock.name == STOCK_CHIP
        assert SHIPPED_NEURONS_PER_CORE == stock.max_neurons
        assert SHIPPED_AXON_SLOTS_PER_CORE == stock.max_axons
        default = KernelCapacity()
        assert default.as_dict() == stock.kernel_capacity().as_dict()
        assert (default.cores, default.capture_events) == (
            SHIPPED_KERNEL_CORES, SHIPPED_CAPTURE_EVENTS)


class TestTheWideConfigurationIsWhatTheMandateNeeds:
    def test_a_whole_mnist_raster_plus_the_bias_row_fits_one_core(self):
        wide = chip_config_named(WIDE_CHIP)
        assert wide.effective_max_axons >= 784
        assert wide.max_axons == 1024
        # ... which the stock fabric refuses outright.
        assert chip_config_named(STOCK_CHIP).effective_max_axons < 784

    def test_the_cell_and_the_threshold_are_both_past_the_stock_limits(self):
        stock = chip_config_named(STOCK_CHIP)
        wide = chip_config_named(WIDE_CHIP)
        assert stock.weight_range == (-7, 7)
        assert wide.weight_range == (-128, 127)
        assert stock.theta_ceiling == 255
        assert wide.theta_ceiling == 65535

    def test_one_logical_slot_costs_one_physical_row(self):
        """A per-synapse-signed cell signs itself, so 1024 declared slots ARE
        1024 crossbar rows -- the stock pair costs the other fabric half its
        declared width."""
        wide = chip_config_named(WIDE_CHIP)
        stock = chip_config_named(STOCK_CHIP)
        assert wide.physical_axon_rows == wide.max_axons
        assert stock.physical_axon_rows == 2 * stock.max_axons

    def test_the_fabric_is_the_shared_wrapper_plus_two_generated_files(self):
        wide = chip_config_named(WIDE_CHIP)
        names = [path.name for path in wide.rtl_sources()]
        assert names == [
            "odin_aer_bridge.v", "odin_fpga_kernel_top.v",
            "odin_gen_core.v", kernel_filename()]

    def test_the_committed_rtl_is_what_the_template_emits_today(self):
        wide = chip_config_named(WIDE_CHIP)
        fabric = generate_fabric(wide.core_spec)
        for name, payload in fabric.files:
            path = wide.committed_rtl_root / name
            assert path.is_file(), (
                f"{path} is missing; emit it with "
                f"scripts/hacc/gen_chip_rtl.py --chip {wide.name}")
            assert path.read_bytes() == payload, (
                f"{path} drifted from hw/gen/odin_gen_kernel.v.tmpl; the package "
                f"ships these bytes and the build node cannot re-expand them")

    def test_the_generated_kernel_declares_the_module_the_wrapper_binds(self):
        """The whole trick: the same module name and port list, a different body,
        so `odin_fpga_kernel_top.v` never has to learn about the variant."""
        wide = chip_config_named(WIDE_CHIP)
        text = (wide.committed_rtl_root / kernel_filename()).read_text()
        assert "module odin_fpga_kernel #(" in text
        top = (REPO_ROOT / "hw" / "fpga" / "kernel"
               / "odin_fpga_kernel_top.v").read_text()
        assert "odin_fpga_kernel #(" in top

    def test_the_capacity_it_declares_names_its_own_fabric(self):
        wide = chip_config_named(WIDE_CHIP)
        capacity = wide.kernel_capacity()
        assert capacity.chip == WIDE_CHIP
        assert capacity.neurons_per_core == 256
        assert capacity.axon_slots_per_core == 1024
        assert capacity.cores == SHIPPED_KERNEL_CORES


class TestTheShellCopyDoesNotDrift:
    """`scripts/hacc/chips.sh` is read by a package that has no `src/`."""

    def test_both_copies_carry_the_same_names(self):
        script = f'source "{CHIPS_SH}"\nodin_chip_names\n'
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True,
            cwd=str(REPO_ROOT))
        assert result.returncode == 0, result.stderr
        assert tuple(result.stdout.split()) == chip_config_names()

    def test_the_default_chip_is_the_stock_one(self):
        script = f'source "{CHIPS_SH}"\nprintf "%s" "${{ODIN_CHIP_DEFAULT}}"\n'
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True,
            cwd=str(REPO_ROOT))
        assert result.stdout == STOCK_CHIP

    @pytest.mark.parametrize("chip", chip_config_names())
    def test_every_field_of_every_profile_agrees(self, chip):
        config = chip_config_named(chip)
        shell = _shell_profile(chip)
        assert shell["chip"] == config.name
        assert shell["core_kind"] == config.core_kind
        assert shell["variant"] == (config.variant or "")
        assert int(shell["max_axons"]) == config.max_axons
        assert int(shell["max_neurons"]) == config.max_neurons
        assert int(shell["physical_axon_rows"]) == config.physical_axon_rows
        assert int(shell["effective_max_axons"]) == config.effective_max_axons
        assert int(shell["weight_bits"]) == int(config.core_spec.weight_bits)
        assert shell["weight_sign_granularity"] == str(
            config.core_spec.weight_sign_granularity)
        assert int(shell["membrane_bits"]) == int(config.core_spec.membrane_bits)
        assert int(shell["theta_ceiling"]) == config.theta_ceiling

    @pytest.mark.parametrize("chip", chip_config_names())
    def test_the_shell_source_list_is_the_python_one(self, chip):
        config = chip_config_named(chip)
        listed = [str(Path(p).resolve()) for p in _shell_sources(chip)]
        assert sorted(listed) == sorted(
            str(path.resolve()) for path in config.rtl_sources())

    def test_only_the_stock_build_directory_has_no_suffix(self):
        """The stock artifact must land where it always did; a second fabric must
        never overwrite it."""
        assert _shell_profile(STOCK_CHIP)["build_suffix"] == ""
        for chip in chip_config_names():
            if chip == STOCK_CHIP:
                continue
            assert _shell_profile(chip)["build_suffix"] == f"_{chip}"

    def test_an_unknown_chip_is_refused_by_both_copies(self):
        with pytest.raises(ChipConfigError, match="not a chip configuration"):
            chip_config_named("odin_imaginary")
        script = f'source "{CHIPS_SH}"\nodin_chip_profile odin_imaginary\n'
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True,
            cwd=str(REPO_ROOT))
        assert result.returncode != 0


class TestABundleAndABitstreamCanBeCompared:
    """One equality, so a bundle mapped for one fabric cannot run on the other."""

    def test_the_committed_bundle_is_a_stock_fabric_bundle(self):
        bundle = json.loads(
            (REPO_ROOT / "scripts" / "hacc" / "package" / "deployment"
             / "nc1_two_core_passes.json").read_text())
        claims = ChipConfig.claims_of_bundle(bundle["chip_config"])
        assert claims == chip_config_named(STOCK_CHIP).bundle_claims()
        assert claims != chip_config_named(WIDE_CHIP).bundle_claims()

    def test_no_two_fabrics_make_the_same_claims(self):
        seen = [tuple(sorted(c.bundle_claims().items())) for c in chip_configs()]
        assert len(set(seen)) == len(seen), (
            "two chip configurations are indistinguishable from a bundle's own "
            "chip_config block, so a bundle could be run on the wrong fabric")

    def test_the_claims_are_the_bundle_s_own_field_names(self):
        """The reader takes a bundle's block verbatim; a renamed field must be a
        KeyError here rather than a silently absent comparison."""
        wide = chip_config_named(WIDE_CHIP)
        block = {
            "weight_bits": 8, "weight_sign_granularity": "per_synapse",
            "effective_max_axons": 1023, "soma_law": {"membrane_bits": 16},
        }
        assert ChipConfig.claims_of_bundle(block) == wide.bundle_claims()
        with pytest.raises(KeyError):
            ChipConfig.claims_of_bundle({k: v for k, v in block.items()
                                         if k != "weight_bits"})


class TestTheCountCurrencyIsNotAGeometryLimit:
    """[ODIN C4] The currency is a REPRESENTATION, and which one is the CHIP's.

    A per-window spike count carries no field on the wire at all -- multiplicity
    is k adjacent AER transactions -- so the crossbar's width cannot and does
    not move it, which is what C2 proved and what stays true here. What a chip
    DOES decide is the width its implementations instantiate its law at, and the
    count travels in a signed word of exactly that width: nevresim's
    ``EventSerialIntegrate<bits>``, the torch fold's assertion and the
    exporter's propagated bound all take it from ``count_ceiling``.
    """

    def test_the_default_and_the_stock_fabric_still_carry_127(self):
        from mimarsinan.mapping.export.odin.feasibility import EMISSION_CEILING
        from mimarsinan.models.spiking.serial.refusals import (
            EMISSION_COUNT_CEILING,
            count_ceiling,
        )
        assert EMISSION_CEILING == EMISSION_COUNT_CEILING == 127
        assert count_ceiling(chip_config_named(STOCK_CHIP)) == 127

    def test_no_fabric_publishes_a_count_field_of_its_own(self):
        for config in chip_configs():
            assert "count" not in config.as_dict()
            # The membrane ceiling is the axis a wider register DOES move.
            assert config.theta_ceiling >= 255

    def test_the_wide_fabric_carries_its_own_declared_word(self):
        from mimarsinan.models.spiking.serial.refusals import count_ceiling

        wide = chip_config_named(WIDE_CHIP)
        assert wide.membrane_bits == 16
        assert count_ceiling(wide) == (1 << 15) - 1

    def test_the_currency_moves_only_with_the_register(self):
        """Every OTHER claim differs between the two fabrics too, so the test
        that isolates the register is the one that names it."""
        from mimarsinan.mapping.export.odin_gen.variants import (
            per_event_law,
            spec_for,
        )
        from mimarsinan.models.spiking.serial.refusals import count_ceiling

        wide_geometry_narrow_register = spec_for(
            per_event_law(8), axons=1024, neurons=256, weight_bits=8)
        assert count_ceiling(wide_geometry_narrow_register) == 127
