"""[U-H seam] The ODIN backends and the release head's pass-boundary rule (U1 W6).

The release head runs a streamed program VERBATIM on every backend: a pass boundary inside a segment replays the
producer's raster, and an enabled backend that cannot replay is refused by name rather than collapsing the run to
counts. The ODIN device and its HACC export buffer no raster across a pass boundary. They therefore run the verbatim
discipline exactly where a program carries NO raster across a pass boundary (there is nothing to buffer), and a
program that does carry one is refused by name at plan time and again at the device runner. The lineage instead let
an enabled ODIN backend downgrade the whole run to COLLAPSE, which the release head's rule forbids.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mimarsinan.models.spiking.hybrid.carry as carry
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, RASTER_FREE_BACKENDS, VERBATIM

_STREAMED = {"spiking_family": "lif", "spiking_variant": "streamed"}
_WINDOWED = {"spiking_family": "lif", "spiking_variant": "synchronized"}
_ODIN_KEYS = ("enable_odin_fpga_simulation", "enable_odin_hacc_export")


def _mapping(carried):
    mapping = SimpleNamespace(stages=[])
    mapping._carried_output_cache = carried
    return mapping


@pytest.mark.parametrize("key", _ODIN_KEYS)
def test_a_streamed_run_with_an_odin_backend_keeps_the_verbatim_discipline(key):
    assert carry.run_pass_transfer({**_STREAMED, key: True}) == VERBATIM


@pytest.mark.parametrize("key", _ODIN_KEYS)
def test_a_windowed_run_with_an_odin_backend_collapses_by_semantics(key):
    assert carry.run_pass_transfer({**_WINDOWED, key: True}) == COLLAPSE


@pytest.mark.parametrize("backend", ["odin_fpga", "odin_hacc"])
def test_the_odin_backends_are_raster_free_not_verbatim_replayers(backend):
    assert backend in RASTER_FREE_BACKENDS
    assert backend not in carry.VERBATIM_BACKENDS
    # Not a run-level backend of the U1 W6 rule (whose every member must replay in core-local time, W33): the run
    # keeps its discipline, and the program-level refusal below is the raster-free backends' own.
    assert backend not in carry._BACKEND_ENABLE_KEYS


@pytest.mark.parametrize("key,backend", [("enable_odin_fpga_simulation", "odin_fpga"),
                                         ("enable_odin_hacc_export", "odin_hacc")])
def test_a_program_that_carries_a_raster_is_refused_by_name_on_an_odin_backend(key, backend):
    with pytest.raises(carry.PassReplayRefusal, match=backend):
        carry.require_replayable_program(_mapping({0: (7, 8)}), {**_STREAMED, key: True}, VERBATIM)


@pytest.mark.parametrize("key", _ODIN_KEYS)
def test_a_program_that_carries_nothing_is_admitted(key):
    carry.require_replayable_program(_mapping({0: ()}), {**_STREAMED, key: True}, VERBATIM)
    carry.require_replayable_program(_mapping({}), {**_STREAMED, key: True}, VERBATIM)


def test_a_collapsed_run_carries_no_raster_and_is_admitted():
    carry.require_replayable_program(_mapping({0: (7,)}), {**_WINDOWED, "enable_odin_fpga_simulation": True},
                                      COLLAPSE)


def test_without_an_odin_backend_nothing_changes():
    carry.require_replayable_program(_mapping({0: (7,)}), dict(_STREAMED), VERBATIM)


class TestTheDeviceRunnerRefusesACarriedRaster:
    def _runner(self, carried, transfer):
        from mimarsinan.chip_simulation.odin_fpga.runner import OdinFpgaRunner

        contract = SimpleNamespace(soma_law=lambda: DEFAULT_SOMA_LAW, spiking_mode="lif")
        return OdinFpgaRunner(_mapping(carried), 4, contract=contract, transport=SimpleNamespace(name="fake"),
                              weight_bits=4, effective_max_axons=127, max_neurons=256,
                              pass_transfer=transfer)

    def test_verbatim_without_a_carried_raster_constructs(self):
        assert self._runner({}, VERBATIM).pass_transfer == VERBATIM

    def test_verbatim_with_a_carried_raster_is_refused_by_name(self):
        with pytest.raises(carry.PassReplayRefusal, match="odin_fpga"):
            self._runner({1: (3,)}, VERBATIM)

    def test_collapse_constructs(self):
        assert self._runner({1: (3,)}, COLLAPSE).pass_transfer == COLLAPSE
