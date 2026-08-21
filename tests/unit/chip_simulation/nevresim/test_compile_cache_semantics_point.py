"""[ODIN P3] the compile cache discriminates the soma point — and only it.

The cache serves a compiled binary on the policy hash ALONE. An axis that
changes the emitted C++ without changing the key hands a caller a stale binary
of different physics with no diagnostic — the highest-severity failure on this
path. The soma point therefore rides in the payload as a ``semantics_point``
sub-object, and it is OMITTED entirely at the default point so no existing
cache entry is invalidated.
"""

import pytest

from mimarsinan.chip_simulation.nevresim.compile_cache import cache_key, policy_hash
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw

_POLICY = dict(
    spiking_mode="lif",
    spike_generation_mode="Uniform",
    firing_mode="Novena",
    thresholding_mode="<=",
    weight_type_name="int",
    threshold_type_name="int",
    simulation_length=8,
    latency=2,
    connectivity_mode="runtime",
)

# Computed from the NINE-key payload as it stood before the soma point existed.
# A change to this literal is a cache-wide invalidation of every pre-P3 build.
_PRE_AXES_HASH = "f7f9f9e217277d4512acf2da549fbe925fdafb20b69483c0f4bca99036ea86e5"


def _point(bits: int) -> SomaLaw:
    return SomaLaw.resolve({
        "spiking_family": "lif", "spiking_variant": "streamed",
        "firing_mode": "Novena", "firing_granularity": "per_event",
        "membrane_bits": bits,
    })


class TestTheDefaultPointIsByteIdentical:
    def test_no_soma_law_reproduces_the_pre_axes_hash(self):
        assert policy_hash(**_POLICY) == _PRE_AXES_HASH

    def test_the_default_point_reproduces_the_pre_axes_hash(self):
        assert policy_hash(**_POLICY, soma_law=DEFAULT_SOMA_LAW) == _PRE_AXES_HASH

    def test_the_default_point_and_no_point_are_the_same_cache_entry(self):
        assert cache_key("m" * 32, policy_hash(**_POLICY)) == cache_key(
            "m" * 32, policy_hash(**_POLICY, soma_law=DEFAULT_SOMA_LAW))


class TestTheSomaPointDiscriminates:
    def test_a_per_event_point_is_a_different_key(self):
        assert policy_hash(**_POLICY, soma_law=_point(8)) != _PRE_AXES_HASH

    def test_membrane_bits_8_and_16_are_different_keys(self):
        """The register WIDTH is physics: the same fold saturates at 255 or at
        65535 and the two produce different counts, so they may never share a
        compiled binary."""
        assert policy_hash(**_POLICY, soma_law=_point(8)) != policy_hash(
            **_POLICY, soma_law=_point(16))

    def test_the_granularity_alone_is_a_different_key(self):
        saturating_only = SomaLaw.resolve({
            "spiking_family": "lif", "spiking_variant": "streamed",
            "membrane_bits": 8,
        })
        assert policy_hash(**_POLICY, soma_law=saturating_only) != policy_hash(
            **_POLICY, soma_law=_point(8))

    @pytest.mark.parametrize("bits", [8, 16])
    def test_the_full_cache_key_carries_the_discrimination(self, bits):
        assert cache_key("m" * 32, policy_hash(**_POLICY, soma_law=_point(bits))) != (
            cache_key("m" * 32, policy_hash(**_POLICY)))
