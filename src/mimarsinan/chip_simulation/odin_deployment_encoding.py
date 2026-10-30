"""The AER-in WORDING of one fabric, dispatched on a bundle's own chip claims.

SHIPPED VERBATIM as ``host/odin_deployment_encoding.py`` beside
``odin_deployment_bundle.py``, which is why it imports nothing but the standard
library. It sits apart from the bundle module because the bundle's token
arithmetic is fabric-AGNOSTIC — it words an event through whatever encoding it
is handed — while this file is the one place that decides which wording a
fabric uses.

CROSS-LANGUAGE CONTRACT: the two wordings are mirrored from
``odin_rtl/stimulus.py`` (``neuron_spike_event`` / ``all_neuron_tref_event`` for
the vendored crossbar, ``variant_axon_event`` / ``variant_tref_event`` for a
generated core), and a unit gate holds the copies byte-equal over the same
vectors the P6 cosimulation gates drive.

WHY IT MUST DISPATCH: a bundle carries no fabric NAME, it carries the claims
that identify one (``ChipConfig.bundle_claims``). Reading a stock stream on a
generated core would deliver the all-neuron time reference 0x7F as a spike on
axon slot 127, and nothing downstream would say so.
"""

from __future__ import annotations

from typing import Any, Mapping

#: AER-in event encodings (ODIN doc/README.md Sec.2.2.1) of the vendored core.
AER_NEURON_SPIKE_SUFFIX = 0x07
AER_ALL_NEURON_TREF = 0x7F

#: The vendored crossbar addresses a PHYSICAL row in eight bits.
STOCK_ROW_ADDRESS_BITS = 8

#: The two wordings, named by the weight-sign granularity that identifies the
#: fabric: a whole row signed once IS the vendored crossbar, and a synapse cell
#: that signs itself IS a generated core.
AER_STOCK = "stock_row17"
AER_VARIANT = "variant_slot"
SIGN_PER_AXON = "per_axon"
SIGN_PER_SYNAPSE = "per_synapse"
PER_EVENT_FIRING = "per_event"


class OdinEncodingRefusal(ValueError):
    """The claims name no fabric whose events this reader can word."""


class AerEncoding:
    """One fabric's AER-in wording: how a row is addressed, and its TREF word."""

    def __init__(self, name: str, address_bits: int, tref_word: int,
                 emits_tref: bool) -> None:
        self.name = str(name)
        self.address_bits = int(address_bits)
        self.tref_word = int(tref_word)
        self.emits_tref = bool(emits_tref)

    def spike_word(self, row: int) -> int:
        """The AER-in address of one event on ``row`` of THIS fabric."""
        if row < 0 or row >= (1 << self.address_bits):
            raise OdinEncodingRefusal(
                f"pre-synaptic row {row} is not a {self.address_bits}-bit "
                f"address on the {self.name} fabric")
        if self.name == AER_STOCK:
            return (int(row) << 8) | AER_NEURON_SPIKE_SUFFIX
        return int(row)

    def as_dict(self) -> dict:
        """What a report says this fabric's events were worded as."""
        return {"encoding": self.name, "address_bits": self.address_bits,
                "tref_word": self.tref_word, "emits_tref": self.emits_tref}


def stock_encoding() -> AerEncoding:
    """The vendored crossbar: a 17-bit ``{row, 0x07}`` and a per-cycle TREF."""
    return AerEncoding(
        AER_STOCK, STOCK_ROW_ADDRESS_BITS, AER_ALL_NEURON_TREF, True)


def variant_encoding(axon_address_bits: int) -> AerEncoding:
    """A generated core: the word IS the axon slot, and the law takes no TREF."""
    bits = int(axon_address_bits)
    if bits < 1:
        raise OdinEncodingRefusal(
            f"a generated core addresses at least two axon rows; "
            f"axon_address_bits={bits}")
    return AerEncoding(AER_VARIANT, bits, 1 << bits, False)


def aer_encoding_of(chip_config: Mapping[str, Any]) -> AerEncoding:
    """THE dispatch: the wording of the fabric this bundle's own CLAIMS name."""
    granularity = str(chip_config["weight_sign_granularity"])
    if granularity == SIGN_PER_AXON:
        return stock_encoding()
    if granularity != SIGN_PER_SYNAPSE:
        raise OdinEncodingRefusal(
            f"weight_sign_granularity {granularity!r} names no fabric this "
            f"reader words events for ({SIGN_PER_AXON!r}, {SIGN_PER_SYNAPSE!r})")
    slots = int(chip_config["effective_max_axons"]) + 1
    bits = int(slots).bit_length() - 1
    if bits < 1 or slots != (1 << bits):
        raise OdinEncodingRefusal(
            f"effective_max_axons {slots - 1} implies {slots} axon rows, which "
            f"is not a power of two; a generated core's AER word IS its axon "
            f"address and a ragged row count has no address width")
    firing = str(chip_config["soma_law"]["firing_granularity"])
    if firing != PER_EVENT_FIRING:
        raise OdinEncodingRefusal(
            f"a generated core takes its time reference only from each core's "
            f"own latency onward and no bundle has frozen that schedule; this "
            f"reader executes {PER_EVENT_FIRING!r}, whose cycle carries no "
            f"TREF, and the bundle declares {firing!r}")
    return variant_encoding(bits)


def encoding_of_bundle(document: Mapping[str, Any]) -> AerEncoding:
    """The AER wording of one sealed bundle, read off its own chip claims."""
    return aer_encoding_of(document["chip_config"])
