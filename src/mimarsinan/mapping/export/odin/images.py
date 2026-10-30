"""Per-core memory images: the neuron word array, the synapse array, and SYN_SIGN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from mimarsinan.mapping.export.odin.expansion import (
    RowPairExpansion,
    expand_row_pairs,
)
from mimarsinan.mapping.export.odin.layout import (
    LIF_NEURON_WORD_FIELDS,
    NEURON_MEMORY_WORDS,
    pack_neuron_word,
)
from mimarsinan.mapping.export.odin.registers import (
    RegisterWrite,
    inference_register_writes,
    pack_syn_sign,
)
from mimarsinan.mapping.export.odin.synapse import (
    CROSSBAR_COLUMNS,
    CROSSBAR_ROWS,
    pack_synapse_words,
)


class OdinImageError(ValueError):
    """A packed core does not fit the declared physical geometry."""


@dataclass(frozen=True)
class OdinCoreImage:
    """One core's complete programming payload."""

    core_index: int
    neuron_words: Tuple[int, ...]
    synapse_words: Tuple[int, ...]
    syn_sign: int
    register_writes: Tuple[RegisterWrite, ...]


def bias_row_count(core: Any) -> int:
    """How many always-on rows sit at the TAIL of this core's slot order."""
    sources = list(core.axon_sources)
    count = 0
    for source in reversed(sources):
        if not getattr(source, "is_always_on_", False):
            break
        count += 1
    return count


def _neuron_word(*, theta: int, membrane_init: int, enabled: bool) -> int:
    values: Dict[str, int] = {field.name: 0 for field in LIF_NEURON_WORD_FIELDS}
    # Leak and calcium/SDSP are OFF in deployment: the exported program is a
    # frozen-weight inference image, not an online-learning one.
    values["lif_izh_sel"] = 1
    values["thr"] = theta
    values["vmem"] = membrane_init if enabled else 0
    values["neur_disable"] = 0 if enabled else 1
    return pack_neuron_word(values)


def build_core_image(
    core: Any,
    *,
    core_index: int,
    theta: int,
    membrane_init: int,
    weight_bits: int,
    gate_activity: int,
) -> Tuple[OdinCoreImage, RowPairExpansion]:
    """Pack one ``HardCore`` into its physical images, without touching its grid."""
    logical_axons = int(core.axons_per_core)
    logical_neurons = int(core.neurons_per_core)
    if len(core.axon_sources) != logical_axons:
        raise OdinImageError(
            f"core {core_index}: {len(core.axon_sources)} axon sources for "
            f"{logical_axons} slots — the positional pairing of weight row a with "
            f"axon_sources[a] is what the whole chain rests on")
    if 2 * logical_axons > CROSSBAR_ROWS:
        raise OdinImageError(
            f"core {core_index}: {logical_axons} logical slots expand to "
            f"{2 * logical_axons} physical rows, over the {CROSSBAR_ROWS}-row "
            f"crossbar")
    if logical_neurons > CROSSBAR_COLUMNS:
        raise OdinImageError(
            f"core {core_index}: {logical_neurons} neurons over the "
            f"{CROSSBAR_COLUMNS}-column crossbar")

    expansion = expand_row_pairs(
        core.get_core_matrix(),
        n_bias_rows=bias_row_count(core),
        weight_bits=weight_bits,
        core_index=core_index,
    )
    magnitudes: List[List[int]] = [
        [0] * CROSSBAR_COLUMNS for _ in range(CROSSBAR_ROWS)
    ]
    signs = [False] * CROSSBAR_ROWS
    for row in expansion.rows:
        signs[row.row] = row.inhibitory
        for neuron, magnitude in enumerate(row.magnitudes.tolist()):
            magnitudes[row.row][neuron] = int(magnitude)

    used_neurons = logical_neurons - int(core.available_neurons or 0)
    neuron_words = tuple(
        _neuron_word(
            theta=theta, membrane_init=membrane_init, enabled=index < used_neurons
        )
        for index in range(NEURON_MEMORY_WORDS)
    )
    syn_sign = pack_syn_sign(signs)
    return (
        OdinCoreImage(
            core_index=core_index,
            neuron_words=neuron_words,
            # Every mapping bit is written 0: the freeze rides on
            # SPI_UPDATE_UNMAPPED_SYN=0 while SPI_PROPAGATE_UNMAPPED_SYN=1 keeps
            # the magnitudes flowing to the soma (registers.py).
            synapse_words=pack_synapse_words(magnitudes, mapped=False),
            syn_sign=syn_sign,
            register_writes=inference_register_writes(
                syn_sign=syn_sign, gate_activity=gate_activity
            ),
        ),
        expansion,
    )
