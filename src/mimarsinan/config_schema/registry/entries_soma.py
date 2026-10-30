"""Registry entries: the soma-law axes and the platform widths that drive them."""

from __future__ import annotations

from mimarsinan.chip_simulation.soma_axes import (
    FIRING_GRANULARITIES,
    MEMBRANE_ARITHMETICS,
    WEIGHT_SIGN_GRANULARITIES,
    derived_firing_granularity,
    derived_membrane_arithmetic,
    legal_firing_granularities,
    legal_membrane_arithmetics,
)
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)

_PC = "platform_constraints"


ENTRIES = (
    _E("firing_granularity", domain="event", group="spiking", owner="SomaLaw",
       type=T.ENUM, options=FIRING_GRANULARITIES, category=Category.ADVANCED,
       exposure="user", label="Firing Granularity",
       effect="WHEN the threshold is evaluated inside one cycle",
       doc="per_cycle: one compare per cycle on the reduced contribution (the "
           "law every deployment runs today). per_event: the threshold is "
           "checked after every arriving event occurrence in canonical order, "
           "emitting one spike and resetting per crossing, so a neuron may emit "
           "0, 1, or more spikes in a cycle. per_event is declarable only under "
           "the streamed LIF variant — a windowed hop collapses counts and "
           "would destroy the multiplicity.",
       provenance="derivation rule",
       derived_default=derived_firing_granularity,
       legal_values=lambda cfg: legal_firing_granularities(cfg),
       empty_means="per_cycle — today's one-compare-per-cycle law"),
    _E("membrane_arithmetic", domain="event", group="spiking", owner="SomaLaw",
       type=T.ENUM, options=MEMBRANE_ARITHMETICS, category=Category.ADVANCED,
       exposure="user", label="Membrane Arithmetic",
       effect="The membrane accumulator: unbounded, or a saturating unsigned register",
       doc="unbounded: today's signed accumulator with no clamp. "
           "saturating_unsigned: the membrane clamps to [0, 2^membrane_bits-1] "
           "on every update. saturating_signed: a two's-complement register "
           "clamping to [-2^(membrane_bits-1), 2^(membrane_bits-1)-1], so a "
           "net-negative cycle keeps its charge instead of flooring at zero. "
           "BITS-DRIVEN like weight quantization — declaring platform "
           "membrane_bits IS declaring a saturating register and "
           "membrane_signed says which one; a contradicting explicit value is "
           "refused by key.",
       provenance="derivation rule",
       derived_default=derived_membrane_arithmetic,
       legal_values=lambda cfg: legal_membrane_arithmetics(cfg),
       empty_means="derived from platform membrane_bits (0 -> unbounded)"),
    _E("membrane_bits", domain="event", section=_PC, group="hardware",
       owner="SomaLaw/platform", type=T.INT, category=Category.ADVANCED,
       exposure="user", label="Membrane Width",
       effect="Declares a fixed-width saturating membrane register",
       doc="Width of the target's membrane register in bits. 0 means the "
           "membrane is not a fixed-width register (the framework's unbounded "
           "accumulator). A positive width declares the saturating unsigned "
           "law, exactly as weight_bits declares a quantized artifact.",
       bounds=(0, 64)),
    _E("membrane_signed", domain="event", section=_PC, group="hardware",
       owner="SomaLaw/platform", type=T.BOOL, category=Category.ADVANCED,
       exposure="user", label="Signed Membrane Register",
       effect="Declares the fixed-width membrane register two's complement",
       doc="Whether the target's membrane register holds negative values. "
           "false (the default) is the unsigned register that floors at zero; "
           "true is the two's-complement register that clamps at "
           "-2^(membrane_bits-1). Signedness is physical structure, so it is "
           "declared beside membrane_bits and membrane_arithmetic falls out of "
           "the pair — nothing changes for a platform that declares no width."),
    _E("weight_sign_granularity", group="hardware", section=_PC,
       owner="mapping/packing", type=T.ENUM,
       options=WEIGHT_SIGN_GRANULARITIES, category=Category.ADVANCED,
       exposure="user", label="Weight Sign Granularity",
       effect="Where the weight sign physically lives on the target",
       doc="per_synapse: every synapse carries its own sign (the framework's "
           "signed weight grid). per_axon: the sign is a per-pre-synaptic-row "
           "property, so the representable magnitude range is symmetric and a "
           "logical row expands into an excitatory/inhibitory physical pair.",
       ),
)
