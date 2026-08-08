"""Registry entries: the core-semantics domain axis and the spiking family/variant taxonomy."""

from __future__ import annotations

from typing import Any, Mapping

from mimarsinan.chip_simulation.activation_semantics import (
    ALL_SPIKING_VARIANTS,
    SPIKING_FAMILIES,
    derived_spiking_variant,
    effective_legacy_spiking_mode,
    legal_spiking_families,
    legal_spiking_variants,
)
from mimarsinan.chip_simulation.core_semantics import (
    CORE_SEMANTICS_OPTIONS, CORE_SEMANTICS_SPIKING,
)
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
    frozen_default as _frozen,
)

SPIKING_MODES = ("lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based")


def _mode(cfg: Mapping[str, Any]) -> str:
    return effective_legacy_spiking_mode(cfg)


ENTRIES = (
    _E("core_semantics", group="spiking", owner="core_semantics",
       type=T.ENUM, options=CORE_SEMANTICS_OPTIONS, category=Category.BASIC,
       exposure="user", label="Core Semantics", important=True,
       effect="Selects the deployment family: spiking cores or value-domain MVM cores",
       doc="spiking: matmul + a neuron nonlinearity (LIF/TTFS event physics); "
           "mvm: pure y=Wx(+b) with value I/O, activations on host, event keys unauthorable.",
       provenance="derivation rule",
       derived_default=_frozen(CORE_SEMANTICS_SPIKING),
       legal_values=lambda cfg: CORE_SEMANTICS_OPTIONS,
       empty_means="spiking — the event-driven deployment family"),
    _E("spiking_family", domain="event", group="spiking", owner="ActivationSemantics",
       type=T.ENUM, options=SPIKING_FAMILIES, category=Category.BASIC,
       exposure="user", label="Spiking Family", important=True,
       effect="Selects the neuron/code family: rate-coded LIF or time-coded TTFS",
       doc="lif: values are spike COUNTS in a T-cycle window (rate code); "
           "ttfs: a value is WHEN a single spike occurs (time code). The "
           "variant picks the temporal discipline within the family.",
       legal_values=lambda cfg: legal_spiking_families(cfg)),
    _E("spiking_variant", domain="event", group="spiking", owner="ActivationSemantics",
       type=T.ENUM, options=ALL_SPIKING_VARIANTS, category=Category.BASIC,
       exposure="user", label="Spiking Variant", important=True,
       effect="Selects the family's temporal discipline (and simulation backends)",
       doc="lif: streamed (per-segment cycle-by-cycle streaming; host ops "
           "between segments see window counts; one segment = end-to-end) — "
           "synchronized (windowed; counts re-encoded at every layer). "
           "ttfs: analytical (closed-form, continuous), "
           "quantized (closed-form, quantized activations), synchronized "
           "(per-cycle, bit-identical to quantized), cascaded (greedy "
           "streamed cascade, lossy).",
       provenance="derivation rule",
       derived_default=derived_spiking_variant,
       legal_values=lambda cfg: legal_spiking_variants(cfg),
       empty_means="derived per family: lif → streamed; ttfs → analytical"),
    _E("spiking_mode", domain="event", group="spiking", owner="ActivationSemantics",
       type=T.ENUM, options=SPIKING_MODES, category=Category.DERIVED,
       derivation="derived", exposure="derived", hidden=True, declarable=False,
       label="Spiking Mode (legacy twin)",
       effect="Internal dispatch string the pipeline reads",
       doc="Derivation-owned legacy twin of (spiking_family, spiking_variant): "
           "lif → 'lif'; ttfs analytical/quantized → 'ttfs'/'ttfs_quantized'; "
           "ttfs synchronized/cascaded → 'ttfs_cycle_based'. RETIRED as a "
           "document key — declare the axes instead.",
       derived_from=("spiking_family", "spiking_variant"),
       why=lambda cfg: (
           f"'{_mode(cfg)}' — folded from (spiking_family="
           f"{cfg.get('spiking_family')!r}, spiking_variant="
           f"{cfg.get('spiking_variant')!r})"
       ),
       provenance="derivation rule"),
    _E("ttfs_cycle_schedule", domain="event", group="spiking", owner="ActivationSemantics",
       type=T.ENUM, options=("cascaded", "synchronized"),
       category=Category.DERIVED, derivation="derived", exposure="derived",
       hidden=True, declarable=False,
       label="TTFS Cycle Schedule (legacy twin)",
       effect="Internal schedule string the pipeline reads",
       doc="Derivation-owned legacy twin of spiking_variant for the "
           "ttfs_cycle_based family ('synchronized' iff (ttfs, synchronized); "
           "the historical 'cascaded' default otherwise, inert off-cycle). "
           "RETIRED as a document key — declare the axes instead.",
       derived_from=("spiking_family", "spiking_variant"),
       why=lambda cfg: (
           f"'{cfg.get('ttfs_cycle_schedule', 'cascaded')}' — folded from "
           f"spiking_variant={cfg.get('spiking_variant')!r}"
       ),
       provenance="derivation rule"),
)
