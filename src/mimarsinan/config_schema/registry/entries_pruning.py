"""Registry entries: structured-pruning / elimination mapping-strategy keys."""

from __future__ import annotations

from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
    frozen_default as _frozen,
)
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    DEFAULT_ELIMINATION_PROPAGATION,
    ELIMINATION_PROPAGATION_MODES,
)

ENTRIES = (
    _E("pruning", group="mapping_strategy", owner="pruning_adaptation",
       type=T.BOOL, category=Category.BASIC, exposure="user", label="Pruning Enabled",
       effect="Adds the Pruning Adaptation step",
       doc="Enable magnitude pruning adaptation (a deployment-side conversion "
           "step, not an architecture property).",
       provenance="consumer frozen default", derived_default=_frozen(False)),
    _E("pruning_fraction", group="mapping_strategy", owner="pruning_adaptation",
       type=T.FLOAT, category=Category.BASIC, exposure="user", label="Pruning Fraction",
       doc="Fraction of weights pruned by the adaptation.", bounds=(0.0, 1.0),
       relevant=R.when_true("pruning"),
       provenance="consumer frozen default", derived_default=_frozen(0.0),
       empty_means="0 — pruning stays inert (no Pruning Adaptation step)"),
    _E("prune_sparsity", group="mapping_strategy", owner="pruning_adaptation",
       type=T.FLOAT, category=Category.ADVANCED, label="Prune Sparsity",
       doc="Legacy sparsity knob consumed by the pruning tuner mask builder.",
       bounds=(0.0, 1.0), relevant=R.when_true("pruning"),
       provenance="consumer frozen default", derived_default=_frozen(0.0)),
    _E("elimination_propagation", group="mapping_strategy",
       owner="pruning_adaptation", type=T.ENUM,
       options=ELIMINATION_PROPAGATION_MODES, category=Category.ADVANCED,
       exposure="user", label="Elimination Propagation",
       doc="Structured-elimination propagation arm for IR pruning: 'masked' "
           "reclaims only the seeded rows/cols (allocation-naive lower bound); "
           "'closure' adds one-hop seed-group coupling (DepGraph-equivalent "
           "baseline); 'cascade' runs the full bidirectional liveness fixpoint.",
       relevant=R.when_true("pruning"), provenance="consumer frozen default",
       derived_default=_frozen(DEFAULT_ELIMINATION_PROPAGATION),
       empty_means="cascade — the full propagative fixpoint (default path)"),
)
