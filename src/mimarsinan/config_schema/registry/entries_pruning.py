"""Registry entries: the pruning mapping-strategy keys (adaptation, the criterion-agnostic
seed seam, and the elimination-propagation / constant-folding axes)."""

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
from mimarsinan.mapping.pruning.liveness_transfer.constant_policy import (
    DEFAULT_ELIMINATION_CONSTANT_FOLDING,
    ELIMINATION_CONSTANT_FOLDING_MODES,
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
    _E("prune_criterion", group="mapping_strategy", owner="pruning_adaptation",
       type=T.ENUM, options=("row_col_l1", "activation", "partial_column_group"),
       category=Category.ADVANCED, exposure="user", label="Prune Criterion",
       doc="Seed criterion for the one-shot pruning at soft-core mapping "
           "(criterion-agnostic cascade seam): 'row_col_l1' keeps the incumbent "
           "structural channel shrink; 'activation' seeds the IR pruning "
           "cascade with measured activation-importance row/col masks; "
           "'partial_column_group' seeds it with Meng-style contiguous row-group "
           "element kills per column — the cascade harvests whichever whole "
           "rows/cols the group kills complete.",
       relevant=R.when_true("pruning"),
       provenance="consumer frozen default",
       derived_default=_frozen("row_col_l1"),
       empty_means="row_col_l1 — the incumbent criterion, byte-identical"),
    _E("prune_group_size", group="mapping_strategy", owner="pruning_adaptation",
       type=T.INT, category=Category.ADVANCED, exposure="user",
       label="Prune Group Size",
       doc="Rows per contiguous column-group for "
           "prune_criterion='partial_column_group' (e.g. 72 reproduces "
           "Meng-style 72x1 groups); groups are scored by L2 norm and the "
           "lowest-scoring fraction is zeroed.",
       bounds=(1, 4096),
       relevant=R.when("prune_criterion", in_=("partial_column_group",)),
       provenance="consumer frozen default", derived_default=_frozen(8),
       empty_means="8 rows per group"),
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
    _E("elimination_constant_folding", group="mapping_strategy",
       owner="pruning_adaptation", type=T.ENUM,
       options=ELIMINATION_CONSTANT_FOLDING_MODES, category=Category.ADVANCED,
       exposure="user", label="Elimination Constant Folding",
       doc="Constant-lattice propagation for IR pruning: 'full' propagates "
           "TOP > CONST(c) across host ops and folds every CONST axon row onto "
           "its core's existing constant carrier (so a non-zero-preserving "
           "activation, a residual join or a bias-only core stops being an "
           "elimination barrier); 'off' is the kill-switch that reproduces the "
           "zero-only cascade byte-identically. Non-zero folds are additionally "
           "gated to the value/MVM chip domain, and 'identity_only' liveness "
           "transfers force this off.",
       relevant=R.when_true("pruning"), provenance="consumer frozen default",
       derived_default=_frozen(DEFAULT_ELIMINATION_CONSTANT_FOLDING),
       empty_means="full — the constant lattice runs (default path)"),
)
