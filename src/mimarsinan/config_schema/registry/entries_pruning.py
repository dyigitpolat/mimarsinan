"""Registry entries: the pruning mapping-strategy keys (adaptation + the W3b
criterion-agnostic seed seam)."""

from __future__ import annotations

from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
    frozen_default as _frozen,
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
)
