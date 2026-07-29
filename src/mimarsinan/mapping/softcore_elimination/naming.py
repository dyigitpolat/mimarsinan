"""[W6b] Generic softcore group naming, derived from the node names alone.

A mapper emits one NeuralCore per (layer, position/column/tile) — the layer
identity plus the positional suffixes it appended (`_col12`, `_pos3_1_g0`,
`_tile0`). The paper table wants ROWS PER LAYER, not per instance, so this
module recovers the identity by peeling those suffixes off the tail:

    ``blocks_0_fc1_col36``      -> layer ``blocks_0_fc1`` -> group ``blocks_*_fc1``
    ``patch_embed_pos0_7_g0``   -> layer ``patch_embed``  -> group ``patch_embed``
    ``head_col0``               -> layer ``head``         -> group ``head``

Two levels, both purely lexical and workload-agnostic — no per-model string
tables live in the framework:

- :func:`softcore_layer_name` peels TRAILING tokens that are a positional
  keyword with an index (``col36``) or a bare integer (``0``), stopping at the
  first token that is neither and never consuming the whole name;
- :func:`softcore_group_name` additionally replaces the remaining bare-integer
  tokens (the repeat index of a stacked block) with ``*``, which is what folds
  seven identical transformer blocks into one table row.

A workload whose layers genuinely differ under one group label keeps its
detail: the report emits the per-layer rows alongside the group rows, and a
group over non-uniform geometry reports ``axons``/``neurons`` as None rather
than inventing a shared shape.
"""

from __future__ import annotations

import re

# Positional keywords a mapper appends with an index. Matching is on the WHOLE
# token, so a layer called `column_proj` or `group_norm` is never touched.
POSITIONAL_KEYWORDS: tuple[str, ...] = (
    "pos", "col", "column", "row", "g", "grp", "group", "tile", "part",
    "seg", "segment", "chunk", "idx", "index", "rep", "inst", "instance",
    "slice", "shard", "patch", "win", "window", "head",
)

_POSITIONAL = re.compile(
    r"^(?:" + "|".join(POSITIONAL_KEYWORDS) + r")\d+$", re.IGNORECASE
)
_BARE_INDEX = re.compile(r"^\d+$")
_SEPARATOR = "_"


def _is_positional(token: str) -> bool:
    return bool(_POSITIONAL.match(token) or _BARE_INDEX.match(token))


def softcore_layer_name(name: str) -> str:
    """The mapped-layer identity of one softcore: its name minus the tail of
    positional suffixes. Always keeps at least the leading token."""
    tokens = str(name).split(_SEPARATOR)
    while len(tokens) > 1 and _is_positional(tokens[-1]):
        tokens.pop()
    return _SEPARATOR.join(tokens)


def softcore_group_name(name: str) -> str:
    """The paper-table row label: the layer identity with repeat indices
    collapsed to ``*`` (``blocks_0_fc1`` -> ``blocks_*_fc1``)."""
    return _SEPARATOR.join(
        "*" if _BARE_INDEX.match(token) else token
        for token in softcore_layer_name(name).split(_SEPARATOR)
    )
