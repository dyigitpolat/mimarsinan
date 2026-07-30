"""[W6c] DISPLAY labels and the repeat-collapse — presentation, never identity.

Rows are keyed structurally (:mod:`...identity`). Everything here runs AFTER
the rows exist and may only decide what they are CALLED and whether two
already-correct rows are printed as one; nothing here can collapse unrelated
layers or shatter a layer, because it never touches the bucketing. A total
label failure costs a row its pretty name, nothing more.

Two steps, both derived from the graph in front of us — there is no word list:

- :func:`display_labels` labels a row with the token-wise LONGEST COMMON PREFIX
  of its instances' names. For a layer mapped onto many crossbars that already
  removes the mapper's positional tail exactly (``blocks_0_fc1_col0`` ...
  ``blocks_0_fc1_col64`` -> ``blocks_0_fc1``). What the LCP cannot reach — the
  suffix of a row with a single instance, or a suffix whose head token is
  constant like ``..._tile_0_16`` — is trimmed against evidence carried by the
  instances THEMSELVES (:func:`positional_stems`), and any label collision
  reverts the trim.
- :func:`collapse_repeats` folds the per-layer rows of a repeated container
  (``blocks_0_fc1`` ... ``blocks_6_fc1`` -> ``blocks_*_fc1``) into one printed
  row. It merges only rows whose label skeleton AND crossbar geometry agree,
  and refuses when the skeleton has no named token left, so the positional
  labels of a bare ``nn.Sequential`` (``_0``, ``_4``, ``_6``) never merge.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

from mimarsinan.mapping.softcore_elimination.facts import InstanceFacts
from mimarsinan.mapping.softcore_elimination.identity import MappedLayerKey
from mimarsinan.mapping.softcore_elimination.types import (
    GroupElimination,
    aggregate_groups,
)

_SEPARATOR = "_"
_BARE_INDEX = re.compile(r"^\d+$")
# A mapper position suffix is `<stem><index>` with a NON-EMPTY alphabetic stem
# (`col36`, `pos4`, `g0`, `tile3`). A bare index is never a stem: in a
# converter-built `nn.Sequential` it is the entire layer identity.
_STEMMED_INDEX = re.compile(r"^([^\W\d_][^\W\d_]*)(\d+)$", re.UNICODE)

InstancesByKey = Mapping[MappedLayerKey, Sequence[InstanceFacts]]


def _tokens(name: str) -> list[str]:
    return str(name).split(_SEPARATOR)


def _common_prefix(token_lists: Sequence[Sequence[str]]) -> list[str]:
    if not token_lists:
        return []
    prefix = list(token_lists[0])
    for tokens in token_lists[1:]:
        keep = 0
        for a, b in zip(prefix, tokens):
            if a != b:
                break
            keep += 1
        prefix = prefix[:keep]
        if not prefix:
            break
    return prefix


def _split_stem(token: str) -> tuple[str, int] | None:
    match = _STEMMED_INDEX.match(token)
    return (match.group(1), int(match.group(2))) if match else None


def positional_stems(instances_by_key: InstancesByKey) -> set[str]:
    """The suffix stems THIS graph shows to be positional, as evidence.

    Three independent witnesses, all read off the mapping:

    - a stem in the VARYING TAIL of a row with several instances — by
      construction those tokens are what the mapper appended to tell instances
      of one and the same mapped layer apart (``col``, ``pos``, ``g``);
    - a stem whose index equals one of the instance's own STRUCTURAL
      COORDINATES (``perceptron_output_column``, the output/input/bank
      windows) in the LAST token of its name — ``head_col0`` on the instance
      whose output column IS 0. Only the last token is admitted, so
      ``blocks_0_fc1_col1`` teaches ``col``, never ``fc``;
    - a stem occurring as the final token of instances of two or more DISTINCT
      mapped layers — a token that fails to distinguish layers cannot be part
      of a layer's identity (``_4_col0`` and ``_6_col0`` in a bare
      ``nn.Sequential`` whose FCs each map to a single crossbar).
    """
    stems: set[str] = set()
    finals: dict[str, set[MappedLayerKey]] = defaultdict(set)
    for key, members in instances_by_key.items():
        token_lists = [_tokens(m.name) for m in members]
        prefix_len = len(_common_prefix(token_lists))
        for member, tokens in zip(members, token_lists):
            if len(members) > 1:
                for token in tokens[prefix_len:]:
                    split = _split_stem(token)
                    if split is not None:
                        stems.add(split[0])
            if not tokens:
                continue
            finals[tokens[-1]].add(key)
            split = _split_stem(tokens[-1])
            if split is not None and (split[1],) in member.coordinates:
                stems.add(split[0])
    for token, keys in finals.items():
        split = _split_stem(token)
        if split is not None and len(keys) > 1:
            stems.add(split[0])
    return stems


def _numeric_suffix_head(
    prefix: Sequence[str], members: Sequence[InstanceFacts]
) -> bool:
    """Is the LCP's last token the HEAD of a numeric mapper suffix?

    ``_map_fc_output_tiled`` writes ``{name}_tile_{start}_{end}``, so on a
    tiled layer the LCP stops at the constant word ``tile`` while the tile
    bounds vary after it. The witness is structural: the tokens that follow
    are bare integers and they spell one of the instance's own coordinates.
    """
    if not prefix or _split_stem(prefix[-1]) is not None:
        return False
    if _BARE_INDEX.match(prefix[-1]):
        return False
    at = len(prefix)
    for member in members:
        tail = _tokens(member.name)[at:]
        run = []
        for token in tail:
            if not _BARE_INDEX.match(token):
                break
            run.append(int(token))
        if not run or tuple(run) not in member.coordinates:
            return False
    return True


def _trim(
    prefix: Sequence[str], members: Sequence[InstanceFacts], stems: Iterable[str]
) -> list[str]:
    """Drop trailing position suffixes off a label, never emptying it."""
    vocabulary = set(stems)
    out = list(prefix)
    head_dropped = False
    while len(out) > 1:
        split = _split_stem(out[-1])
        drop = split is not None and split[0] in vocabulary
        if not drop and not head_dropped and _numeric_suffix_head(out, members):
            drop, head_dropped = True, True
        if not drop:
            break
        if not any(token for token in out[:-1]):
            break  # trimming would leave nothing readable
        out.pop()
    return out


def display_labels(instances_by_key: InstancesByKey) -> dict[MappedLayerKey, str]:
    """A human-readable, COLLISION-FREE label per structurally-keyed row."""
    stems = positional_stems(instances_by_key)
    raw: dict[MappedLayerKey, str] = {}
    trimmed: dict[MappedLayerKey, str] = {}
    for key, members in instances_by_key.items():
        ordered = sorted(members, key=lambda m: m.name)
        prefix = _common_prefix([_tokens(m.name) for m in ordered])
        # No shared prefix at all (or nothing but separators): the names carry
        # no usable identity, so the structural key speaks for itself.
        raw[key] = _SEPARATOR.join(prefix) if any(prefix) else str(key)
        trimmed[key] = (
            _SEPARATOR.join(_trim(prefix, ordered, stems))
            if any(prefix) else raw[key]
        )

    collisions = {
        label for label, n in Counter(trimmed.values()).items() if n > 1
    }
    labels = {
        key: (raw[key] if label in collisions else label)
        for key, label in trimmed.items()
    }
    # Reverting can only restore the fuller name, but two rows may genuinely
    # carry identical names; then only the structural key separates them.
    still_colliding = {
        label for label, n in Counter(labels.values()).items() if n > 1
    }
    return {
        key: (f"{label} <{key}>" if label in still_colliding else label)
        for key, label in labels.items()
    }


def _skeleton(label: str) -> tuple[str, ...]:
    """The label with repeat indices replaced by ``*``."""
    return tuple(
        "*" if _BARE_INDEX.match(token) else token for token in _tokens(label)
    )


def _is_collapsible(skeleton: Sequence[str]) -> bool:
    """A repeat pattern is only meaningful when a NAMED token survives it."""
    return "*" in skeleton and any(token and token != "*" for token in skeleton)


def collapse_repeats(
    rows: Sequence[GroupElimination]
) -> tuple[GroupElimination, ...]:
    """Fold the per-layer rows of a repeated container into one printed row.

    Purely a presentation post-step over structurally-correct rows: rows merge
    only when their label skeletons AND their ``(axons, neurons)`` geometry all
    agree, so a bucket can never mix unrelated layers, and a single layer is
    never split (it is one row going in and at most one row coming out). The
    uncollapsed rows stay available beside these.
    """
    buckets: dict[tuple[str, ...], list[GroupElimination]] = defaultdict(list)
    for row in rows:
        buckets[_skeleton(row.group)].append(row)

    out: list[GroupElimination] = []
    for skeleton, members in buckets.items():
        geometries = {(m.axons, m.neurons) for m in members}
        if (
            len(members) > 1
            and len(geometries) == 1
            and _is_collapsible(skeleton)
        ):
            out.append(
                aggregate_groups(tuple(members), label=_SEPARATOR.join(skeleton))
            )
        else:
            out.extend(members)
    return tuple(sorted(out, key=lambda row: row.group))
