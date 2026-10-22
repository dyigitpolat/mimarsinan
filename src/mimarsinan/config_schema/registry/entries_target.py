"""Registry entries: the target bundle selection, its overrides and the two labels it writes."""

from __future__ import annotations

from mimarsinan.config_schema.registry.decision_classes import COMPOSITE, NOT_A_DECISION
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)
from mimarsinan.config_schema.registry.value_schema import ANY, ValueSchema

_PC = "platform_constraints"
_OWNER = "mapping/platform/target_bundles"

#: ``{key: {value, note}}``: each override replaces one bundle key's whole value and says why.
TARGET_OVERRIDES_SCHEMA = ValueSchema(
    kind=T.JSON,
    entries=ValueSchema(
        kind=T.JSON,
        fields={
            "value": ValueSchema(kind=ANY, decision_class=COMPOSITE,
                                 doc="the overriding value: the class of the key it overrides"),
            "note": ValueSchema(kind=T.STR, min_length=1, decision_class=NOT_A_DECISION,
                                doc="why the bundle value is overridden: read by people only"),
        },
        required=("value", "note"),
    ),
)


def _why_bundle_label(cfg: dict) -> str:
    return f"written by the target expansion for target={cfg.get('target')!r}"


ENTRIES = (
    _E("target", section=_PC, group="hardware", owner=_OWNER, type=T.STR,
       category=Category.ADVANCED, exposure="user", label="Target",
       effect="One name selects geometry, capabilities, semantics, physics and faithfulness class",
       doc="Name of a target bundle. Its expansion writes the bundle's keys into the flat "
           "config at every entry point and refuses a conflicting explicit key; an unknown "
           "name refuses listing the known names.",
       passthrough=True, empty_means="no bundle — the hand-written declaration stands"),
    _E("target_overrides", section=_PC, group="hardware", owner=_OWNER, type=T.JSON,
       category=Category.ADVANCED, exposure="user", label="Target Overrides",
       effect="Replaces named bundle keys, each with a stated reason",
       doc="Sparse map of bundle key -> {value, note}: each entry replaces that key's whole "
           "bundle value and its note says why, so a run states how it departs from the "
           "published target.",
       value_schema=TARGET_OVERRIDES_SCHEMA, passthrough=True,
       empty_means="{} — no bundle key overridden"),
    _E("target_claim_class", section=_PC, group="hardware", owner=_OWNER, type=T.STR,
       category=Category.DERIVED, derivation="derived", exposure="derived",
       label="Target Claim Class", declarable=False,
       doc="Claim class of the selected target bundle (which claims its numbers may back), "
           "written by the target expansion on flat configs only, never into a document.",
       derived_from=("target",), why=_why_bundle_label, provenance="derivation rule",
       passthrough=True),
    _E("target_faithfulness_class", section=_PC, group="hardware", owner=_OWNER, type=T.STR,
       category=Category.DERIVED, derivation="derived", exposure="derived",
       label="Target Faithfulness Class", declarable=False,
       doc="Faithfulness class of the selected target bundle (how closely the declaration "
           "follows the published chip), written by the target expansion on flat configs only.",
       derived_from=("target",), why=_why_bundle_label, provenance="derivation rule",
       passthrough=True),
)
