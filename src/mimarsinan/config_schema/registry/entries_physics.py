"""Registry entries: the target's declared physical constants (platform physics)."""

from __future__ import annotations

from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
    frozen_default as _frozen,
)

_PC = "platform_constraints"


ENTRIES = (
    _E("platform_physics_profile", section=_PC, group="hardware",
       owner="deployment_record.platform_physics", type=T.STR,
       category=Category.ADVANCED, exposure="user",
       label="Platform Physics Profile",
       effect="Which target's per-unit area/energy/timing constants price this run",
       doc="Named physics profile (a <name>.json + <name>.md pair shipped under "
           "deployment_record/platform_physics/profiles). Its constants are what the "
           "cost model multiplies the record's counts by, so absolute area, energy, "
           "latency and throughput are attributable to a real chip. Empty means no "
           "physics is declared: the absolute objectives are then unavailable rather "
           "than computed from framework defaults wearing a vendor's name.",
       empty_means="no physics declared — absolute objectives unavailable"),
    _E("platform_physics_overrides", section=_PC, group="hardware",
       owner="deployment_record.platform_physics", type=T.JSON,
       category=Category.ADVANCED, exposure="user",
       label="Platform Physics Overrides",
       effect="Operator-declared constants replacing (or adding to) the profile's",
       doc="Sparse map of physics constant -> declaration, applied on top of the "
           "selected profile and recorded with the run, so a report states exactly "
           "what physics priced it. Each entry needs a nominal value and a note "
           "saying why it deviates; low/high and unit default to the profile's. With "
           "no profile selected these constants alone form a custom target.",
       provenance="consumer frozen default", derived_default=_frozen({})),
    _E("activity_factor", section=_PC, group="hardware",
       owner="deployment_record.quantities", type=T.FLOAT,
       category=Category.ADVANCED, exposure="user",
       bounds=(0.0, 1.0), default=0.0,
       label="Switching Activity Factor",
       effect="Declared per-synapse spike probability per timestep for "
              "candidate-time spike-dependent pricing",
       doc="The EDA switching-activity assumption: at search time no spikes have "
           "been measured, so spike-dependent quantities (synaptic events, NoC "
           "messages) are modeled as activity_factor x timesteps over the "
           "structural census, with provenance 'modeled'. 0 means undeclared: "
           "those quantities stay absent rather than resting on an assumption "
           "nobody stated. Sealed records always carry measured counts instead.",
       empty_means="undeclared — spike-dependent modeled quantities stay absent"),
)
