"""Registry entries: the endpoint-recovery family (floors, ledgers, budgets)."""

from __future__ import annotations

from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
    frozen_default as _frozen,
)
from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY


ENTRIES = (
    _E("endpoint_floor_steps", group="tuning", owner="endpoint_recovery/steps_ledger",
       type=T.INT, category=Category.ADVANCED, unit="steps", label="Endpoint Floor Steps",
       doc="Per-cell RUN-total training-step budget for armed 5u endpoint-floor "
           "stages (one ledger shared by every armed endpoint; steps, never wall "
           "seconds — the reproducibility contract).", bounds=(0, None),
       provenance="TUNING_POLICY",
       derived_default=_frozen(TUNING_POLICY.endpoint_floor_steps),
       empty_means="the frozen TUNING_POLICY run-total budget"),
    _E("endpoint_floor_min_cover_steps", group="tuning",
       owner="endpoint_recovery/convergence_stop",
       type=T.INT, category=Category.ADVANCED, unit="steps",
       label="Endpoint Floor Min-cover Steps",
       doc="[C1/C3'] absolute min-cover before the armed endpoint convergence "
           "stop may vote (covers the lr dip). Calibrated to small-model step "
           "costs; large-backbone cells bound it down (2000 steps is a "
           "multi-hour mandatory burn at their step costs).", bounds=(0, None),
       provenance="TUNING_POLICY",
       derived_default=_frozen(TUNING_POLICY.endpoint_floor_min_cover_steps),
       empty_means="the frozen TUNING_POLICY lr-dip cover"),
    _E("endpoint_target_floor", group="tuning", owner="endpoint_recovery",
       type=T.FLOAT, category=Category.ADVANCED, label="Endpoint Target Floor",
       doc="Every-endpoint D-hat target floor. The ConversionPolicy recipe sets it "
           "only for the bit-parity-lossless family (analytical ttfs); every other "
           "mode floors at 0 and takes its floor from the WQ endpoint.", bounds=(0.0, 1.0),
       provenance="ConversionPolicy recipe", derived_default=_frozen(0.0),
       empty_means="the recipe floor where the mode has one, else 0 (no floor)"),
    _E("endpoint_target_margin", group="tuning", owner="endpoint_recovery",
       type=T.FLOAT, category=Category.ADVANCED, label="Endpoint Target Margin",
       doc="Headroom added above the endpoint target to BANK measured "
           "downstream conversion debt (calculus 17.9): anchoring at the "
           "D-hat high-water guarantees ending that debt below origin. "
           "0 = anchored (byte-identical).",
       bounds=(0.0, 1.0), provenance="consumer frozen default",
       derived_default=_frozen(0.0), empty_means="0 (anchored target)"),
    _E("wq_endpoint_recovery_steps", group="tuning", owner="wq_endpoint_recovery",
       type=T.INT, category=Category.ADVANCED, unit="steps",
       label="WQ Endpoint Recovery Steps",
       doc="Per-cell cap on the WQ endpoint recovery stage (recipe default stays "
           "for families that pass by the floor climb).", bounds=(0, None),
       provenance="ConversionPolicy recipe", derived_default=_frozen(0),
       empty_means="the ConversionPolicy recipe cap for the mode"),
    _E("endpoint_floor_lr", group="tuning", owner="workload_profile/tuning_policy",
       type=T.FLOAT, category=Category.ADVANCED, label="Endpoint Floor LR",
       doc="Floor-chasing endpoint LR ceiling. Builders register it via "
           "ModelWorkloadProfile; explicit value wins; absent = the frozen "
           "TUNING_POLICY value.", bounds=(0.0, None),
       provenance="builder profile",
       derived_default=_frozen(TUNING_POLICY.endpoint_floor_lr),
       empty_means="the builder's registration, else the frozen TUNING_POLICY value"),
    _E("endpoint_recovery_steps", group="tuning", owner="endpoint_recovery",
       type=T.INT, category=Category.ADVANCED, unit="steps",
       label="Mode Endpoint Recovery Steps",
       doc="Per-cell cap on the MODE stage's endpoint recovery (LIF / TTFS "
           "cycle), the sibling of wq_/aa_endpoint_recovery_steps. The recipe "
           "cap is convergence-grounded on healthy endpoints; a cell whose leg "
           "exhausts it while still climbing funds the leg here rather than by "
           "editing the shared recipe.", bounds=(0, None),
       provenance="ConversionPolicy recipe", derived_default=_frozen(0),
       empty_means="the ConversionPolicy recipe cap for the mode"),
    _E("aa_endpoint_recovery_steps", group="tuning", owner="endpoint_recovery",
       type=T.INT, category=Category.ADVANCED, unit="steps",
       label="AA Endpoint Recovery Steps",
       doc="[PR14b] funded endpoint-recovery leg for Activation Adaptation — the "
           "largest conversion swap; budget follows swap distance (calculus 13.2 "
           "L-C). 0 disables (byte-identical).", bounds=(0, None),
       provenance="consumer frozen default", derived_default=_frozen(0),
       empty_means="0 (no AA endpoint leg)"),
)
