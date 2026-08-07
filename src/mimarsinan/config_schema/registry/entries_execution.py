"""Registry entries: deployed-composition execution physics [calculus 15.11/16]."""

from __future__ import annotations

from mimarsinan.config_schema.registry.relevance import Relevance as R
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
    frozen_default as _frozen,
)


ENTRIES = (
    _E("lif_execution_discipline", domain="event", group="spiking", owner="LifSegmentPolicy",
       type=T.ENUM, options=("streaming", "synchronized"),
       category=Category.DERIVED, derivation="derived", exposure="derived",
       hidden=True, declarable=False,
       label="LIF Execution Discipline (internal)",
       effect="Executor evaluation form inside the windowed-lif semantics",
       doc="synchronized = two-window integrate-then-emit: the emitted count "
           "is exactly the strict staircase (genuine == analytic) [calculus 16]. "
           "RETIRED as a document key: the temporal discipline is the "
           "spiking_variant axis; this internal executor form stays "
           "derivation-owned (streaming tick loop).",
       derived_from=("spiking_family", "spiking_variant"),
       why=lambda cfg: "streaming — the deployed cycle-accurate tick loop",
       provenance="consumer frozen default",
       derived_default=_frozen("streaming")),
    _E("spike_phase_dither", domain="event", group="spiking", owner="spike_trains",
       type=T.BOOL, category=Category.ADVANCED, exposure="user",
       label="Spike Phase Dither",
       doc="Count-exact per-channel comb rotation mod T at uniform encodes: "
           "decorrelates arrival vs rectified-transient overfire [calculus 15.11].",
       provenance="consumer frozen default", derived_default=_frozen(False),
       empty_means="off"),
    _E("lif_membrane_init", domain="event", group="spiking", owner="LIFActivation",
       type=T.FLOAT, category=Category.ADVANCED, exposure="user",
       label="LIF Membrane Init",
       doc="Window-start membrane guard (normalized); negative recenters the "
           "signed-charge rectifier [calculus 15.11].", empty_means="0.0",
       provenance="consumer frozen default", derived_default=_frozen(0.0)),
    _E("spike_count_parity_samples", domain="event", group="spiking", owner="certification",
       type=T.INT, category=Category.ADVANCED, unit="samples",
       label="Spike-count Parity Samples",
       doc="Samples per backend spike-count certificate [calculus 17]: counts "
           "are integer-exact, so 1-2 samples x millions of neuron-windows "
           "out-powers argmax parity at any n.", bounds=(1, None),
       provenance="consumer frozen default", derived_default=_frozen(2),
       empty_means="2 samples"),
    _E("activation_bits", domain="value", section="platform_constraints", group="hardware",
       owner="boundary_quantization", type=T.INT, category=Category.ADVANCED,
       exposure="user", label="Activation Bits",
       effect="Arms value-domain (mvm) boundary activation quantization",
       doc="[mvm AQ] symmetric signed grid width at host<->chip boundaries; "
           "absent = float boundary I/O. Event-domain platforms declare "
           "target_tq instead — this key is value-domain only.",
       bounds=(2, None), empty_means="absent — float boundary I/O"),
    _E("schedule_policy", group="mapping_strategy",
       owner="hybrid_build_scheduled", type=T.ENUM,
       options=("pool", "bank_clustered"), category=Category.ADVANCED,
       exposure="user", label="Schedule Policy",
       effect="Pass composition under scheduling: capacity split vs "
              "weight-stationary bank streaming",
       doc="[wsm V2] pool: the historical capacity split (fresh pool per "
           "pass, weights reprogram). bank_clustered: same-bank instances "
           "stream over a resident core-set — weights program once, verified "
           "by placement-geometry identity; segments outside the policy's "
           "class fall back to pool.",
       relevant=R.when_true("allow_scheduling"),
       provenance="consumer frozen default", derived_default=_frozen("pool"),
       empty_means="pool — the historical scheduled build"),
    _E("core_value_granularity", section="platform_constraints", group="hardware",
       owner="mapping.platform.core_residency", type=T.JSON,
       category=Category.ADVANCED, exposure="user", label="Core Value Granularity",
       effect="Which per-core values constrain what may share a hardware core",
       doc="Per-value grain: absent / per_core / per_neuron, by name. Only per_core "
           "constrains residency; per_neuron stores one per neuron range; absent is "
           "vacuous. Defaults follow the deployment domain, so name only exceptions.",
       provenance="consumer frozen default", derived_default=_frozen({})),
    _E("allow_weight_reuse", section="platform_constraints",
       group="hardware", owner="ChipCapabilities/weight_reuse",
       type=T.BOOL, category=Category.ADVANCED, exposure="user",
       label="Allow Weight Reuse",
       effect="Capability gate: programmed banks stay resident across passes",
       doc="Hardware capability [wsm]: the chip can keep a programmed weight "
           "bank resident across scheduled passes; today arms the reuse-phase "
           "report, the bank-aware schedule policy consumes it next.",
       provenance="consumer frozen default", derived_default=_frozen(False),
       empty_means="off — passes reprogram freely, report unarmed"),
    _E("value_parity_samples", domain="value", group="deployment_target", owner="value_gates",
       type=T.INT, category=Category.ADVANCED, unit="samples",
       label="Value Parity Samples",
       doc="Samples per value-domain (mvm) certificate edge: the fp64 twin is "
           "window-exact, so a handful of samples x every neuron-window "
           "out-powers argmax parity at any n. 0 disarms the gates.",
       bounds=(0, None),
       provenance="consumer frozen default", derived_default=_frozen(2),
       empty_means="2 samples"),
)
