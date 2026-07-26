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
    _E("lif_execution_discipline", group="spiking", owner="LifSegmentPolicy",
       type=T.ENUM, options=("streaming", "synchronized"),
       category=Category.ADVANCED, exposure="user",
       label="LIF Execution Discipline",
       doc="synchronized = two-window integrate-then-emit: the emitted count "
           "is exactly the strict staircase (genuine == analytic) [calculus 16]. "
           "Latency (D+1)*T, pipelined throughput unchanged.",
       provenance="consumer frozen default", derived_default=_frozen("streaming"),
       empty_means="fire-during-integrate streaming"),
    _E("spike_phase_dither", group="spiking", owner="spike_trains",
       type=T.BOOL, category=Category.ADVANCED, exposure="user",
       label="Spike Phase Dither",
       doc="Count-exact per-channel comb rotation mod T at uniform encodes: "
           "decorrelates arrival vs rectified-transient overfire [calculus 15.11].",
       provenance="consumer frozen default", derived_default=_frozen(False),
       empty_means="off"),
    _E("lif_membrane_init", group="spiking", owner="LIFActivation",
       type=T.FLOAT, category=Category.ADVANCED, exposure="user",
       label="LIF Membrane Init",
       doc="Window-start membrane guard (normalized); negative recenters the "
           "signed-charge rectifier [calculus 15.11].", empty_means="0.0",
       provenance="consumer frozen default", derived_default=_frozen(0.0)),
    _E("spike_count_parity_samples", group="spiking", owner="certification",
       type=T.INT, category=Category.ADVANCED, unit="samples",
       label="Spike-count Parity Samples",
       doc="Samples per backend spike-count certificate [calculus 17]: counts "
           "are integer-exact, so 1-2 samples x millions of neuron-windows "
           "out-powers argmax parity at any n.", bounds=(1, None),
       provenance="consumer frozen default", derived_default=_frozen(2),
       empty_means="2 samples"),
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
    _E("value_parity_samples", group="deployment_target", owner="value_gates",
       type=T.INT, category=Category.ADVANCED, unit="samples",
       label="Value Parity Samples",
       doc="Samples per value-domain (mvm) certificate edge: the fp64 twin is "
           "window-exact, so a handful of samples x every neuron-window "
           "out-powers argmax parity at any n. 0 disarms the gates.",
       bounds=(0, None),
       provenance="consumer frozen default", derived_default=_frozen(2),
       empty_means="2 samples"),
)
