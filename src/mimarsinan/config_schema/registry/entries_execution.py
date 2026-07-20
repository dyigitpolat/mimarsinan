"""Registry entries: deployed-composition execution physics [calculus 15.11/16]."""

from __future__ import annotations

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
)
