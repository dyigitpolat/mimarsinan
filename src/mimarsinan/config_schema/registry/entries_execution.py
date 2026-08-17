"""Registry entries: deployed-composition execution physics [calculus 15.11/16]."""

from __future__ import annotations

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
    _E("core_value_granularity", section="platform_constraints", group="hardware",
       owner="mapping.platform.core_residency", type=T.JSON,
       category=Category.ADVANCED, exposure="user", label="Core Value Granularity",
       effect="Which per-core values constrain what may share a hardware core",
       doc="Per-value grain: absent / per_core / per_neuron, by name. Only per_core "
           "constrains residency; per_neuron stores one per neuron range; absent is "
           "vacuous. Defaults follow the deployment domain, so name only exceptions.",
       provenance="consumer frozen default", derived_default=_frozen({})),
    _E("cores_per_tile", section="platform_constraints", group="hardware",
       owner="sanafe_arch_synth", type=T.INT, category=Category.ADVANCED,
       exposure="user", label="Cores per Tile",
       effect="Fixes the SANA-FE NoC tile grouping independently of the packed model",
       doc="NoC floorplan: cores per tile of the synthesized SANA-FE "
           "architecture. 0 derives it from the declared platform (preset "
           "tile wiring: loihi 4/tile, truenorth 1/tile; else "
           "ceil(sqrt(declared core capacity))).",
       bounds=(0, None), empty_means="0 — derived from the declared platform"),
    _E("tile_grid_rows", section="platform_constraints", group="hardware",
       owner="sanafe_arch_synth", type=T.INT, category=Category.ADVANCED,
       exposure="user", label="Tile Grid Rows",
       effect="Fixes the SANA-FE NoC mesh height independently of the packed model",
       doc="NoC floorplan: tile-grid rows (mesh height). 0 derives the "
           "most-square exact grid; declare together with tile_grid_cols "
           "(both 0 or both > 0 — a half-declared grid is a config error).",
       bounds=(0, None), empty_means="0 — the derived most-square exact grid"),
    _E("tile_grid_cols", section="platform_constraints", group="hardware",
       owner="sanafe_arch_synth", type=T.INT, category=Category.ADVANCED,
       exposure="user", label="Tile Grid Cols",
       effect="Fixes the SANA-FE NoC mesh width independently of the packed model",
       doc="NoC floorplan: tile-grid columns (mesh width). 0 derives the "
           "most-square exact grid; declare together with tile_grid_rows "
           "(both 0 or both > 0 — a half-declared grid is a config error).",
       bounds=(0, None), empty_means="0 — the derived most-square exact grid"),
    _E("pass_buffer_capacity_bytes", section="platform_constraints",
       group="hardware", owner="deployment_record.pass_carry", type=T.INT,
       category=Category.ADVANCED, exposure="user",
       label="Pass Buffer Capacity (bytes)",
       effect="Optional ceiling on the host-side pass-carry buffer; a "
              "schedule needing more refuses at mapping time",
       doc="[B] The platform's declared buffer for rasters carried across "
           "schedule pass boundaries. 0 = undeclared: the required buffer "
           "stays a reported metric (carry_peak_live_bytes) and nothing "
           "gates. When declared, a program whose worst pass boundary needs "
           "more live raster bytes refuses BEFORE any simulation, naming "
           "both numbers. Never a search axis.",
       bounds=(0, None), empty_means="0 — undeclared; metric only, no gate"),
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
