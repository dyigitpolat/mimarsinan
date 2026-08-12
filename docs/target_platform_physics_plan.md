# Target Platform Physics — integration plan

2026-08-13 · status: **plan, for owner review** · prerequisite: the streamed
mixed-domain seam fix must land first.

## Why

The deployment formalization (W4) records *counts*: cells used, MACs, spikes,
packets, passes, programming bytes, timesteps. The cost model prices them, but
with **global default coefficient bands living in code**. A target cannot say
"my chip's numbers are these", so no absolute Area / Energy / Latency /
Throughput can be attributed to a real platform, and cross-platform comparison
would be comparing defaults rather than chips.

This plan adds the missing half — the per-target physics — the way EDA/CAD
tools do it: the vendor declares per-unit constants; the framework multiplies
them by the quantities it already measures.

Non-goal: inventing numbers. **Research-first**: every constant is grounded in
published work or a datasheet, and anything estimated is marked as such, with
its reasoning, in the profile's description file.

## 1. The object

A `PlatformPhysics` profile is a typed, versioned declaration. Every constant
carries evidence, not just a value:

```python
PhysicsConstant:
    low / nominal / high : float        # a band, always (a point value sets all three)
    unit                 : str          # "um^2/cell", "pJ/MAC", "ns", "mW", ...
    evidence_kind        : "published" | "datasheet" | "derived" | "estimated"
    citation             : str          # paper id + table/figure/page, or datasheet rev
    derivation           : str          # how a per-unit number was obtained from a system-level one
    note                 : str          # why an estimate is reasonable, and its expected error
PlatformPhysicsValidity:
    technology_node_nm, supply_v, temperature_c, array_size_assumed, ...
```

Grouped by the category each constant prices (the grouping is also the wizard's
panel layout and the objectives' availability keys):

| Group | Constants (unit) |
|---|---|
| `array` | `area_per_cell` (µm²), `area_per_cell_per_weight_bit`, `e_mac` (pJ), `t_array_read` (ns), `conductance_levels`, `write_sigma`, `read_sigma` |
| `periphery` | `area_per_adc`, `adc_sharing_factor` (cols/ADC), `e_adc_conversion`, `t_adc_conversion`, `area_per_row_driver`, `e_row_drive` |
| `neuron` | `area_per_neuron_logic`, `area_per_state_bit`, `membrane_bits`, `e_neuron_update`, `e_leak_per_neuron_step` |
| `interconnect` | `area_per_router`, `area_per_tile_fixed`, `e_intra_tile_packet`, `e_inter_tile_hop`, `t_hop` |
| `programming` | `e_dma_per_byte`, `bytes_per_connectivity_entry`, `e_core_program`, `e_core_init`, `t_program_per_byte`, `t_core_init` |
| `sync` | `e_sync_barrier`, `t_sync_barrier` |
| `global` | `t_cycle` (ns/timestep), `area_global_fixed`, `p_static_per_core` (mW), `p_static_global` |
| `host` | `p_host` (W), `host_compute_rate` (scale vs the measuring machine) |
| `execution` | `weights_resident_across_batch` (bool), `pipeline_depth`, `overlap_policy` |

`t_cycle` deserves emphasis: it is the single constant that converts our
integer `latency_steps` into seconds, and therefore gates E2E latency and
throughput entirely.

## 2. Where it lives (SSOT)

- **Schema + registry + loader**: `src/mimarsinan/deployment_record/physics/`
  — the cost model is the primary consumer and the record must carry the
  profile. Import direction is preserved (`deployment_record` may import
  `mapping`/`config_schema`; never `pipelining`/`search`/`gui`).
- **Profiles as data, not code**: `physics_profiles/<name>.json` plus
  `physics_profiles/<name>.md` — the *profile description file* the owner asked
  for, holding citations, per-constant derivations, estimate rationales, and
  the validity domain in prose. A vendor adds a target by dropping in two files;
  no `src/` edit.
- **One resolution owner**: `pipelining/core/platform_constraints_resolver.py`
  resolves `profile + overrides → concrete PlatformPhysics` and surfaces it in
  `platform_constraints_resolved`, so it lands verbatim in the record's
  `identity.platform` and the run is reproducible and comparable.
- **One pricing owner**: `deployment_record/cost/` consumes the resolved
  physics; today's default bands remain as the *documented fallback*, and every
  `CostTerm.basis` states which was used.

## 3. Config surface

Two keys, declared the only legal way (config_schema registry entries,
`section="platform_constraints"`, `group="hardware"`):

- `platform_physics_profile` — profile name, or empty for "none declared".
- `platform_physics_overrides` — sparse map `constant → value/band` for the
  configurator panel's edits, recorded so a run states exactly what physics it
  used.

Golden resolution snapshot regenerated; the registry symmetry tests apply as
usual.

## 4. Objectives

New **absolute** axes registered in the objectives registry v2 — additions,
not surgery:

`chip_area_mm2` (min), `energy_per_inference_mj` (min), `e2e_latency_s` (min),
`throughput_inferences_s` (max), plus decomposition terms (array/periphery/
interconnect area; dynamic/static/programming energy).

Each declares an `availability` predicate naming the constants it needs, so an
incomplete profile disables **exactly** the axes it cannot back and says why —
the same mechanism that today reports *"requires the sealed record's energy
fragment"*. A run with no profile must report **no** area number at all rather
than a default masquerading as vendor data.

## 5. Three quantities the record must add

Everything else on the constant tables already has its multiplicand sealed into
every run. Missing:

1. **ADC/conversion counts** (or a declared conversion model) — the one
   energy *and* latency multiplicand not currently derivable.
2. **Membrane/state bit width** as a recorded quantity rather than an implicit
   config value.
3. **Total NoC hop count** as a first-class scalar (computable today by summing
   `traffic.noc.link_loads`, just not surfaced).

## 6. Research-first grounding

Five of the repo's twelve literature platforms already carry provenance quotes,
so their physics profiles *extend an audited declaration* rather than inventing
one. Proposed first set, in order of how well-documented their numbers are:

| Profile | Source | Why first |
|---|---|---|
| `truenorth` | Merolla 2014 (already registered with a quote) | per-spike energy and area are headline results |
| `loihi` | Davies 2018 | energy/synaptic-op, per-core area, 14 nm; already a SANA-FE preset |
| `isaac_like` | Shafiee 2016 (registered) | the classic full ADC/area/energy breakdown for analog IMC |
| `prime_like` | Chi 2016 (registered) | second analog point, different sharing regime |
| `neurram_like` | Wan 2022 (registered) | modern RRAM measurements |

Per profile the research pass produces: the constant table with a quote or
table/figure reference per value; a shown derivation wherever a per-unit number
is obtained from a system-level one; and an explicit `estimated` mark plus
rationale where the paper is silent. Paper retrieval and citation injection go
through the existing `asta-papers` tooling so the bibliography stays machine-
maintained.

A sixth profile, `generic_estimated_22nm`, exists purely as a documented
estimate baseline — clearly labelled, so nobody mistakes it for a measurement.

## 7. Wizard (Co-design tab)

- **Profile selector** — registered profiles with name, technology node, and a
  one-line provenance summary; plus `custom`.
- **Configurator panel** — grouped by the categories in §1. Each row: constant,
  band (low/nominal/high), unit, an evidence badge
  (`published`/`datasheet`/`derived`/`estimated`), the citation on hover, and an
  override field. Overridden rows are visibly marked as deviating from the
  profile.
- **Completeness readout** — which objectives this profile can back, driven by
  the *same* availability predicates the registry uses, so the operator sees
  "area: available · energy: needs `e_adc_conversion`" before launching rather
  than discovering it after a run.
- Written UX spec + owner pixel review, per the standing GUI rule.

## 8. Staging

| Stage | Content | Kind |
|---|---|---|
| P0 | schema + registry + loader + ONE fully-researched profile + validity tests | pure addition |
| P1 | config keys + resolver surfacing + record carries resolved physics (+ golden regen) | additive; records gain a field |
| P2 | the three missing record quantities | additive |
| P3 | cost model consumes physics; absolute objectives registered with availability | behavior change (new axes) |
| P4 | wizard selector + configurator + UX spec + pixel review | GUI |
| P5 | remaining researched profiles (data + description file + test each) | data |
| P6 | first cross-platform co-optimization study | research |

P0–P2 are byte-identical to today's outputs for any run that declares no
profile.

## 9. Discipline notes

- No physics constant may be hardcoded at a consumer; the resolver is the only
  reader of the profile, the cost model the only pricer.
- Every constant carries evidence; every `estimated` constant carries a note. A
  test enforces both, and that every profile validates against the schema.
- Objectives gate on availability — never a silent default. A cross-platform
  comparison where one side is `estimated` must say so in the report.
- Bands propagate: an absolute objective computed from a banded constant is
  itself banded, and the existing cost-report machinery already carries that.
