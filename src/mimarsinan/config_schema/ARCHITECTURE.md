# config_schema/ — Deployment config SSOT: defaults, derivation, validation, provenance, display views

Every pipeline run consumes one flat config dict merged from deployment parameters and
platform constraints; this module is the single source of truth for its default values,
pipeline-mode presets, and the derivation rules that fold spiking-mode implications and
the `ConversionPolicy` recipe into that dict. It also validates both the deployment
config JSON (`main.py` input) and the merged runtime config, records each key's concern
group and provenance in a namespaced registry, and builds presentation-only config views
for the GUI monitor. `DeploymentPipeline` and the wizard both build configs through these
helpers so they stay in sync.

## Quantization assembly

AQ/WQ are derived pipeline-assembly modes, not free config booleans.
`derive_deployment_parameters` derives `activation_quantization` from the mode:
ON for `lif`, `ttfs_quantized`, and `ttfs_cycle_based` (cascaded and
synchronized); OFF for analytical `ttfs` and for every float-weight (vanilla)
deployment. Weight quantization is bits-driven: `weight_bits` declares a
quantized artifact, and a float-weight deployment is declared via the vanilla
mechanism (`pipeline_mode: "vanilla"`, or `weight_quantization: false` without
`weight_bits`) — `enforce_quantization_assembly_contract` (called at session
parse) rejects `weight_quantization: false` alongside `weight_bits` under a
non-vanilla pipeline. An explicit `activation_quantization` that contradicts
the derivation raises a loud config error; an absent key derives silently.
`MIMARSINAN_UNSAFE_QUANT_OVERRIDES=1` (accessor in `common/env.py`) is the
research escape: contradictions are honored verbatim and logged with an
`[UNSAFE-OVERRIDE]` line. Pipeline-mode presets never inject AQ/WQ — a
preset-injected value would masquerade as an explicit one under this contract.

## The configurability SSOT (`registry/`)

`registry/` is the one declarative table of EVERY config key (the live
`CONFIG_KEYS_SET` plus top-level document and platform structural keys —
coverage is validated at import, both directions, never hand-trusted).
Workload-profile-injectable keys (`common/workload_profile.py`) are registered
here with NO schema default: absence is meaningful — explicit config > model
registration > data registration > the consumer's frozen workload-neutral
default.
Each `ConfigKeySchema` carries provenance (group/owner/derivation/exposure,
the KeySpec heritage), widget knowledge (type/options/bounds/unit), docs
(label/doc/effect), wizard altitude (`category`: basic/advanced/derived/
runtime + `declarable`), declarative `Relevance` predicates (JSON-codable so
the frontend evaluates them without Python), derived-key `why`
explanations plus optional machine-readable `meta` facts, the SSOT-source
`provenance` of a derived-by-default value (one of the `PROVENANCE_SOURCES`
vocabulary: TUNING_POLICY / provider registration / builder profile /
ConversionPolicy recipe / derivation rule / consumer frozen default —
rendered generically as a badge beside every green derived value), the
`derived_default` that SOURCE produces for a given config state, the
`legal_values` set (see below), and `hidden` (a derivation-owned key that
renders on NO UI surface: `pipeline_mode`, `pretrained_weight_sets`).

BUILD-TIME INVARIANTS (`registry/validate_registry`, `ConfigKeySchema.__post_init__`):
a DERIVED key MUST name a `provenance` source; a `derived_default` MUST name one
too; and a key naming a source MUST be able to produce a CONCRETE value (schema
default, `derived_default`, or DERIVED — the wizard renders the VALUE, never prose
about where it comes from). `legal_values` requires a discrete type (enum/str/bool)
and may return `None` (legality does not apply in this state — neither locked nor
judged, distinct from an empty set).
`effective_value(config, key)` is THE accessor consumers use for a frozen
fallback (explicit > schema default > `derived_default`), so the number a
consumer reads and the number the wizard renders cannot drift.

THE LEGAL-VALUE-SET LAW. A key whose legality depends on other config declares
`legal_values(cfg) -> tuple`; `legal_values_view` exposes the set for the current
state and the UI is driven by its SIZE alone: `|legal| == 1` LOCKS the field
(read-only, rendered as the derived value); `|legal| > 1` allows an override that
offers ONLY those options; an explicit illegal value is a keyed, remediable
`legal_value_set` error (`validation.legality_errors`, remedies prescribed by the
rule). The rules live in the SSOTs (`spiking_semantics.legal_firing_modes` /
`legal_spike_generation_modes` / `legal_thresholding_modes`, cross-checked in test
against `FiringStrategy`; `temporal_allocation.S_ALLOCATION_SUPPORTED_MODES`),
never as per-mode ladders in the derivation, the validators, or JS.
The wizard renders entirely from `serialize_registry()`;
`parse_deployment_document` classifies any config document against it
(unknown keys are reported as dotted paths, never dropped); the
representability unit test (`tests/unit/config_schema/test_wizard_representability.py`)
round-trips every tier-0/0.1/1/2 config through parse -> emit and pins the
wizard as the configurability SSOT.

## Key files
| File | Purpose |
|---|---|
| `defaults.py` | `DEFAULT_DEPLOYMENT_PARAMETERS` / `DEFAULT_PLATFORM_CONSTRAINTS` / training+tuning recipes, `PIPELINE_MODE_PRESETS` + `apply_preset`, `CONFIG_KEYS_SET`, and copy-returning getters (incl. user/system splits driven by exposure). |
| `deployment_derivation.py` | `legal_value_error` / `legal_values_for` / `legality_bearing_keys` (the legal-value-set law's derivation surface), `derive_deployment_parameters` (quantization/pipeline-mode rules — the ONLY derivation implementation; delegates the recipe fold to `recipe_fold.py`; `mirror_training_recipe` reflects the training recipe as-is into the tuning recipe, a document-declared `tuning_recipe` alongside it conflicts loudly), `enforce_quantization_assembly_contract` (bits-driven WQ contract, see “Quantization assembly”), `derive_pipeline_runtime_parameters` (GENERIC over the registry: each legality-bearing key takes its `derived_default` when absent and is judged against its `legal_values` set when declared — an illegal value raises for programmatic callers while the wizard surfaces the same rule as a keyed error — and defaults `negative_value_shift` ON: the boundary-lossless policy whose OFF position is the mapper's subsume-forward path, never a silent clamp), and `derive_platform_constraints` (max_axons/max_neurons from the declared core grid; consistent explicit accepted, contradiction rejected; scalar-only documents skipped). |
| `recipe_fold.py` | `fold_conversion_recipe` — folds the `ConversionPolicy.derive` recipe into deployment parameters under the TWO-TIER contract: internal recipe knobs and the recipe-owned correctness keys (`cycle_accurate_lif_forward`) are written authoritatively, while registry-declarable knobs take the recipe value as their mode-aware DEFAULT — an explicit document value wins (`explicit_keys` names the declared keys, so merged-in generic defaults never masquerade as declarations); sim enables are capability-derived — unsupported is authoritative-off with an explicit ON rejected loudly, supported defaults ON with a declared OFF honored as a stored user override; and the `lif_exact_qat` → `lif_per_hop_retiming` auto-pairing (the exact-QAT arm deploys as the per-hop RE-TIMED pair; an explicit contradiction, a non-lif mode, or Novena fails loud — lif_exact_qat_program §6). |
| `registry/` | The configurability SSOT (see above): `types.py` (`ConfigKeySchema`/`FieldType`/`Category`, plus `promote_when` — mode-aware prominence: an ADVANCED key renders primary while the predicate holds — `empty_means` — what an absent value resolves to — and `provided_by` — ownership transfer: while the key is irrelevant, this other group produces its value and the card renders an ownership chip where the hand field would be), `relevance.py` (predicate combinators + JSON codec), `groups.py` (the eleven `CONCERN_GROUPS`; the wizard workbench sections host whole groups, so the taxonomy IS the placement; `co_search` hosts the search concern — it co-optimizes model AND hardware, so it never nests under either — and declares `empty_state`, the quiet one-line card shown when none of a group's keys exist; `mapping_strategy` splits the platform concern by the capability-vs-strategy principle: what the hardware CAN do stays in `hardware`, what we CHOOSE when mapping — scheduling, encoding placement (no schema default: an explicit choice configs pin as data), the pruning family, the `s_allocation` temporal allocation — renders on the Co-Design mapping-strategy panel above the full-width mapping panel; `conversion` keeps only the derived AQ/WQ semantics — its former knobs re-homed by concern (calibration/adaptation knobs to `tuning`), so no conversion panel renders), `entries_*.py` (the per-key table; the recipe-defaulted sim enables carry `meta` — the machine-readable capability flag the vehicles card renders toggles vs muted lines from — and the cores-derived max_axons/max_neurons plus the correctness mechanism `cycle_accurate_lif_forward` is DERIVED non-declarable (validation rejects a document that pins it), `negative_value_shift` is a `mapping_strategy` BASIC knob with two value-sound positions, and the pretrained-weight regime: `preload_weights` (BASIC switch, legality-bearing — disabled when the builder registers no applicable set), `pretrained_weight_set` (DERIVED-but-declarable choice among the builder's registered sets, legality-bearing over the registered ids), and `weight_source` DERIVED-but-declarable (the chosen set's source; an explicit path wins) while the injection key `pretrained_weight_sets` (the tuple of `PretrainedWeightSet` records the builder registers) is DERIVED non-declarable + hidden; `arch_search`/`search_space` live in `co_search` while `model_config_mode`/`hw_config_mode` stay with the concern whose provenance they declare), `build.py` (assembly, defaults injection from `defaults.py`, import-time coverage validation, `serialize_registry`), `parse.py` (document classification, loud unknown-key report). |
| `resolve.py` | `resolve_draft(draft) -> Resolution` — one draft resolved the way a run would resolve it: merged flat config, DERIVED keys with WHY, errors attached to their keys (`rule_id`), explicit-key diff vs schema defaults. A legality error supersedes the plain structural message for its key and STOPS the derivation (no hypothetical values). Powers `/api/config/resolve`. `derived_view(resolved)` is public so a caller holding an ENRICHED config (the wizard folds the model builder's registrations in) re-derives the rows against the values the run would actually see; `derived_values_view(resolved)` serves the CONCRETE prospective value of every source-bearing key (the green in-field text; `{}` while erroring); `legal_values_view(cfg)` + `effective_view(dp)` serve the always-computable legal sets. |
| `runtime.py` | `build_flat_pipeline_config` — merges defaults + overrides + preset + the derivation passes (deployment, runtime, platform-constraints) the same way `DeploymentPipeline` does, without device I/O. |
| `validation.py` | `legality_errors` (THE legal-value-set check + rule-prescribed one-click remedies, shared by both validators and the wizard), `validate_against_registry` (THE field-domain check: every present, registry-known key judged against its declared scalar type / inclusive numeric `bounds` / enum `options` — keyed, remediable `field_domain` rows in the same channel as the legality law; legality-bearing keys are owned by `legality_errors` and skipped, so the sets never overlap; wired into BOTH doors so a bad hand-authored value gets the same verdict at runtime and in the wizard resolve payload), `validate_deployment_config` (JSON shape, spiking-mode membership, generic legality, generic field-domain over the flattened document, coalescing-key rejection, `non_declarable_key_errors` — a document declaring a RUNTIME key or a non-declarable DERIVED key is rejected, so unexposed knobs cannot be set programmatically), `validate_merged_config` (runtime flat config: spiking legality + field domain), `s_allocation_config_errors` (loud-rejects reserved `explicit`/`budget` temporal-allocation modes). |
| `namespaced_schema.py` | KeySpec provenance table + byte-identical `to_namespaced`/`to_flat` bijection, DERIVED from the registry (`KeySpec`, `KEY_SPECS`, `LEGACY_KEY_TABLE`, `keys_with_derivation` / `keys_with_exposure` / `provenance_table`, re-exported `CONCERN_GROUPS`). |
| `display_view.py` | `build_config_display_view` / `build_pipeline_config_view` / `load_saved_config_from_run_dir` — structured, JSON-safe monitor payloads with per-field source annotation (explicit/default/derived/preset/runtime). |
| `display_view_meta.py` | Display metadata resolved from the registry (no hand tables): `CONFIG_DISPLAY_GROUPS` (= concern groups + `other`), `RUNTIME_KEYS` / `DERIVED_KEYS` / `TOP_LEVEL_RUN_KEYS` derived from categories, field-source and default-value resolution. |
| `display_view_build.py` | Nested display-block builders: recipe, model-config (via model registry schema), cores, arch-search blocks, and the pipeline-steps preview. |

## Dependencies
- `chip_simulation` — `spiking_semantics` predicates (`require_known_spiking_mode`, `requires_ttfs_firing`, `forces_activation_quantization`, `is_cycle_based`, `is_cascaded_ttfs`) and the legal-value-set / derived-default functions (`legal_firing_modes`, `legal_spike_generation_modes`, `legal_thresholding_modes`, `derived_firing_mode`, `derived_spike_generation_mode`) used by derivation, validation, and the registry entries.
- `tuning` — `orchestration.conversion_policy.ConversionPolicy` (the mode→recipe SSOT folded into derived parameters), `orchestration.temporal_allocation` (`s_allocation` mode constants, the supported-mode legal set, and the unsupported-mode error), and `orchestration.tuning_policy.TUNING_POLICY` (the frozen fallbacks `endpoint_floor_steps` / `prefix_stage_lr` / `endpoint_floor_lr` render as derived values).
- `mapping` — `platform.coalescing.coalescing_config_errors` rejects deprecated coalescing keys during config validation.
- `pipelining` — lazy display-view imports: `core.registry.model_registry.get_model_config_schema` (model-config field schema) and `core.pipelines.deployment_specs` (pipeline-steps preview).
- `gui` — lazy display-view imports: `wizard.config_builder.build_deployment_config_from_state` (expand saved wizard configs) and `wizard.schema.get_wizard_nas_schema` (arch-search block labels).

## Dependents
- `pipelining` — `core.pipelines.deployment_pipeline` (defaults + both derivation passes) and `core.platform_constraints_resolver` (`DEFAULT_PLATFORM_CONSTRAINTS`).
- `gui` — `wizard/schema.py` (form defaults), `wizard/validation.py` (`validate_deployment_config`), `wizard/config_builder.py` (`KEY_SPECS` / `keys_with_exposure`), `server/routes_wizard.py` (`build_flat_pipeline_config` preview), and `runs.py` / `runtime/process_monitor.py` / `runtime/collector/mixins/read_api.py` (monitor config views).

## Exported API
`__init__.py` re-exports:
- Defaults and presets: `DEFAULT_DEPLOYMENT_PARAMETERS`, `DEFAULT_PLATFORM_CONSTRAINTS`, `DEFAULT_TRAINING_RECIPE`, `DEFAULT_TUNING_RECIPE`, `PIPELINE_MODE_PRESETS`, `CONFIG_KEYS_SET`, the `get_default_*` / `get_pipeline_mode_presets` / `get_config_keys_set` getters, `apply_preset`.
- Runtime merge: `build_flat_pipeline_config`.
- Validation: `validate_deployment_config`, `validate_merged_config`, `s_allocation_config_errors`.
- Display views: `build_config_display_view`, `build_pipeline_config_view`.
- Provenance registry: `CONCERN_GROUPS`, `KEY_SPECS`, `KeySpec`, `LEGACY_KEY_TABLE`, `keys_with_exposure`, `keys_with_derivation`, `provenance_table`, `to_flat`, `to_namespaced`.
- Registry accessor: `effective_value` (via `config_schema.registry`) — the one read of a key's frozen fallback, used by pipeline steps.
