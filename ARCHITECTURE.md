# mimarsinan — Architecture Overview

mimarsinan deploys trained PyTorch models onto neuromorphic hardware cores. It
takes a native `nn.Module`, converts it into a graph of perceptrons, gradually
adapts that graph toward hardware constraints (quantization, spiking
conversion, pruning) while recovering accuracy at every step, lowers it to an
IR of neural cores, packs the cores onto a constrained chip, and then runs the
mapped chip on cycle-level simulators (nevresim C++, SANA-FE, Lava Loihi) to
produce the deployed-forward accuracy number the pipeline reports.

This file is the lean overview. Detail lives in the 16 per-module
`src/mimarsinan/<module>/ARCHITECTURE.md` files (see the module map below);
a machine test keeps this file at or under 400 lines and keeps the per-module
docs in sync with the files they describe.

## The 30-second mental model

1. **One flat config dict** drives everything. `config_schema/` owns its
   defaults, derivation, and validation; `DeploymentPlan` resolves it once into
   typed deployment axes; steps read the plan, never raw keys.
2. **A declarative step registry** (`StepSpec` list) filters itself per plan
   and is contract-validated (requires/promises) before anything runs.
3. **Adaptation is progressive**: each hardware constraint is applied at a
   fractional rate by a tuner, with accuracy recovery between increments.
4. **Spiking semantics are centralized**: four deployable spiking modes
   (`lif`, `ttfs`, `ttfs_quantized`, `ttfs_cycle_based` × `cascaded |
   synchronized` schedule) are dispatched through SSOT predicate/policy
   modules — never through scattered `if spiking_mode == ...` checks.
5. **The deployed number comes from a simulator**, and NF↔SCM parity (torch
   spiking forward vs. simulated chip) holds by construction because both
   sides share one boundary-transcoding SSOT (`spiking/segment_boundary.py`).

## Repository layout

- `src/mimarsinan/` — the Python package (16 modules, one doc each).
- `run.py` / `src/main.py` — the deployment entry points (below).
- `tests/` — unit suite (`tests/unit/`), integration tests, shared fixtures.
- `test_configs/` + `scripts/run_tier.py` — tiered end-to-end run matrices.
- `scripts/` — commit gates: typecheck, module budget, undefined names.
- `nevresim/`, `spikingjelly/` — vendored simulator / spiking dependencies.
- `generated/` — per-run working directories (configs, caches, artifacts).

## Entry points and execution flow

Run the deployment pipeline from the project root with the `env/` virtualenv
active:

```bash
source env/bin/activate
python run.py <deployment_config.json>   # CLI run, monitor GUI attached
python run.py --ui                       # wizard + run manager (spawns headless runs)
python run.py --headless <config.json>   # one run, file-based monitoring, exit code
```

`run.py` pins `CUBLAS_WORKSPACE_CONFIG` before any CUDA context (deterministic
matmul reduction order — the spiking forward flips on near-boundary neurons
otherwise) and honors `--debug` for CUDA diagnostics. All three modes converge
on the same composition root:

```
run.py  (--ui | --headless <cfg> | <cfg> via src/main.py)
   │
   ▼
PipelineSession                       pipelining/session.py — composition root
   parse_deployment_config(json) → ParsedDeploymentConfig
   apply_preset(pipeline_mode) → DeploymentPipeline(params, constraints, ...)
   │
   ▼
DeploymentPlan.resolve(config)        the single config-resolution layer
   │
   ▼
StepPlan registry                     core/pipelines/deployment_specs.py
   ordered StepSpecs, filtered per-step by applies_to(plan);
   requires/promises DAG validated up-front, failing loud;
   simulation tail spliced from BACKEND_REGISTRY.selected_step_specs(plan)
   │
   ▼
Pipeline engine                       core/engine/ — cache, contracts, hooks
   configuration → pretraining → torch mapping →
   adaptation steps (run tuners from tuning/) →
   quantization / normalization fusion → soft & hard core mapping →
   simulation backends (chip_simulation/: nevresim, SANA-FE, Lava Loihi)
```

The full step order (each step self-selects via `applies_to(plan)`):
Architecture Search, Model Configuration, Model Building, Weight Preloading,
Pretraining, Torch Mapping, Pruning Adaptation, Activation
Analysis/Adaptation, Clamp Adaptation, Activation Shifting, Activation
Quantization, LIF Adaptation, TTFS Cycle Fine-Tuning, Noise Adaptation, Weight
Quantization (+ Verification), Normalization Fusion, Soft Core Mapping, Core
Quantization Verification, Hard Core Mapping, then the capability-selected
simulation steps. Every step's outputs are persisted in the `PipelineCache`,
so runs resume from any step (`start_step` / `stop_step` in the config).

## SSOTs — where behavior lives

Each cross-cutting decision has exactly one home. Extend these; never fork
their logic locally.

| Decision | SSOT module |
|---|---|
| Config resolution: flat config → typed deployment axes | `pipelining/core/deployment_plan.py` (`DeploymentPlan.resolve`) |
| Config defaults, derivation rules, validation, key provenance | `config_schema/` (`defaults.py`, `deployment_derivation.py`, `namespaced_schema.py`) |
| Deployment mode → proven conversion recipe (driver, knobs, sim enables) | `tuning/orchestration/conversion_policy.py` (`ConversionPolicy.derive`) |
| Tuning-loop behavior constants (checkpoint, recovery, rollback, commit gate) | `tuning/orchestration/tuning_policy.py` (frozen `TUNING_POLICY`) |
| Spiking-mode taxonomy, mode predicates, per-backend capability matrix | `chip_simulation/spiking_semantics.py` |
| Behavior-carrying per-`(firing × sync)` mode dispatch | `chip_simulation/spiking_mode_policy.py` (`policy_for_spiking_mode`) |
| Segment-boundary encode/decode shared by torch forward and simulators | `spiking/segment_boundary.py` |
| Deployed-bias compensation (negative shift, TTFS half-step) | `mapping/support/bias_compensation.py` |
| `MIMARSINAN_*` environment variables (one call-time accessor each) | `common/env.py` |
| The only sanctioned log-and-degrade seam | `common/best_effort.py` (`best_effort`) |

Notes:

- `ConversionPolicy` is a deterministic `(spiking_mode, schedule) →
  ConversionRecipe` table; each special-case row carries a written rationale
  and its capability-derived backend-enable set. `config_schema` folds the
  recipe into derived parameters, so config, pipeline, and wizard agree.
- `TuningPolicy` freezes the formerly config-readable `tuning_*` knobs at
  their proven values; tuning-loop behavior is code, not configuration.
- `best_effort` is only for telemetry/rendering side work. Verification,
  mapping, and training failures always propagate; a ratchet test keeps the
  broad-except allowlist shrinking.

## Module map

One row per top-level module under `src/mimarsinan/`. Read the linked doc
before editing that module.

| Module | Purpose | Detail |
|---|---|---|
| `pipelining` | Pipeline engine: session composition root, `DeploymentPlan`, step registry, cache, and all concrete pipeline steps | [doc](src/mimarsinan/pipelining/ARCHITECTURE.md) |
| `config_schema` | Deployment-config SSOT: defaults, presets, derivation, validation, key provenance, display views | [doc](src/mimarsinan/config_schema/ARCHITECTURE.md) |
| `data_handling` | `DataProvider` registry, preprocessing spec, and `DataLoaderFactory` (torch + opt-in FFCV) | [doc](src/mimarsinan/data_handling/ARCHITECTURE.md) |
| `models` | Torch model zoo, NN building blocks (activations/decorators), and the deployable spiking simulator (`SpikingHybridCoreFlow`) | [doc](src/mimarsinan/models/ARCHITECTURE.md) |
| `model_training` | `BasicTrainer` facade, training recipes, losses, aux-model QAT trainers, pretrained-weight loading | [doc](src/mimarsinan/model_training/ARCHITECTURE.md) |
| `search` | Multi-objective NAS + hardware co-search: problems, optimizers (NSGA-II, LLM), search-space SSOT | [doc](src/mimarsinan/search/ARCHITECTURE.md) |
| `torch_mapping` | Native torch model → Mapper DAG conversion: FX trace, normalization, representability, mapper emission | [doc](src/mimarsinan/torch_mapping/ARCHITECTURE.md) |
| `tuning` | Progressive hardware adaptation: `SmoothAdaptationTuner` loop, adaptation axes, concrete tuners, `ConversionPolicy`/`TuningPolicy` | [doc](src/mimarsinan/tuning/ARCHITECTURE.md) |
| `transformations` | Pure weight/activation transforms: effective-parameter view, quantization, normalization fusion, pruning | [doc](src/mimarsinan/transformations/ARCHITECTURE.md) |
| `mapping` | Mapper graph → IR → packed hard cores: pruning, packing, latency, layout estimation, verification, chip export | [doc](src/mimarsinan/mapping/ARCHITECTURE.md) |
| `spiking` | Spike-train encoding, segment-boundary transcoding SSOT, unified segment-aware NF forward, cascade calibration | [doc](src/mimarsinan/spiking/ARCHITECTURE.md) |
| `chip_simulation` | Simulation backends (nevresim, SANA-FE, Lava Loihi, TTFS), spiking-semantics SSOTs, certification/coverage/Pareto instruments | [doc](src/mimarsinan/chip_simulation/ARCHITECTURE.md) |
| `code_generation` | nevresim C++ source generation from mapped chips (`ChipModel`, main templates, span export) | [doc](src/mimarsinan/code_generation/ARCHITECTURE.md) |
| `visualization` | Write-to-file Graphviz/matplotlib/Plotly renderings of pipeline artifacts | [doc](src/mimarsinan/visualization/ARCHITECTURE.md) |
| `gui` | Browser-based run monitor, run manager, and configuration wizard (FastAPI + SPA) | [doc](src/mimarsinan/gui/ARCHITECTURE.md) |
| `common` | Leaf utilities: env-var SSOT, `best_effort`, file I/O, compiler discovery, `Reporter`, diagnostics | [doc](src/mimarsinan/common/ARCHITECTURE.md) |

Dependency shape: `common` is the leaf everyone may use; `config_schema`,
`data_handling`, `transformations`, and `code_generation` are near-leaves;
`pipelining` sits at the top and composes everything; `chip_simulation` and
`tuning` host the semantics SSOTs the rest consult. Per-module docs list exact
dependencies and dependents.

## Testing & gates

### The unit suite (≤ ~2 minutes, run it after every change)

```bash
source env/bin/activate
python -m pytest            # default profile from pytest.ini
```

`pytest.ini` defaults: parallel (`-n auto --dist worksteal`), quiet, slow
tests deselected (`-m "not slow"`), 60 s per-test timeout. Conventions the
suite enforces:

- **CPU-only by default** — `tests/conftest.py` blanks `CUDA_VISIBLE_DEVICES`
  unless `MIMARSINAN_TEST_CUDA=1`, and pins one torch thread per xdist worker.
- **No network** — `tests/unit/conftest.py` installs an autouse socket guard;
  anything beyond loopback raises. Use local fixtures or an integration tier.
- Slow/multi-module tests carry the `slow` / `integration` markers; opt in
  with `pytest -m slow`.
- **Golden resolution snapshot** — `tests/unit/config_schema/
  test_golden_resolution_snapshot.py` pins the full resolved numeric surface
  (config, plan, budgets, recipe/policy tables) of every tier config against a
  checked-in snapshot; regenerate deliberately with
  `scripts/regen_golden_resolution_snapshot.py` and audit the diff (a changed
  line = a changed resolved value).

### Architecture ratchets (`tests/unit/architecture/`)

Codebase-shape invariants that only tighten:

- `test_lazy_imports.py` — function-level imports live on a shrinking
  allowlist (cycle-breakers and optional heavy backends only).
- `test_broad_excepts.py` — `except Exception:` is a tracked, shrinking
  allowlist (`best_effort` or documented fallbacks only).
- `test_module_budget.py` — runs `scripts/check_module_budget.py --strict`:
  ≤ 300 LOC per file, ≤ 10 sibling `.py` files per directory, with shrinking
  allowlists for modules scheduled to split.
- `test_entrypoint_imports.py` — entrypoint/public-package import surfaces
  stay importable; runs `scripts/check_undefined_names.py` (pyflakes
  `undefined name` findings against a checked-in baseline — new ones fail).
- `test_architecture_docs.py` — docs drift guard: this file stays ≤ 400
  lines, every top-level module has exactly one `ARCHITECTURE.md`, and each
  module doc's Key-files table covers exactly its direct children.
- `test_legacy_imports.py` / `test_no_network_guard.py` — banned legacy import
  paths stay dead; the network guard itself is tested.

### Type checking

```bash
./scripts/typecheck.sh      # basedpyright --level error, curated rule set
```

The curated `pyrightconfig.json` gate must report **zero errors**. It is too
slow for the pytest suite; run it before committing.

### Tiered integration runs (`test_configs/`)

`test_configs/generate.py` is the SSOT that emits every tier's config JSONs
and manifests — never hand-edit the generated files; edit the generator and
re-run it. The tiers trade coverage for wall clock:

| Tier | Runs | Dataset | Wall budget / run |
|---|---|---|---|
| `tier0` | 25 | MNIST | 5 min |
| `tier1` | 8 | CIFAR-10 | 120 min |
| `tier2` | 3 | ImageNet / CIFAR-100 | 360 min |

Each run is one full deployment pipeline over a `(mode × quantization ×
vehicle × S × platform)` cell; manifests record the hypervolume cell and tags
(`pruned`, `sched`, `offload`, `wall_risk`, ...). Execute a tier with
`python scripts/run_tier.py` — it runs each config headlessly under the wall
budget and prints a result table from the run's persisted status/metrics.

## Contributing rules

1. **Tests first.** Write contained unit tests covering the design before
   adding or changing code; the tests dictate the design. Run the suite after
   every change.
2. **Never weaken checks.** Do not remove assertions or silence failing
   checks to match potentially incorrect code — fix the code.
3. **Read, then update, the module doc.** Read the relevant
   `ARCHITECTURE.md` before editing a module; update it when you add/remove
   files, change public API, or change cross-module imports.
4. **New config keys go to `config_schema/`**: a default in `defaults.py`, a
   `KeySpec` in `namespaced_schema.py`, derivation in
   `deployment_derivation.py` if implied by mode, validation if user-facing —
   and steps read them through `DeploymentPlan`, not raw dict lookups.
5. **Mode behavior goes to the SSOTs.** New spiking-mode behavior extends
   `spiking_semantics` predicates / `SpikingModePolicy` / `ConversionPolicy`;
   no local `if spiking_mode == ...` branches.
6. **Failures propagate.** Wrap only non-critical telemetry/rendering in
   `best_effort`; everything else raises. New broad excepts fail the ratchet.
7. **Ratchets only tighten.** Allowlists (lazy imports, broad excepts, module
   budget, undefined-names baseline) may shrink, never grow.
8. **Respect the module budget.** Keep files ≤ 300 LOC and directories ≤ 10
   sibling `.py` files; split into subpackages with an `__init__.py`
   (conservative exports) and an `ARCHITECTURE.md`.
9. **Boy-scout rule.** When you meet duplicate or near-duplicate logic, make
   a shared abstraction and test it — don't add a third copy.
10. **Comment style.** Self-documenting names over commentary; one-line module
    docstrings; 1–3-line class/method docstrings of intent; no multiline `#`
    blocks unless they state a non-obvious invariant or cross-language
    contract.
11. **Gates before commit.** `python -m pytest` green and
    `./scripts/typecheck.sh` at zero errors; for pipeline-behavior changes,
    a `test_configs/` tier-0 spot check is the integration safety net.
12. **Run from the root.** Activate `env/` and drive the pipeline via
    `run.py`; new env vars get a call-time accessor in `common/env.py`.
