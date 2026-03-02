# Mimarsinan — Software Architecture

> **Mimarsinan** is a framework for deploying deep neural networks onto neuromorphic (spiking neural network) hardware. It provides an end-to-end pipeline that takes a high-level model description, performs architecture search, trains and quantizes the model, maps it onto hardware cores, and verifies correctness through spiking simulation.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Project Layout](#2-project-layout)
3. [Entry Point and Configuration](#3-entry-point-and-configuration)
4. [Pipeline Architecture](#4-pipeline-architecture)
   - [Pipeline Engine](#41-pipeline-engine)
   - [Pipeline Step Contract](#42-pipeline-step-contract)
   - [Pipeline Cache](#43-pipeline-cache)
   - [Pipeline Modes](#44-pipeline-modes)
5. [Pipeline Steps in Detail](#5-pipeline-steps-in-detail)
   - [Architecture Search](#51-architecture-search)
   - [Model Building](#52-model-building)
   - [Pretraining](#53-pretraining)
   - [Activation Analysis](#54-activation-analysis)
   - [Clamp Adaptation](#55-clamp-adaptation)
   - [Input Activation Analysis](#56-input-activation-analysis)
   - [Activation Shifting](#57-activation-shifting)
   - [Activation Quantization](#58-activation-quantization)
   - [Pruning Adaptation](#59-pruning-adaptation)
   - [Weight Quantization](#510-weight-quantization)
   - [Quantization Verification](#511-quantization-verification)
   - [Normalization Fusion](#512-normalization-fusion)
   - [Soft Core Mapping](#513-soft-core-mapping)
   - [Core Quantization Verification](#514-core-quantization-verification)
   - [CoreFlow Tuning](#515-coreflow-tuning)
   - [Hard Core Mapping](#516-hard-core-mapping)
   - [Simulation](#517-simulation)
6. [Model Subsystem](#6-model-subsystem)
   - [Perceptron](#61-perceptron)
   - [PerceptronFlow and PerceptronMixer](#62-perceptronflow-and-perceptronmixer)
   - [Supermodel](#63-supermodel)
   - [Activation Decorators](#64-activation-decorators)
   - [Model Builders](#65-model-builders)
7. [Mapper Graph — ModelRepresentation](#7-mapper-graph--modelrepresentation)
   - [Mapper Hierarchy](#71-mapper-hierarchy)
   - [Dual Purpose: Forward Pass and Hardware Mapping](#72-dual-purpose-forward-pass-and-hardware-mapping)
8. [Intermediate Representation (IR)](#8-intermediate-representation-ir)
   - [IRGraph, IRNode, IRSource](#81-irgraph-irnode-irsource)
   - [NeuralCore](#82-neuralcore)
   - [ComputeOp](#83-computeop)
   - [IRMapping](#84-irmapping)
9. [Hardware Mapping](#9-hardware-mapping)
   - [SoftCore and SoftCoreMapping (Legacy)](#91-softcore-and-softcoremapping-legacy)
   - [HardCore and HardCoreMapping](#92-hardcore-and-hardcoremapping)
   - [HybridHardCoreMapping](#93-hybridhardcoremapping)
10. [Training and Tuning](#10-training-and-tuning)
    - [BasicTrainer](#101-basictrainer)
    - [AdaptationManager](#102-adaptationmanager)
    - [SmartSmoothAdaptation](#103-smartsmoothadaptation)
    - [Tuner Hierarchy](#104-tuner-hierarchy)
    - [CoreFlowTuner](#105-coreflowtuner)
11. [Spiking Simulation](#11-spiking-simulation)
    - [SpikingUnifiedCoreFlow](#111-spikingunifiedcoreflow)
    - [SpikingHybridCoreFlow](#111b-spikinghybridcoreflow)
    - [Nevresim (C++ Simulator)](#112-nevresim-c-simulator)
    - [SimulationRunner](#113-simulationrunner)
12. [Architecture Search](#12-architecture-search)
    - [Search Framework](#121-search-framework)
    - [NSGA-II Optimizer](#122-nsga-ii-optimizer)
    - [Kedi Optimizer (LLM-based)](#123-kedi-optimizer-llm-based)
    - [JointArchHwProblem](#124-jointarchhwproblem)
13. [Data Handling](#13-data-handling)
14. [Visualization](#14-visualization)
    - [Static Visualizations](#141-static-visualizations)
    - [Browser-Based Pipeline Monitor (GUI)](#142-browser-based-pipeline-monitor-gui)
15. [Code Generation](#15-code-generation)
16. [Key Data Flow Diagram](#16-key-data-flow-diagram)
17. [Dependency Graph](#17-dependency-graph)
18. [Conventions and Patterns](#18-conventions-and-patterns)
19. [Contributing Guide](#19-contributing-guide)

---

## 1. High-Level Overview

Mimarsinan converts a conventional deep learning model into a spiking neural network deployable on neuromorphic hardware. The framework is organized around a **pipeline** that executes a sequence of **steps**, each transforming the model closer to its final hardware-mapped form:

```
Configuration  →  Architecture Search  →  Model Build  →  Train
     →  Quantize (activations & weights)  →  Fuse Normalizations
     →  Map to IR  →  Tune Spike Thresholds  →  Map to Hardware Cores
     →  Simulate & Verify
```

Each step reads from and writes to a **pipeline cache**, which persists intermediate artifacts to disk. Steps declare explicit data contracts (`requires`, `promises`, `updates`, `clears`) that the pipeline engine verifies at assembly time, ensuring sound data flow.

---

## 2. Project Layout

```
mimarsinan/
├── run.py                          # CLI entry point (alternative to src/main.py)
├── requirements.txt                # Python dependencies
├── src/
│   ├── main.py                     # Primary entry point with run_pipeline()
│   ├── init.py                     # Global initialization (CUDA, nevresim path)
│   └── mimarsinan/
│       ├── pipelining/             # Pipeline engine and all pipeline steps
│       │   ├── pipeline.py         # Pipeline class (engine)
│       │   ├── pipeline_step.py    # PipelineStep base class
│       │   ├── cache/              # PipelineCache + load/store strategies
│       │   ├── pipelines/
│       │   │   └── deployment_pipeline.py  # Unified configurable pipeline
│       │   └── pipeline_steps/     # Individual step implementations
│       ├── models/                 # Neural network model definitions
│       │   ├── layers.py           # Perceptron, activations, decorators
│       │   ├── supermodel.py       # Top-level model wrapper
│       │   ├── perceptron_mixer/   # Model architectures
│       │   │   ├── perceptron_mixer.py   # MLP-Mixer architecture
│       │   │   └── vision_transformer.py # Vision Transformer (ViT) architecture
│       │   ├── builders/           # Model builder classes
│       │   │   ├── perceptron_mixer_builder.py
│       │   │   └── vit_builder.py  # VisionTransformer builder
│       │   ├── unified_core_flow.py # Spiking simulation for IRGraph
│       │   └── hybrid_core_flow.py  # Spiking simulation for HybridMapping
│       ├── mapping/                # Model → hardware mapping
│       │   ├── ir.py               # Unified IR (IRGraph, NeuralCore, ComputeOp)
│       │   ├── ir_mapping.py       # ModelRepresentation → IRGraph
│       │   ├── ir_pruning.py       # Post-pruning IR compaction (remove zeroed rows/cols)
│       │   ├── mapping_utils.py    # Mapper classes + SoftCoreMapping
│       │   ├── softcore_mapping.py # SoftCore, HardCore, HardCoreMapping
│       │   ├── core_packing.py     # Generic best-fit greedy bin-packing
│       │   ├── hybrid_hardcore_mapping.py  # Hybrid neural+compute mapping
│       │   └── layout/             # Shape-only layout estimation for search
│       │       ├── layout_ir_mapping.py  # Collect LayoutSoftCoreSpecs from mapper graph
│       │       ├── layout_packer.py      # Pack layout softcores into layout hardcores
│       │       └── layout_types.py       # LayoutSoftCoreSpec, LayoutHardCoreType, etc.
│       ├── tuning/                 # Training-aware tuning subsystem
│       │   ├── adaptation_manager.py       # Manages activation decorators
│       │   ├── smart_smooth_adaptation.py  # Gradual adaptation framework
│       │   └── tuners/             # Specialized tuner implementations
│       ├── model_training/         # Training utilities
│       │   ├── basic_trainer.py    # Standard training loop
│       │   └── training_utilities.py
│       ├── transformations/        # Model transformations
│       │   ├── perceptron_transformer.py   # Weight/bias fusion utilities
│       │   └── weight_quantization.py
│       ├── search/                 # Architecture search subsystem
│       │   ├── problem.py          # SearchProblem protocol
│       │   ├── results.py          # SearchResult, Candidate, ObjectiveSpec
│       │   ├── optimizers/         # NSGA-II, Kedi LLM optimizer
│       │   └── problems/           # Concrete search problems
│       ├── data_handling/          # Dataset management
│       │   ├── data_provider.py    # DataProvider base class
│       │   ├── data_provider_factory.py    # Factory with registry
│       │   ├── data_loader_factory.py
│       │   └── data_providers/     # MNIST, CIFAR-10, CIFAR-100, ECG
│       ├── chip_simulation/        # Nevresim C++ simulator interface
│       │   ├── nevresim_driver.py  # Python ↔ C++ bridge
│       │   ├── simulation_runner.py
│       │   ├── compile_nevresim.py
│       │   └── execute_nevresim.py
│       ├── code_generation/        # C++ code generation for chip model
│       │   ├── cpp_chip_model.py   # ChipModel, Core, Neuron, SpikeSource
│       │   ├── generate_main.py    # main.cpp template instantiation
│       │   └── main_cpp_template*.py
│       ├── visualization/          # Graphviz, matplotlib visualizations
│       │   ├── mapping_graphviz.py # IR/SoftCore/HardCore/Hybrid DOT graphs
│       │   ├── softcore_flowchart.py
│       │   ├── activation_function_visualization.py
│       │   ├── histogram_visualization.py
│       │   ├── hardcore_visualization.py
│       │   └── search_visualization.py
│       ├── gui/                    # Browser-based pipeline monitoring GUI
│       │   ├── __init__.py         # start_gui(), GUIHandle (hook registration)
│       │   ├── data_collector.py   # Thread-safe metric/snapshot store
│       │   ├── reporter.py         # GUIReporter (Reporter protocol impl)
│       │   ├── composite_reporter.py # Dispatches to multiple reporters
│       │   ├── server.py           # FastAPI + WebSocket server (daemon thread)
│       │   ├── snapshot.py         # Pure snapshot extractors (model, IR, HW, search)
│       │   └── static/             # Frontend SPA (ES modules + Plotly.js)
│       │       ├── index.html
│       │       ├── style.css
│       │       └── js/             # Modular JS: main, util, overview, tabs
│       └── common/                 # Shared utilities
│           ├── file_utils.py
│           ├── build_utils.py
│           └── wandb_utils.py      # Weights & Biases reporter + Reporter protocol
```

---

## 3. Entry Point and Configuration

### CLI

The primary entry point is `src/main.py`:

```bash
python src/main.py <deployment_config.json>
```

The JSON configuration specifies:

| Key | Description |
|-----|-------------|
| `data_provider_name` | Registered dataset name (e.g. `"mnist"`, `"cifar10"`) |
| `experiment_name` | Name for WandB logging and output directory |
| `pipeline_mode` | `"vanilla"` or `"phased"` (selects a preset; see §4.4) |
| `deployment_parameters` | Hyperparameters, spiking mode, quantization flags, `max_simulation_samples` (see §4.4, §5.17) |
| `platform_constraints` | Hardware constraints (max_axons, max_neurons, weight_bits, target_tq) |
| `start_step` | Step name to resume from (uses cached state) |
| `stop_step` | Optional step name to stop after |
| `target_metric_override` | Override the pipeline's tracked accuracy target |
| `generated_files_path` | Base output directory |
| `seed` | Random seed for reproducibility |
| `pruning` | Boolean; enables the Pruning Adaptation step (see §5.9) |
| `pruning_fraction` | Float (0–1); fraction of least-significant rows/columns to prune (e.g. `0.05`) |
| `generate_visualizations` | Boolean (default `false`); enables SVG/PNG visualization output in Soft Core and Hard Core mapping steps |

### Platform Constraints Protocol

Platform constraints support two modes:

- **`"user"` mode**: Constraints are directly specified as a flat dict.
- **`"auto"` mode**: Constraints split into `fixed` (passed to pipeline) and `search_space` (merged into `deployment_parameters.arch_search` for architecture search to explore).

### GUI Integration

`run_pipeline()` automatically starts a browser-based monitoring GUI on port 8501 (configurable via `gui_port`). The integration follows a non-invasive design:

1. A `GUIHandle` is created via `start_gui(pipeline)`, which spawns a FastAPI/Uvicorn server in a daemon thread
2. The pipeline's `reporter` is wrapped in a `CompositeReporter` that dispatches metrics to both WandB and the GUI's `DataCollector`
3. Pre/post-step hooks are registered on the pipeline to track step lifecycle and extract rich snapshots after each step
4. When the pipeline completes, WandB is explicitly finished via `reporter.finish()` to avoid process hangs
5. The process then waits for user input (Enter key) before exiting, keeping the GUI server alive for post-run inspection

If the GUI fails to start (e.g., port conflict, missing dependencies), a warning is printed and the pipeline continues without monitoring.

### Initialization (`src/init.py`)

Before pipeline execution, `init()` configures:
- Path to the `nevresim` C++ simulator
- Force cuDNN initialization
- Multiprocessing settings

---

## 4. Pipeline Architecture

### 4.1 Pipeline Engine

**File**: `pipelining/pipeline.py`

The `Pipeline` class is the orchestration engine. It:

1. Maintains an ordered list of `(name, PipelineStep)` tuples
2. Manages a `PipelineCache` for inter-step data transfer
3. Resolves key translations so each step accesses the correct cached version of a datum
4. Verifies data contracts at assembly time (when steps are added)
5. Enforces performance tolerance between steps (each step must not degrade accuracy below `tolerance * previous_metric`)

**Key methods:**

- `add_pipeline_step(name, step)` — Registers a step and runs verification
- `run()` / `run_from(step_name)` — Executes the pipeline (optionally from a midpoint)
- `set_up_requirements()` — Builds the key translation table mapping virtual keys to real cache keys
- `verify()` — Checks that every `requires` has been `promises`d by a prior step
- `_run_step()` — Executes one step: checks requirements, runs `step.run()`, calls `step.validate()`, saves cache, asserts contracts, checks performance tolerance
- `register_pre_step_hook(fn)` / `register_post_step_hook(fn)` — Registers callbacks invoked before/after each step. Multiple hooks can be registered; they are called in order. Used by the GUI to track step lifecycle and extract snapshots

**Key design: Namespaced cache keys.** Each cache entry is stored under `"{step_name}.{key}"`. When step B requires `"model"` which was promised by step A, the pipeline translates B's virtual key `"model"` to the real key `"A.model"`. When B *updates* `"model"`, the old entry `"A.model"` is removed and replaced with `"B.model"`.

### 4.2 Pipeline Step Contract

**File**: `pipelining/pipeline_step.py`

Every pipeline step declares four sets:

| Contract | Meaning |
|----------|---------|
| `requires` | Cache entries this step reads (must exist from prior steps) |
| `promises` | New cache entries this step creates |
| `updates` | Existing entries this step modifies (reads old value, writes new value) |
| `clears` | Entries this step removes from the cache |

At runtime, `PipelineStep.run()`:
1. Resets tracking sets
2. Calls `self.process()` (the step's implementation)
3. Asserts all `requires` were accessed and all `updates` were written

Steps interact with the cache exclusively through:
- `self.get_entry(key)` — Read (must be in `requires`)
- `self.add_entry(key, obj, strategy)` — Write (must be in `promises`)
- `self.update_entry(key, obj, strategy)` — Update (must be in `updates`)

Every step must also implement `validate()` → returns a metric (typically accuracy) that becomes the pipeline's target metric for the next step.

### 4.3 Pipeline Cache

**File**: `pipelining/cache/pipeline_cache.py`

`PipelineCache` is a dict-like store that supports three serialization strategies:

| Strategy | Format | Use Case |
|----------|--------|----------|
| `"basic"` | JSON | Scalars, lists, dicts |
| `"torch_model"` | `.pt` (torch.save) | PyTorch nn.Module models |
| `"pickle"` | `.pickle` | Complex Python objects (IRGraph, AdaptationManager, etc.) |

The cache is saved to the working directory after each step completes, enabling pipeline resumption from any step.

### 4.4 Pipeline Modes

A single `DeploymentPipeline` class (`pipelines/deployment_pipeline.py`) assembles steps dynamically from two orthogonal configuration axes:

| Axis | Values | Description |
|------|--------|-------------|
| **Pipeline mode** (`pipeline_mode`) | `"vanilla"`, `"phased"` | How much transformation to apply |
| **Spiking mode** (`spiking_mode` in `deployment_parameters`) | `"rate"`, `"ttfs"`, `"ttfs_quantized"` | SNN activation strategy |

**Pipeline mode** selects a preset that enables step groups:

- **`"vanilla"`** — Pretraining + direct mapping.  Suitable for ANNs, rate-coded SNNs, or TTFS SNNs.
- **`"phased"`** — Pretraining + activation quantization + weight quantization.  The full transformation chain for rate-coded quantised SNN platforms.

**Spiking mode** controls which SNN simulation strategy is used:

- **`"rate"`** — Rate-coded SNN.  CoreFlow tuning step is included to adjust spiking thresholds.
- **`"ttfs"`** — Time-to-first-spike SNN (continuous / analytical).  Analytical ReLU↔TTFS mapping; no threshold tuning needed.
- **`"ttfs_quantized"`** — Time-to-first-spike SNN (time-step quantised).  Analytical closed-form computation with fire-once semantics and `S` discrete time steps per layer. Spike times are computed exactly and quantized to the nearest discrete step.

For both TTFS variants, `firing_mode` and `spike_generation_mode` are automatically set to `"TTFS"` and validated — the analytical vs cycle-based distinction is controlled exclusively by `spiking_mode`. If a config JSON explicitly sets `firing_mode` or `spike_generation_mode` to a value inconsistent with the TTFS spiking mode, a `ValueError` is raised at pipeline initialisation.

Two quantization flags (booleans in `deployment_parameters`) provide fine-grained control:

| Flag | What it enables |
|------|-----------------|
| `activation_quantization` | Activation Analysis → Clamp Adaptation → Input Activation Analysis → Activation Shifting → Activation Quantization.  Configured via `target_tq`. |
| `weight_quantization` | Weight Quantization → Quantization Verification.  Configured via `weight_bits`. |

An additional pruning flag controls dimension reduction:

| Flag | What it enables |
|------|-----------------|
| `pruning` + `pruning_fraction` | Pruning Adaptation (between activation quantization and weight quantization). Gradually zeros the least-significant rows/columns. Configured via `pruning_fraction` (0–1). |

Preset defaults are applied with `setdefault`, so explicit values in `deployment_parameters` always win.

**Example: vanilla + TTFS + weight quantization:**

```
Model Configuration → Model Building → Pretraining
→ Weight Quantization → Quantization Verification
→ Normalization Fusion → Soft Core Mapping
→ Core Quantization Verification → Hard Core Mapping → Simulation
```

**Example: phased + rate (full quantisation chain):**

```
Model Configuration → Model Building → Pretraining
→ Activation Analysis → Clamp Adaptation → Input Activation Analysis
→ Activation Shifting → Activation Quantization
→ Weight Quantization → Quantization Verification
→ Normalization Fusion → Soft Core Mapping
→ Core Quantization Verification → CoreFlow Tuning
→ Hard Core Mapping → Simulation
```

**Example: phased + TTFS + pruning + weight quantization:**

```
Model Configuration → Model Building → Pretraining
→ Activation Analysis → Clamp Adaptation → Input Activation Analysis
→ Activation Shifting → Activation Quantization
→ Pruning Adaptation
→ Weight Quantization → Quantization Verification
→ Normalization Fusion → Soft Core Mapping
→ Core Quantization Verification → Hard Core Mapping → Simulation
```

Either mode works with `configuration_mode: "nas"` (replaces Model Configuration with Architecture Search).

---

## 5. Pipeline Steps in Detail

### 5.1 Architecture Search

**File**: `pipeline_steps/architecture_search_step.py`

- **Requires**: (none)
- **Promises**: `model_config`, `model_builder`, `platform_constraints_resolved`, `architecture_search_result`, `scaled_simulation_length`

Operates in two modes based on `configuration_mode`:

- **`"user"`**: Uses the model config and core topology from `deployment_parameters` directly.
- **`"nas"`**: Runs multi-objective optimization (NSGA-II or Kedi) via a `JointArchHwProblem` that jointly optimizes architecture parameters and hardware core-type dimensions. The problem is model-agnostic; model-specific search spaces, config assemblers, and validation functions are injected as parameters.

Objectives: `accuracy` (max), `wasted_area` (min), `total_params` (min).

The step produces a `PerceptronMixerBuilder` and the resolved platform constraints (including `cores` — a list of core types with `{count, max_axons, max_neurons}`). The builder is created directly from the search-resolved constraints; **no side-effect writes** are made to `pipeline.config`. Downstream steps read hardware dimensions from the cached `platform_constraints_resolved` entry.

After the search completes, the step **validates the best candidate**: if the search failed to find any feasible configuration, a `RuntimeError` is raised with the constraint violation details, rather than silently passing an infeasible config to later steps.

### 5.2 Model Building

**File**: `pipeline_steps/model_building_step.py`

- **Requires**: `model_config`, `model_builder`
- **Promises**: `model`, `adaptation_manager`

Builds the actual PyTorch model using the builder from the previous step. Initializes an `AdaptationManager` for each perceptron's activation function. Performs a warmup forward pass to initialize lazy modules (e.g., `LazyBatchNorm1d`).

### 5.3 Pretraining

**File**: `pipeline_steps/pretraining_step.py`

- **Requires**: `model`
- **Updates**: `model`

Trains the model from scratch using `BasicTrainer` for `training_epochs` with a warmup phase. This establishes a baseline accuracy before quantization transformations begin.

### 5.4 Activation Analysis

**File**: `pipeline_steps/activation_analysis_step.py`

- **Requires**: `model`
- **Promises**: `activation_scales`

Decorates each perceptron's activation with a `SavedTensorDecorator`, runs validation, and computes activation scales based on the 80th percentile of the cumulative sum of sorted activations. These scales determine the clamping range for each perceptron.

### 5.5 Clamp Adaptation

**File**: `pipeline_steps/clamp_adaptation_step.py`

- **Requires**: `model`, `adaptation_manager`, `activation_scales`
- **Updates**: `model`, `adaptation_manager`

Uses `ClampTuner` to gradually introduce clamping to each perceptron's activation, guided by the previously computed activation scales. The `SmartSmoothAdaptation` framework ensures the clamping is applied incrementally to minimize accuracy loss.

### 5.6 Activation Shifting

**File**: `pipeline_steps/activation_shift_step.py`

- **Requires**: `model`, `adaptation_manager`
- **Updates**: `model`, `adaptation_manager`

Shifts activation functions so they align with quantization levels. Computes a shift amount based on `target_tq` and `activation_scale`, applies it to biases via `PerceptronTransformer.apply_effective_bias_transform`, and trains to recover accuracy.

### 5.7 Activation Quantization

**File**: `pipeline_steps/activation_quantization_step.py`

- **Requires**: `model`, `adaptation_manager`
- **Updates**: `model`, `adaptation_manager`

Uses `ActivationQuantizationTuner` to gradually quantize activations to `target_tq` levels. The tuner uses `SmartSmoothAdaptation` to incrementally increase quantization strength while maintaining accuracy.

### 5.9 Pruning Adaptation

**File**: `pipeline_steps/pruning_adaptation_step.py`

- **Requires**: `model`, `adaptation_manager`
- **Updates**: `model`, `adaptation_manager`

Conditionally included when `pruning` is enabled and `pruning_fraction > 0`. Uses `PruningTuner` (extends `PerceptronTuner`) to gradually zero the least-significant rows and columns of each perceptron's weight matrix:

1. Computes row/column significance masks based on absolute weight magnitude sums
2. Identifies the bottom `pruning_fraction` rows and columns as pruning candidates
3. Uses `SmartSmoothAdaptation` to progressively scale candidate weights toward zero (at adaptation rate `r`, weights are multiplied by `1 − r`)
4. When adaptation completes (`r = 1.0`), pruned rows/columns are fully zeroed

The zeroed structure is later physically removed from the IR graph by `ir_pruning.prune_ir_graph()` during Soft Core Mapping, which compacts `NeuralCore` weight matrices and rewires source references.

### 5.10 Weight Quantization

**File**: `pipeline_steps/weight_quantization_step.py`

- **Requires**: `model`, `adaptation_manager`
- **Updates**: `model`

Calls `compute_per_source_scales` first so that `per_input_scales` is available during quantization (effective-weight calibration depends on it). Freezes normalization layer statistics, then uses `NormalizationAwarePerceptronQuantizationTuner` to quantize weights to `weight_bits` precision. The quantization is normalization-aware: it computes effective weights (fusing normalization) before quantizing.

### 5.11 Quantization Verification

**File**: `pipeline_steps/quantization_verification_step.py`

- **Requires**: `model`

Verifies that all perceptron effective weights and biases are correctly quantized: `w * parameter_scale` must be close to integer values within tolerance. This is a sanity check before mapping.

### 5.12 Normalization Fusion

**File**: `pipeline_steps/normalization_fusion_step.py`

- **Requires**: `model`, `adaptation_manager`
- **Updates**: `model`

Fuses BatchNorm layers into preceding linear layers. Only `Perceptron`s whose `normalization` is a `BatchNorm1d` (not `nn.Identity()`) are processed. The fusion folds BatchNorm parameters directly into `Perceptron.layer.weight.data` and `Perceptron.layer.bias.data`:

```
W_fused = diag(γ / √(σ² + ε)) @ W
b_fused = γ * (b - μ) / √(σ² + ε) + β
```

Scaling factors (`activation_scale`, `parameter_scale`) are **not modified**, ensuring mathematical equivalence — the fused network produces identical outputs to the pre-fusion network. After fusion, the normalization is replaced with `nn.Identity()`. LayerNorm and post-activation normalizations are not fusable and are skipped.

### 5.13 Soft Core Mapping

**File**: `pipeline_steps/soft_core_mapping_step.py`

- **Requires**: `model`, `platform_constraints_resolved`
- **Promises**: `ir_graph`

This critical step converts the PyTorch model into an `IRGraph`:

1. Extracts the `ModelRepresentation` (mapper graph) from the model
2. Reads `max_axons`, `max_neurons`, and `allow_axon_tiling` from the `platform_constraints_resolved` cache entry (produced by Architecture Search). When heterogeneous `cores` are present, `max_axons` and `max_neurons` are resolved as the **maximum** across all core types — this allows softcores as large as the biggest core type, relying on the greedy packer's scarcity-aware metric (§9.2) to place them correctly
3. Creates an `IRMapping` with these hardware constraints
4. Traverses the mapper graph, converting each mapper to `NeuralCore`s and/or `ComputeOp`s
5. Optionally quantizes weights (rounds `core_matrix *= parameter_scale` to integers) only when `weight_quantization` is enabled
6. Optionally generates Graphviz visualizations when `generate_visualizations` is enabled in the config (disabled by default)
7. Runs a soft-core spiking simulation for early verification

### 5.14 Core Quantization Verification

**File**: `pipeline_steps/core_quantization_verification_step.py`

- **Requires**: `ir_graph`

Conditionally included in the pipeline only when `weight_quantization` is enabled. Verifies that all `NeuralCore` weight matrices in the IR graph are properly quantized: `core_matrix * parameter_scale` must produce integers within the allowed range for the specified `weight_bits`.

### 5.15 CoreFlow Tuning

**File**: `pipeline_steps/core_flow_tuning_step.py`

- **Requires**: `model`, `ir_graph`
- **Updates**: `ir_graph`
- **Promises**: `scaled_simulation_length`

Included only for rate-coded spiking mode; TTFS modes use analytical or cycle-based thresholds and do not require tuning.

Uses `CoreFlowTuner` to adjust thresholds of `NeuralCore`s in the IR graph. The tuner:

1. Computes "stable" (rate-based) spike rates for each core
2. Runs event-based spiking simulation and compares spike rates
3. Iteratively adjusts thresholds to match stable and event-based behaviors
4. Determines the optimal `scaled_simulation_length` for the chip

### 5.16 Hard Core Mapping

**File**: `pipeline_steps/hard_core_mapping_step.py`

- **Requires**: `model`, `ir_graph`, `scaled_simulation_length`, `platform_constraints_resolved`
- **Promises**: `hard_core_mapping`

Converts the `IRGraph` into a `HybridHardCoreMapping`:

1. Segments the IR graph at `ComputeOp` boundaries
2. Allocates a **single shared pool** of hardware cores from the `cores_config` (see §9.3)
3. Converts each neural segment to a `SoftCoreMapping` → `HardCoreMapping`, drawing cores from the shared pool
4. Packs `SoftCore`s into physical `HardCore`s using **best-fit** greedy bin-packing (smallest feasible core first)
5. Produces the final deployable hybrid program
6. Runs hard-core spiking simulation for verification
7. Optionally generates Graphviz visualizations when `generate_visualizations` is enabled in the config (disabled by default)

### 5.17 Simulation

**File**: `pipeline_steps/simulation_step.py`

- **Requires**: `model`, `hard_core_mapping`, `scaled_simulation_length`

Runs a full chip simulation using the `NevresimDriver` (C++ simulator). For single-segment mappings (`HardCoreMapping`), a single nevresim invocation is used. For multi-segment `HybridHardCoreMapping`s, the `SimulationRunner` orchestrates per-segment nevresim calls with host-side `ComputeOp` execution between them (see §11.3).

Key configuration parameters:
- **`max_simulation_samples`** (in `deployment_parameters`): When set to a positive value less than the test set size, the runner randomly subsamples that many test examples (using the pipeline seed for reproducibility), reducing simulation wall time for large datasets.
- **`weight_quantization`**: Controls the C++ weight type — `int` when quantization is enabled, `float` when disabled. This ensures non-quantized models (e.g., ViT) retain floating-point precision through the C++ simulation.

---

## 6. Model Subsystem

### 6.1 Perceptron

**File**: `models/perceptron_mixer/perceptron.py`

The `Perceptron` is the fundamental building block — a module that encapsulates:

```
forward(x):
    x = input_activation(x)       # Optional input transform
    out = layer(x)                 # nn.Linear (weights + bias)
    out = normalization(out)       # BatchNorm or Identity
    out = scaler(out)              # MaxValueScaler or Identity
    out = activation(out)          # LeakyGradReLU + decorators
    out = regularization(out)      # NoisyDropout (training only)
    return out
```

Key parameters stored on each `Perceptron`:

| Parameter | Purpose |
|-----------|---------|
| `activation_scale` | Output clamping range; also determines spike threshold |
| `parameter_scale` | Quantization scale for weights and biases |
| `per_input_scales` | Per-input-channel scale tensor (set at mapping time by `compute_per_source_scales`) |
| `input_activation_scale` | Observed input activation range (diagnostic) |

### 6.2 PerceptronFlow and PerceptronMixer

**`PerceptronFlow`** (`perceptron_mixer/perceptron_flow.py`) is the abstract base class for models composed of `Perceptron`s. It defines the interface:
- `get_perceptrons()` — Flat list of all perceptrons
- `get_perceptron_groups()` — Grouped by layer for input scale analysis
- `get_mapper_repr()` — Returns the `ModelRepresentation` for mapping

**`PerceptronMixer`** (`perceptron_mixer/perceptron_mixer.py`) implements an MLP-Mixer architecture:

```
Input → Patch Embedding → [Token Mixer → Channel Mixer] × N → Output Projection
```

Where:
- **Patch Embedding**: Rearranges input into patches via `einops`, projects each patch through a `Perceptron`
- **Token Mixer**: Transposes to mix across patch positions via two `Perceptron`s
- **Channel Mixer**: Mixes across channels via two `Perceptron`s
- **Output Projection**: Flattens and projects to class logits

**`VisionTransformer`** (`perceptron_mixer/vision_transformer.py`) implements a Vision Transformer (ViT) architecture:

```
Input → Patch Embedding (Conv2D) → CLS Prepend → Positional Embedding
  → [LayerNorm → MHSA → Add (residual) → LayerNorm → FFN → Add (residual)] × N
  → Final LayerNorm → CLS Select → Classification Head
```

Where:
- **Patch Embedding**: `Conv2DPerceptronMapper` with `use_batchnorm=True` (the only fusable normalization in the model)
- **CLS Token**: A `ConstantPrependMapper` that prepends a learnable class token (`concat_constant` ComputeOp)
- **Positional Embedding**: A `ConstantAddMapper` that adds learned position embeddings (`add_constant` ComputeOp)
- **Layer Normalization**: `LayerNormMapper` emitting `layer_norm` ComputeOps (not fusable)
- **Multi-Head Self-Attention**: `MultiHeadAttentionComputeMapper` wrapping Q/K/V projections (`PerceptronMapper`s), attention computation (`multi_head_attention` ComputeOp), and output projection
- **FFN**: Two `PerceptronMapper`s with GELU activation between them
- **Residual Connections**: `AddMapper` emitting `add` ComputeOps, creating skip connections in the IR graph
- **CLS Select**: `SelectMapper` extracting the CLS token output (`select` ComputeOp)

All `Perceptron`s except the patch embedding use `normalization=nn.Identity()`, since LayerNorm is handled via ComputeOps. The model is parametrizable for architecture search: `embed_dim`, `num_heads`, `depth`, `mlp_ratio`, `patch_size` are all configurable.

Both `PerceptronMixer` and `VisionTransformer` construct a **mapper graph** (DAG of `Mapper` objects) that serves as the single source of truth for both:
1. The PyTorch forward pass
2. The hardware mapping

### 6.3 Supermodel

**File**: `models/supermodel.py`

`Supermodel` wraps a `PerceptronFlow` with:
- A **preprocessor** (e.g., `InputCQ` for input quantization)
- An **input activation** (`TransformedActivation` with Clamp + Quantize decorators)

```
forward(x):
    out = preprocessor(x)      # e.g., normalize to [0, 1]
    out = in_act(out)           # Clamp to [0, 1], quantize to Tq levels
    out = perceptron_flow(out)  # The actual model computation
    return out
```

### 6.4 Activation Decorators

**File**: `models/layers.py`

Activations are composable via a decorator pattern:

```
TransformedActivation
  ├── base_activation (LeakyGradReLU)
  └── decorators[]
       ├── ClampDecorator         — Clamps output to [min, max]
       ├── QuantizeDecorator      — Applies staircase quantization
       ├── ShiftDecorator         — Shifts input by an offset
       ├── ScaleDecorator         — Scales output
       ├── NoiseDecorator         — Adds noise during training
       ├── SavedTensorDecorator   — Records activations for analysis
       ├── StatsDecorator         — Computes activation statistics
       └── RateAdjustedDecorator  — Gradually blends base ↔ decorated output
```

Each decorator has `input_transform(x)` and `output_transform(x)` methods. `DecoratedActivation` composes them: `output_transform(base_activation(input_transform(x)))`.

`TransformedActivation` supports dynamic decorator management via `decorate()` and `pop_decorator()`, which is used extensively during analysis and tuning steps.

Key custom functions:
- **`LeakyGradReLU`**: ReLU in forward, leaky gradient in backward (avoids dead neurons during quantization-aware training)
- **`DifferentiableClamp`**: Clamp in forward, exponential decay gradients outside bounds
- **`StaircaseFunction`**: Quantization (floor) in forward, straight-through gradient in backward

### 6.5 Model Builders

**Directory**: `models/builders/`

Builders are created during Architecture Search and cached for later use:

- `PerceptronMixerBuilder` — Builds `Supermodel(preprocessor=InputCQ, perceptron_flow=PerceptronMixer)`
- `VitBuilder` — Builds `Supermodel(preprocessor=InputCQ, perceptron_flow=VisionTransformer)`
- `SimpleMlpBuilder` — Simple multi-layer perceptron
- `SimpleConvBuilder` — Convolutional model
- `VGG16Builder` — VGG-16 architecture

Each builder's `build(configuration)` method takes a model config dict and returns a `Supermodel`.

---

## 7. Mapper Graph — ModelRepresentation

### 7.1 Mapper Hierarchy

**File**: `mapping/mapping_utils.py`

The mapper graph is a DAG of `Mapper` objects that mirrors the model's computation graph. Each mapper knows how to:
1. Execute the PyTorch forward pass (`forward()`)
2. Map itself to SoftCores via `map(mapping)` (legacy)
3. Map itself to the unified IR via `map_to_ir(ir_mapping)` (new)

```
Mapper (abstract base)
├── InputMapper                        — Marks the input tensor
├── ModuleMapper                       — Wraps an arbitrary nn.Module
├── PerceptronMapper                   — Maps a Perceptron to NeuralCores
├── Conv2DPerceptronMapper             — Conv2d as im2col + matmul
├── Conv1DPerceptronMapper             — Conv1d as im2col + matmul
├── EinopsRearrangeMapper              — Tensor rearrangement
├── MergeLeadingDimsMapper             — (B, N, D) → (B*N, D)
├── SplitLeadingDimMapper              — (B*N, D) → (B, N, D)
├── Ensure2DMapper                     — Reshapes to (B, D)
├── ReshapeMapper                      — Generic reshape
├── StackMapper                        — Concatenates multiple mapper outputs
├── AddMapper                          — Element-wise addition → "add" ComputeOp
├── ConstantAddMapper                  — Adds a learnable constant → "add_constant" ComputeOp
├── ConstantPrependMapper              — Prepends a learnable constant → "concat_constant" ComputeOp
├── SelectMapper                       — Selects a slice of the token sequence → "select" ComputeOp
├── DropoutMapper                      — Dropout (identity during eval/mapping)
├── LayerNormMapper                    — Layer normalization → "layer_norm" ComputeOp
├── MultiHeadAttentionComputeMapper    — MHSA: Q/K/V projections + "multi_head_attention" ComputeOp + output projection
├── PoolingMapper                      — MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
└── (others)
```

### 7.2 Dual Purpose: Forward Pass and Hardware Mapping

The mapper graph is the **single source of truth** for both computation and mapping:

**Forward pass** (training/evaluation):
```python
model.forward(x)
  → supermodel.forward(x)
    → perceptron_flow.forward(x)
      → model_representation(x)  # Traverses mapper graph
```

**Hardware mapping** (IR generation):
```python
ir_mapping.map(model.get_mapper_repr())
  → model_representation.map_to_ir(ir_mapping)
    → output_mapper.map_to_ir(ir_mapping)  # Recursively maps all mappers
```

Each mapper's `map_to_ir()` method:
1. Recursively maps its input mappers
2. Receives input source arrays (describing which core outputs feed this mapper)
3. Creates `NeuralCore`s and/or `ComputeOp`s in the IR
4. Returns output source arrays for downstream mappers

For convolutions, `Conv2DPerceptronMapper._map_to_ir()` implements the im2col transformation: it registers one `WeightBank` per output-channel group and creates one bank-backed `NeuralCore` per spatial position.  Each core shares the bank's weight matrix and differs only in its input wiring (receptive field position).  This avoids O(h_out × w_out) weight replication in the IR and in simulation.

---

## 8. Intermediate Representation (IR)

### 8.1 IRGraph, IRNode, IRSource

**File**: `mapping/ir.py`

The unified IR represents the entire mapped network:

```
IRGraph
├── nodes: List[IRNode]       # Topologically sorted
└── output_sources: np.ndarray[IRSource]  # Which node outputs form the final output
```

**`IRSource`** describes where an input element comes from:
- `node_id >= 0`: Output index from another node
- `node_id == -1`: Always off (zero)
- `node_id == -2`: From the original input tensor
- `node_id == -3`: Always on (constant 1.0)

### 8.2 NeuralCore

A hardware-mappable crossbar-based computation:

```
Computation: output = ReLU(core_matrix @ input)
```

Key fields:
- `core_matrix: np.ndarray | None` — shape `(axons, neurons)`, the weight matrix.  `None` when using a shared weight bank.
- `threshold: float` — Spiking threshold (tuned by CoreFlowTuner)
- `activation_scale`, `parameter_scale`, `input_activation_scale` — Scaling factors
- `psum_group_id`, `psum_role` — For partial-sum decomposition when a layer exceeds hardware limits
- `weight_bank_id: int | None` — If set, this core references a `WeightBank` stored on the `IRGraph` instead of owning its `core_matrix`.
- `weight_row_slice: tuple[int,int] | None` — Optional neuron-axis slice into the bank's matrix (for output-channel tiling).

**Weight ownership modes:**
1. *Owned weights* (FC layers): `core_matrix` is a concrete array, `weight_bank_id` is `None`.
2. *Shared weights* (conv layers): `weight_bank_id` references a `WeightBank` on the `IRGraph`; `core_matrix` is `None`.  Use `get_core_matrix(graph)` to resolve.

### 8.2b WeightBank

A shared weight matrix stored once on the `IRGraph` and referenced by many `NeuralCore`s.  This avoids O(h_out × w_out) weight replication for convolutional layers.

Key fields:
- `id: int`
- `core_matrix: np.ndarray` — shape `(axons, neurons)`, same layout as NeuralCore
- `activation_scale`, `parameter_scale`, `input_activation_scale`

### 8.3 ComputeOp

Non-neural operations that cannot be mapped to crossbar cores:

- **Spatial pooling**: `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`
- **Shape manipulation**: `flatten`, `identity`
- **Transformer ops**: `layer_norm`, `multi_head_attention`, `add`, `add_constant`, `concat_constant`, `select`, `gelu`
- Store `input_shape`, `output_shape`, and operation-specific `params` (e.g., `num_heads`, `embed_dim` for MHSA; `weight`, `bias`, `eps` for LayerNorm; learnable tensors for `add_constant` and `concat_constant`)

Each ComputeOp implements `execute(input_tensor)` and `execute_on_gathered(flat_input)`. The latter accepts a flat `(B, N)` tensor from the spiking simulation, dispatches to the appropriate `_exec_*` method (which handles any internal reshaping), and returns a flat output. For spatial operations (`max_pool2d`, `avg_pool2d`, etc.), the method reshapes internally using the stored `input_shape`.

In spiking simulation, `ComputeOp`s act as **synchronization barriers**: spike counts are converted to rates, the operation is applied, and rates are converted back to spikes.

### 8.4 IRMapping

**File**: `mapping/ir_mapping.py`

`IRMapping` converts a `ModelRepresentation` to an `IRGraph`:

- `add_neural_core(...)` — Creates a `NeuralCore` with source wiring (owned weights)
- `add_shared_neural_core(...)` — Creates a bank-backed `NeuralCore` that references a `WeightBank` instead of storing its own `core_matrix`
- `register_weight_bank(...)` — Stores a shared weight matrix and returns its `bank_id`
- `add_compute_op(...)` — Creates a `ComputeOp`
- `map_fc(...)` — Maps a fully-connected layer, handling:
  - **Output tiling**: Splits neurons across multiple cores when `neurons > max_neurons`
  - **Axon tiling**: Splits axons across cores (partial sums) when `axons > max_axons` and `allow_axon_tiling` is enabled
  - **Partial sum accumulation**: Creates `psum_role="partial_pos"/"partial_neg"` cores and an `"accum"` core

**Heterogeneous tiling**: When multiple core types exist, `max_axons` and `max_neurons` are resolved as the **maximum** across all core types. This allows softcores as large as the biggest core type. The greedy packer's scarcity-aware placement metric (§9.2) then ensures flexible softcores (those that fit multiple core types) are directed toward more abundant types, preserving scarce large-capacity types for softcores that strictly require them.

---

## 9. Hardware Mapping

### 9.1 SoftCore and SoftCoreMapping (Legacy)

**File**: `mapping/softcore_mapping.py`, `mapping/mapping_utils.py`

`SoftCore` is a logical neural core with:
- `core_matrix` — Weight matrix
- `axon_sources` — List of `SpikeSource` objects (the connectivity)
- Scaling parameters and threshold

`SoftCoreMapping` contains a list of `SoftCore`s and output sources. The legacy `map()` method traverses the `ModelRepresentation` via the old `Mapper.map()` interface.

### 9.2 HardCore and HardCoreMapping

**Files**: `mapping/softcore_mapping.py`, `mapping/core_packing.py`

`HardCore` represents a physical core on the chip with fixed capacity:
- `axons_per_core`, `neurons_per_core` — Capacity
- `available_axons`, `available_neurons` — Remaining capacity

`HardCoreMapping` packs `SoftCore`s into `HardCore`s using the generic `greedy_pack_softcores` algorithm (`core_packing.py`):
1. Sort soft cores by neuron count (descending)
2. For each soft core, try to fit it into an already-used hardware core.  Among all feasible used cores, pick the one with the minimum **remaining capacity** after placement `(avail_a − s_a) × (avail_n − s_n)`, concentrating softcores into tightly-fitting cores and leaving others available for differently-shaped softcores.
3. If no used core has room, pick an unused core using a **scarcity-aware** metric: `waste / abundance`, where `waste` is the L-shaped dead zone `h_a · s_n + s_a · h_n − 2 · s_a · s_n` and `abundance` is the count of remaining unused cores of that type. This directs flexible softcores (those fitting multiple core types) toward more abundant types, preserving scarce types for dimensionally-constrained softcores that can only fit a specific type. Without this, greedy waste minimisation can exhaust scarce types early, causing later placements to fail.
4. Place the soft core, update axon sources to point to hardware core positions
5. Track the mapping `(soft_core_id, soft_neuron) → (hard_core_idx, hard_neuron)`

The same `greedy_pack_softcores` function is used by both the real `HardCoreMapping` and the layout-only `pack_layout` (architecture search), ensuring consistent packing behaviour.

`HardCoreMapping` also exposes `axons_per_core` and `neurons_per_core` properties that return the **maximum** axon and neuron counts among its cores. These are used to determine the uniform dimensions for nevresim code generation (see §11.2).

### 9.3 HybridHardCoreMapping

**File**: `mapping/hybrid_hardcore_mapping.py`

The deployable program representation, supporting skip connections and complex data flow (e.g., transformer residual connections) via a **state-buffer** approach:

```
HybridHardCoreMapping
├── stages: List[HybridStage]
│   ├── HybridStage(kind="neural", hard_core_mapping, input_slices, output_slices)
│   ├── HybridStage(kind="compute", compute_op, input_slices, output_slices)
│   ├── HybridStage(kind="neural", ...)
│   └── ...
├── state_buffer_size: int          # Total entries in global state buffer
└── output_slices: List[SegmentIOSlice]  # Final output extraction
```

**State buffer design**: A global dictionary maps `node_id → activations`. Each `HybridStage` declares its I/O via `SegmentIOSlice` dataclasses:

```
SegmentIOSlice(node_id, seg_offset, seg_count)
```

- **Input slices**: Which state-buffer entries feed into this stage's local input buffer
- **Output slices**: Which of this stage's outputs are written back to the state buffer

This allows skip connections: a residual `AddMapper`'s ComputeOp reads two state-buffer entries (the pre-block output and the block output) that may have been produced by non-adjacent stages.

Built by `build_hybrid_hard_core_mapping(ir_graph, cores_config)`:
1. Allocates a **single shared pool** of hardware cores from `cores_config` upfront; all segments draw from this same pool, ensuring the total core budget is respected across the entire hybrid program
2. Walks the IR graph in topological order, grouping consecutive `NeuralCore`s
3. At each `ComputeOp`, flushes the current neural segment into a `HardCoreMapping` (packed from the shared pool)
4. External source references (from earlier stages or the original input) are resolved via the state buffer
5. Each stage's `input_slices` and `output_slices` are computed to wire the state buffer correctly

---

## 10. Training and Tuning

### 10.1 BasicTrainer

**File**: `model_training/basic_trainer.py`

Standard PyTorch training loop with:
- `train_n_epochs(lr, epochs, warmup_epochs)` — Fixed-duration training with cosine annealing and optional warmup
- `train_until_target_accuracy(lr, max_epochs, target_accuracy, warmup_epochs)` — Early stopping when target is reached
- `validate()` / `test()` — Evaluation on validation/test sets
- Mixed precision training via `torch.cuda.amp`
- WandB reporting via `report_function`

### 10.2 AdaptationManager

**File**: `tuning/adaptation_manager.py`

Manages the progressive application of transformations to perceptron activations:

```python
class AdaptationManager:
    clamp_rate: float      # 0.0 → 1.0 (no clamp → full clamp)
    shift_rate: float      # 0.0 → 1.0
    quantization_rate: float
    scale_rate: float
    noise_rate: float
```

`update_activation(pipeline_config, perceptron)` rebuilds the perceptron's activation as a `TransformedActivation` with `RateAdjustedDecorator`s for clamping, quantization, and shifting. The rates control how aggressively each transformation is applied, allowing gradual introduction.

### 10.3 SmartSmoothAdaptation

**File**: `tuning/smart_smooth_adaptation.py`

A framework for gradually applying a transformation while maintaining accuracy:

1. Start with `t = 0` (no transformation)
2. Find the largest step size that keeps accuracy above the tolerable threshold
3. Apply the transformation at rate `t + step_size`
4. Train to recover accuracy
5. Repeat until `t = 1.0` (full transformation)

The framework includes:
- Binary search for step size
- State save/restore (clone/restore model state)
- Target adjustment (dynamically adjusts expected accuracy based on observed degradation)

### 10.4 Tuner Hierarchy

**Directory**: `tuning/tuners/`

All tuners extend `BasicTuner`, which uses `SmartSmoothAdaptation`:

| Tuner | Purpose |
|-------|---------|
| `ClampTuner` | Introduces activation clamping |
| `ActivationQuantizationTuner` | Quantizes activations to Tq levels |
| `PruningTuner` | Gradually zeros least-significant weight rows/columns |
| `NormalizationAwarePerceptronQuantizationTuner` | Quantizes weights (normalization-aware) |
| `NoiseTuner` | Introduces training noise |
| `CoreFlowTuner` | Tunes spiking thresholds (operates on IRGraph, not model) |

`PruningTuner` extends `PerceptronTuner` and applies pruning masks (from `transformations/pruning.py`) that scale the bottom `pruning_fraction` of rows and columns by `(1 − rate)`. The zeroed structure is later compacted from the IR graph by `ir_pruning.prune_ir_graph()` (see §8.4).

Each tuner defines:
- `_update_and_evaluate(rate)` — Apply transformation at given rate and evaluate
- Learning rate exploration and training to recover accuracy

### 10.5 CoreFlowTuner

**File**: `tuning/tuners/core_flow_tuner.py`

A specialized tuner that adjusts `NeuralCore` thresholds in the IR graph to optimize spiking simulation accuracy. Unlike other tuners, this operates on the `IRGraph` rather than the PyTorch model:

1. Computes "stable" spike rates (using continuous-valued simulation)
2. Runs event-based spiking simulation
3. Compares per-core spike rates between stable and spiking modes
4. Calculates perturbations and applies them to thresholds
5. Iterates to find optimal thresholds
6. Determines `scaled_simulation_length` for deployment

---

## 11. Spiking Simulation

### 11.1 SpikingUnifiedCoreFlow

**File**: `models/unified_core_flow.py`

A PyTorch-based spiking simulator for `IRGraph`:

- Implements membrane potential dynamics with configurable firing modes (`"Default"`, `"Novena"`, `"TTFS"`)
- Supports multiple spike generation modes (`"Stochastic"`, `"Deterministic"`, `"FrontLoaded"`, `"Uniform"`, `"TTFS"`)
- Handles `ComputeOp`s as sync barriers: converts spike counts to rates, applies the operation via `node.execute_on_gathered(flat_rates)`, converts back to spikes
- Uses range-compressed `IRSourceSpan`s for efficient input gathering
- Supports both `"<"` and `"<="` thresholding modes
- Dispatches between rate-coded and TTFS forward paths based on `spiking_mode` (not `firing_mode`)
- **TTFS continuous** (`spiking_mode="ttfs"`): Analytical spike-time computation — `relu(W @ x + b) / θ` per core in topological order. Equivalent to standard ReLU; no time-stepping.
- **TTFS quantized** (`spiking_mode="ttfs_quantized"`): **Analytical closed-form** computation (not cycle-based). For each neuron: `t_exact = θ / (W @ x + b)`, then quantized to the nearest discrete time step `k = round((t_exact − t_min) / Δt)`, clamped to `[0, S−1]`. The output activation is `(S − k) / S`. Non-firing neurons (where `W @ x + b ≤ 0`) are masked to output 0. This matches the nevresim C++ TTFS quantized behaviour exactly while being orders of magnitude faster than cycle-based simulation.

### 11.1b SpikingHybridCoreFlow

**File**: `models/hybrid_core_flow.py`

A PyTorch-based spiking simulator for `HybridHardCoreMapping`, using the state-buffer approach:

- Maintains a global `state_buffer: Dict[int, Tensor]` keyed by IR node IDs
- Iterates through `HybridStage`s in order:
  - **Neural stages**: Assembles the local input from state-buffer entries via `input_slices`, runs through `NeuralCore`s, writes outputs back via `output_slices`
  - **Compute stages**: Gathers inputs from the state buffer, executes the `ComputeOp`, writes the result back
- Supports both rate-coded and TTFS (continuous + analytical quantized) forward paths
- The analytical TTFS quantized path mirrors `SpikingUnifiedCoreFlow`'s closed-form computation

### 11.2 Nevresim (C++ Simulator)

**Files**: `chip_simulation/nevresim_driver.py`, `code_generation/cpp_chip_model.py`

For final verification, the framework generates C++ code and runs it through the `nevresim` simulator:

1. **Code Generation** (`cpp_chip_model.py`): Converts `HardCoreMapping` to `ChipModel` → generates C++ structs (`SpikeSource`, `Neuron`, `Core`, `ChipModel`). When a segment contains heterogeneous `HardCore` sizes, individual core matrices and axon sources are **padded** to the segment's maximum dimensions (`HardCoreMapping.axons_per_core`, `neurons_per_core`) so that nevresim receives uniform core geometry. The `ChipModel` is parameterized by `weight_type` (`int` or `float`): when `weight_quantization` is disabled, weights are written as floating-point values and the C++ template instantiates with `float` weight arithmetic.
2. **Template Instantiation** (`generate_main.py`): Generates `main.cpp` from a template with simulation parameters
3. **Compilation** (`compile_nevresim.py`): Compiles with C++20 (`-std=c++20 -O3`). Prefers Clang ≥ 17; falls back to `g++-11` when no suitable Clang is found.
4. **Execution** (`execute_nevresim.py`): Runs the binary in parallel processes, collects output

`NevresimDriver` provides two entry points:
- `predict_spiking(data_loader, ...)` — Standard batch evaluation from a `DataLoader`
- `predict_spiking_raw(input_data, ...)` — Lower-level entry for per-segment simulation, accepting `(input_tensor, target)` tuples directly (used by `SimulationRunner` for multi-segment hybrid mapping)

### 11.3 SimulationRunner

**File**: `chip_simulation/simulation_runner.py`

Orchestrates the end-to-end simulation. Supports two modes:

**Single-segment** (`HardCoreMapping`):
1. Preprocesses test data through the model's preprocessor and input activation
2. Saves inputs to files
3. Calculates chip latency
4. Runs a single `NevresimDriver` invocation
5. Parses simulator output and computes accuracy

**Multi-segment** (`HybridHardCoreMapping`):
1. Preprocesses test data as above; flattens input images to `(N, features)` for consistency with the state buffer
2. Optionally subsamples test data to `max_simulation_samples` (seeded) for faster execution
3. Determines `weight_type` (`int` when `weight_quantization` is enabled, `float` otherwise) and passes it to each `NevresimDriver` — this ensures non-quantized models retain floating-point weight precision through C++ simulation
4. Maintains a NumPy state buffer (`Dict[int, np.ndarray]`) mirroring the PyTorch state-buffer approach
5. Iterates through hybrid stages in order:
   - **Neural stages**: Assembles segment input from the state buffer via `input_slices`, instantiates a per-segment `NevresimDriver` with the correct `weight_type`, runs `predict_spiking_raw`, writes outputs back via `output_slices`
   - **Compute stages**: Gathers inputs from the state buffer, executes the `ComputeOp` on the host using PyTorch, writes the result back
6. Extracts final outputs via `output_slices` and evaluates classification accuracy

---

## 12. Architecture Search

### 12.1 Search Framework

**Files**: `search/problem.py`, `search/results.py`, `search/optimizers/base.py`

A clean, protocol-based search framework:

```
SearchProblem[ConfigT]  ←  Protocol
  ├── objectives: Sequence[ObjectiveSpec]   # name + goal (min/max)
  ├── validate(config) → bool               # Fast feasibility check
  ├── constraint_violation(config) → float  # Continuous violation (≤0 feasible, >0 infeasible)
  ├── evaluate(config) → Dict[str, float]   # Compute objective values
  └── meta(config) → Dict[str, Any]         # Optional metadata

SearchOptimizer[ConfigT]  ←  Base class
  └── optimize(problem) → SearchResult[ConfigT]

SearchResult[ConfigT]
  ├── objectives: Sequence[ObjectiveSpec]
  ├── best: Candidate[ConfigT]
  ├── pareto_front: List[Candidate]
  ├── all_candidates: List[Candidate]
  └── history: List[Dict]

EncodedProblem[ConfigT] extends SearchProblem
  ├── n_var, xl, xu    # Continuous variable encoding
  └── decode(x) → ConfigT
```

The `constraint_violation` method returns 0.0 when `validate()` passes (default). Subclasses override it to return a **continuous** violation metric (e.g., how many axons exceed the limit) so that evolutionary optimizers can guide the search toward feasibility rather than treating all infeasible candidates equally.

### 12.2 NSGA-II Optimizer

**File**: `search/optimizers/nsga2_optimizer.py`

Uses `pymoo`'s NSGA-II implementation. Handles:
- Encoding/decoding via `EncodedProblem.decode()`
- **Native constraint handling** via pymoo's `G` output: calls `problem.constraint_violation()` and populates `n_ieq_constr=1` inequality constraint. pymoo's constraint-domination principle ensures: (a) feasible solutions always dominate infeasible ones, (b) among infeasible solutions those with smaller violation are preferred. This guides the search toward feasible regions even when the initial random population is entirely infeasible.
- Invalid candidate penalty (`1e18`) for objective values when infeasible
- Pareto front extraction
- Best candidate selection (accuracy-first, then cores, unused area, params)

### 12.3 Kedi Optimizer (LLM-based)

**File**: `search/optimizers/kedi_optimizer.py`

An innovative LLM-based optimizer that uses agentic reasoning:
1. Generates initial candidates using an LLM
2. Learns from validation/evaluation failures
3. Consolidates constraint knowledge across generations
4. Uses performance analysis to guide search direction
5. Generates offspring from Pareto front patterns

### 12.4 JointArchHwProblem

**File**: `search/problems/joint_arch_hw_problem.py`

A **model-agnostic** joint architecture and hardware co-search problem. Model-specific concerns (search spaces, config assemblers, validation functions) are injected as parameters, allowing the same problem class to handle PerceptronMixer, VisionTransformer, or any future architecture.

Jointly optimizes:
- **Architecture parameters**: Model-specific (e.g., `patch_rows/cols/channels`, `fc_w1/w2` for MLP-Mixer; `embed_dim`, `num_heads`, `depth`, `mlp_ratio`, `patch_size` for ViT)
- **Hardware core-type dimensions**: Number of core types (heterogeneous), axon/neuron counts per type, threshold grouping

Objectives: `accuracy` (max), `wasted_area` (min), `total_params` (min).

Key design decisions:

- **Tiling to maximum core type**: `decode()` computes `max_axons = max(cores[*].max_axons)` and `max_neurons = max(cores[*].max_neurons)`. Softcores can be as large as the biggest core type; the scarcity-aware packer (§9.2) handles heterogeneous placement.
- **Continuous constraint violation**: `constraint_violation()` returns `max(0, max_in_features - (max_axons - 1))`, giving NSGA-II a smooth gradient toward feasibility (see §12.2).
- **Layout-only hardware estimation**: Each candidate is evaluated by collecting shape-only `LayoutSoftCoreSpec`s via `LayoutIRMapping` (which computes inter-core latencies and assigns latency-stratified random threshold groups using a deterministic `threshold_seed`) and then packing them via `pack_layout` using the same best-fit algorithm as real mapping.
- **Quick validation**: `_quick_validate()` checks patch divisibility and axon/neuron feasibility without building the full model.

---

## 13. Data Handling

**Directory**: `data_handling/`

```
DataProvider (abstract)
  ├── _get_training_dataset()
  ├── _get_validation_dataset()
  ├── _get_test_dataset()
  ├── get_prediction_mode() → ClassificationMode | RegressionMode
  └── get_input_shape(), get_output_shape(), batch sizes, etc.

DataProviderFactory
  └── create() → DataProvider

BasicDataProviderFactory
  ├── _provider_registry: Dict[str, Type[DataProvider]]  # Class-level registry
  ├── register(name) → decorator       # @BasicDataProviderFactory.register("mnist")
  └── create() → DataProvider           # Cached per factory instance
```

Built-in providers: `mnist`, `mnist32`, `cifar10`, `cifar100`, `ecg`

The factory uses a **class-level registry** pattern: providers self-register via the `@BasicDataProviderFactory.register(name)` decorator.

`DataLoaderFactory` creates PyTorch `DataLoader`s for training, validation, and test splits with appropriate batch sizes and multiprocessing settings.

---

## 14. Visualization

### 14.1 Static Visualizations

**Directory**: `visualization/`

| Module | Purpose |
|--------|---------|
| `mapping_graphviz.py` | Graphviz DOT generation for IRGraph, SoftCoreMapping, HardCoreMapping, and HybridHardCoreMapping |
| `softcore_flowchart.py` | Flowchart-style visualization of SoftCore connectivity |
| `activation_function_visualization.py` | Plots activation functions over a range |
| `histogram_visualization.py` | Activation distribution histograms |
| `hardcore_visualization.py` | Heatmaps and utilization charts for HardCore mappings |
| `search_visualization.py` | Architecture search result plots |

The `mapping_graphviz` module is particularly extensive, providing:
- **Per-node detailed views**: Show individual NeuralCore/ComputeOp properties
- **Summary views**: Group NeuralCores into "layer stacks" for readability
- **Hybrid combined views**: Multi-stage program overview with embedded thumbnails
- Automatic SVG/PNG rendering via the system `dot` binary

**Note:** Visualization generation in Soft Core and Hard Core mapping steps is disabled by default for performance. Set `"generate_visualizations": true` in the config JSON to enable DOT/SVG/PNG output.

### 14.2 Browser-Based Pipeline Monitor (GUI)

**Directory**: `gui/`

A real-time browser-based dashboard that launches automatically with every pipeline run, providing live monitoring and post-run inspection.

**Architecture** (isolated from the core pipeline via clean interfaces):

| Component | File | Purpose |
|-----------|------|---------|
| `GUIHandle` | `__init__.py` | Facade: creates collector, reporter, and step hooks |
| `DataCollector` | `data_collector.py` | Thread-safe in-memory store for metrics, step lifecycle, and snapshots. Broadcasts updates to WebSocket listeners |
| `GUIReporter` | `reporter.py` | Implements the `Reporter` protocol; forwards `report()` calls to `DataCollector` |
| `CompositeReporter` | `composite_reporter.py` | Dispatches `report()`/`console_log()` to multiple reporters (WandB + GUI) |
| `server.py` | `server.py` | FastAPI application with REST endpoints (`/api/pipeline`, `/api/steps/{name}`) and a WebSocket (`/ws`) for real-time push. Runs in a daemon thread via Uvicorn |
| `snapshot.py` | `snapshot.py` | Pure functions that extract JSON-serializable snapshots from pipeline artifacts: model weights/stats, IR graph topology with latency-tier grouping, hardware mapping with per-core heatmaps and connectivity, search results, adaptation rates |

**Frontend** (`static/`): A single-page application built with ES modules and Plotly.js. Modularized into focused files:

| Module | Responsibility |
|--------|---------------|
| `js/main.js` | State management, WebSocket, refresh loop |
| `js/util.js` | HTML helpers, Plotly safe-wrapper with `uirevision` for zoom/pan persistence |
| `js/overview.js` | Pipeline bar, target metric progression, step timing charts |
| `js/step-detail.js` | Step detail panel, tab routing, metrics tab |
| `js/model-tab.js` | Model layer stats, weight distributions |
| `js/ir-graph-tab.js` | Hierarchical latency-tier DAG with 3D stacked blocks (PlotNeuralNet/NN-SVG style, left-to-right). Always-collapsed tiers in the SVG; clicking a tier opens a stacked detail panel below showing groups as clickable cards, then clicking a group drills into core/op details beneath |
| `js/hardware-tab.js` | Stage flow with proportional inline heatmaps, vertical input buffer (left), horizontal output buffer (bottom). Click-to-overlay span connectivity: input spans on core left edge (axon side), output spans on core bottom edge (neuron side), midpoint arrowheads, hover brightens all span elements |
| `js/search-tab.js` | Pareto front, search history, candidate tables |
| `js/scales-tab.js` | Activation/parameter scales, adaptation rates, quantization staircase |

**Key design decisions:**
- **Non-invasive integration**: The GUI attaches to the existing `Reporter` protocol and pipeline hooks; no core pipeline code is modified beyond adding hook registration support
- **Thread-safe**: `DataCollector` uses a lock; the FastAPI server runs in its own daemon thread and event loop
- **NaN/Inf safety**: All JSON responses use a custom `_SafeJSONResponse` that recursively replaces non-finite floats with `null`, preventing `ValueError: Out of range float values are not JSON compliant` crashes when training metrics contain NaN or Inf
- **Snapshot extraction**: `snapshot.py` contains pure functions that accept pipeline artifacts and return JSON-safe dicts. Failures are logged at debug level and never crash the pipeline
- **Zoom/pan persistence**: All Plotly charts use `uirevision: 'persist'` on layout and axes to preserve user interactions across data updates
- **Stable DOM**: The detail panel only rebuilds when structural data changes (step name, status, snapshot keys), not on every metric or duration update

---

## 15. Code Generation

**Directory**: `code_generation/`

Generates C++ code for the `nevresim` chip simulator:

**`cpp_chip_model.py`** defines the chip's data model:
```
ChipModel
├── cores: List[Core]
│   └── neurons: List[Neuron]
│       ├── weights: List[float]
│       ├── threshold: float
│       └── bias: float
├── connections: List[Connection]
│   └── axon_sources: List[SpikeSource]
├── output_sources: List[SpikeSource]
└── Metadata (axons_per_core, neurons_per_core, etc.)
```

`SpikeSource` identifies the origin of each axon input:
- `(core, neuron)` — Output from a specific core's neuron
- `is_input` — From the external input buffer
- `is_off` — Disconnected (zero)
- `is_always_on` — Constant 1

**`generate_main.py`** instantiates a C++ template with simulation parameters (input count, simulation length, spike generation mode, firing mode, latency). C++ chip and execution policy selection is dispatched by `spiking_mode`:

| `spiking_mode` | C++ Compute Policy | C++ Execution Policy |
|---|---|---|
| `"ttfs"` | `TTFSAnalyticalCompute` | `TTFSContinuousExecution` (single-pass) |
| `"ttfs_quantized"` | `TTFSQuantizedCompute<S>` | `TTFSExecution<S>` (single-pass, neuron-internal cycle loop) |
| `"rate"` (Default/Novena) | `SpikingCompute<FirePolicy>` | `SpikingExecution` (cycle-based) |

---

## 16. Key Data Flow Diagram

```
                        ┌─────────────────┐
                        │  JSON Config     │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  Architecture   │
                        │  Search         │──▶ model_config, model_builder
                        └────────┬────────┘    platform_constraints_resolved
                                 │
                        ┌────────▼────────┐
                        │  Model Building │──▶ Supermodel (PyTorch nn.Module)
                        └────────┬────────┘    AdaptationManager
                                 │
           ┌─────────────────────▼─────────────────────┐
           │         Training & Quantisation Phase         │
           │         (steps enabled via config flags)      │
           │                                               │
           │  Pretrain                                     │
           │  ┌─ if activation_quantization ──────────┐    │
           │  │ Analyze → Clamp → Shift → Act Quant   │    │
           │  └───────────────────────────────────────┘    │
           │  ┌─ if pruning ─────────────────────────┐    │
           │  │ Pruning Adaptation                    │    │
           │  └───────────────────────────────────────┘    │
           │  ┌─ if weight_quantization ──────────────┐    │
           │  │ Weight Quant → Verification            │    │
           │  └───────────────────────────────────────┘    │
           │  Fuse Normalization                           │
           └─────────────────────┬─────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Soft Core Mapping       │
                    │  Model → IRGraph         │
                    │  (NeuralCores+ComputeOps)│
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  CoreFlow Tuning         │
                    │  Adjust spike thresholds │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Hard Core Mapping       │
                    │  IRGraph → HybridMapping │
                    │  (packed physical cores) │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Simulation / Deploy     │
                    │  C++ codegen → nevresim  │
                    └─────────────────────────┘
```

---

## 17. Dependency Graph

```
data_handling ◄──── model_training ◄──── tuning ◄──── pipelining
                         │                  │              │
                         ▼                  ▼              ▼
                      models ──────► mapping ────► code_generation
                         │              │              │
                         ▼              ▼              ▼
                    transformations  visualization  chip_simulation
                                                       │
                                        gui ◄───       ▼
                                        │    └── nevresim (C++)
                                        ▼
                                      common
```

Module dependency rules:
- `pipelining` depends on everything (orchestration layer)
- `models` depends on `mapping` (for Mapper classes in `mapping_utils.py`)
- `mapping` depends on `models` (for Perceptron), `transformations`, and `code_generation` (for SpikeSource)
- `tuning` depends on `model_training` and `models`
- `chip_simulation` depends on `code_generation` and `mapping`
- `gui` depends on `common` (for `Reporter` protocol), reads pipeline artifacts (model, IR graph, hardware mapping, search results) via pure snapshot functions; integrated at the entry-point level (`main.py`), not within the pipeline engine
- `data_handling` and `common` are leaf dependencies

---

## 18. Conventions and Patterns

### Naming Conventions
- **Pipeline steps**: `{Action}Step` (e.g., `PretrainingStep`, `WeightQuantizationStep`)
- **Tuners**: `{Type}Tuner` (e.g., `ClampTuner`, `CoreFlowTuner`)
- **Mappers**: `{Type}Mapper` (e.g., `PerceptronMapper`, `Conv2DPerceptronMapper`)
- **Builders**: `{Model}Builder` (e.g., `PerceptronMixerBuilder`)

### Design Patterns
- **Pipeline + Step**: Command pattern for sequential execution with dependency injection via cache
- **Decorator**: Extensively used for activation function composition (`TransformedActivation`, `RateAdjustedDecorator`)
- **Registry**: `BasicDataProviderFactory` uses class-level registration for data providers
- **Dual-Purpose Graph**: `ModelRepresentation`/`Mapper` graph serves both forward pass and hardware mapping
- **Protocol-Based Interfaces**: `SearchProblem`, `SearchOptimizer` use Python Protocols for flexibility
- **Bridge/Adapter**: IR ↔ SoftCore conversion utilities bridge old and new mapping systems

### Scale and Quantization Conventions
- `activation_scale`: The clamping range for a perceptron's output (ReLU output is clamped to `[0, activation_scale]`)
- `parameter_scale`: `weight * parameter_scale ≈ integer` after weight quantization
- `per_input_scales`: 1-D tensor set at mapping time; each element is the source's `activation_scale` for that input channel
- Effective weight: `(per_input_scales[j] * layer.weight[i,j] * normalization_factor) / activation_scale`
- Effective bias: `(layer.bias - running_mean) * normalization_factor + beta) / activation_scale`

### Cache Strategy Selection
- Use `"basic"` for JSON-serializable scalars and configs
- Use `"torch_model"` for `nn.Module` instances (preserves device info)
- Use `"pickle"` for complex objects (`IRGraph`, `AdaptationManager`, `SearchResult`)

---

## 19. Contributing Guide

### Adding a New Pipeline Step

1. Create a new file in `pipelining/pipeline_steps/`
2. Subclass `PipelineStep`, declaring `requires`, `promises`, `updates`, `clears`
3. Implement `process()` (the computation) and `validate()` (returns a metric)
4. Register the step in `pipelining/pipeline_steps/__init__.py`
5. Add it to the appropriate pipeline(s) in `pipelining/pipelines/`
6. Ensure the data contracts are satisfied (promises/requires chain)

### Adding a New Model Architecture

1. Create a new `PerceptronFlow` subclass in `models/perceptron_mixer/` (see `VisionTransformer` as an example of a complex architecture with non-neural operations)
2. Build the computation using the `Mapper` graph (mappers from `mapping/mapping_utils.py`). For operations that cannot be mapped to crossbar cores (e.g., LayerNorm, attention, element-wise add), use the appropriate `ComputeOp`-emitting mappers (`LayerNormMapper`, `MultiHeadAttentionComputeMapper`, `AddMapper`, etc.)
3. Implement `get_perceptrons()`, `get_perceptron_groups()`, `get_mapper_repr()`
4. Create a builder in `models/builders/` with a `build(configuration)` method
5. Register the builder in the architecture search step if using NAS
6. Use `normalization=nn.Identity()` for `Perceptron`s where normalization is handled externally (e.g., via `LayerNormMapper`); only use `use_batchnorm=True` where BatchNorm fusion (§5.12) is desired

### Adding a Native PyTorch Model (torch_mapping)

Instead of hand-building a `PerceptronFlow` + Mapper graph, you can use a standard
PyTorch `nn.Module` and let the `torch_mapping` module convert it automatically:

1. Create a builder in `models/builders/` whose `build()` returns a plain `nn.Module`
   (not a `Supermodel`).  See `TorchVGG16Builder` or `TorchCustomBuilder` as examples.
2. Register the builder in `ModelConfigurationStep` with a `"torch_*"` model type name.
3. The pipeline will automatically insert `TorchMappingStep` after Pretraining, which
   traces the trained model with `torch.fx`, validates representability, converts to a
   Mapper DAG with Perceptron wrappers, and wraps everything in a `Supermodel`.
4. The conversion transfers trained weights and absorbs BatchNorm / activation into
   Perceptrons.  The resulting Supermodel is compatible with all downstream pipeline
   steps (normalization fusion, quantization, core mapping, etc.).
5. Use `check_representability()` from `torch_mapping` to verify your model before
   running the full pipeline.

### Adding a New Data Provider

1. Create a new file in `data_handling/data_providers/`
2. Subclass `DataProvider`
3. Implement `_get_training_dataset()`, `_get_validation_dataset()`, `_get_test_dataset()`, `get_prediction_mode()`
4. Register with `@BasicDataProviderFactory.register("your_name")`
5. Import the module in `data_handling/data_providers/__init__.py`

### Adding a New Mapper (for IR Mapping)

1. Subclass `Mapper` in `mapping/mapping_utils.py`
2. Implement `_forward_impl(x)` for the PyTorch forward pass
3. Implement `_map_to_ir(ir_mapping)` to create `NeuralCore`s and/or `ComputeOp`s
4. Optionally implement `_map(mapping)` for the legacy SoftCoreMapping path
5. Use the mapper in your model's mapper graph

### Adding a New ComputeOp Type

1. Add the operation name to `ComputeOp._dispatch()` in `mapping/ir.py`
2. Implement the `_exec_{op_type}(flat_input, batch_size)` method — receives flat `(B, N)` tensor, reshapes internally as needed, returns flat output
3. Both `SpikingUnifiedCoreFlow` and `SpikingHybridCoreFlow` use `node.execute_on_gathered()` generically, so no changes are needed in the simulators unless the operation requires special spiking semantics
4. Create the appropriate `Mapper` subclass in `mapping_utils.py` that emits the new ComputeOp via `ir_mapping.add_compute_op()`

### Running the Pipeline

```bash
# Activate virtual environment
source venv/bin/activate  # or your environment

# Run with a config file
python src/main.py configs/your_config.json
```

### Key Environment Requirements
- Python 3.10+
- CUDA-capable GPU (strongly recommended)
- C++20 compiler: Clang ≥ 17 (preferred) or `g++-11` (fallback) — required for `std::ranges` support in nevresim
- Dependencies: `torch`, `torchvision`, `einops`, `numpy`, `wandb`, `pymoo`, `matplotlib`, `plotly`, `fastapi`, `uvicorn`, `websockets`

