# Pipeline Pruning Compaction Investigation Report

## Purpose

Determine why pruned soft cores still appear in the final hard core mapping without compaction (all-zero rows and columns present). This report documents the instrumentation added and how to complete the investigation to produce a root-cause summary and implementation plan.

## Investigation Phases Implemented

### Phase 1: Config and pipeline composition

**Instrumentation added:**

- **DeploymentPipeline** ([src/mimarsinan/pipelining/pipelines/deployment_pipeline.py](src/mimarsinan/pipelining/pipelines/deployment_pipeline.py)): When assembling steps, logs whether pruning is enabled and whether `PruningAdaptationStep` was added: `[DeploymentPipeline] Pruning enabled: pruning=True, pruning_fraction=0.1; PruningAdaptationStep added.` or `[DeploymentPipeline] Pruning not in pipeline: ...`.
- **SoftCoreMappingStep** ([src/mimarsinan/pipelining/pipeline_steps/soft_core_mapping_step.py](src/mimarsinan/pipelining/pipeline_steps/soft_core_mapping_step.py)): At start of `process()`, when `PRUNING_INVESTIGATION=1`, logs effective `pruning` and `pruning_fraction` from config, and for the first 5 perceptrons logs presence and shape of `layer.prune_mask`, `layer.prune_bias_mask`, and `layer.weight.shape`.

**Evidence from run:** A pipeline run with `examples/mnist_arch_search_nsga2_ttfs.json` (pruning true, pruning_fraction 0.1) confirmed: `[DeploymentPipeline] Pruning enabled: pruning=True, pruning_fraction=0.1; PruningAdaptationStep added.`

### Phase 2: IR pruning and initial masks

**Instrumentation added:**

- **get_initial_pruning_masks_from_model** ([src/mimarsinan/mapping/ir_pruning.py](src/mimarsinan/mapping/ir_pruning.py)): When `PRUNING_INVESTIGATION=1`:
  - **Tiled branch:** Logs one example per first few nodes: `node_id`, `col_offset`, `end`, `ay_node`, `len(col_slice)`. Before return, logs `branch=tiled`, `neural_cores`, `perceptrons`, `initial_pruned_per_node_count`, and for up to 3 node_ids: `row_len`, `row_pruned`, `col_len`, `col_pruned`.
  - **1:1 branch:** Before return, logs `branch=1:1`, same counts, and for up to 3 node_ids the same mask stats.
- **SoftCoreMappingStep** (existing + enhancement): After `prune_ir_graph`, when `PRUNING_INVESTIGATION=1`, for each neural core logs `node_id`, `get_core_matrix.shape`, `pruned_row_mask` / `pruned_col_mask` lengths and sums, and **WARNING** if mask length does not equal matrix rows/columns (so that later mask drop in `neural_core_to_soft_core` can be correlated).

### Phase 3: Segment graph → soft cores → compaction

**Instrumentation added:**

- **_flush_neural_segment** ([src/mimarsinan/mapping/hybrid_hardcore_mapping.py](src/mimarsinan/mapping/hybrid_hardcore_mapping.py)): When `PRUNING_INVESTIGATION=1`, before and after `_remap_external_sources_to_segment_inputs`, logs the first node’s `pruned_row_mask` and `pruned_col_mask` lengths (to confirm deepcopy preserves masks). Existing logs for each soft core before/after `compact_soft_core_mapping` (shape and mask presence) remain.
- **neural_core_to_soft_core** ([src/mimarsinan/mapping/ir.py](src/mimarsinan/mapping/ir.py)): When `PRUNING_INVESTIGATION=1`, for each core logs `node_id`, `core_matrix.shape`, `pruned_row_mask` len, `pruned_col_mask` len, and `attach=True/False`. If masks are dropped due to length mismatch, logs `dropped masks (length mismatch)`.
- **compact_soft_core_mapping** ([src/mimarsinan/mapping/softcore_mapping.py](src/mimarsinan/mapping/softcore_mapping.py)): Counts `n_compacted` vs `n_skipped`; when `PRUNING_INVESTIGATION=1`, at end logs `compact_soft_core_mapping summary: n_compacted=... n_skipped=... total=...`. Existing per-core “skip compaction” logs remain.
- **HardCoreMappingStep** ([src/mimarsinan/pipelining/pipeline_steps/hard_core_mapping_step.py](src/mimarsinan/pipelining/pipeline_steps/hard_core_mapping_step.py)): When `PRUNING_INVESTIGATION=1`, at start of `process()` logs for the first 5 neural nodes in `ir_graph` the presence and length of `pruned_row_mask` and `pruned_col_mask` (to confirm what is passed into `build_hybrid_hard_core_mapping`).

### Phase 4: End-to-end verification

**How to reproduce and collect evidence:**

1. Run the pipeline with pruning enabled and `PRUNING_INVESTIGATION=1`:
   ```bash
   PRUNING_INVESTIGATION=1 PYTHONPATH=src python scripts/investigate_pruning_flow.py examples/mnist_arch_search_nsga2_ttfs.json
   ```
   Or run the full pipeline (e.g. `python run.py examples/mnist_arch_search_nsga2_ttfs.json`) with `PRUNING_INVESTIGATION=1` set in the environment, and capture stdout to a log file.

2. In the log, collect:
   - **Phase 1:** `[DeploymentPipeline]` and `[PRUNING_INVESTIGATION] SoftCoreMappingStep` (config and first perceptrons’ buffers).
   - **Phase 2:** `[PRUNING_INVESTIGATION] get_initial_pruning_masks` (branch, counts, sample node mask stats), `[PRUNING_INVESTIGATION] After prune_ir_graph` (per-node shape and masks), and any `WARNING ... row_mask length ... != matrix rows` or `col_mask length ... != matrix cols`.
   - **Phase 3:** `[PRUNING_INVESTIGATION] _flush_neural_segment before/after segment build`, `[PRUNING_INVESTIGATION] neural_core_to_soft_core` (and “dropped masks”), `[PRUNING_INVESTIGATION] compact_soft_core_mapping skip compaction` vs `compact_soft_core_mapping summary`, `[PRUNING_INVESTIGATION] _flush_neural_segment before/after compact`, and `[PRUNING_INVESTIGATION] HardCoreMappingStep ir_graph input`.

3. From the script output, use `_report_ir_graph` and `_report_hard_core_mapping` to see final IR node shapes/masks and hard core dimensions and all-zero row/column counts.

## Hypothesis table (filled from out.log)

| Hypothesis | Description | Result from out.log |
|------------|-------------|---------------------|
| **H1** | Pruning disabled or PruningAdaptation not in pipeline → no masks on model. | **Ruled out.** `[DeploymentPipeline] Pruning enabled: pruning=True, pruning_fraction=0.1`. All 10 perceptrons have `prune_mask` and `prune_bias_mask`. |
| **H2** | Model has buffers but `get_initial_pruning_masks_from_model` returns empty (1:1 mismatch, tiled logic bug, or shape mismatch). | **CONFIRMED.** Tiled branch: `neural_cores=135 perceptrons=10 initial_pruned_per_node_count=1`. Only node 0 received initial masks; 134 nodes got no model-derived masks. |
| **H3** | Initial masks exist but `prune_ir_graph` does not set or incorrectly sets masks on some nodes (e.g. bank-backed slice length wrong). | **CONFIRMED.** Node 0: matrix (1, 14) but row_mask len=57, col_mask len=16 → WARNING. Nodes 30–45: matrix (127, 14) but row_mask len=129 → WARNING (off-by-2, likely bias row). Masks not updated to post-compaction dimensions. |
| **H4** | Masks correct on IR but lost in segment deepcopy. | **Ruled out.** Before/after segment build: first node keeps pruned_row_mask len=57, pruned_col_mask len=16. |
| **H5** | Masks lost in `neural_core_to_soft_core` due to length mismatch between `get_core_matrix(graph)` shape and mask length. | **CONFIRMED.** Core 0: shape (1, 14) vs mask (57, 16) → masks dropped. Cores 30–45, 90–105, 134: missing masks at compact (length mismatch or no initial masks). |
| **H6** | `compact_soft_core_mapping` skips due to missing/mismatched masks on SoftCore. | **CONFIRMED.** `compact_soft_core_mapping summary: n_compacted=0 n_skipped=135 total=135`. 46 cores skipped due to missing masks; 89 had masks but all-False (no model pruning). |

## Counts recorded (from out.log)

- **Initial masks:** `initial_pruned_per_node_count=1` (only node 0; 135 neural cores, 10 perceptrons).
- **After prune_ir_graph:** 1 node with non-zero mask sums (node 0: 56 rows, 2 cols); many nodes with length mismatch (node 0: 57/16 vs matrix 1×14; nodes 30–45: row 129 vs 127).
- **Before compact:** 89 soft cores with `present=True len_match=True` (masks all-False); 46 with `present=False` (core 0, 30–45, 90–105, 134).
- **Compaction:** `n_compacted=0`, `n_skipped=135`.
- **Hard core mapping:** (Use `_report_hard_core_mapping` section in log for all-zero row/column counts if present.)

## Root-cause summary (from out.log)

1. **Primary (H2):** In the **tiled branch** of `get_initial_pruning_masks_from_model`, only **one node** (node 0) receives initial pruning masks. The loop consumes one perceptron and assigns masks to one IR node, then stops assigning for the remaining 134 nodes. So the vast majority of IR nodes never get model-derived pruning; they get all-False masks from propagation or default, and no compaction is possible for them.

2. **Secondary (H3 + H5):** For the single node that did get masks (node 0), `prune_ir_graph` **compacts the matrix** to (1, 14) but **leaves** `pruned_row_mask` / `pruned_col_mask` at **pre-compaction lengths** (57, 16). When building soft cores, `neural_core_to_soft_core` sees a length mismatch and **drops** the masks, so core 0 has no masks at compaction time. Similarly, nodes 30–45 have row mask length 129 vs matrix rows 127 (off-by-2, likely bias row), so their masks are dropped and 16 more cores skip compaction.

3. **Result (H6):** No soft core is compacted (`n_compacted=0`). Pruned rows/columns therefore remain in the soft (and thus hard) core mapping.

**Recommended fixes (see implementation plan):** (1) Fix the tiled branch so **every** tiled node receives correctly sliced initial masks (one perceptron → many nodes with column/row slices). (2) After physical compaction in `prune_ir_graph`, **update** each node's `pruned_row_mask` and `pruned_col_mask` to the **post-compaction** matrix dimensions (or slice masks to match `get_core_matrix` shape) so that `neural_core_to_soft_core` can attach them. (3) Fix the row mask length for nodes 30–45 (129 vs 127) so that mask length equals matrix rows (e.g. exclude bias from row mask when matrix already has bias folded, or use the same convention everywhere).
## Once the log is collected, state here at which step the compaction chain fails (e.g. “initial_node empty for tiled IR”, “neural_core_to_soft_core drops masks for bank-backed cores due to shape mismatch”, “config pruning false”, etc.).*

## Files modified for instrumentation

- [deployment_pipeline.py](src/mimarsinan/pipelining/pipelines/deployment_pipeline.py) — pruning step addition log
- [soft_core_mapping_step.py](src/mimarsinan/pipelining/pipeline_steps/soft_core_mapping_step.py) — Phase 1 model buffers; Phase 2 post–prune_ir_graph and length mismatch WARNING
- [ir_pruning.py](src/mimarsinan/mapping/ir_pruning.py) — get_initial_pruning_masks_from_model branch and sample logs
- [hybrid_hardcore_mapping.py](src/mimarsinan/mapping/hybrid_hardcore_mapping.py) — segment build before/after mask lengths
- [ir.py](src/mimarsinan/mapping/ir.py) — neural_core_to_soft_core shape/mask/attach and drop log
- [softcore_mapping.py](src/mimarsinan/mapping/softcore_mapping.py) — n_compacted/n_skipped and summary log
- [hard_core_mapping_step.py](src/mimarsinan/pipelining/pipeline_steps/hard_core_mapping_step.py) — IR input node mask presence/lengths

All instrumentation is gated by `os.environ.get("PRUNING_INVESTIGATION")` so it does not affect production runs unless explicitly enabled.
