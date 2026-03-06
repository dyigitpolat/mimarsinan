# Pipeline Pruning Compaction — Implementation Plan

## Objective

Fix the issue where pruned soft cores are mapped during hardware core mapping **without compaction**, so that all-zero rows and columns do not appear in the final hard core mapping. This document specifies concrete code changes and tests to apply once the investigation report has identified the root cause.

## Prerequisites

1. **Complete the investigation:** Run the pipeline with `PRUNING_INVESTIGATION=1` (see [pipeline_pruning_compaction_investigation_report.md](pipeline_pruning_compaction_investigation_report.md)) and capture the full log through Soft Core Mapping and Hard Core Mapping.
2. **Fill the hypothesis table** in the investigation report and document the **root-cause summary** (which of H1–H6 is confirmed and at which step the chain breaks).

## Root-cause–dependent fixes

Apply the fix that matches the confirmed hypothesis.

### H1: Pruning disabled or PruningAdaptation not in pipeline

- **Cause:** Config has `pruning: false` or `pruning_fraction: 0`, so `PruningAdaptationStep` is never added and the model never gets `prune_mask` / `prune_bias_mask`. SoftCoreMappingStep never calls `prune_ir_graph`.
- **Fix:** Ensure the run uses a config with `pruning: true` and `pruning_fraction > 0`. No code change required unless the intent is to support “compaction from zero-threshold only” when pruning is off; in that case, keep the existing behavior (no compaction when pruning is disabled) and document it.

### H2: get_initial_pruning_masks_from_model returns empty

- **Cause:** 1:1 branch skipped due to shape mismatch (`nr != n_axons or nc != n_neurons`); or tiled branch fails to consume nodes (e.g. `ax_node != n_axons`, or column slice length mismatch).
- **Fix (ir_pruning.py):**
  - **1:1:** Log or relax the shape check so that IR matrix dimensions are aligned with model layer dimensions (e.g. account for bias row). Ensure `node.get_core_matrix(ir_graph)` shape matches `(in_f+1, out_f)` from the layer’s `prune_mask`.
  - **Tiled:** Verify column slicing: `end - col_offset == ay_node` and `len(col_slice) == ay_node`. If the mapper produces nodes with different column counts, ensure `ir_col_mask_full` is sliced so that each node gets a mask of length equal to its `get_core_matrix` column count. Adjust the slice (e.g. `col_slice = ir_col_mask_full[col_offset:col_offset+ay_node]`) and padding so that every tiled node receives a valid mask.

### H3: prune_ir_graph does not set or incorrectly sets masks (e.g. bank-backed)

- **Cause:** Phase 5 in `prune_ir_graph` sets per-node `pruned_col_mask` for bank-backed cores; the slice length may not match the node’s effective matrix columns (e.g. `weight_row_slice` or `get_core_matrix` columns).
- **Fix (ir_pruning.py):** For bank-backed nodes, set `pruned_col_mask` (and row mask if applicable) so that their lengths exactly match `get_core_matrix(ir_graph).shape` for that node. Use the node’s effective matrix shape (e.g. from `weight_row_slice` or bank slice) when building the slice from `pruned_col_mask_full` / row mask. After setting masks, assert or validate `len(node.pruned_row_mask) == mat.shape[0]` and `len(node.pruned_col_mask) == mat.shape[1]` for each node.

### H4: Masks lost in segment deepcopy

- **Cause:** `copy.deepcopy(n)` in `_remap_external_sources_to_segment_inputs` does not copy `pruned_row_mask` / `pruned_col_mask` (e.g. if they are not dataclass fields or are stored in a way that is not deep-copied).
- **Fix (ir.py / hybrid_hardcore_mapping.py):** Ensure `NeuralCore` defines `pruned_row_mask` and `pruned_col_mask` as dataclass fields (or otherwise that they are copied in `deepcopy`). If they are not fields, add them to the dataclass or explicitly copy them in `_remap_external_sources_to_segment_inputs` when building `n2` from `n`.

### H5: neural_core_to_soft_core drops masks (length mismatch)

- **Cause:** For segment graph, `get_core_matrix(graph)` returns a shape (e.g. for bank-backed cores a slice) that does not match the length of `pruned_row_mask` / `pruned_col_mask` on the node (e.g. full-layer mask length vs slice length), so the check at ir.py lines 642–644 sets masks to None.
- **Fix (ir.py):** Ensure that when the node has masks, they are defined for the **same** matrix layout as `get_core_matrix(graph)`. For bank-backed nodes, `prune_ir_graph` must set masks with length equal to the node’s effective matrix (Phase 5 slice). If the segment graph passes the same nodes (or deep copies) and the same `weight_banks`, `get_core_matrix(seg_graph)` should return the same shape as in the full graph; if not, align mask lengths with the segment graph’s matrix shape (e.g. by slicing the existing masks to the effective matrix size before the length check), or ensure `prune_ir_graph` always writes masks with the node’s effective shape so that no slicing is needed in `neural_core_to_soft_core`.

### H6: compact_soft_core_mapping skips (missing/mismatched masks on SoftCore)

- **Cause:** SoftCores reach compaction without `pruned_row_mask` / `pruned_col_mask` or with length mismatch (e.g. due to H5). So compaction is skipped and dimensions are not reduced.
- **Fix:** Resolve H5 (and any earlier step that leaves masks missing or wrong length). Optionally, in `compact_soft_core_mapping`, add an assertion or log when a core has no masks but the pipeline config had pruning enabled, so that regressions are visible.

## Tests and assertions to add

1. **Integration test (pipeline with pruning):** In [tests/unit/mapping/test_pruning_verification_flow.py](tests/unit/mapping/test_pruning_verification_flow.py) (or a new integration test module), add a test that:
   - Builds a small model, enables pruning, runs through Soft Core Mapping and Hard Core Mapping (or at least through `build_hybrid_hard_core_mapping` with a pruned IR),
   - Asserts that every soft core in the segment has `pruned_row_mask` and `pruned_col_mask` with lengths matching `core_matrix.shape` when pruning is enabled and the model had prune buffers,
   - And/or asserts that the total used axons/neurons in the hard core mapping is strictly less than the unpruned case (as in existing `TestPruningVerificationIntegration`).

2. **Unit test: get_initial_pruning_masks_from_model (tiled):** When `len(neural_cores) != len(perceptrons)`, assert that every node in the tiled layer gets an entry in `initial_pruned_per_node` and that the column mask length for each node equals that node’s `get_core_matrix(ir_graph).shape[1]`.

3. **Unit test: prune_ir_graph bank-backed:** For a bank-backed node with `weight_row_slice`, after `prune_ir_graph` assert `len(node.pruned_col_mask) == get_core_matrix(ir_graph).shape[1]` and `len(node.pruned_row_mask) == get_core_matrix(ir_graph).shape[0]`.

4. **Assertion in compact_soft_core_mapping (optional):** When config or a global indicates pruning was enabled, assert that cores that have no masks (and are skipped) are expected (e.g. from a layer without prune_mask); or log a warning so that “all cores skipped” is visible in CI.

## Summary

- Complete the investigation using the instrumentation in the report and fill the hypothesis table and root-cause summary.
- Apply the fix corresponding to the confirmed hypothesis (H1–H6) in the files listed above.
- Add or extend tests so that pruning compaction is covered and regressions (e.g. masks dropped, compaction skipped for all cores) are caught.
- Optionally keep `PRUNING_INVESTIGATION` logging for future debugging; it is already gated by the environment variable.
