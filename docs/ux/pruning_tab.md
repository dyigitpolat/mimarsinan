# Pruning tab — UX spec (owner pixel review)

Scope: the "Pruning" tab on the Pruning Adaptation step-detail page
(`gui/static/js/pruning-tab.js`, data from
`gui/snapshot/model_snapshot.py::snapshot_pruning_layers` via
`build_step_snapshot`). The tab is mounted only for the step named by the
registry constant `PRUNING_ADAPTATION_STEP`
(`pipelining/core/step_plan.py`; re-exported by `core/pipelines/deployment_specs.py`).

Conventions the tab inherits:

- Masks follow the model convention **True = PRUNED**; `prune_row_mask`
  indexes out_features (neurons), `prune_col_mask` in_features (axons).
- Committed weights stay full-width on the model (zeroed rows), so
  **pre** dims come from `weight.shape` and **post** dims are the kept counts.
- Red row/col lines over matrices mark pruned rows/cols — same convention
  as the IR Graph and Hardware tabs.

## State 1 — pruned run (`pruning: true`, `pruning_fraction > 0`)

Layout: a left layer list (220 px), then a details card, a **Mask map** card,
and a **Weight matrix** card (cards wrap on narrow windows); any skipped
layers appear in a warning card below the browse area.

1. Layer list — one entry per surviving layer:
   - layer name (e.g. `perceptron_0`), bold;
   - pre → post neurons line (e.g. `120 → 96 neurons`);
   - pre → post axons line (e.g. `784 → 600 axons`);
   - pruned counts line (e.g. `24 rows, 184 cols pruned`).
   Clicking an entry selects it (accent highlight) and updates the panels.
2. Layer details card — a table with:
   - Layer (name + index);
   - Neurons `pre → post`; Axons `pre → post`;
   - Pruned rows; Pruned cols;
   - Achieved sparsity (neurons) and (axons) as percentages
     (`pruned / pre`, e.g. `20.0%`);
   - Configured fraction (`pruning_fraction` from the run config, e.g.
     `20.0%`) — the achieved-vs-configured comparison is read from these
     two adjacent rows.
3. Mask map card — a PNG built from the masks only: the outer product of
   the keep-masks. Kept weights render as the positive band (green in the
   default BrBG colormap), any cell whose row OR col is pruned renders as
   the negative band (brown), and the red row/col lines repeat the pruned
   coordinates. A fully unpruned layer is a solid positive field.
4. Weight matrix card — the existing post-pruning weight heatmap with red
   pruned-row/col lines.
5. Image failure: if either PNG fails to load (404, dead resource), the
   `<img>` is replaced by a visible red chip reading
   `mask map failed to load` / `weight heatmap failed to load` — never a
   blank frame.

## State 2 — unpruned run (`pruning: true`, `pruning_fraction: 0`)

The step ran but pruned nothing: masks exist and are all-False.

- Layer list shows `N → N neurons` / `M → M axons`, `0 rows, 0 cols pruned`.
- Details show achieved sparsity `0.0%` against the configured `0.0%`.
- Mask map is a uniform positive (kept) field with no red lines.
- Weight heatmap renders normally with no red lines.

(When `pruning: false`, the Pruning Adaptation step is not in the pipeline
at all, so no step page and no tab exist — that absence is the expected
state, not a defect.)

## State 3 — skip diagnostics (never a silently empty tab)

Every skip path surfaces a structured `{layer, reason}` entry rendered in a
warning-bordered **Skipped layers (n)** card listing the layer name (or
`<model>` for model-level conditions) and the reason:

- missing `prune_row_mask`/`prune_col_mask` buffers on a layer;
- mask length mismatch (reason names both lengths);
- no weight tensor on a layer;
- bare `nn.Linear` fallback (model exposes no perceptrons — synthesized
  wrappers never carry masks);
- no model in the step snapshot (model-level, emitted by the builder).

When ALL layers are skipped (or the model is missing), the tab shows an
explanatory empty-state line ("No pruning layer data — every layer was
skipped.") directly above the skipped-layers card. The tab itself always
mounts for the Pruning Adaptation step.

## Review checklist

- [ ] Pruned run: pre→post dims correct against the run log; achieved vs
      configured rows adjacent and plausible.
- [ ] Mask map and weight heatmap agree on which rows/cols are pruned.
- [ ] Unpruned run: all-kept mask map, 0% sparsity rows.
- [ ] Kill the resource endpoint (or open a stale historical run) and
      confirm the red failure chips replace the images.
- [ ] Skip case: maskless layer shows up in the warning card with a
      readable reason.
