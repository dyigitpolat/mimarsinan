# Hardware heatmaps — target rendering spec (W2.2)

The written UX contract for weight-heatmap rendering across the monitor
(hardware workbench grid, hard-core inspector, soft-core inspector, pruning
tab). Owner pixel review judges screenshots against THIS document.

## Rendering guarantees (backend: `gui/rendering/`)

1. **Square cells.** One integer pooling factor `k = ceil(long_side / target)`
   applies to BOTH axes of a matrix, so a cell is always square in PNG pixel
   space. When the dims divide by `k` the pixel grid is an exact integer
   fraction of the matrix grid.
2. **Zero margins.** PNG pixel `[0, 0]` IS matrix cell `[0, 0]`; the last
   pixel row/column is the last matrix row/column. No axes, ticks, padding, or
   tight-layout drift — percent-positioned frontend overlays (constituent
   boundaries, placement regions, fused lines) are correct by construction.
3. **Native size below the target.** A matrix whose long side is at or below
   `DEFAULT_TARGET_LONG_SIDE = 400` renders at 1 px per cell; the browser
   upscales it (`image-rendering: pixelated` when upscaling, smooth sampling
   when downscaling). No server-side upscaling ever.
4. **Mask survival.** Pruned rows/columns paint the reserved red `#e53935`.
   Decimation ANY-pools the line masks and dilates isolated lines to >= 2 px,
   so a SINGLE pruned line in a 1000x1000 matrix is visible in a 200 px tile.
   Values pool by signed max-|value|, so extreme weights survive any zoom.
5. **Empty matrices** render a distinct 64x64 placeholder tile (background +
   border + diagonal cross), never a stretched 1x1.
6. **NaN** cells render the background color `#2b303c` at native resolution.
7. **Purity.** Rendering is a pure function of arrays (numpy + stdlib zlib) —
   no matplotlib/pyplot, thread-safe, byte-deterministic.

## Shared color scale (one per family per snapshot)

- Families: hardware cores (`hard_core_mapping`), IR cores + pre-pruning +
  weight banks (`ir_graph`), pruning layers (`pruning_layers`). Bias strips
  keep per-strip autoscale (different units).
- The family scale is the **max of per-matrix p98 symmetric scales**: each
  matrix saturates at most 2% of its own cells while every tile in the family
  is directly comparable. Values beyond the scale clip to the LUT endpoints.
- The colormap is the 256-entry BrBG LUT (brown = negative, neutral centre,
  teal = positive).
- The snapshot summary carries `heatmap_scale: {vmin, vmax,
  colorbar_resource}`; a colorbar PNG is served as its own resource
  (`heatmap_colorbar` kind).
- **Display**: the hardware workbench card header shows the legend
  (`-vmax … colorbar … +vmax · pruned chip`) for the hard-core family; the
  soft-core inspector shows the IR-family legend above its heatmap row. The
  two scales may differ — each legend labels its own family.

## Alignment guarantees (frontend)

- `.hw-core` and `.hw-insp-heatmap-wrap` are `content-box`: inline
  width/height/aspect-ratio target the IMAGE box; borders add outside and the
  grid cell budget compensates (`CORE_BORDER_PX`).
- `.hw-core-constituent-overlay` is `position: absolute; inset: 0` — pinned to
  the image box. With margin-free PNGs, percentage overlays land on the exact
  matrix cells.
- Images use `object-fit: fill` intentionally: the PNG aspect equals the
  matrix aspect exactly, so fill introduces zero distortion.

## Instant-load budget

- Snapshot-persist pre-renders the UI-resolution artifact (<= 400 px long
  side) for BOTH render policies and persists the source (.mimsrc); a browser
  attach — first or later — is a plain file read per tile, never a
  1024 px matplotlib render storm (was: 474–555 per-request renders).
- All three resource routes answer conditional requests: `ETag` on every 200,
  empty-body `304` on `If-None-Match` match. Revisits revalidate, not
  re-download.
- The live in-memory payload cache is LRU-bounded at 64 MB
  (`RESOURCE_STORE_MAX_PAYLOAD_BYTES`); evicted payloads re-materialise on
  demand.
- On-demand zoom: `?res=full` serves a near-native render (<= 1600 px long
  side) from the persisted source, cached as a `.full.png` sibling; absent a
  source it falls back to the UI artifact.

## Explicit diagnostics

- A soft core with a post-pruning heatmap but no stored pre-pruning
  matrix/masks shows an inline "Pre-pruning view unavailable" tile instead of
  silently dropping the view.

## Owner pixel review checklist (post-merge)

- [ ] Grid tiles: square cells, no letterboxing, no smearing (`object-fit`).
- [ ] Constituent/fused overlays sit exactly on cell boundaries at small and
      large tile sizes.
- [ ] Colorbar legend visible in the workbench header; numbers match
      `heatmap_scale.vmin/vmax`; soft inspector shows the IR legend.
- [ ] A pruned row/col reads clearly red at every zoom level.
- [ ] Small cores (e.g. 8x8) render crisp (pixelated), large cores smooth.
- [ ] First attach to a finished headless run paints the hardware tab without
      a visible render stall; second visit returns 304s (devtools network).
