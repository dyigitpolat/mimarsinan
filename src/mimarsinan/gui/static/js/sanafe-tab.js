/* SANA-FE Simulation tab.
 *
 * Two side-by-side floorplan panels per segment.  Each panel has
 * **independent** metric (cell colour) and overlay (lines on top)
 * selectors, so you can compare "spikes + tile-to-tile traffic" next
 * to "energy + NoC congestion" at a glance.  Defaults:
 *
 *   left  panel: spikes_fired + NoC routes
 *   right panel: energy_j     + NoC congestion
 *
 * Overlay options (per panel):
 *   - None
 *   - NoC routes       (cubic-bezier curves between tile centres,
 *                       A→B and B→A on opposite sides, arrowheads at
 *                       the destination, per-curve + per-tile hover)
 *   - NoC congestion   (per-mesh-edge XY-routed packet density)
 *   - Live connectivity (static core-to-core weight edges)
 *   - Critical cores   (markers on the per-cycle critical core)
 *
 * The Plotly-driven cycle panels (cascade, waterfall, playback,
 * efficiency scatter) and per-core bar charts are unchanged.
 *
 * Playback: backed by ``noc_traffic_per_cycle``.  The chip floorplan
 * stays visible as a low-opacity underlay so the chart never blanks
 * out even on cycles with no inter-tile traffic; an explicit
 * "no traffic" notice replaces the panel when the run has zero NoC
 * activity at all (e.g. legacy single-tile arch dumps).
 *
 * All snapshot data is read straight from ``snap.sanafe_simulation`` —
 * no lazy resources are used by this tab.
 */
import { esc, safeReact } from './util.js';

function fmtMj(j) { return (j * 1000).toFixed(3); }
function fmtSec(s) { return s.toExponential(3); }
function fmtInt(n) { return Number(n).toLocaleString(); }

/* Per-cell colour metric (drives the chip's cell-fill colour). */
const CELL_METRICS = [
  { key: 'energy_j',         label: 'Energy (J, log10)',     log: true,  cmap: 'YlOrRd' },
  { key: 'spikes_fired',     label: 'Spikes Fired',          log: false, cmap: 'Viridis' },
  { key: 'core_latency',     label: 'Latency Layer',         log: false, cmap: 'Portland' },
  { key: 'n_neurons',        label: 'Live Neurons',          log: false, cmap: 'Plasma' },
  { key: 'n_axons_used',     label: 'Live Axons',            log: false, cmap: 'Plasma' },
  { key: 'has_hardware_bias',label: 'Has Hardware Bias',     log: false, cmap: 'Greys' },
  { key: 'n_always_on_axons',label: 'Always-On Axons',       log: false, cmap: 'Cividis' },
  { key: 'hcm_diff_input',   label: 'HCM Δ input (SF−HCM)',  log: false, cmap: 'RdBu',  diverging: true },
  { key: 'hcm_diff_output',  label: 'HCM Δ output (SF−HCM)', log: false, cmap: 'RdBu',  diverging: true },
];

/* Overlay choices — drives the SVG layer drawn on top of the cells. */
const OVERLAY_OPTIONS = [
  { key: 'none',           label: 'No Overlay'             },
  { key: 'noc_routes',     label: 'NoC Routes (tile→tile)' },
  { key: 'noc_congestion', label: 'NoC Congestion (mesh edges)' },
  { key: 'connectivity',   label: 'Live Connectivity (core→core)' },
  { key: 'critical_cores', label: 'Critical Cores'         },
];

const NOC_THICKNESS_METRICS = [
  { key: 'packet_count', label: 'Packet Count' },
  { key: 'spike_count',  label: 'Total Spikes' },
  { key: 'total_hops',   label: 'Total Hops'   },
];

/* Per-panel defaults. */
const PANEL_DEFAULTS = [
  { metric: 'spikes_fired', overlay: 'noc_routes',     thickness: 'packet_count' },
  { metric: 'energy_j',     overlay: 'noc_congestion', thickness: 'packet_count' },
];

/* ---------------------------------------------------------------- */
/* Colormap helpers (hand-rolled — same scales Plotly ships)         */
/* ---------------------------------------------------------------- */

const COLORMAPS = {
  YlOrRd:   [[255,255,178],[254,217,118],[254,178,76],[253,141,60],[252,78,42],[227,26,28],[177,0,38]],
  Viridis:  [[68,1,84],[59,82,139],[33,144,140],[94,201,98],[253,231,37]],
  Portland: [[12,51,131],[10,136,186],[242,211,56],[242,143,56],[217,30,30]],
  Plasma:   [[13,8,135],[126,3,168],[204,71,120],[248,149,64],[240,249,33]],
  Cividis:  [[0,32,76],[68,77,114],[124,123,120],[181,176,98],[253,231,55]],
  Greys:    [[40,40,40],[120,120,120],[200,200,200],[245,245,245]],
  RdBu:     [[33,102,172],[103,169,207],[247,247,247],[244,165,130],[178,24,43]],
  /* Tab-internal palette for the NoC-congestion overlay.  Earlier
   * cyan→amber→magenta version went through bright yellow in the
   * middle, which clashed loudly with the orange/red energy
   * heatmap underneath.  This is a softer cool-blue → indigo →
   * pink → hot-magenta progression that stays in the cyberpunk
   * register without screaming over the cells. */
  CyberHeat: [[80,120,200],[120,90,210],[190,75,200],[255,65,200]],
  /* Traffic-load palette for NoC-congestion edges: green = light
   * load, yellow = moderate, red = hot.  Standard "speedometer"
   * intuition; reads at a glance against the dark floorplan. */
  Traffic:   [[60,200,100],[240,220,50],[255,60,60]],
};

/* CSS linear-gradient string from a named colormap — used by the
 * inline legend bars in each panel sidebar. */
function _cmapGradient(name) {
  const stops = COLORMAPS[name] || COLORMAPS.Viridis;
  const parts = stops.map((rgb, i) => {
    const pct = (100 * i / (stops.length - 1)).toFixed(1);
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]}) ${pct}%`;
  });
  return `linear-gradient(to right, ${parts.join(', ')})`;
}
function _lerp(a, b, t) { return Math.round(a + (b - a) * t); }
function _sampleCmap(name, t) {
  const stops = COLORMAPS[name] || COLORMAPS.Viridis;
  if (!Number.isFinite(t)) return null;
  if (t <= 0) return stops[0];
  if (t >= 1) return stops[stops.length - 1];
  const idx = t * (stops.length - 1);
  const lo = Math.floor(idx), hi = Math.min(stops.length - 1, lo + 1);
  const f = idx - lo;
  return [
    _lerp(stops[lo][0], stops[hi][0], f),
    _lerp(stops[lo][1], stops[hi][1], f),
    _lerp(stops[lo][2], stops[hi][2], f),
  ];
}
function _rgbStr(rgb, alpha = 1.0) {
  if (!rgb) return 'transparent';
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
}

/* ---------------------------------------------------------------- */
/* Tooltip singleton                                                  */
/* ---------------------------------------------------------------- */

let _tooltipEl = null;
function _getTooltip() {
  if (_tooltipEl && _tooltipEl.isConnected) return _tooltipEl;
  _tooltipEl = document.createElement('div');
  _tooltipEl.className = 'sf-tooltip';
  _tooltipEl.style.display = 'none';
  document.body.appendChild(_tooltipEl);
  return _tooltipEl;
}
function _showTooltip(html, x, y) {
  const el = _getTooltip();
  el.innerHTML = html;
  el.style.display = 'block';
  const margin = 12;
  const rect = el.getBoundingClientRect();
  let left = x + margin, top = y + margin;
  if (left + rect.width > window.innerWidth)  left = x - margin - rect.width;
  if (top  + rect.height > window.innerHeight) top = y - margin - rect.height;
  el.style.left = `${Math.max(4, left)}px`;
  el.style.top  = `${Math.max(4, top)}px`;
}
function _hideTooltip() {
  if (_tooltipEl) _tooltipEl.style.display = 'none';
}

/* ---------------------------------------------------------------- */
/* Floorplan geometry                                                */
/* ---------------------------------------------------------------- */

function _tileGrid(perTile) {
  const maxCores = perTile.reduce((m, t) => Math.max(m, t.cores.length), 1);
  const w = Math.max(1, Math.ceil(Math.sqrt(maxCores)));
  const h = Math.max(1, Math.ceil(maxCores / w));
  return { tileW: w, tileH: h };
}

function _coreMetricValue(core, metricKey, hcmDiffByCore) {
  if (metricKey === 'hcm_diff_input')
    return hcmDiffByCore.get(core.core_index)?.input_delta_sum ?? 0;
  if (metricKey === 'hcm_diff_output')
    return hcmDiffByCore.get(core.core_index)?.output_delta_sum ?? 0;
  return core[metricKey] ?? 0;
}

function _metricRange(seg, metric) {
  const cores = Array.isArray(seg.per_core) ? seg.per_core : [];
  const diff = new Map((seg.hcm_diff || []).map(d => [d.core_index, d]));
  let lo = Infinity, hi = -Infinity;
  cores.forEach(c => {
    let v = _coreMetricValue(c, metric.key, diff);
    if (!Number.isFinite(v)) return;
    if (metric.log) { if (v <= 0) return; v = Math.log10(v); }
    if (v < lo) lo = v; if (v > hi) hi = v;
  });
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = 0; hi = 1; }
  if (hi === lo) hi = lo + 1;
  if (metric.diverging) {
    const m = Math.max(Math.abs(lo), Math.abs(hi));
    lo = -m; hi = m;
  }
  return { lo, hi };
}

function _cellColor(core, metric, range, hcmDiffByCore) {
  let v = _coreMetricValue(core, metric.key, hcmDiffByCore);
  if (!Number.isFinite(v)) return 'rgba(60,60,60,0.4)';
  if (metric.log) {
    if (v <= 0) return 'rgba(60,60,60,0.4)';
    v = Math.log10(v);
  }
  const t = (v - range.lo) / (range.hi - range.lo);
  return _rgbStr(_sampleCmap(metric.cmap, Math.max(0, Math.min(1, t))));
}

function _coreTooltipHtml(core, hcmDiffByCore) {
  const d = hcmDiffByCore.get(core.core_index);
  return (
    `<strong>core ${core.core_index}</strong>` +
    `<div class="sf-tooltip-row"><span class="sf-tooltip-label">energy</span><span>${fmtMj(core.energy_j ?? 0)} mJ</span></div>` +
    `<div class="sf-tooltip-row"><span class="sf-tooltip-label">spikes fired</span><span>${fmtInt(core.spikes_fired ?? 0)}</span></div>` +
    `<div class="sf-tooltip-row"><span class="sf-tooltip-label">latency layer</span><span>${core.core_latency ?? 0}</span></div>` +
    `<div class="sf-tooltip-row"><span class="sf-tooltip-label">live neurons</span><span>${fmtInt(core.n_neurons ?? 0)}</span></div>` +
    `<div class="sf-tooltip-row"><span class="sf-tooltip-label">live axons</span><span>${fmtInt(core.n_axons_used ?? 0)}</span></div>` +
    `<div class="sf-tooltip-row"><span class="sf-tooltip-label">hw bias</span><span>${core.has_hardware_bias ? 'yes' : 'no'}</span></div>` +
    `<div class="sf-tooltip-row"><span class="sf-tooltip-label">always-on axons</span><span>${fmtInt(core.n_always_on_axons ?? 0)}</span></div>` +
    (d ? `<div class="sf-tooltip-row"><span class="sf-tooltip-label">HCM Δ in / out</span><span>${d.input_delta_sum} / ${d.output_delta_sum}</span></div>` : '')
  );
}

/* Tile-aggregate body (used inside the cell tooltip, since tile
 * borders are pointer-events:none and never receive mousemove). */
function _tileTooltipBody(tile, inLinks, outLinks) {
  const inboundPkts = inLinks.reduce((s, l) => s + l.packet_count, 0);
  const outboundPkts = outLinks.reduce((s, l) => s + l.packet_count, 0);
  const fmtLink = (L, fromKey, toKey) =>
    `${L[fromKey]} → ${L[toKey]}  ${L.packet_count} pkt · ${L.spike_count} spk · ${L.total_hops} hops`;
  const inboundList = inLinks
    .slice().sort((a, b) => b.packet_count - a.packet_count)
    .slice(0, 6)
    .map(L => `<div class="sf-tooltip-list">${esc(fmtLink(L, 'src_tile', 'dst_tile'))}</div>`)
    .join('');
  const outboundList = outLinks
    .slice().sort((a, b) => b.packet_count - a.packet_count)
    .slice(0, 6)
    .map(L => `<div class="sf-tooltip-list">${esc(fmtLink(L, 'src_tile', 'dst_tile'))}</div>`)
    .join('');
  return (
    `<div class="sf-tooltip-row"><span class="sf-tooltip-label">cores</span><span>${tile.cores.length}</span></div>` +
    `<div class="sf-tooltip-row"><span class="sf-tooltip-label">tile energy</span><span>${fmtMj(tile.energy_j ?? 0)} mJ</span></div>` +
    `<div class="sf-tooltip-row"><span class="sf-tooltip-label">inbound pkts</span><span style="color:#ff4be8">${fmtInt(inboundPkts)}</span></div>` +
    `<div class="sf-tooltip-row"><span class="sf-tooltip-label">outbound pkts</span><span style="color:#00ffff">${fmtInt(outboundPkts)}</span></div>` +
    (inboundList ? `<div class="sf-tooltip-section-title" style="color:#ff4be8;margin-top:5px">incoming (top ${inLinks.length > 6 ? 6 : inLinks.length})</div>${inboundList}` : '') +
    (outboundList ? `<div class="sf-tooltip-section-title" style="color:#00ffff;margin-top:5px">outgoing (top ${outLinks.length > 6 ? 6 : outLinks.length})</div>${outboundList}` : '')
  );
}

/* ---------------------------------------------------------------- */
/* Overlay renderers (one per overlay kind)                          */
/* ---------------------------------------------------------------- */

/* Compute the (x, y, tangent) of a cubic Bezier at parameter t. */
function _bezierPointAndTangent(p0, p1, p2, p3, t) {
  const mt = 1 - t;
  const x =
    mt*mt*mt*p0.x + 3*mt*mt*t*p1.x + 3*mt*t*t*p2.x + t*t*t*p3.x;
  const y =
    mt*mt*mt*p0.y + 3*mt*mt*t*p1.y + 3*mt*t*t*p2.y + t*t*t*p3.y;
  // dB/dt = 3(1-t)²(p1-p0) + 6(1-t)t(p2-p1) + 3t²(p3-p2)
  const dx =
    3*mt*mt*(p1.x-p0.x) + 6*mt*t*(p2.x-p1.x) + 3*t*t*(p3.x-p2.x);
  const dy =
    3*mt*mt*(p1.y-p0.y) + 6*mt*t*(p2.y-p1.y) + 3*t*t*(p3.y-p2.y);
  const len = Math.hypot(dx, dy) || 1;
  return { x, y, ux: dx/len, uy: dy/len };
}

/* Mesh-graph nodes — one ring per tile, parked at the tile centre.
 * Drawn by both the noc_routes and noc_congestion overlays so the
 * graph structure (vertices = tiles, edges = mesh links) reads
 * unambiguously.  Hover-interactive; the wiring in _renderFloorplan
 * surfaces the per-tile inbound/outbound breakdown. */
function _renderMeshNodes(perTile, tileCenters, geom, meshOriginTx, meshOriginTy, cellPx, tileW, tileH, nodeR) {
  /* Render a node at EVERY mesh position the chip occupies — both
   * tile-bearing positions (e.g. (0,0), (1,0)) and intermediate hop
   * positions (e.g. (1,2) in a 3×3 mesh with only 7 tiles).  XY
   * routing passes through every mesh cell along the path, so the
   * NoC graph view needs to show all of them; otherwise hop arrows
   * dangle into invisible vertices.  ``perTile`` entries get
   * data-tile so they remain hover-interactive for the tile
   * aggregate; intermediate-mesh positions get only a coords-data
   * attribute so they read as "graph waypoint, no tile here".
   * ``nodeR`` is the ring radius; callers scale it with ``cellPx`` so
   * the graph reads at any chip size. */
  const r = (nodeR != null) ? nodeR : 7;
  const tileByPos = new Map();
  perTile.forEach(t => tileByPos.set(`${t.mesh_x},${t.mesh_y}`, t));
  const pad = (cellPx + 1) * tileW;
  const padH = (cellPx + 1) * tileH;
  const centerAt = (mx, my) => ({
    x: 1 + (mx - meshOriginTx) * pad + pad / 2 - 0.5,
    y: 1 + (my - meshOriginTy) * padH + padH / 2 - 0.5,
  });
  const out = [];
  for (let my = 0; my < geom.height; my++) {
    for (let mx = 0; mx < geom.width; mx++) {
      const key = `${mx},${my}`;
      const tile = tileByPos.get(key);
      if (tile) {
        const c = tileCenters.get(tile.tile_index);
        if (!c) continue;
        out.push(
          `<circle class="sf-mesh-node" cx="${c.x.toFixed(2)}" cy="${c.y.toFixed(2)}" r="${r}" data-tile="${tile.tile_index}" data-mesh="${mx},${my}"/>`,
        );
      } else {
        /* Unallocated mesh position — same size + styling as tile
         * nodes (no dashed/dimmer look).  ``data-mesh`` lets the
         * hover handler look up pass-through packet counts; no
         * ``data-tile`` because there is no tile here. */
        const c = centerAt(mx, my);
        out.push(
          `<circle class="sf-mesh-node sf-mesh-node-empty" cx="${c.x.toFixed(2)}" cy="${c.y.toFixed(2)}" r="${r}" data-mesh="${mx},${my}"/>`,
        );
      }
    }
  }
  return out.join('\n');
}

function _renderOverlayNocRoutes(seg, tileCenters, opts) {
  const links = Array.isArray(seg.noc_links) ? seg.noc_links : [];
  /* Don't bail on empty links — still want idle dashed topology +
   * mesh-node rings so the chip's NoC graph stays visible even when
   * a segment carries zero tile-to-tile messages. */
  const perTile = opts.perTile || [];
  const geom = opts.geom || { width: 1, height: 1 };
  const meshOriginTx = opts.meshOriginTx || 0;
  const meshOriginTy = opts.meshOriginTy || 0;
  const cellPx = opts.cellPx || 16;
  const tileW = opts.tileW || 1;
  const tileH = opts.tileH || 1;
  const thickKey = opts.nocThicknessKey || 'packet_count';
  const maxVal = Math.max(...links.map(l => l[thickKey] || 0), 1);
  /* Graph metrics shared with the congestion overlay so the two
   * views are visually consistent (same ring size, lane positions,
   * arrow shape rule).  arrSize itself is computed per-edge below
   * as 2.5*w so the head-length / body-thickness ratio stays locked
   * and the head base = 2*w. */
  const {
    nodeR: curveNodeR,
    laneOffset: curveLaneOffset,
  } = _nocGraphMetrics(cellPx);
  const inboundByTile = new Map(), outboundByTile = new Map();
  links.forEach(L => {
    if (!inboundByTile.has(L.dst_tile)) inboundByTile.set(L.dst_tile, []);
    if (!outboundByTile.has(L.src_tile)) outboundByTile.set(L.src_tile, []);
    inboundByTile.get(L.dst_tile).push(L);
    outboundByTile.get(L.src_tile).push(L);
  });

  /* Pre-compute the two-direction offset for every (src, dst) pair so
   * A→B and B→A never collapse on top of each other.  The sign is
   * stable per unordered pair (anchored to ``min(src,dst)``); we no
   * longer use it for default *colour* (that comes only at hover-tile
   * time) — just for spatial separation. */
  const edgeDescs = links.map(L => {
    const a = tileCenters.get(L.src_tile);
    const b = tileCenters.get(L.dst_tile);
    if (!a || !b) return null;
    const dx = b.x - a.x, dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    /* Lane offset is the chord-perpendicular itself — A→B and B→A
     * naturally land on opposite sides because the chord direction
     * flips between them.  The old ``sign`` multiplier flipped along
     * with ``nx`` and cancelled out, so both directions ended up on
     * the same line. */
    const nx = -dy / len, ny = dx / len;
    const offset = Math.min(34, 12 + 14 * Math.sqrt((L[thickKey] || 0) / maxVal));
    const ox = nx * offset;
    const oy = ny * offset;
    const mx = (a.x + b.x) / 2 + ox;
    const my = (a.y + b.y) / 2 + oy;
    /* Stroke scales with cellPx and traffic so curves get thinner on
     * denser chips and thicker on sparse ones — same shrinking
     * behaviour as the mesh nodes + arrow heads. */
    const wBase = Math.max(0.4, cellPx * 0.06);
    const wScale = Math.max(0.8, cellPx * 0.24);
    const w = wBase + wScale * Math.sqrt((L[thickKey] || 0) / maxVal);
    /* Head length locked to body thickness so the arrow stays pointy
     * across all traffic levels: length 2.5*w, base 2*w → aspect 1.25. */
    const arrSize = w * 2.5;
    /* Pull endpoint back from destination tile centre so the bezier's
     * tip + the rendered arrow head land at the destination MESH-NODE
     * ring edge.
     *
     * Naive ``pullback = curveNodeR + arrSize + 1`` only works when
     * the bezier's tangent at its endpoint is aligned with the chord
     * (src→dst direction).  With a perpendicular control-point offset
     * the tangent rotates off-chord, so the arrow tip (which advances
     * along the *tangent*) lands SHORT of the ring — visible as a
     * gap between the arrow tip and the dest mesh node.
     *
     * Fix: iterate a Newton-like step on pullback until the arrow tip
     * lies on the ring (|tip − dst_center| ≈ curveNodeR).  Converges
     * in 2–3 iterations for realistic offsets; capped at 8 with a
     * 0.1px tolerance.  ``pullback`` is also clamped so the bezier
     * endpoint never crosses past the chord midpoint. */
    let pullback = curveNodeR + arrSize + 1;
    let bx = 0, by = 0, c2x = 0, c2y = 0;
    const maxPullback = Math.max(curveNodeR + arrSize, len * 0.5);
    for (let iter = 0; iter < 8; iter++) {
      pullback = Math.max(2, Math.min(pullback, maxPullback));
      bx = b.x - (dx / len) * pullback;
      by = b.y - (dy / len) * pullback;
      c2x = bx + (mx - bx) * 0.5;
      c2y = by + (my - by) * 0.5;
      const tdx = bx - c2x, tdy = by - c2y;
      const tlen = Math.hypot(tdx, tdy);
      if (tlen < 1e-6) break;
      const tipx = bx + (tdx / tlen) * arrSize;
      const tipy = by + (tdy / tlen) * arrSize;
      const tipDist = Math.hypot(tipx - b.x, tipy - b.y);
      const delta = tipDist - curveNodeR;
      if (Math.abs(delta) < 0.1) break;
      /* Increasing pullback retreats the endpoint along the chord →
       * tip retreats with it → tipDist grows.  So to drive tipDist
       * toward curveNodeR we step pullback by ``-delta``. */
      pullback -= delta;
    }
    const c1x = a.x + (mx - a.x) * 0.5;
    const c1y = a.y + (my - a.y) * 0.5;
    return {
      L, a, b: { x: bx, y: by }, mx, my, c1x, c1y, c2x, c2y, w, arrSize,
    };
  }).filter(Boolean);

  /* Default class ``sf-edge`` only — no per-direction colour at
   * render time.  Hover handlers below add ``sf-edge-in`` /
   * ``sf-edge-out`` against the *currently-hovered tile*. */
  const pathEls = edgeDescs.map((e, i) =>
    `<path class="sf-edge" d="M${e.a.x.toFixed(2)},${e.a.y.toFixed(2)} ` +
    `C${e.c1x.toFixed(2)},${e.c1y.toFixed(2)} ` +
    `${e.c2x.toFixed(2)},${e.c2y.toFixed(2)} ` +
    `${e.b.x.toFixed(2)},${e.b.y.toFixed(2)}" fill="none" ` +
    `stroke-width="${e.w.toFixed(2)}" opacity="0.82" ` +
    `data-edge-idx="${i}" data-src="${e.L.src_tile}" data-dst="${e.L.dst_tile}" />`,
  );
  const sourceR = Math.max(1.5, cellPx * 0.16);
  const sourceEls = edgeDescs.map((e, i) =>
    `<circle class="sf-noc-source" cx="${e.a.x.toFixed(2)}" cy="${e.a.y.toFixed(2)}" ` +
    `r="${sourceR.toFixed(2)}" stroke="rgba(0,0,0,0.55)" stroke-width="0.7" ` +
    `data-edge-idx="${i}" data-src="${e.L.src_tile}" data-dst="${e.L.dst_tile}"/>`,
  );
  const arrowEls = edgeDescs.map((e, i) => {
    const p0 = e.a;
    const p1 = { x: e.c1x, y: e.c1y };
    const p2 = { x: e.c2x, y: e.c2y };
    const p3 = e.b;
    const tip = _bezierPointAndTangent(p0, p1, p2, p3, 1.0);
    /* Head base = 2*w (twice body stroke), length = 2.5*w → pointy
     * aspect locked at 1.25, independent of cellPx. */
    const halfW = e.w;
    const px = -tip.uy * halfW, py = tip.ux * halfW;
    const tipx = tip.x + tip.ux * e.arrSize, tipy = tip.y + tip.uy * e.arrSize;
    return `<polygon class="sf-arrow" ` +
      `points="${tipx},${tipy} ${tip.x + px},${tip.y + py} ${tip.x - px},${tip.y - py}" ` +
      `data-edge-idx="${i}" data-src="${e.L.src_tile}" data-dst="${e.L.dst_tile}"/>`;
  });
  /* Group each edge's bezier-body path + arrow-head polygon so the
   * pair gets ONE drop-shadow on its composite silhouette — no halo
   * leaking from body onto head or vice versa.  Body is drawn AFTER
   * head inside the group so the curve sits on top of the polygon's
   * base, which is the visually-natural overlap (the head's wings
   * stick out past the curve's edges only at the tip). */
  const groupEls = edgeDescs.map((_, i) =>
    `<g class="sf-arrow-group">${arrowEls[i]}${pathEls[i]}</g>`,
  );
  /* Lane-positioned dashed underlay — drawn FIRST so the bezier
   * curves + tile-pair arrows + mesh nodes paint over the top. */
  const idleEls = _renderAllIdleLanes(
    geom, meshOriginTx, meshOriginTy, cellPx, tileW, tileH, curveNodeR, curveLaneOffset,
  );
  /* Mesh-node rings rendered last so they sit on top of the
   * curve-end stubs.  Same nodes as the congestion overlay so the
   * graph-structure reading is consistent across overlays.
   *
   * Final z-order:
   *   idle dashed → arrow-groups (body+head silhouette) →
   *   source dots → mesh-node rings. */
  const nodeEls = _renderMeshNodes(
    perTile, tileCenters, geom, meshOriginTx, meshOriginTy, cellPx, tileW, tileH, curveNodeR,
  );
  return {
    svg: idleEls + '\n' +
         groupEls.join('\n') + '\n' +
         sourceEls.join('\n') + '\n' +
         nodeEls,
    state: { kind: 'noc_routes', edgeDescs, inboundByTile, outboundByTile, thickKey },
  };
}

/* Scale parameters for the NoC graph view, derived from the mesh cell
 * size.  Centralised so noc_routes and noc_congestion stay visually
 * consistent at any chip size.
 *
 *   nodeR        — mesh-node ring radius
 *   laneOffset   — perpendicular distance from chord centreline to a
 *                  directional lane (so the two opposite-direction
 *                  arrows on the same mesh edge sit on opposite sides)
 *   surfaceBack  — chord-along distance from a dest node's centre to
 *                  the point on its ring at perpendicular distance
 *                  ``laneOffset``.  Tips land exactly on the ring.
 *   swBase       — base arrow stroke width (scales with traffic on top)
 *   arrSizeBase  — base arrow-head size (scales with traffic on top) */
function _nocGraphMetrics(cellPx) {
  /* Indexed purely to cellPx (the per-core panel allotment) so denser
   * chips draw thinner arrows + smaller nodes: as the user adds more
   * cores into the same panel real estate, cellPx shrinks → graph
   * elements shrink alongside it.  Floors are tiny so the geometry
   * doesn't collapse to zero at extreme aspect ratios but never
   * dominate the scaling at normal sizes. */
  const nodeR = Math.max(2.5, cellPx * 0.40);
  const laneOffset = Math.max(1.5, cellPx * 0.20);
  const surfaceBack = Math.sqrt(Math.max(0, nodeR * nodeR - laneOffset * laneOffset));
  const swBase = Math.max(0.4, cellPx * 0.08);
  const arrSizeBase = Math.max(1.5, cellPx * 0.26);
  return { nodeR, laneOffset, surfaceBack, swBase, arrSizeBase };
}

/* For a single directed mesh hop (mx,my) → (mx+dx,my+dy), return the
 * lane-positioned (ax,ay,bx,by) tuple shared by every renderer:
 *   - tail at source-centre on the chord axis (so a horizontal arrow's
 *     tail x = source center_x, a vertical arrow's tail y = source
 *     center_y); offset perpendicular to that axis by laneOffset.
 *   - tip on the dest ring's surface in the same lane. */
function _laneEndpoints(meshCenter, mx, my, dx, dy, laneOffset, surfaceBack) {
  const a = meshCenter(mx, my);
  const b = meshCenter(mx + dx, my + dy);
  let offX = 0, offY = 0;
  if (dx > 0)      offY = -laneOffset;   // east  → top lane
  else if (dx < 0) offY =  laneOffset;   // west  → bottom lane
  else if (dy > 0) offX = -laneOffset;   // south → left lane
  else             offX =  laneOffset;   // north → right lane
  const ax = a.x + offX;
  const ay = a.y + offY;
  let bx, by;
  if (dx > 0)      { bx = b.x - surfaceBack; by = b.y + offY; }
  else if (dx < 0) { bx = b.x + surfaceBack; by = b.y + offY; }
  else if (dy > 0) { bx = b.x + offX;        by = b.y - surfaceBack; }
  else             { bx = b.x + offX;        by = b.y + surfaceBack; }
  return { ax, ay, bx, by, dx, dy };
}

/* Dashed-white lane lines for every directed mesh-adjacent pair in the
 * geom.  Used as a topology underlay in noc_routes (where the main
 * info is at the bezier / tile-pair level, so the lane info itself is
 * always "no live single-hop arrow in this view"). */
function _renderAllIdleLanes(geom, originTx, originTy, cellPx, tileW, tileH, nodeR, laneOffset) {
  const pad = (cellPx + 1) * tileW;
  const padH = (cellPx + 1) * tileH;
  const meshCenter = (mx, my) => ({
    x: 1 + (mx - originTx) * pad + pad / 2 - 0.5,
    y: 1 + (my - originTy) * padH + padH / 2 - 0.5,
  });
  const surfaceBack = Math.sqrt(Math.max(0, nodeR * nodeR - laneOffset * laneOffset));
  const dirs = [{dx:1,dy:0},{dx:-1,dy:0},{dx:0,dy:1},{dx:0,dy:-1}];
  const out = [];
  for (let my = 0; my < geom.height; my++) {
    for (let mx = 0; mx < geom.width; mx++) {
      dirs.forEach(d => {
        if (mx + d.dx < 0 || my + d.dy < 0) return;
        if (mx + d.dx >= geom.width || my + d.dy >= geom.height) return;
        const e = _laneEndpoints(meshCenter, mx, my, d.dx, d.dy, laneOffset, surfaceBack);
        out.push(
          `<line class="sf-mesh-edge-idle" ` +
          `x1="${e.ax.toFixed(2)}" y1="${e.ay.toFixed(2)}" ` +
          `x2="${e.bx.toFixed(2)}" y2="${e.by.toFixed(2)}" />`,
        );
      });
    }
  }
  return out.join('\n');
}

function _renderOverlayNocCongestion(seg, tileCenters, cellPx, tileW, tileH, originTx, originTy, perTile, geom) {
  const links = Array.isArray(seg.noc_link_load) ? seg.noc_link_load : [];
  /* Don't early-return on empty links: even with zero traffic we
   * still want the lane-positioned dashed topology + mesh-node rings
   * visible so the chip's NoC graph is always legible. */

  const pad = (cellPx + 1) * tileW;
  const padH = (cellPx + 1) * tileH;
  const meshCenter = (mx, my) => ({
    x: 1 + (mx - originTx) * pad + pad / 2 - 0.5,
    y: 1 + (my - originTy) * padH + padH / 2 - 0.5,
  });

  const { nodeR, laneOffset, surfaceBack, swBase } = _nocGraphMetrics(cellPx);

  /* Live directed-hop lookup keyed by ``fx,fy-tx,ty``.  Lets the lane
   * iterator decide per-direction whether to render a coloured live
   * arrow or a dashed idle line, all in the same lane position. */
  const liveByDir = new Map();
  links.forEach((L, i) => {
    if ((L.packet_count || 0) > 0) {
      liveByDir.set(`${L.from_x},${L.from_y}-${L.to_x},${L.to_y}`, { L, i });
    }
  });
  /* Normalise traffic over the live links only; idle lanes never feed
   * the colormap. */
  const liveCounts = Array.from(liveByDir.values()).map(e => e.L.packet_count);
  const minLoad = liveCounts.length ? Math.min(...liveCounts) : 0;
  const maxLoad = liveCounts.length ? Math.max(...liveCounts, minLoad + 1) : 1;
  const tRange = (maxLoad - minLoad) || 1;
  const normT = v => Math.max(0, Math.min(1, (v - minLoad) / tRange));

  /* Iterate over every directed mesh-adjacent pair.  Each (mx,my,dx,dy)
   * tuple is a lane: live → coloured silhouette path, idle → dashed
   * white line at the SAME lane position.
   *
   * The live arrow is one filled ``<path>`` tracing the union of the
   * rectangle body (width sw, length tail→base) and the triangle
   * head (base 2*sw, length 2.5*sw → pointy aspect locked at 1.25).
   * Single element → single drop-shadow on the actual outer outline,
   * so the body never casts a halo onto the head. */
  const dirs = [{dx:1,dy:0},{dx:-1,dy:0},{dx:0,dy:1},{dx:0,dy:-1}];
  const idleEls = [];
  const arrowEls = [];
  for (let my = 0; my < geom.height; my++) {
    for (let mx = 0; mx < geom.width; mx++) {
      dirs.forEach(d => {
        const nxm = mx + d.dx, nym = my + d.dy;
        if (nxm < 0 || nym < 0 || nxm >= geom.width || nym >= geom.height) return;
        const e = _laneEndpoints(meshCenter, mx, my, d.dx, d.dy, laneOffset, surfaceBack);
        const entry = liveByDir.get(`${mx},${my}-${nxm},${nym}`);
        if (!entry) {
          idleEls.push(
            `<line class="sf-mesh-edge-idle" ` +
            `x1="${e.ax.toFixed(2)}" y1="${e.ay.toFixed(2)}" ` +
            `x2="${e.bx.toFixed(2)}" y2="${e.by.toFixed(2)}" />`,
          );
          return;
        }
        const { L, i: idx } = entry;
        const t = normT(L.packet_count);
        const rgb = _sampleCmap('Traffic', t);
        const fill = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
        /* sw = body thickness; head base = 2*sw; head length = 2.5*sw.
         * sw itself scales with cellPx (denser chips → thinner) and
         * traffic intensity. */
        const sw = swBase + cellPx * 0.14 * Math.sqrt(t);
        const arrSize = sw * 2.5;
        const op = 0.82 + 0.18 * t;
        const ux = d.dx, uy = d.dy;            // unit chord (cardinal)
        const perpX = -uy, perpY = ux;
        const bhw = sw / 2;                    // body half-width
        const hhw = sw;                        // head half-width = 2x body-half
        const baseX = e.bx - ux * arrSize;
        const baseY = e.by - uy * arrSize;
        const fmt = v => v.toFixed(2);
        /* Silhouette vertices, clockwise starting at tail-right:
         *   tail-r → body-base-r → head-base-r → tip
         *          → head-base-l → body-base-l → tail-l → close */
        const dAttr =
          `M${fmt(e.ax + perpX*bhw)},${fmt(e.ay + perpY*bhw)} ` +
          `L${fmt(baseX + perpX*bhw)},${fmt(baseY + perpY*bhw)} ` +
          `L${fmt(baseX + perpX*hhw)},${fmt(baseY + perpY*hhw)} ` +
          `L${fmt(e.bx)},${fmt(e.by)} ` +
          `L${fmt(baseX - perpX*hhw)},${fmt(baseY - perpY*hhw)} ` +
          `L${fmt(baseX - perpX*bhw)},${fmt(baseY - perpY*bhw)} ` +
          `L${fmt(e.ax - perpX*bhw)},${fmt(e.ay - perpY*bhw)} Z`;
        arrowEls.push(
          `<path class="sf-noc-arrow" data-cong-idx="${idx}" ` +
          `d="${dAttr}" fill="${fill}" opacity="${op.toFixed(2)}" />`,
        );
      });
    }
  }
  /* Mesh nodes rendered LAST so the rings sit on top of the silhouette
   * stubs that emerge from inside them.  Final z-order:
   *   idle dashed → live arrow silhouettes → mesh-node rings. */
  const nodeEls = (perTile && geom)
    ? _renderMeshNodes(perTile, tileCenters, geom, originTx, originTy, cellPx, tileW, tileH, nodeR)
    : '';
  return {
    svg: idleEls.join('\n') + '\n' +
         arrowEls.join('\n') + '\n' +
         nodeEls,
    state: { kind: 'noc_congestion', links, perTile },
  };
}
function _renderOverlayConnectivity(seg, corePositions, cellPx) {
  const edges = Array.isArray(seg.connectivity) ? seg.connectivity : [];
  if (edges.length === 0) return { svg: '', state: null };
  const maxW = Math.max(...edges.map(e => e.weight_sum_abs), 1);
  const px = (col) => col * (cellPx + 1) + cellPx / 2 + 1;
  const py = (row) => row * (cellPx + 1) + cellPx / 2 + 1;
  const lineEls = edges.map((e, i) => {
    const src = corePositions.get(e.src_core);
    const dst = corePositions.get(e.dst_core);
    if (!src || !dst) return '';
    const t = e.weight_sum_abs / maxW;
    return `<line x1="${px(src.col).toFixed(2)}" y1="${py(src.row).toFixed(2)}"
         x2="${px(dst.col).toFixed(2)}" y2="${py(dst.row).toFixed(2)}"
         stroke="rgba(76,175,80,${(0.15 + 0.6 * t).toFixed(3)})"
         stroke-width="${(0.5 + 2 * t).toFixed(2)}"
         data-conn-idx="${i}" />`;
  }).join('\n');
  return { svg: lineEls, state: { kind: 'connectivity', edges } };
}

function _renderOverlayCriticalCores(seg, corePositions, cellPx) {
  const cpts = Array.isArray(seg.critical_cores) ? seg.critical_cores : [];
  if (cpts.length === 0) return { svg: '', state: null };
  const freq = new Map();
  cpts.forEach(p => freq.set(p.core_index, (freq.get(p.core_index) || 0) + 1));
  const maxFreq = Math.max(...freq.values(), 1);
  const px = (col) => col * (cellPx + 1) + cellPx / 2 + 1;
  const py = (row) => row * (cellPx + 1) + cellPx / 2 + 1;
  const rects = [];
  freq.forEach((n, coreIdx) => {
    const p = corePositions.get(coreIdx);
    if (!p) return;
    const sz = (cellPx + 1) * (0.5 + 0.8 * (n / maxFreq));
    rects.push(
      `<rect x="${(px(p.col) - sz/2).toFixed(2)}" y="${(py(p.row) - sz/2).toFixed(2)}"
         width="${sz.toFixed(2)}" height="${sz.toFixed(2)}"
         fill="none" stroke="#000" stroke-width="2" rx="2"
         data-crit-core="${coreIdx}" data-crit-freq="${n}"/>`,
    );
  });
  return { svg: rects.join('\n'), state: { kind: 'critical_cores', freq } };
}

/* ---------------------------------------------------------------- */
/* Floorplan renderer                                                 */
/* ---------------------------------------------------------------- */

function _renderFloorplan(host, seg, metric, opts) {
  const cellPx = opts.cellPx || 16;
  const overlay = opts.overlay || 'none';
  const onCellClick = opts.onCellClick || null;

  const perTile = Array.isArray(seg.per_tile) ? seg.per_tile : [];
  const perCore = Array.isArray(seg.per_core) ? seg.per_core : [];
  if (perTile.length === 0 || perCore.length === 0) {
    host.innerHTML = '<div class="empty-state">No tiles to render.</div>';
    return;
  }
  const { tileW, tileH } = _tileGrid(perTile);
  const coreById = new Map(perCore.map(c => [c.core_index, c]));
  const hcmDiff = new Map((seg.hcm_diff || []).map(d => [d.core_index, d]));
  const range = _metricRange(seg, metric);

  /* Bound the floorplan to the actually-used tiles instead of the
   * full mesh_width × mesh_height rectangle.  Avoids the empty space
   * on the right when the last mesh row is partially filled. */
  let tx_min = Infinity, ty_min = Infinity, tx_max = -Infinity, ty_max = -Infinity;
  perTile.forEach(tile => {
    const tx = tile.mesh_x >= 0 ? tile.mesh_x : 0;
    const ty = tile.mesh_y >= 0 ? tile.mesh_y : 0;
    if (tx < tx_min) tx_min = tx;
    if (ty < ty_min) ty_min = ty;
    if (tx > tx_max) tx_max = tx;
    if (ty > ty_max) ty_max = ty;
  });
  if (!Number.isFinite(tx_min)) { tx_min = 0; ty_min = 0; tx_max = 0; ty_max = 0; }
  const meshTilesW = tx_max - tx_min + 1;
  const meshTilesH = ty_max - ty_min + 1;
  const cellRows = meshTilesH * tileH;
  const cellCols = meshTilesW * tileW;
  const meshPx = {
    w: cellCols * cellPx + cellCols + 1,
    h: cellRows * cellPx + cellRows + 1,
  };

  /* Cell map keyed by (row, col) using tile origin shifted to (0, 0). */
  const corePositions = new Map();
  const cellAt = new Map();
  perTile.forEach(tile => {
    const tx = (tile.mesh_x >= 0 ? tile.mesh_x : 0) - tx_min;
    const ty = (tile.mesh_y >= 0 ? tile.mesh_y : 0) - ty_min;
    tile.cores.forEach((coreIdx, pos) => {
      const ix = pos % tileW, iy = Math.floor(pos / tileW);
      const col = tx * tileW + ix;
      const row = ty * tileH + iy;
      cellAt.set(`${row},${col}`, { coreIdx, tile });
      corePositions.set(coreIdx, { col, row });
    });
  });
  let gridHtml = `<div class="sf-mesh" style="grid-template-columns:repeat(${cellCols},${cellPx}px);grid-auto-rows:${cellPx}px;width:${meshPx.w}px">`;
  for (let row = 0; row < cellRows; row++) {
    for (let col = 0; col < cellCols; col++) {
      const slot = cellAt.get(`${row},${col}`);
      if (!slot) { gridHtml += `<div class="sf-cell empty"></div>`; continue; }
      const core = coreById.get(slot.coreIdx);
      const bg = core ? _cellColor(core, metric, range, hcmDiff) : 'transparent';
      gridHtml += `<div class="sf-cell" data-core="${slot.coreIdx}" data-tile="${slot.tile.tile_index}" style="background:${bg}"></div>`;
    }
  }
  gridHtml += `</div>`;

  /* Tile borders — positioned to occupy *exactly* the tile's cell
   * block; the visible ring is drawn outside the box via
   * ``box-shadow`` (see .sf-tile-border CSS) so it lands in the
   * 1px inter-cell gap without distorting any cell positions. */
  let tileBordersHtml = '';
  perTile.forEach(tile => {
    const tx = (tile.mesh_x >= 0 ? tile.mesh_x : 0) - tx_min;
    const ty = (tile.mesh_y >= 0 ? tile.mesh_y : 0) - ty_min;
    const left = 1 + tx * tileW * (cellPx + 1);
    const top  = 1 + ty * tileH * (cellPx + 1);
    const w = tileW * cellPx + (tileW - 1);
    const h = tileH * cellPx + (tileH - 1);
    tileBordersHtml += `<div class="sf-tile-border" data-tile="${tile.tile_index}"
       style="left:${left}px;top:${top}px;width:${w}px;height:${h}px"></div>`;
  });

  /* Tile-centre lookup (in floorplan-pixel space) for NoC overlays. */
  const tileCenters = new Map();
  perTile.forEach(tile => {
    const tx = (tile.mesh_x >= 0 ? tile.mesh_x : 0) - tx_min;
    const ty = (tile.mesh_y >= 0 ? tile.mesh_y : 0) - ty_min;
    const cx = 1 + tx * tileW * (cellPx + 1) + (tileW * (cellPx + 1)) / 2 - 0.5;
    const cy = 1 + ty * tileH * (cellPx + 1) + (tileH * (cellPx + 1)) / 2 - 0.5;
    tileCenters.set(tile.tile_index, { x: cx, y: cy });
  });

  /* Cross-overlay aggregates — computed once here so every hover
   * handler can read them regardless of which overlay is active.
   * The previous bug was building these inside _renderOverlayNocRoutes
   * only, so the noc_congestion overlay's tile / mesh-node hover
   * tooltips showed 0 inbound / 0 outbound for every node. */
  const _nocLinks = Array.isArray(seg.noc_links) ? seg.noc_links : [];
  const inboundByTile = new Map(), outboundByTile = new Map();
  _nocLinks.forEach(L => {
    if (!inboundByTile.has(L.dst_tile)) inboundByTile.set(L.dst_tile, []);
    if (!outboundByTile.has(L.src_tile)) outboundByTile.set(L.src_tile, []);
    inboundByTile.get(L.dst_tile).push(L);
    outboundByTile.get(L.src_tile).push(L);
  });
  /* Per-mesh-position pass-through traffic — built from the per-hop
   * load.  Lets non-tile mesh waypoints show how much traffic flows
   * THROUGH them, even though they have no tile aggregate to display. */
  const _nocHops = Array.isArray(seg.noc_link_load) ? seg.noc_link_load : [];
  const meshFlowByPos = new Map();
  const _ensureMeshPos = (x, y) => {
    const k = `${x},${y}`;
    if (!meshFlowByPos.has(k)) meshFlowByPos.set(k, { in: [], out: [] });
    return meshFlowByPos.get(k);
  };
  _nocHops.forEach(L => {
    if ((L.packet_count || 0) <= 0) return;
    _ensureMeshPos(L.from_x, L.from_y).out.push(L);
    _ensureMeshPos(L.to_x, L.to_y).in.push(L);
  });

  /* Dispatch overlay. */
  let overlaySvgInner = '';
  let overlayState = null;
  if (overlay === 'noc_routes') {
    const r = _renderOverlayNocRoutes(seg, tileCenters, {
      nocThicknessKey: opts.nocThicknessKey,
      perTile,
      geom: { width: meshTilesW, height: meshTilesH },
      meshOriginTx: tx_min, meshOriginTy: ty_min,
      cellPx, tileW, tileH,
    });
    overlaySvgInner = r.svg; overlayState = r.state;
  } else if (overlay === 'noc_congestion') {
    const r = _renderOverlayNocCongestion(
      seg, tileCenters, cellPx, tileW, tileH, tx_min, ty_min, perTile,
      { width: meshTilesW, height: meshTilesH },
    );
    overlaySvgInner = r.svg; overlayState = r.state;
  } else if (overlay === 'connectivity') {
    const r = _renderOverlayConnectivity(seg, corePositions, cellPx);
    overlaySvgInner = r.svg; overlayState = r.state;
  } else if (overlay === 'critical_cores') {
    const r = _renderOverlayCriticalCores(seg, corePositions, cellPx);
    overlaySvgInner = r.svg; overlayState = r.state;
  }

  const overlaySvg = overlaySvgInner
    ? `<svg class="sf-noc-overlay" width="${meshPx.w}" height="${meshPx.h}" viewBox="0 0 ${meshPx.w} ${meshPx.h}">${overlaySvgInner}</svg>`
    : '';

  /* The inner wrapper exists so abs-positioned tile borders + SVG
   * overlay use the mesh's own coordinate system (no parent
   * padding to fight).  Without this wrapper, contours sat at the
   * outer floorplan padding-box edge instead of at the cells. */
  host.innerHTML = `
    <div class="sf-floorplan">
      <div class="sf-floorplan-inner">
        ${gridHtml}
        ${tileBordersHtml}
        ${overlaySvg}
      </div>
    </div>`;

  /* --- Wire interactions ------------------------------------------------ */
  const fpRoot = host.querySelector('.sf-floorplan');
  /* Reset edge highlighting whenever the mouse leaves the floorplan. */
  function _clearEdgeHighlights() {
    fpRoot.querySelectorAll('.sf-noc-overlay path, .sf-noc-overlay polygon, .sf-noc-overlay circle')
      .forEach(el => {
        el.classList.remove(
          'sf-edge-in', 'sf-edge-out', 'sf-edge-dim',
          'sf-arrow-in', 'sf-arrow-out', 'sf-arrow-dim',
          'sf-noc-source-in', 'sf-noc-source-out', 'sf-noc-source-dim',
        );
      });
  }
  /* Apply tile-relative direction colouring to every edge / arrow /
   * source dot.  Edges where the hovered tile is the source become
   * "outgoing"; where it's the destination become "incoming"; the
   * rest dim to ~15% opacity.  Triggered on cell hover (and thus
   * effectively on hover-of-the-tile). */
  function _highlightByTile(tileIdx) {
    fpRoot.querySelectorAll('.sf-noc-overlay path[data-edge-idx]').forEach(p => {
      const src = Number(p.dataset.src), dst = Number(p.dataset.dst);
      p.classList.remove('sf-edge-in', 'sf-edge-out', 'sf-edge-dim');
      if (src === tileIdx)      p.classList.add('sf-edge-out');
      else if (dst === tileIdx) p.classList.add('sf-edge-in');
      else                      p.classList.add('sf-edge-dim');
    });
    fpRoot.querySelectorAll('.sf-noc-overlay polygon[data-edge-idx]').forEach(p => {
      const src = Number(p.dataset.src), dst = Number(p.dataset.dst);
      p.classList.remove('sf-arrow-in', 'sf-arrow-out', 'sf-arrow-dim');
      if (src === tileIdx)      p.classList.add('sf-arrow-out');
      else if (dst === tileIdx) p.classList.add('sf-arrow-in');
      else                      p.classList.add('sf-arrow-dim');
    });
    fpRoot.querySelectorAll('.sf-noc-overlay circle[data-edge-idx]').forEach(c => {
      const src = Number(c.dataset.src), dst = Number(c.dataset.dst);
      c.classList.remove('sf-noc-source-in', 'sf-noc-source-out', 'sf-noc-source-dim');
      if (src === tileIdx)      c.classList.add('sf-noc-source-out');
      else if (dst === tileIdx) c.classList.add('sf-noc-source-in');
      else                      c.classList.add('sf-noc-source-dim');
    });
  }

  fpRoot.addEventListener('mouseleave', () => {
    _hideTooltip();
    _clearEdgeHighlights();
  });

  /* Cell hover → core + tile-aggregate tooltip + edge recolour by
   * tile.  Tile borders are pointer-events:none so cells get all
   * mouseover events even though the borders sit on top visually. */
  fpRoot.querySelectorAll('.sf-cell[data-core]').forEach(cell => {
    const ci = Number(cell.dataset.core);
    const tileIdx = Number(cell.dataset.tile);
    const tile = perTile.find(t => t.tile_index === tileIdx);
    cell.addEventListener('mouseenter', () => {
      if (overlayState && overlayState.kind === 'noc_routes') {
        _highlightByTile(tileIdx);
      }
    });
    cell.addEventListener('mousemove', ev => {
      const core = coreById.get(ci);
      if (!core) return;
      /* Always populate inbound/outbound from the floorplan-level
       * aggregates — both noc_routes and noc_congestion overlays
       * (and even ``none``) get the same accurate per-tile counts. */
      const inb = inboundByTile.get(tileIdx) || [];
      const out = outboundByTile.get(tileIdx) || [];
      const coreSection = _coreTooltipHtml(core, hcmDiff);
      const tileSection = tile
        ? `<div class="sf-tooltip-section">` +
          `<div class="sf-tooltip-section-title" style="color:#7aa6d8">Tile ${tile.tile_index} @ (${tile.mesh_x}, ${tile.mesh_y})</div>` +
          _tileTooltipBody(tile, inb, out) +
          `</div>`
        : '';
      _showTooltip(coreSection + tileSection, ev.clientX, ev.clientY);
    });
    cell.addEventListener('mouseleave', () => {
      _hideTooltip();
      _clearEdgeHighlights();
    });
    if (onCellClick) {
      cell.addEventListener('click', () => {
        const core = coreById.get(ci);
        if (core) onCellClick(core);
      });
    }
  });

  /* Overlay-specific hover handlers. */
  if (overlayState && overlayState.kind === 'noc_routes') {
    fpRoot.querySelectorAll('.sf-noc-overlay path[data-edge-idx]').forEach(p => {
      p.addEventListener('mousemove', ev => {
        const idx = Number(p.dataset.edgeIdx);
        const L = overlayState.edgeDescs[idx]?.L;
        if (!L) return;
        _showTooltip(
          `<strong>tile ${L.src_tile} → ${L.dst_tile}</strong>` +
          `<div class="sf-tooltip-row"><span class="sf-tooltip-label">packets</span><span>${fmtInt(L.packet_count)}</span></div>` +
          `<div class="sf-tooltip-row"><span class="sf-tooltip-label">spikes</span><span>${fmtInt(L.spike_count)}</span></div>` +
          `<div class="sf-tooltip-row"><span class="sf-tooltip-label">total hops</span><span>${fmtInt(L.total_hops)}</span></div>` +
          `<div style="font-size:10px;color:var(--text-muted);margin-top:4px">thickness = ${esc(overlayState.thickKey)}</div>`,
          ev.clientX, ev.clientY,
        );
      });
      p.addEventListener('mouseleave', _hideTooltip);
    });
  } else if (overlayState && overlayState.kind === 'noc_congestion') {
    /* Congestion overlay edges are <line> elements connecting mesh
     * nodes; hover the line for that directed hop's stats. */
    fpRoot.querySelectorAll('.sf-noc-overlay path[data-cong-idx]').forEach(p => {
      p.addEventListener('mousemove', ev => {
        const idx = Number(p.dataset.congIdx);
        const L = overlayState.links[idx];
        if (!L) return;
        _showTooltip(
          `<strong>mesh edge (${L.from_x},${L.from_y}) → (${L.to_x},${L.to_y})</strong>` +
          `<div class="sf-tooltip-row"><span class="sf-tooltip-label">packets</span><span>${fmtInt(L.packet_count)}</span></div>` +
          `<div style="font-size:10px;color:var(--text-muted);margin-top:4px">XY-routed inter-tile traffic on this single mesh hop.  Lanes are offset perpendicular so A→B and B→A are visually distinct.</div>`,
          ev.clientX, ev.clientY,
        );
      });
      p.addEventListener('mouseleave', _hideTooltip);
    });
  }
  /* Mesh-node hover (rings at each tile centre, present on both
   * noc_routes and noc_congestion overlays).
   *
   * Tile-bearing nodes show the per-tile aggregate (energy / cores /
   * inbound + outbound tile-pair traffic) using the lifted
   * inboundByTile / outboundByTile maps — accurate regardless of
   * which overlay is active.
   *
   * Non-tile mesh waypoints have no tile to summarise; instead we
   * show the PASS-THROUGH packet traffic, computed from the per-hop
   * ``meshFlowByPos``.  That way the user can hover any ring and
   * always learn what's going through that NoC position. */
  if (overlayState && (overlayState.kind === 'noc_routes' || overlayState.kind === 'noc_congestion')) {
    fpRoot.querySelectorAll('.sf-noc-overlay circle.sf-mesh-node').forEach(node => {
      const hasTile = node.dataset.tile !== undefined && node.dataset.tile !== '';
      const tileIdx = hasTile ? Number(node.dataset.tile) : null;
      const tile = hasTile ? perTile.find(t => t.tile_index === tileIdx) : null;
      const meshKey = node.dataset.mesh || '';
      node.addEventListener('mouseenter', () => {
        if (hasTile && overlayState.kind === 'noc_routes') _highlightByTile(tileIdx);
      });
      node.addEventListener('mousemove', ev => {
        if (tile) {
          const inb = inboundByTile.get(tileIdx) || [];
          const out = outboundByTile.get(tileIdx) || [];
          _showTooltip(
            `<strong>tile ${tile.tile_index} @ (${tile.mesh_x}, ${tile.mesh_y})</strong>` +
            _tileTooltipBody(tile, inb, out),
            ev.clientX, ev.clientY,
          );
          return;
        }
        /* Non-tile waypoint — show per-hop pass-through stats from
         * the position-keyed flow map. */
        const flow = meshFlowByPos.get(meshKey) || { in: [], out: [] };
        const inSum = flow.in.reduce((s, l) => s + (l.packet_count || 0), 0);
        const outSum = flow.out.reduce((s, l) => s + (l.packet_count || 0), 0);
        const fmtHop = (L) =>
          `(${L.from_x},${L.from_y}) → (${L.to_x},${L.to_y})  ${L.packet_count} pkt`;
        const inboundList = flow.in.slice().sort((a, b) => b.packet_count - a.packet_count)
          .slice(0, 6)
          .map(L => `<div class="sf-tooltip-list">${esc(fmtHop(L))}</div>`).join('');
        const outboundList = flow.out.slice().sort((a, b) => b.packet_count - a.packet_count)
          .slice(0, 6)
          .map(L => `<div class="sf-tooltip-list">${esc(fmtHop(L))}</div>`).join('');
        _showTooltip(
          `<strong>mesh waypoint @ (${esc(meshKey)})</strong>` +
          `<div style="font-size:10px;color:var(--text-muted);margin-top:2px">No tile mapped here — XY-routed packets still pass through.</div>` +
          `<div class="sf-tooltip-row"><span class="sf-tooltip-label">pkts in</span><span style="color:#ff4be8">${fmtInt(inSum)}</span></div>` +
          `<div class="sf-tooltip-row"><span class="sf-tooltip-label">pkts out</span><span style="color:#00ffff">${fmtInt(outSum)}</span></div>` +
          (inboundList ? `<div class="sf-tooltip-section-title" style="color:#ff4be8;margin-top:5px">incoming hops</div>${inboundList}` : '') +
          (outboundList ? `<div class="sf-tooltip-section-title" style="color:#00ffff;margin-top:5px">outgoing hops</div>${outboundList}` : ''),
          ev.clientX, ev.clientY,
        );
      });
      node.addEventListener('mouseleave', () => {
        _hideTooltip();
        _clearEdgeHighlights();
      });
    });
  } else if (overlayState && overlayState.kind === 'connectivity') {
    fpRoot.querySelectorAll('.sf-noc-overlay line[data-conn-idx]').forEach(line => {
      line.addEventListener('mousemove', ev => {
        const idx = Number(line.dataset.connIdx);
        const e = overlayState.edges[idx];
        if (!e) return;
        _showTooltip(
          `<strong>core ${e.src_core} → ${e.dst_core}</strong>` +
          `<div class="sf-tooltip-row"><span class="sf-tooltip-label">|w| sum</span><span>${e.weight_sum_abs.toExponential(2)}</span></div>` +
          `<div class="sf-tooltip-row"><span class="sf-tooltip-label">fanin entries</span><span>${e.fan_count}</span></div>`,
          ev.clientX, ev.clientY,
        );
      });
      line.addEventListener('mouseleave', _hideTooltip);
    });
  } else if (overlayState && overlayState.kind === 'critical_cores') {
    fpRoot.querySelectorAll('.sf-noc-overlay rect[data-crit-core]').forEach(rect => {
      rect.addEventListener('mousemove', ev => {
        const ci = Number(rect.dataset.critCore);
        const n  = Number(rect.dataset.critFreq);
        _showTooltip(
          `<strong>critical core ${ci}</strong>` +
          `<div class="sf-tooltip-row"><span class="sf-tooltip-label">critical on</span><span>${n} cycle(s)</span></div>` +
          `<div style="font-size:10px;color:var(--text-muted);margin-top:4px">this core drove sim_time more often than any other</div>`,
          ev.clientX, ev.clientY,
        );
      });
      rect.addEventListener('mouseleave', _hideTooltip);
    });
  }
}

/* ---------------------------------------------------------------- */
/* Summary / decomposition headers                                   */
/* ---------------------------------------------------------------- */

function summaryCardsHtml(agg, archPreset) {
  return `
    <div class="grid-3" style="margin-bottom:20px">
      <div class="card"><div class="big-metric">
        <div class="value" style="color:#ff9800">${fmtMj(agg.total_energy_j ?? 0)}</div>
        <div class="label">Total Energy (mJ)</div>
      </div></div>
      <div class="card"><div class="big-metric">
        <div class="value" style="color:#5b8af5">${fmtSec(agg.max_sim_time_s ?? 0)}</div>
        <div class="label">Max Sim Time (s)</div>
      </div></div>
      <div class="card"><div class="big-metric">
        <div class="value" style="color:#4caf50">${fmtInt(agg.total_spikes ?? 0)}</div>
        <div class="label">Total Spikes</div>
      </div></div>
    </div>
    <div class="grid-3" style="margin-bottom:20px">
      <div class="card"><div class="big-metric">
        <div class="value" style="color:#9c27b0">${fmtInt(agg.total_packets ?? 0)}</div>
        <div class="label">NoC Packets</div>
      </div></div>
      <div class="card"><div class="big-metric">
        <div class="value" style="color:#00bcd4">${fmtInt(agg.sample_count ?? 0)}</div>
        <div class="label">Samples Run</div>
      </div></div>
      <div class="card"><div class="big-metric">
        <div class="value" style="color:#888;font-size:22px">${esc(archPreset || '')}</div>
        <div class="label">Arch Preset</div>
      </div></div>
    </div>`;
}

function energyBreakdownHtml() {
  return `
    <div class="card" style="margin-bottom:20px">
      <div class="card-header">Energy Decomposition (Joules, log scale)</div>
      <div class="card-body"><div id="sanafe-eb" style="min-height:280px"></div></div>
    </div>`;
}

function renderEnergyBreakdown(eb) {
  if (!eb) return;
  const components = ['synapse', 'dendrite', 'soma', 'network'];
  const colors = { synapse: '#5b8af5', dendrite: '#4caf50',
                   soma: '#ff9800', network: '#9c27b0' };
  const traces = components.map(c => ({
    x: [c.charAt(0).toUpperCase() + c.slice(1)],
    y: [Math.max(eb[c] ?? 0, 1e-18)],
    name: c,
    type: 'bar',
    marker: { color: colors[c] },
  }));
  safeReact('sanafe-eb', traces, {
    height: 280, barmode: 'group',
    yaxis: { title: 'Energy (J)', type: 'log' },
    showlegend: false,
  });
}

/* ---------------------------------------------------------------- */
/* Plotly per-cycle / per-core charts (unchanged)                    */
/* ---------------------------------------------------------------- */

function renderCascadeTimeline(elId, seg) {
  const cascade = Array.isArray(seg.cascade) ? seg.cascade : [];
  if (cascade.length === 0) return;
  const depths = [...new Set(cascade.map(p => p.depth))].sort((a, b) => a - b);
  const cycles = [...new Set(cascade.map(p => p.cycle))].sort((a, b) => a - b);
  const traces = depths.map(d => {
    const yByCycle = new Map(cascade.filter(p => p.depth === d).map(p => [p.cycle, p.firings]));
    return {
      x: cycles, y: cycles.map(c => yByCycle.get(c) || 0),
      name: `depth ${d}`, type: 'bar',
    };
  });
  safeReact(elId, traces, {
    height: 240, barmode: 'stack',
    xaxis: { title: 'Cycle' }, yaxis: { title: 'Firings' },
    margin: { l: 50, r: 20, t: 10, b: 40 },
    legend: { orientation: 'h' },
  });
}

function renderEnergyWaterfall(elId, seg) {
  const series = Array.isArray(seg.cycle_energy) ? seg.cycle_energy : [];
  if (series.length === 0) return;
  const cycles = series.map(p => p.cycle);
  const cum = { synapse: [], dendrite: [], soma: [], network: [] };
  const running = { synapse: 0, dendrite: 0, soma: 0, network: 0 };
  series.forEach(p => {
    running.synapse += p.synapse_j;
    running.dendrite += p.dendrite_j;
    running.soma += p.soma_j;
    running.network += p.network_j;
    cum.synapse.push(running.synapse);
    cum.dendrite.push(running.dendrite);
    cum.soma.push(running.soma);
    cum.network.push(running.network);
  });
  const colors = { synapse: '#5b8af5', dendrite: '#4caf50',
                   soma: '#ff9800', network: '#9c27b0' };
  const traces = ['synapse', 'dendrite', 'soma', 'network'].map(k => ({
    x: cycles, y: cum[k], name: k.charAt(0).toUpperCase() + k.slice(1),
    type: 'scatter', mode: 'lines', stackgroup: 'one',
    line: { width: 0.5, color: colors[k] }, fillcolor: colors[k],
  }));
  safeReact(elId, traces, {
    height: 260,
    xaxis: { title: 'Cycle' }, yaxis: { title: 'Cumulative Energy (J)' },
    margin: { l: 60, r: 20, t: 10, b: 40 },
    legend: { orientation: 'h' },
  });
}

function renderEfficiencyScatter(elId, seg) {
  const cores = Array.isArray(seg.per_core) ? seg.per_core : [];
  if (cores.length === 0) return;
  const xs = cores.map(c => c.spikes_fired);
  const ys = cores.map(c => c.energy_j);
  const text = cores.map(c =>
    `core ${c.core_index}<br>energy: ${fmtMj(c.energy_j)} mJ<br>spikes: ${fmtInt(c.spikes_fired)}<br>latency: ${c.core_latency}`,
  );
  safeReact(elId, [{
    x: xs, y: ys, mode: 'markers', type: 'scatter',
    marker: {
      size: 8,
      color: cores.map(c => c.core_latency),
      colorscale: 'Portland', showscale: true,
      colorbar: { title: 'Latency Layer', titleside: 'right' },
    },
    text, hoverinfo: 'text',
  }], {
    height: 280,
    xaxis: { title: 'Spikes Fired' },
    yaxis: { title: 'Energy (J)', type: 'log' },
    margin: { l: 70, r: 60, t: 10, b: 40 },
  });
}

/* NoC playback — always shows a low-opacity chip underlay so it can
 * never be visually empty.  Replaced with a static "no NoC traffic"
 * notice when ``noc_traffic_per_cycle`` carries zero packets across
 * every cycle (e.g. single-tile arch dumps). */
function renderNocAnimation(elId, seg) {
  const traffic = Array.isArray(seg.noc_traffic_per_cycle) ? seg.noc_traffic_per_cycle : [];
  const totalPackets = traffic.reduce((s, c) => s + c.length, 0);
  const el = document.getElementById(elId);
  if (!el || !window.Plotly) return;
  if (traffic.length === 0 || totalPackets === 0) {
    el.innerHTML =
      '<div class="empty-state" style="padding:30px;font-size:13px">No NoC traffic recorded for this segment.<br>' +
      '<span style="font-size:11px;color:var(--text-muted)">Either ' +
      '<code>log_message_trace=False</code> or the architecture is single-tile ' +
      '(no inter-tile messages cross the NoC). Re-run with <code>cores_per_tile</code> ' +
      'auto-defaulted to get a 2D mesh.</span></div>';
    return;
  }
  const perTile = Array.isArray(seg.per_tile) ? seg.per_tile : [];
  const { tileW, tileH } = _tileGrid(perTile);

  /* Bound to actually-used tiles, same as the static floorplan. */
  let tx_min = Infinity, ty_min = Infinity, tx_max = -Infinity, ty_max = -Infinity;
  perTile.forEach(tile => {
    const tx = tile.mesh_x >= 0 ? tile.mesh_x : 0;
    const ty = tile.mesh_y >= 0 ? tile.mesh_y : 0;
    if (tx < tx_min) tx_min = tx; if (ty < ty_min) ty_min = ty;
    if (tx > tx_max) tx_max = tx; if (ty > ty_max) ty_max = ty;
  });
  if (!Number.isFinite(tx_min)) { tx_min = 0; ty_min = 0; tx_max = 0; ty_max = 0; }
  const meshTilesW = tx_max - tx_min + 1;
  const meshTilesH = ty_max - ty_min + 1;
  const cellRows = meshTilesH * tileH;
  const cellCols = meshTilesW * tileW;

  const tileCenter = (mx, my) => [
    ((mx - tx_min) + 0.5) * tileW - 0.5,
    ((my - ty_min) + 0.5) * tileH - 0.5,
  ];

  /* Per-core energy heatmap underlay so the chip is always visible.
   * Explicit x/y arrays so Plotly always treats z[i][j] as the value
   * at integer (j, i); without them Plotly autostretches by axis
   * range and the contour rectangles end up half a cell off. */
  const coreById = new Map((seg.per_core || []).map(c => [c.core_index, c]));
  const underlayZ = Array.from({ length: cellRows }, () => new Array(cellCols).fill(null));
  perTile.forEach(tile => {
    const tx = (tile.mesh_x >= 0 ? tile.mesh_x : 0) - tx_min;
    const ty = (tile.mesh_y >= 0 ? tile.mesh_y : 0) - ty_min;
    tile.cores.forEach((coreIdx, pos) => {
      const ix = pos % tileW, iy = Math.floor(pos / tileW);
      const col = tx * tileW + ix, row = ty * tileH + iy;
      const c = coreById.get(coreIdx);
      if (!c) return;
      const v = c.energy_j ?? 0;
      underlayZ[row][col] = v > 0 ? Math.log10(v) : null;
    });
  });
  const underlay = {
    z: underlayZ, type: 'heatmap',
    x: Array.from({ length: cellCols }, (_, i) => i),
    y: Array.from({ length: cellRows }, (_, i) => i),
    colorscale: 'YlOrRd', showscale: false,
    opacity: 0.45, hoverinfo: 'skip',
  };

  /* Tile contours — drawn as Plotly ``shapes`` over the heatmap. */
  const tileShapes = perTile.map(tile => {
    const tx = (tile.mesh_x >= 0 ? tile.mesh_x : 0) - tx_min;
    const ty = (tile.mesh_y >= 0 ? tile.mesh_y : 0) - ty_min;
    return {
      type: 'rect',
      x0: tx * tileW - 0.5,
      x1: (tx + 1) * tileW - 0.5,
      y0: ty * tileH - 0.5,
      y1: (ty + 1) * tileH - 0.5,
      line: { color: 'rgba(122,192,255,0.55)', width: 1.4 },
      fillcolor: 'rgba(0,0,0,0)',
      layer: 'above',
    };
  });

  function frameForCycle(cycleIdx) {
    const cycle = traffic[cycleIdx] || [];
    const xs = [], ys = [], hover = [];
    cycle.forEach(([sx, sy, dx, dy, n]) => {
      const [fx, fy] = tileCenter(sx, sy);
      const [tx, ty] = tileCenter(dx, dy);
      const label = `tile (${sx},${sy}) → (${dx},${dy}) · ${n} pkt`;
      xs.push(fx, tx, null);
      ys.push(fy, ty, null);
      hover.push(label, label, '');
    });
    return {
      data: [
        underlay,
        {
          x: xs, y: ys,
          mode: 'lines+markers',
          line: { color: '#00ffff', width: 2 },
          marker: { size: 6, color: '#00ffff',
                    line: { color: 'rgba(0,0,0,0.5)', width: 0.6 } },
          hoverinfo: 'text', text: hover,
          showlegend: false,
        },
      ],
      name: String(cycleIdx),
    };
  }
  const frames = traffic.map((_, i) => frameForCycle(i));
  const initialIdx = Math.max(0, traffic.findIndex(c => c.length > 0));

  /* Reverse the Y axis so z[0] / mesh_y=0 lands at the TOP of the
   * plot — matches the static floorplan's CSS-grid layout where row 0
   * is also at the top.  Plotly's default Y axis goes up, which put
   * the chip upside-down. */
  try { Plotly.purge(el); } catch (_) { /* ignore */ }
  Plotly.newPlot(el, frames[initialIdx].data, {
    height: Math.max(240, cellRows * 14 + 30),
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(8,12,20,0.6)',
    xaxis: { range: [-0.5, cellCols - 0.5],
             showgrid: false, zeroline: false, showticklabels: false, fixedrange: true },
    yaxis: { range: [cellRows - 0.5, -0.5],
             showgrid: false, zeroline: false, showticklabels: false,
             scaleanchor: 'x', scaleratio: 1, fixedrange: true },
    margin: { l: 10, r: 10, t: 10, b: 10 },
    shapes: tileShapes,
  });
  Plotly.addFrames(el, frames);

  /* Custom HTML controls — single play/pause toggle + slider +
   * cycle-counter in one tight row, replacing Plotly's bulky
   * updatemenus + sliders. */
  const toggle = document.getElementById(`${elId}-toggle`);
  const slider = document.getElementById(`${elId}-slider`);
  const label  = document.getElementById(`${elId}-label`);
  if (toggle && slider && label) {
    slider.min = '0';
    slider.max = String(frames.length - 1);
    slider.step = '1';
    slider.value = String(initialIdx);
    label.textContent = `cycle ${initialIdx} / ${frames.length - 1}`;

    let cur = initialIdx;
    let playing = false;
    let intervalId = null;

    function jumpTo(idx) {
      cur = idx;
      slider.value = String(idx);
      label.textContent = `cycle ${idx} / ${frames.length - 1}`;
      Plotly.animate(el, [String(idx)], {
        mode: 'immediate',
        frame: { duration: 0, redraw: true },
        transition: { duration: 0 },
      });
    }
    function stopPlaying() {
      if (intervalId !== null) { clearInterval(intervalId); intervalId = null; }
      playing = false;
      toggle.textContent = '▶';
    }
    function startPlaying() {
      playing = true;
      toggle.textContent = '⏸';
      intervalId = window.setInterval(() => {
        const next = (cur + 1) % frames.length;
        jumpTo(next);
      }, 500);
    }
    toggle.onclick = () => { playing ? stopPlaying() : startPlaying(); };
    slider.oninput = () => {
      if (playing) stopPlaying();
      jumpTo(Number(slider.value) | 0);
    };
    /* Stop the timer if the user navigates away (the host div gets
     * re-rendered).  MutationObserver on the chart parent triggers
     * cleanup when our div is removed from the DOM. */
    const obs = new MutationObserver(() => {
      if (!document.body.contains(el)) { stopPlaying(); obs.disconnect(); }
    });
    obs.observe(document.body, { childList: true, subtree: true });
  }
}

function renderPerCoreBars(sampleIdx, seg) {
  const ceId = `sanafe-core-energy-${sampleIdx}-${seg.stage_index}`;
  const csId = `sanafe-core-spikes-${sampleIdx}-${seg.stage_index}`;
  const coreLabels = seg.per_core.map(c => `c${c.core_index}`);
  const coreEnergies = seg.per_core.map(c => c.energy_j);
  const coreSpikes = seg.per_core.map(c => c.spikes_fired);
  safeReact(ceId, [{
    x: coreLabels, y: coreEnergies, type: 'bar',
    marker: { color: '#ff9800' },
    hovertemplate: '%{x}<br>%{y:.3e} J<extra></extra>',
  }], {
    height: 240, yaxis: { title: 'Energy (J)' },
    margin: { l: 60, r: 20, t: 10, b: 40 },
  });
  safeReact(csId, [{
    x: coreLabels, y: coreSpikes, type: 'bar',
    marker: { color: '#4caf50' },
    hovertemplate: '%{x}<br>%{y} spikes<extra></extra>',
  }], {
    height: 240, yaxis: { title: 'Spikes' },
    margin: { l: 60, r: 20, t: 10, b: 40 },
  });
}

/* ---------------------------------------------------------------- */
/* Segment composition                                              */
/* ---------------------------------------------------------------- */

function metricSelectorHtml(id, options, current) {
  return `<select id="${id}" style="margin-left:6px">` +
    options.map(o =>
      `<option value="${o.key}"${o.key === current ? ' selected' : ''}>${esc(o.label)}</option>`,
    ).join('') +
    `</select>`;
}

function segmentHtml(sampleIdx, seg) {
  const si = sampleIdx, st = seg.stage_index;
  const cascadeId = `sanafe-cascade-${si}-${st}`;
  const waterId   = `sanafe-water-${si}-${st}`;
  const animId    = `sanafe-anim-${si}-${st}`;
  const scatterId = `sanafe-scatter-${si}-${st}`;
  /* Spike-raster popup is a global modal launched from cell clicks
   * (see ``renderRaster``); no per-segment id is needed here. */
  const hasCascade = Array.isArray(seg.cascade) && seg.cascade.length > 0;
  const hasWater = Array.isArray(seg.cycle_energy) && seg.cycle_energy.length > 0;

  /* Per-panel controls.  Sidebar layout: the selects and the inline
   * legend sit to the LEFT of the floorplan so the floorplan keeps
   * all its vertical real estate. */
  const panelHtml = (panelIdx, defaults) => {
    const mId = `sanafe-pn-metric-${si}-${st}-${panelIdx}`;
    const oId = `sanafe-pn-overlay-${si}-${st}-${panelIdx}`;
    const tId = `sanafe-pn-thick-${si}-${st}-${panelIdx}`;
    const fId = `sanafe-pn-floor-${si}-${st}-${panelIdx}`;
    const lId = `sanafe-pn-legend-${si}-${st}-${panelIdx}`;
    return `
      <div class="sf-panel">
        <div class="sf-panel-body">
          <div class="sf-panel-sidebar">
            <div class="sf-panel-title">// P${panelIdx + 1}</div>
            <div class="sf-control">
              <span class="sf-control-label">Colour</span>
              ${metricSelectorHtml(mId, CELL_METRICS, defaults.metric)}
            </div>
            <div class="sf-control">
              <span class="sf-control-label">Overlay</span>
              ${metricSelectorHtml(oId, OVERLAY_OPTIONS, defaults.overlay)}
            </div>
            <div class="sf-control sf-thick-wrap">
              <span class="sf-control-label">Edge Thickness</span>
              ${metricSelectorHtml(tId, NOC_THICKNESS_METRICS, defaults.thickness)}
            </div>
            <div id="${lId}"></div>
          </div>
          <div class="sf-panel-floor" id="${fId}"></div>
        </div>
      </div>`;
  };

  /* Help-chip — replaces explanatory paragraphs.  Shows the
   * ``data-help`` text in the shared sf-tooltip on hover. */
  const helpChip = (helpText) =>
    `<span class="sf-help" data-help="${esc(helpText)}">?</span>`;

  return `
    <div class="card" style="margin-bottom:16px">
      <div class="card-header">Segment ${st} — <span style="font-weight:normal">${esc(seg.stage_name)}</span></div>
      <div class="card-body">
        <div class="grid-3" style="gap:12px;margin-bottom:12px">
          <div class="big-metric"><div class="value" style="font-size:18px;color:#ff9800">${fmtMj(seg.energy_j)}</div><div class="label">Energy (mJ)</div></div>
          <div class="big-metric"><div class="value" style="font-size:18px;color:#00ffff">${fmtSec(seg.sim_time_s)}</div><div class="label">Sim Time (s)</div></div>
          <div class="big-metric"><div class="value" style="font-size:18px;color:#4caf50">${fmtInt(seg.spikes)}</div><div class="label">Spikes</div></div>
        </div>
        <div class="grid-3" style="gap:12px;margin-bottom:12px">
          <div class="big-metric"><div class="value" style="font-size:18px;color:#ff4be8">${fmtInt(seg.packets_sent)}</div><div class="label">NoC Packets</div></div>
          <div class="big-metric"><div class="value" style="font-size:18px;color:#00bcd4">${fmtInt(seg.neurons_fired)}</div><div class="label">Neurons Fired</div></div>
          <div class="big-metric"><div class="value" style="font-size:18px;color:#7aa6d8">${seg.per_tile.length}</div><div class="label">Tiles</div></div>
        </div>

        <div class="sf-section-title">
          Chip Floorplan
          ${helpChip(
            'Two independently-configurable views of the same chip. ' +
            'Each panel picks its own per-core colour metric and its own overlay. ' +
            'Hover a cell for per-core stats + tile aggregate + direction-coded NoC edges. ' +
            'Hover a curve / line for that edge\'s stats. ' +
            'NoC routes draw A→B and B→A on opposite sides of the chord, ' +
            'neutral colour by default, recoloured cyan = outgoing / magenta = incoming ' +
            'relative to the hovered tile.  Arrowheads at the destination end.',
          )}
        </div>
        <div class="grid-2" style="gap:14px;margin-bottom:14px">
          ${panelHtml(0, PANEL_DEFAULTS[0])}
          ${panelHtml(1, PANEL_DEFAULTS[1])}
        </div>
        <!-- spike raster shows up as a global modal on click (see renderRaster), no inline div needed -->


        <div class="grid-2" style="gap:12px;margin-bottom:14px">
          ${hasCascade ? `
          <div>
            <div class="sf-section-title">Latency-Cascade Timeline
              ${helpChip('Per-cycle firing count, stacked by core latency depth. Depth-0 are the input-pool cores, deeper layers fire later in the cascade.')}
            </div>
            <div id="${cascadeId}" style="min-height:240px"></div>
          </div>` : '<div></div>'}
          ${hasWater ? `
          <div>
            <div class="sf-section-title">Energy Waterfall
              ${helpChip('Cumulative joules per category (synapse / dendrite / soma / network), reconstructed per cycle from event counters × YAML per-event constants.')}
            </div>
            <div id="${waterId}" style="min-height:240px"></div>
          </div>` : '<div></div>'}
        </div>

        <div class="grid-2" style="gap:12px;margin-bottom:14px">
          <div>
            <div class="sf-section-title">NoC Playback
              ${helpChip('Per-cycle packet flow over the chip mesh. Press ▶ to play, drag the slider to scrub. Energy heatmap stays visible as an underlay so you always see the chip even on quiet cycles.')}
            </div>
            <div id="${animId}" style="min-height:240px"></div>
            <div class="sf-playback-controls">
              <button class="sf-playback-toggle" id="${animId}-toggle" type="button">▶</button>
              <input class="sf-playback-slider" id="${animId}-slider" type="range" min="0" max="0" value="0">
              <span class="sf-playback-cycle" id="${animId}-label">cycle 0 / 0</span>
            </div>
          </div>
          <div>
            <div class="sf-section-title">Efficiency Scatter
              ${helpChip('Each dot is one core: x = output firings, y = activity-derived energy (log). Latency-layer colour highlights which depth lights up which corner. High-energy / low-spike outliers are inefficiency hotspots.')}
            </div>
            <div id="${scatterId}" style="min-height:300px"></div>
          </div>
        </div>

        <div class="grid-2" style="gap:12px">
          <div>
            <div class="sf-section-title">Per-Core Energy
              ${helpChip('Activity-derived energy per core (J). Sums synapse × incoming spikes + dendrite/soma × n_neurons × T_eff + soma_spike_out × firings + axon_in/out × packets — same per-event constants SANA-FE uses.')}
            </div>
            <div id="sanafe-core-energy-${si}-${st}" style="min-height:240px"></div>
          </div>
          <div>
            <div class="sf-section-title">Per-Core Spikes
              ${helpChip('Total output firings per core.')}
            </div>
            <div id="sanafe-core-spikes-${si}-${st}" style="min-height:240px"></div>
          </div>
        </div>
      </div>
    </div>`;
}

function renderRaster(panelId, coreIdx, core) {
  if (!core || !core.spike_raster) return;
  const raster = core.spike_raster;
  /* Reuse a single global modal element regardless of which panel
   * fired the click.  Drop any previous instance so the click on a
   * different core re-renders cleanly with a fresh Plotly chart. */
  const prev = document.getElementById('sf-raster-modal');
  if (prev) prev.remove();
  const modal = document.createElement('div');
  modal.id = 'sf-raster-modal';
  modal.className = 'sf-raster-modal';
  modal.innerHTML =
    `<div class="sf-raster-modal-content">
      <div class="sf-raster-modal-header">
        <div>
          <span class="sf-raster-modal-title">Spike Raster · core ${coreIdx}</span>
          <span class="sf-raster-modal-sub">${raster.length} neurons × ${raster[0]?.length || 0} cycles</span>
        </div>
        <button class="sf-raster-modal-close" type="button" aria-label="Close raster" title="Close (Esc)">×</button>
      </div>
      <div id="${panelId}-chart" style="min-height:340px"></div>
    </div>`;
  document.body.appendChild(modal);
  const close = () => {
    if (modal.isConnected) modal.remove();
    document.removeEventListener('keydown', escHandler);
  };
  const escHandler = ev => { if (ev.key === 'Escape') close(); };
  document.addEventListener('keydown', escHandler);
  modal.addEventListener('click', ev => {
    /* Click on backdrop (not inside the content) closes. */
    if (ev.target === modal) close();
  });
  modal.querySelector('.sf-raster-modal-close').onclick = close;
  safeReact(`${panelId}-chart`, [{
    z: raster, type: 'heatmap',
    colorscale: [[0, '#101418'], [1, '#4caf50']],
    showscale: false, hoverinfo: 'x+y+z',
  }], {
    height: Math.max(180, raster.length * 4 + 60),
    xaxis: { title: 'Cycle' },
    yaxis: { title: 'Neuron', autorange: 'reversed' },
    margin: { l: 60, r: 20, t: 10, b: 40 },
  });
}

/* HTML for the per-panel inline legend.  Lives at the bottom of the
 * sidebar so the floorplan stays unblocked.  Compact (~80px tall)
 * gradient bar + min/max labels for the cell metric, plus a few
 * coloured-swatch rows describing the overlay-specific colours. */
function _buildLegendHtml(metric, range, overlay) {
  /* Numeric formatter for the legend min/max labels. */
  const fmt = v => {
    if (!Number.isFinite(v)) return '—';
    if (Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3)) return v.toExponential(1);
    return Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 });
  };
  let lo = range.lo, hi = range.hi;
  if (metric.log) { lo = Math.pow(10, lo); hi = Math.pow(10, hi); }
  let html = `
    <div class="sf-legend">
      <div class="sf-legend-title">${esc(metric.label)}</div>
      <div class="sf-legend-bar" style="background:${_cmapGradient(metric.cmap)}"></div>
      <div class="sf-legend-labels"><span>${fmt(lo)}</span><span>${fmt(hi)}</span></div>
    </div>`;
  /* Overlay legend block. */
  if (overlay === 'noc_routes') {
    html += `
    <div class="sf-legend">
      <div class="sf-legend-title">NoC Routes</div>
      <div class="sf-legend-row"><span class="sf-legend-swatch" style="background:#ff9800"></span>neutral (default)</div>
      <div class="sf-legend-row"><span class="sf-legend-swatch" style="background:#00ffff"></span>outgoing (hover)</div>
      <div class="sf-legend-row"><span class="sf-legend-swatch" style="background:#ff4be8"></span>incoming (hover)</div>
    </div>`;
  } else if (overlay === 'noc_congestion') {
    html += `
    <div class="sf-legend">
      <div class="sf-legend-title">Mesh-edge Packets</div>
      <div class="sf-legend-bar" style="background:${_cmapGradient('Traffic')}"></div>
      <div class="sf-legend-labels"><span>low</span><span>high</span></div>
    </div>`;
  } else if (overlay === 'connectivity') {
    html += `
    <div class="sf-legend">
      <div class="sf-legend-title">Live Connectivity</div>
      <div class="sf-legend-row"><span class="sf-legend-swatch" style="background:rgba(76,175,80,0.85)"></span>weight magnitude</div>
    </div>`;
  } else if (overlay === 'critical_cores') {
    html += `
    <div class="sf-legend">
      <div class="sf-legend-title">Critical Cores</div>
      <div class="sf-legend-row"><span class="sf-legend-swatch" style="background:none;border:1.5px solid #000;width:12px;height:12px;border-radius:2px"></span>critical-on-cycle frequency</div>
    </div>`;
  }
  return html;
}

/* Derive a cellPx that lets the floorplan FILL the available width
 * and height of its host container.  Falls back to 28 if we can't
 * measure (host detached from DOM, no per_tile data yet). */
function _optimalCellPx(seg, hostEl) {
  const perTile = Array.isArray(seg.per_tile) ? seg.per_tile : [];
  if (perTile.length === 0) return 28;
  const { tileW, tileH } = _tileGrid(perTile);
  let tx_min = Infinity, ty_min = Infinity, tx_max = -Infinity, ty_max = -Infinity;
  perTile.forEach(tile => {
    const tx = tile.mesh_x >= 0 ? tile.mesh_x : 0;
    const ty = tile.mesh_y >= 0 ? tile.mesh_y : 0;
    if (tx < tx_min) tx_min = tx; if (ty < ty_min) ty_min = ty;
    if (tx > tx_max) tx_max = tx; if (ty > ty_max) ty_max = ty;
  });
  if (!Number.isFinite(tx_min)) return 28;
  const cellRows = (ty_max - ty_min + 1) * tileH;
  const cellCols = (tx_max - tx_min + 1) * tileW;
  /* Available space: host container's clientWidth + a generous
   * vertical budget that's roughly 60% of the viewport height (so
   * the floorplan grows tall on big screens, stays modest on small
   * ones).  Subtract the floorplan's own 12px padding (×2) so the
   * cells actually fit inside the painted panel. */
  const availW = Math.max(120, (hostEl.clientWidth || 500) - 28);
  const availH = Math.max(220, Math.round(window.innerHeight * 0.6) - 28);
  /* Floorplan width = cellCols * (cellPx + 1) + 1 (mesh gaps + 1px
   * padding).  Solve for cellPx given availW:
   *   cellPx ≤ (availW - 1) / cellCols - 1   */
  const pxByW = Math.floor((availW - 1) / cellCols) - 1;
  const pxByH = Math.floor((availH - 1) / cellRows) - 1;
  const cellPx = Math.max(20, Math.min(64, Math.min(pxByW, pxByH)));
  return cellPx;
}

function wirePanel(panelIdx, sampleIdx, seg) {
  const si = sampleIdx, st = seg.stage_index;
  const mSel = document.getElementById(`sanafe-pn-metric-${si}-${st}-${panelIdx}`);
  const oSel = document.getElementById(`sanafe-pn-overlay-${si}-${st}-${panelIdx}`);
  const tSel = document.getElementById(`sanafe-pn-thick-${si}-${st}-${panelIdx}`);
  const host = document.getElementById(`sanafe-pn-floor-${si}-${st}-${panelIdx}`);
  const legendHost = document.getElementById(`sanafe-pn-legend-${si}-${st}-${panelIdx}`);
  if (!host) return;

  const refreshThickVisibility = () => {
    const isNocRoutes = oSel && oSel.value === 'noc_routes';
    if (tSel) {
      /* Hide the whole .sf-control row so the label disappears too. */
      const wrap = tSel.closest('.sf-thick-wrap');
      if (wrap) wrap.style.display = isNocRoutes ? '' : 'none';
    }
  };

  const draw = () => {
    const metric = CELL_METRICS.find(m => m.key === (mSel?.value || 'energy_j')) || CELL_METRICS[0];
    const overlay = oSel?.value || 'none';
    const thickness = tSel?.value || 'packet_count';
    refreshThickVisibility();
    const cellPx = _optimalCellPx(seg, host);
    _renderFloorplan(host, seg, metric, {
      cellPx,
      overlay,
      nocThicknessKey: thickness,
      onCellClick: core => renderRaster(`sanafe-raster-${si}-${st}`, core.core_index, core),
    });
    if (legendHost) {
      legendHost.innerHTML = _buildLegendHtml(metric, _metricRange(seg, metric), overlay);
    }
  };
  if (mSel) mSel.onchange = draw;
  if (oSel) oSel.onchange = draw;
  if (tSel) tSel.onchange = draw;
  draw();
  /* Re-fit on window resize so the floorplan tracks the viewport.
   * Debounced via animation frame; cleans itself up if the host
   * is removed from the DOM (sample switch). */
  let pendingFrame = null;
  const onResize = () => {
    if (!host.isConnected) {
      window.removeEventListener('resize', onResize);
      return;
    }
    if (pendingFrame) cancelAnimationFrame(pendingFrame);
    pendingFrame = requestAnimationFrame(() => { pendingFrame = null; draw(); });
  };
  window.addEventListener('resize', onResize, { passive: true });
}

function wireSegmentControls(sampleIdx, seg) {
  wirePanel(0, sampleIdx, seg);
  wirePanel(1, sampleIdx, seg);
  renderCascadeTimeline(`sanafe-cascade-${sampleIdx}-${seg.stage_index}`, seg);
  renderEnergyWaterfall(`sanafe-water-${sampleIdx}-${seg.stage_index}`, seg);
  renderNocAnimation(`sanafe-anim-${sampleIdx}-${seg.stage_index}`, seg);
  renderEfficiencyScatter(`sanafe-scatter-${sampleIdx}-${seg.stage_index}`, seg);
  renderPerCoreBars(sampleIdx, seg);
}

/* (?) help-chip hover handler — single delegated listener on the
 * tab container so we don't have to attach per-chip after each
 * re-render (sample switch, panel redraw, …).  Reads ``data-help``
 * off the target and shows it in the shared sf-tooltip. */
function _wireHelpChips(root) {
  root.addEventListener('mousemove', ev => {
    const chip = ev.target.closest && ev.target.closest('.sf-help');
    if (!chip) return;
    _showTooltip(
      `<strong style="font-size:11px">HELP</strong>` +
      `<div style="margin-top:4px;font-size:11px">${esc(chip.dataset.help || '')}</div>`,
      ev.clientX, ev.clientY,
    );
  });
  root.addEventListener('mouseout', ev => {
    if (ev.target && ev.target.classList && ev.target.classList.contains('sf-help')) {
      _hideTooltip();
    }
  });
}

export function renderSanafeTab(snap, container) {
  if (!snap) {
    container.innerHTML =
      '<div class="empty-state">No SANA-FE simulation data available.<br>' +
      '<span style="font-size:12px;color:var(--text-muted)">Enable ' +
      '<code>enable_sanafe_simulation</code> in deployment parameters.</span></div>';
    return;
  }

  const agg = snap.aggregate || {};
  const archPreset = snap.arch_preset || '';
  const perSample = Array.isArray(snap.per_sample) ? snap.per_sample : [];
  const eb = agg.energy_breakdown_j || null;

  let html = summaryCardsHtml(agg, archPreset);
  html += energyBreakdownHtml();

  if (perSample.length === 0) {
    html += '<div class="empty-state">No per-sample data.</div>';
    container.innerHTML = html;
    return;
  }

  if (perSample.length > 1) {
    html += '<div class="card" style="margin-bottom:16px"><div class="card-header">Sample</div><div class="card-body">';
    html += '<select id="sanafe-sample-select">';
    perSample.forEach((s, i) => {
      html += `<option value="${i}">Sample ${esc(String(s.sample_index))}</option>`;
    });
    html += '</select></div></div>';
  }

  const initial = perSample[0];
  html += `<div id="sanafe-segments">`;
  initial.segments.forEach(seg => { html += segmentHtml(0, seg); });
  html += `</div>`;

  container.innerHTML = html;
  renderEnergyBreakdown(eb);
  _wireHelpChips(container);
  initial.segments.forEach(seg => wireSegmentControls(0, seg));

  const select = document.getElementById('sanafe-sample-select');
  if (select) {
    select.onchange = () => {
      const idx = Number(select.value) || 0;
      const segContainer = document.getElementById('sanafe-segments');
      const entry = perSample[idx];
      if (!segContainer || !entry) return;
      segContainer.innerHTML = entry.segments.map(seg => segmentHtml(idx, seg)).join('');
      entry.segments.forEach(seg => wireSegmentControls(idx, seg));
    };
  }
}
