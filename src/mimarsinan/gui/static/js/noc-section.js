/* Hardware/NoC section: spike traffic on the deployed mesh over simulated
   time. Three instruments on one clock: a per-tile traffic heatmap with a
   time scrubber + playback, inter-node link traffic (mesh edges via XY
   routing), and a source→target flow inspector. Data: the SANA-FE step's
   snapshot (per-cycle records persisted by the runtime). */
import { esc, safeReact, emptyAnnotation } from './util.js';

const SANAFE_STEP_HINT = 'SANA-FE';

// ── Module state (one NoC view per page) ─────────────────────────────────
const view = {
  snap: null,          // sanafe_simulation snapshot
  stepName: null,
  sampleIdx: 0,
  segmentIdx: 0,
  cycle: 0,
  playing: null,       // interval handle
  overlay: 'flows',    // 'flows' | 'links'
  cumulative: false,
  selA: null,
  selB: null,
  loadedFor: null,     // cache key
};

export async function renderNocSection(state, apiUrl, fetchJSON) {
  const host = document.getElementById('noc-root');
  if (!host) return;

  const steps = (state.pipeline && state.pipeline.steps) || [];
  const sanafeStep = steps.find(s => s.name.includes(SANAFE_STEP_HINT));
  if (!sanafeStep) {
    host.innerHTML = '<div class="empty-state">This pipeline has no SANA-FE simulation step — no NoC records to show.</div>';
    return;
  }
  if (sanafeStep.status === 'pending' || sanafeStep.status === 'running') {
    host.innerHTML = `<div class="empty-state">SANA-FE simulation ${esc(sanafeStep.status)} — NoC records appear when it completes.</div>`;
    return;
  }

  const cacheKey = `${state.historicalRunId || 'live'}:${sanafeStep.name}`;
  if (view.loadedFor !== cacheKey) {
    host.innerHTML = '<div class="empty-state">Loading SANA-FE records…</div>';
    let detail;
    try {
      detail = await fetchJSON(apiUrl('/steps/' + encodeURIComponent(sanafeStep.name)));
    } catch (e) {
      host.innerHTML = '<div class="empty-state">Failed to load the SANA-FE step snapshot</div>';
      return;
    }
    const snap = detail && detail.snapshot && detail.snapshot.sanafe_simulation;
    if (!snap || !Array.isArray(snap.per_sample) || snap.per_sample.length === 0) {
      host.innerHTML = '<div class="empty-state">The SANA-FE step persisted no simulation records.</div>';
      return;
    }
    stopPlayback();
    view.snap = snap;
    view.stepName = sanafeStep.name;
    view.sampleIdx = 0;
    view.segmentIdx = 0;
    view.cycle = 0;
    view.selA = null;
    view.selB = null;
    view.loadedFor = cacheKey;
  }
  buildNocDom(host);
  renderAll();
}

// ── Data accessors ───────────────────────────────────────────────────────
function currentSegment() {
  const sample = view.snap.per_sample[view.sampleIdx];
  if (!sample) return null;
  const segments = sample.segments || [];
  return segments[view.segmentIdx] || null;
}

function segmentCycleCount(seg) {
  return Math.max(
    (seg.noc_traffic_per_cycle || []).length,
    (seg.tile_packets_per_cycle || []).length,
    (seg.noc_link_load_per_cycle || []).length,
  );
}

function tileGeometry(seg) {
  // tile_index -> {x, y}; prefer per_tile mesh coords, fall back to
  // arch_geometry.tiles_xy order.
  const map = new Map();
  for (const t of seg.per_tile || []) {
    map.set(t.tile_index, { x: t.mesh_x, y: t.mesh_y, tile: t });
  }
  const geom = seg.arch_geometry || null;
  if (map.size === 0 && geom) {
    (geom.tiles_xy || []).forEach(([x, y], i) => map.set(i, { x, y, tile: null }));
  }
  return { map, width: geom ? geom.width : 0, height: geom ? geom.height : 0 };
}

function tilePacketsAtCycle(seg, cycle) {
  const bins = (seg.tile_packets_per_cycle || [])[cycle];
  const out = new Map();
  if (bins) for (const [k, v] of Object.entries(bins)) out.set(Number(k), v);
  return out;
}

function tilePacketsCumulative(seg) {
  const out = new Map();
  for (const t of seg.per_tile || []) out.set(t.tile_index, t.packets_sent);
  return out;
}

function flowsAtCycle(seg, cycle) {
  return ((seg.noc_traffic_per_cycle || [])[cycle]) || [];
}

function flowsCumulative(seg) {
  return (seg.noc_links || []).map(L => [L.src_x, L.src_y, L.dst_x, L.dst_y, L.packet_count]);
}

function linksAtCycle(seg, cycle) {
  const perCycle = seg.noc_link_load_per_cycle || [];
  if (!perCycle.length) return null; // legacy record — per-cycle load absent
  return perCycle[cycle] || [];
}

function linksCumulative(seg) {
  return (seg.noc_link_load || []).map(L => [L.from_x, L.from_y, L.to_x, L.to_y, L.packet_count]);
}

// ── Color scale: quiet → cyan → amber → rose ─────────────────────────────
function heat(t) {
  const stops = [
    [0.0, [26, 34, 48]],
    [0.25, [21, 94, 117]],
    [0.55, [34, 211, 238]],
    [0.8, [245, 158, 11]],
    [1.0, [244, 63, 94]],
  ];
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) {
    if (x <= stops[i][0]) {
      const [t0, c0] = stops[i - 1];
      const [t1, c1] = stops[i];
      const f = (x - t0) / (t1 - t0 || 1);
      const c = c0.map((v, j) => Math.round(v + (c1[j] - v) * f));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return 'rgb(244,63,94)';
}

// ── DOM scaffold ─────────────────────────────────────────────────────────
function buildNocDom(host) {
  if (host.querySelector('#noc-mesh-svg')) return; // already mounted
  host.innerHTML = `
    <div class="noc-toolbar" id="noc-toolbar"></div>
    <div class="noc-layout">
      <div>
        <div class="card">
          <div class="card-header"><span>NoC traffic heatmap</span>
            <span class="note" id="noc-mesh-note"></span></div>
          <div class="card-body no-pad">
            <div class="noc-canvas-wrap"><svg id="noc-mesh-svg" style="display:block;width:100%"></svg></div>
            <div class="noc-legend" id="noc-legend"></div>
          </div>
        </div>
        <div class="card" style="margin-top:16px">
          <div class="card-header"><span>Traffic over simulated time</span>
            <span class="note">packets per timestep · click to scrub</span></div>
          <div class="card-body no-pad"><div id="noc-timeline" style="min-height:120px"></div></div>
        </div>
      </div>
      <div>
        <div class="card">
          <div class="card-header"><span>Flow inspector</span>
            <span class="note">click tile A, then tile B</span></div>
          <div class="card-body" id="noc-flow-inspector">
            <div class="empty-state">Pick a source tile on the mesh</div>
          </div>
        </div>
        <div class="card" style="margin-top:16px">
          <div class="card-header">Traffic taxonomy</div>
          <div class="card-body" id="noc-totals"></div>
        </div>
        <div class="card" style="margin-top:16px">
          <div class="card-header">Per-tile totals</div>
          <div class="card-body no-pad" id="noc-tile-table"></div>
        </div>
      </div>
    </div>`;
}

function renderAll() {
  renderToolbar();
  renderMesh();
  renderTimeline();
  renderFlowInspector();
  renderTotals();
  renderTileTable();
}

// ── Toolbar ──────────────────────────────────────────────────────────────
function renderToolbar() {
  const bar = document.getElementById('noc-toolbar');
  const seg = currentSegment();
  if (!bar || !seg) return;
  const sample = view.snap.per_sample[view.sampleIdx];
  const cycles = segmentCycleCount(seg);
  const sampleOpts = view.snap.per_sample.map((s, i) =>
    `<option value="${i}" ${i === view.sampleIdx ? 'selected' : ''}>sample ${esc(String(s.sample_index))}</option>`).join('');
  const segmentOpts = (sample.segments || []).map((s, i) =>
    `<option value="${i}" ${i === view.segmentIdx ? 'selected' : ''}>${esc(s.stage_name || ('segment ' + s.stage_index))}</option>`).join('');
  bar.innerHTML = `
    <span class="noc-toolbar-label">Scope</span>
    <select id="noc-sample-sel" class="btn-sm" style="cursor:pointer">${sampleOpts}</select>
    <select id="noc-segment-sel" class="btn-sm" style="cursor:pointer">${segmentOpts}</select>
    <span class="divider" style="width:1px;height:20px;margin:0 2px"></span>
    <span class="noc-toolbar-label">Overlay</span>
    <div class="seg-control" style="flex:none">
      <button type="button" class="seg-btn ${view.overlay === 'flows' ? 'active' : ''}" data-overlay="flows">src→dst flows</button>
      <button type="button" class="seg-btn ${view.overlay === 'links' ? 'active' : ''}" data-overlay="links">mesh links</button>
    </div>
    <button type="button" class="btn-sm ${view.cumulative ? 'primary' : ''}" id="noc-cumulative-btn"
      title="Show whole-segment totals instead of one timestep">Σ cumulative</button>
    <span class="divider" style="width:1px;height:20px;margin:0 2px"></span>
    <button type="button" class="noc-play-btn ${view.playing ? 'playing' : ''}" id="noc-play-btn"
      title="${view.playing ? 'Pause' : 'Play'} traffic over simulated time">${view.playing ? '❚❚' : '▶'}</button>
    <input type="range" class="noc-scrubber" id="noc-scrubber"
      min="0" max="${Math.max(0, cycles - 1)}" value="${Math.min(view.cycle, Math.max(0, cycles - 1))}"
      ${view.cumulative ? 'disabled' : ''}>
    <span class="noc-time-readout" id="noc-time-readout"></span>`;

  document.getElementById('noc-sample-sel').addEventListener('change', (e) => {
    view.sampleIdx = Number(e.target.value);
    view.segmentIdx = 0; view.cycle = 0; view.selA = view.selB = null;
    stopPlayback(); renderAll();
  });
  document.getElementById('noc-segment-sel').addEventListener('change', (e) => {
    view.segmentIdx = Number(e.target.value);
    view.cycle = 0; view.selA = view.selB = null;
    stopPlayback(); renderAll();
  });
  bar.querySelectorAll('[data-overlay]').forEach(btn => {
    btn.addEventListener('click', () => {
      view.overlay = btn.dataset.overlay;
      renderToolbar(); renderMesh();
    });
  });
  document.getElementById('noc-cumulative-btn').addEventListener('click', () => {
    view.cumulative = !view.cumulative;
    stopPlayback(); renderToolbar(); renderMesh(); renderTimeline();
  });
  document.getElementById('noc-play-btn').addEventListener('click', togglePlayback);
  document.getElementById('noc-scrubber').addEventListener('input', (e) => {
    view.cycle = Number(e.target.value);
    updateTimeReadout();
    renderMesh();
    renderTimelineCursor();
  });
  updateTimeReadout();
}

function updateTimeReadout() {
  const el = document.getElementById('noc-time-readout');
  const seg = currentSegment();
  if (!el || !seg) return;
  const cycles = segmentCycleCount(seg);
  el.textContent = view.cumulative
    ? `Σ ${cycles} timesteps`
    : `t = ${view.cycle} / ${Math.max(0, cycles - 1)}`;
}

function togglePlayback() {
  if (view.playing) { stopPlayback(); renderToolbar(); return; }
  const seg = currentSegment();
  if (!seg) return;
  const cycles = segmentCycleCount(seg);
  if (cycles <= 1 || view.cumulative) return;
  view.playing = setInterval(() => {
    const svg = document.getElementById('noc-mesh-svg');
    if (!svg || !document.body.contains(svg)) { stopPlayback(); return; }
    view.cycle = (view.cycle + 1) % cycles;
    const scrubber = document.getElementById('noc-scrubber');
    if (scrubber) scrubber.value = String(view.cycle);
    updateTimeReadout();
    renderMesh();
    renderTimelineCursor();
  }, 350);
  renderToolbar();
}

function stopPlayback() {
  if (view.playing) clearInterval(view.playing);
  view.playing = null;
}

// ── Mesh SVG ─────────────────────────────────────────────────────────────
const TILE = 84, GAP = 26, PAD = 24;

function tileCenter(x, y) {
  return [PAD + x * (TILE + GAP) + TILE / 2, PAD + y * (TILE + GAP) + TILE / 2];
}

function renderMesh() {
  const svg = document.getElementById('noc-mesh-svg');
  const seg = currentSegment();
  if (!svg || !seg) return;
  const { map, width, height } = tileGeometry(seg);
  if (map.size === 0) {
    svg.innerHTML = '';
    document.getElementById('noc-mesh-note').textContent = 'no tile geometry in this record';
    return;
  }
  const W = Math.max(width, 1) * (TILE + GAP) - GAP + 2 * PAD;
  const H = Math.max(height, 1) * (TILE + GAP) - GAP + 2 * PAD;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.style.maxHeight = '520px';

  const packets = view.cumulative
    ? tilePacketsCumulative(seg)
    : tilePacketsAtCycle(seg, view.cycle);
  const maxPackets = Math.max(1, ...packets.values());

  const parts = [];

  // Tiles.
  for (const [tileIdx, pos] of map) {
    const [cx, cy] = tileCenter(pos.x, pos.y);
    const n = packets.get(tileIdx) || 0;
    const fill = n > 0 ? heat(n / maxPackets) : 'var(--bg-input)';
    const isA = view.selA === tileIdx, isB = view.selB === tileIdx;
    const ring = isA ? '#22d3ee' : (isB ? '#f59e0b' : 'var(--border-strong)');
    const cores = pos.tile ? (pos.tile.cores || []).length : 0;
    // Bright heat fills need dark ink; quiet fills keep the light ink.
    const darkInk = n > 0 && n / maxPackets > 0.45;
    parts.push(`<g class="noc-tile" data-tile="${tileIdx}" style="cursor:pointer">
      <rect x="${cx - TILE / 2}" y="${cy - TILE / 2}" width="${TILE}" height="${TILE}" rx="8"
        fill="${fill}" stroke="${ring}" stroke-width="${isA || isB ? 2.5 : 1}"></rect>
      <text x="${cx}" y="${cy - 6}" text-anchor="middle" font-size="13" font-weight="700"
        fill="${darkInk ? '#0a0c10' : '#e2e8f0'}" style="pointer-events:none">T${tileIdx}</text>
      <text x="${cx}" y="${cy + 12}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace"
        fill="${darkInk ? '#0a0c10' : '#8494a7'}" style="pointer-events:none">${n} pkt</text>
      <title>tile ${tileIdx} @ (${pos.x},${pos.y}) — ${n} packets${view.cumulative ? ' (segment total)' : ` at t=${view.cycle}`} · ${cores} cores</title>
    </g>`);
  }

  // Overlay edges.
  const meshNote = document.getElementById('noc-mesh-note');
  if (view.overlay === 'flows') {
    const flows = view.cumulative ? flowsCumulative(seg) : flowsAtCycle(seg, view.cycle);
    parts.push(...flowEdges(flows));
    if (meshNote) meshNote.textContent = view.cumulative ? 'tile fill + flows: segment totals' : 'tile fill + flows at the scrubbed timestep';
  } else {
    let links = view.cumulative ? linksCumulative(seg) : linksAtCycle(seg, view.cycle);
    let note = view.cumulative ? 'mesh-edge load: segment totals (XY routing)' : 'mesh-edge load at the scrubbed timestep (XY routing)';
    if (links === null) {
      links = linksCumulative(seg);
      note = 'segment-total link load — this run predates per-cycle link records';
    }
    parts.push(...linkEdges(links));
    if (meshNote) meshNote.textContent = note;
  }

  // Route highlight for the flow inspector.
  if (view.selA != null && view.selB != null && view.selA !== view.selB) {
    const a = map.get(view.selA), b = map.get(view.selB);
    if (a && b) parts.push(routeHighlight(a, b));
  }

  svg.innerHTML = parts.join('');
  svg.querySelectorAll('.noc-tile').forEach(g => {
    g.addEventListener('click', () => onTileClick(Number(g.dataset.tile)));
  });
  renderLegend(maxPackets);
}

// Fixed-size arrowhead triangle at (x,y) pointing along (ux,uy).
function arrowHead(x, y, ux, uy, color) {
  const s = 7;
  const px = -uy, py = ux;
  return `<path d="M${x},${y} L${(x - ux * s * 1.6 + px * s * 0.7).toFixed(1)},${(y - uy * s * 1.6 + py * s * 0.7).toFixed(1)} L${(x - ux * s * 1.6 - px * s * 0.7).toFixed(1)},${(y - uy * s * 1.6 - py * s * 0.7).toFixed(1)} z" fill="${color}" style="pointer-events:none"/>`;
}

// Trim a center-to-center segment to the tile boundaries and offset it
// perpendicular so A→B and B→A render side by side.
function trimmedEdge(x1, y1, x2, y2, offset) {
  const len = Math.hypot(x2 - x1, y2 - y1) || 1;
  const ux = (x2 - x1) / len, uy = (y2 - y1) / len;
  const ox = -uy * offset, oy = ux * offset;
  const trim = TILE / 2 + 3;
  return {
    x1: x1 + ux * trim + ox, y1: y1 + uy * trim + oy,
    x2: x2 - ux * (trim + 6) + ox, y2: y2 - uy * (trim + 6) + oy,
    ux, uy,
  };
}

function flowEdges(quints) {
  const parts = [];
  if (!quints || !quints.length) return parts;
  const maxN = Math.max(1, ...quints.map(q => q[4]));
  for (const [sx, sy, dx, dy, n] of quints) {
    const [cx1, cy1] = tileCenter(sx, sy);
    const [cx2, cy2] = tileCenter(dx, dy);
    const e = trimmedEdge(cx1, cy1, cx2, cy2, 12);
    // Long flows arc over intermediate tiles instead of crossing them.
    const span = Math.hypot(cx2 - cx1, cy2 - cy1) / (TILE + GAP);
    const bow = 12 * (span - 1);
    const mx = (e.x1 + e.x2) / 2 - e.uy * bow;
    const my = (e.y1 + e.y2) / 2 + e.ux * bow;
    const w = 1.5 + 3.5 * (n / maxN);
    const color = heat(0.35 + 0.65 * (n / maxN));
    parts.push(`<path d="M${e.x1.toFixed(1)},${e.y1.toFixed(1)} Q${mx.toFixed(1)},${my.toFixed(1)} ${e.x2.toFixed(1)},${e.y2.toFixed(1)}"
      fill="none" stroke="${color}" stroke-width="${w.toFixed(1)}" stroke-linecap="round" opacity="0.85">
      <title>(${sx},${sy}) → (${dx},${dy}) — ${n} packets</title></path>`);
    parts.push(arrowHead(e.x2, e.y2, e.ux, e.uy, color));
  }
  return parts;
}

function linkEdges(quints) {
  const parts = [];
  if (!quints || !quints.length) return parts;
  const maxN = Math.max(1, ...quints.map(q => q[4]));
  for (const [fx, fy, tx, ty, n] of quints) {
    const [cx1, cy1] = tileCenter(fx, fy);
    const [cx2, cy2] = tileCenter(tx, ty);
    const e = trimmedEdge(cx1, cy1, cx2, cy2, 8);
    const w = 1.5 + 5 * (n / maxN);
    const color = heat(0.3 + 0.7 * (n / maxN));
    parts.push(`<line x1="${e.x1.toFixed(1)}" y1="${e.y1.toFixed(1)}" x2="${e.x2.toFixed(1)}" y2="${e.y2.toFixed(1)}"
      stroke="${color}" stroke-width="${w.toFixed(1)}" stroke-linecap="round" opacity="0.9">
      <title>edge (${fx},${fy}) → (${tx},${ty}) — ${n} packets</title></line>`);
    parts.push(arrowHead(e.x2, e.y2, e.ux, e.uy, color));
  }
  return parts;
}

function routeHighlight(a, b) {
  // XY route: x first, then y — mirrors the emission-side aggregation.
  const points = [];
  let cx = a.x, cy = a.y;
  points.push(tileCenter(cx, cy));
  const stepX = b.x > cx ? 1 : -1;
  while (cx !== b.x) { cx += stepX; points.push(tileCenter(cx, cy)); }
  const stepY = b.y > cy ? 1 : -1;
  while (cy !== b.y) { cy += stepY; points.push(tileCenter(cx, cy)); }
  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${p[1]}`).join(' ');
  return `<path d="${d}" fill="none" stroke="#22d3ee" stroke-width="2"
    stroke-dasharray="6 5" opacity="0.7" style="pointer-events:none"/>`;
}

function renderLegend(maxPackets) {
  const el = document.getElementById('noc-legend');
  if (!el) return;
  const stops = Array.from({ length: 11 }, (_, i) => heat(i / 10)).join(',');
  el.innerHTML = `<span>0</span>
    <span class="noc-legend-swatch" style="background:linear-gradient(90deg,${stops})"></span>
    <span>${maxPackets} pkt</span>
    <span style="margin-left:auto">edge thickness ∝ packets · arrows show direction</span>`;
}

// ── Timeline (packets per timestep, click to scrub) ──────────────────────
function renderTimeline() {
  const el = document.getElementById('noc-timeline');
  const seg = currentSegment();
  if (!el || !seg) return;
  const cycles = segmentCycleCount(seg);
  const perCycleTiles = seg.tile_packets_per_cycle || [];
  const totals = Array.from({ length: cycles }, (_, i) => {
    const bins = perCycleTiles[i];
    if (!bins) return 0;
    return Object.values(bins).reduce((s, v) => s + v, 0);
  });
  const interTile = Array.from({ length: cycles }, (_, i) =>
    flowsAtCycle(seg, i).reduce((s, q) => s + q[4], 0));
  const traces = cycles ? [
    { x: totals.map((_, i) => i), y: totals, type: 'bar', name: 'all packets', marker: { color: '#155e75' } },
    { x: interTile.map((_, i) => i), y: interTile, type: 'bar', name: 'inter-tile', marker: { color: '#22d3ee' } },
  ] : [];
  safeReact(el, traces, {
    height: 150,
    barmode: 'overlay',
    margin: { t: 8, r: 12, b: 28, l: 44 },
    xaxis: { title: '', dtick: Math.max(1, Math.ceil(cycles / 24)) },
    yaxis: { title: '' },
    showlegend: true,
    legend: { orientation: 'h', y: 1.15, font: { size: 10 } },
    annotations: cycles ? [] : emptyAnnotation('No per-cycle records'),
    shapes: cursorShape(),
  });
  el.on && el.removeAllListeners && el.removeAllListeners('plotly_click');
  if (el.on) {
    el.on('plotly_click', (data) => {
      const x = data.points && data.points[0] && data.points[0].x;
      if (typeof x === 'number' && !view.cumulative) {
        view.cycle = x;
        const scrubber = document.getElementById('noc-scrubber');
        if (scrubber) scrubber.value = String(x);
        updateTimeReadout();
        renderMesh();
        renderTimelineCursor();
      }
    });
  }
}

function cursorShape() {
  if (view.cumulative) return [];
  return [{
    type: 'line', xref: 'x', yref: 'paper',
    x0: view.cycle, x1: view.cycle, y0: 0, y1: 1,
    line: { color: '#f59e0b', width: 1.5, dash: 'dot' },
  }];
}

function renderTimelineCursor() {
  const el = document.getElementById('noc-timeline');
  if (!el || !el.layout) return;
  Plotly.relayout(el, { shapes: cursorShape() });
}

// ── Flow inspector ───────────────────────────────────────────────────────
function onTileClick(tileIdx) {
  if (view.selA == null || (view.selA != null && view.selB != null)) {
    view.selA = tileIdx;
    view.selB = null;
  } else if (tileIdx !== view.selA) {
    view.selB = tileIdx;
  } else {
    view.selA = null;
    view.selB = null;
  }
  renderMesh();
  renderFlowInspector();
}

function flowStats(seg, aIdx, bIdx) {
  const link = (seg.noc_links || []).find(L => L.src_tile === aIdx && L.dst_tile === bIdx);
  return link || { packet_count: 0, spike_count: 0, total_hops: 0 };
}

function flowSeries(seg, a, b) {
  const cycles = segmentCycleCount(seg);
  return Array.from({ length: cycles }, (_, i) =>
    flowsAtCycle(seg, i)
      .filter(q => q[0] === a.x && q[1] === a.y && q[2] === b.x && q[3] === b.y)
      .reduce((s, q) => s + q[4], 0));
}

function renderFlowInspector() {
  const host = document.getElementById('noc-flow-inspector');
  const seg = currentSegment();
  if (!host || !seg) return;
  const { map } = tileGeometry(seg);
  if (view.selA == null) {
    host.innerHTML = '<div class="empty-state">Pick a source tile on the mesh</div>';
    return;
  }
  const a = map.get(view.selA);
  if (view.selB == null) {
    const t = a && a.tile;
    host.innerHTML = `
      <div class="noc-flow-line"><span class="tk-chip info mono">A = T${view.selA}</span>
        <span class="note">now pick a target tile</span></div>
      ${t ? `<div class="divider"></div>
      <div class="noc-flow-line"><span>cores</span><span class="noc-flow-count">${(t.cores || []).length}</span></div>
      <div class="noc-flow-line"><span>packets sent (segment)</span><span class="noc-flow-count">${t.packets_sent}</span></div>
      <div class="noc-flow-line"><span>spikes fired</span><span class="noc-flow-count">${t.spikes_fired}</span></div>
      <div class="noc-flow-line"><span>energy</span><span class="noc-flow-count">${(t.energy_j * 1e6).toFixed(3)} µJ</span></div>` : ''}`;
    return;
  }
  const b = map.get(view.selB);
  const ab = flowStats(seg, view.selA, view.selB);
  const ba = flowStats(seg, view.selB, view.selA);
  host.innerHTML = `
    <div class="noc-flow-line">
      <span class="tk-chip info mono">T${view.selA}</span><span>→</span>
      <span class="tk-chip warn mono">T${view.selB}</span>
      <button type="button" class="btn-sm" id="noc-flow-clear" style="margin-left:auto">clear</button>
    </div>
    <div class="divider"></div>
    <div class="noc-flow-line"><span>A→B packets</span><span class="noc-flow-count">${ab.packet_count}</span></div>
    <div class="noc-flow-line"><span>A→B spikes · hops</span><span class="noc-flow-count">${ab.spike_count} · ${ab.total_hops}</span></div>
    <div class="noc-flow-line"><span>B→A packets</span><span class="noc-flow-count">${ba.packet_count}</span></div>
    <div class="noc-flow-line"><span>B→A spikes · hops</span><span class="noc-flow-count">${ba.spike_count} · ${ba.total_hops}</span></div>
    <div class="divider"></div>
    <div class="note" style="margin-bottom:4px">routed traffic over simulated time (XY route highlighted on the mesh)</div>
    <div id="noc-flow-series" style="min-height:110px"></div>`;
  document.getElementById('noc-flow-clear').addEventListener('click', () => {
    view.selA = view.selB = null;
    renderMesh(); renderFlowInspector();
  });
  if (a && b) {
    const sAB = flowSeries(seg, a, b);
    const sBA = flowSeries(seg, b, a);
    safeReact('noc-flow-series', [
      { x: sAB.map((_, i) => i), y: sAB, type: 'bar', name: 'A→B', marker: { color: '#22d3ee' } },
      { x: sBA.map((_, i) => i), y: sBA, type: 'bar', name: 'B→A', marker: { color: '#f59e0b' } },
    ], {
      height: 130, barmode: 'group',
      margin: { t: 4, r: 8, b: 24, l: 34 },
      showlegend: true, legend: { orientation: 'h', y: 1.25, font: { size: 9 } },
    });
  }
}

// ── Totals + per-tile table ──────────────────────────────────────────────
function renderTotals() {
  const host = document.getElementById('noc-totals');
  const seg = currentSegment();
  if (!host || !seg) return;
  const sample = view.snap.per_sample[view.sampleIdx];
  const rows = [
    ['inter-tile packets', seg.inter_tile_packets],
    ['intra-tile packets', seg.intra_tile_packets],
    ['input-path packets', seg.input_path_packets],
    ['spikes (segment)', seg.spikes],
    ['timesteps', seg.timesteps_executed],
    ['sim time', seg.sim_time_s != null ? (seg.sim_time_s * 1e6).toFixed(2) + ' µs' : '—'],
    ['segment energy', seg.energy_j != null ? (seg.energy_j * 1e6).toFixed(3) + ' µJ' : '—'],
    ['arch', view.snap.arch_preset + (sample.arch_name ? ` · ${sample.arch_name}` : '')],
  ];
  host.innerHTML = '<div class="rail-stats" style="grid-template-columns:1fr 1fr">'
    + rows.map(([k, v]) => `<div class="rail-stat"><div class="rail-stat-value" style="font-size:12px">${esc(String(v ?? '—'))}</div><div class="rail-stat-label">${esc(k)}</div></div>`).join('')
    + '</div>';
}

function renderTileTable() {
  const host = document.getElementById('noc-tile-table');
  const seg = currentSegment();
  if (!host || !seg) return;
  const tiles = seg.per_tile || [];
  if (!tiles.length) {
    host.innerHTML = '<div class="empty-state">No per-tile records</div>';
    return;
  }
  host.innerHTML = `<table class="data-table compact">
    <thead><tr><th>Tile</th><th>xy</th><th>Cores</th><th>Packets</th><th>Spikes</th><th>µJ</th></tr></thead>
    <tbody>${tiles.map(t => `<tr>
      <td class="num">T${t.tile_index}</td>
      <td class="num">(${t.mesh_x},${t.mesh_y})</td>
      <td class="num">${(t.cores || []).length}</td>
      <td class="num">${t.packets_sent}</td>
      <td class="num">${t.spikes_fired}</td>
      <td class="num">${(t.energy_j * 1e6).toFixed(2)}</td>
    </tr>`).join('')}</tbody>
  </table>`;
}
