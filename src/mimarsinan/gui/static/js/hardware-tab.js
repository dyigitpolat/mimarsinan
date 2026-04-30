/* Hardware mapping tab — three-pane workbench.
 *
 * Layout:
 *   ┌──────┬──────────────────────┬──────────────┐
 *   │ rail │ canvas (stage flow)  │ inspector    │
 *   └──────┴──────────────────────┴──────────────┘
 *
 * The pixel-sizing rule for cores is preserved verbatim: the longest core
 * dimension across all types in a segment is mapped to MAX_CORE_DISPLAY_PX,
 * every other dimension scales proportionally with a MIN_CORE_DISPLAY_PX
 * floor. Heatmaps and connectivity spans stay lazy-loaded from
 * /api/.../resources/{kind}/{rid}; the snapshot only carries {kind, rid}.
 */
import { imgSrcAttr, resourceUrl, getResourceContext } from './resource-urls.js';
import { esc, safeReact, plotHistogram } from './util.js';

// ── Lazy connectivity (per-core span list) ───────────────────────────────
const _connectivityCache = new Map();
const _connectivityInflight = new Map();

function getConnectivitySpans(stage, coreIdx, onLoaded) {
  if (!stage) return null;
  if (Array.isArray(stage.connectivity)) {
    return stage.connectivity.filter(
      sp => sp.src_core === coreIdx || sp.dst_core === coreIdx,
    );
  }
  if (!stage.has_connectivity || coreIdx == null || !stage.cores) return [];
  const core = stage.cores.find(c => c.core_index === coreIdx);
  if (!core || !core.connectivity_resource) return [];
  const url = resourceUrl(core.connectivity_resource, getResourceContext());
  if (!url) return [];
  if (_connectivityCache.has(url)) return _connectivityCache.get(url);
  if (!_connectivityInflight.has(url)) {
    _connectivityInflight.set(url, (async () => {
      try {
        const res = await fetch(url, { headers: { Accept: 'application/json' } });
        if (!res.ok) { _connectivityCache.set(url, []); return []; }
        const data = await res.json();
        const spans = Array.isArray(data?.spans) ? data.spans
          : (Array.isArray(data) ? data : []);
        _connectivityCache.set(url, spans);
        return spans;
      } catch (_) {
        _connectivityCache.set(url, []);
        return [];
      } finally {
        _connectivityInflight.delete(url);
      }
    })());
  }
  if (typeof onLoaded === 'function') {
    _connectivityInflight.get(url).then(onLoaded);
  }
  return null;
}

// ── Helpers ──────────────────────────────────────────────────────────────
function _resolveLayerLabel(irNodeId, irGraph) {
  if (!irGraph || !irGraph.nodes) return '—';
  const node = irGraph.nodes.find(n => n.id === irNodeId || n.id === parseInt(irNodeId, 10));
  if (!node) return '—';
  if (node.weight_bank_id != null) return `WB${node.weight_bank_id}`;
  if (node.layer_group) return node.layer_group;
  if (node.name) {
    const parts = node.name.split('.');
    return parts.length > 1 ? parts[parts.length - 1] : node.name;
  }
  return '—';
}

function compactCount(value) {
  const v = Number(value || 0);
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1).replace(/\.0$/, '')}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(1).replace(/\.0$/, '')}k`;
  return String(v);
}

function nodeTag(nodeId) {
  if (nodeId === -2) return 'input';
  if (nodeId === -3) return 'output';
  return `n${nodeId}`;
}

function utilColor(util) {
  if (util >= 0.66) return '#34d399';
  if (util >= 0.33) return '#fbbf24';
  return '#6b7280';
}

function segKey(stage, fallbackIdx) {
  return stage.segment_index ?? fallbackIdx;
}

// ── Sizing constants — keep IDENTICAL to preserve relative dims ──────────
const MAX_CORE_DISPLAY_PX = 200;
const MIN_CORE_DISPLAY_PX = 8;
// SEG_VIEW_WIDTH is the per-segment "column budget" — used to compute how many
// core cells fit per row of the grid. It used to be a hardcoded 580; we now
// measure the actual canvas width at render time so the grid expands into the
// available centre column instead of leaving empty space on the right. The
// constant below is just a sane fallback when measurement isn't available
// yet (e.g. before first layout).
const SEG_VIEW_WIDTH_FALLBACK = 580;
const ID_WIDTH = 20;
const UTIL_HEIGHT = 14;
const GRID_GAP = 6;
const HW_SOFT_MAX = 200;
const MIN_HEATMAP_VIEW_WIDTH = 80;
const MIN_HEATMAP_VIEW_HEIGHT = 80;

function computeGlobalLayout(hw) {
  const layout = hw.global_core_layout || [];
  if (layout.length === 0) return null;
  return layout.map(cl => ({
    axons: cl.axons_per_core, neurons: cl.neurons_per_core, count: cl.count,
    key: `${cl.axons_per_core}x${cl.neurons_per_core}`,
  }));
}

// ── Entry point ──────────────────────────────────────────────────────────
export function renderHardwareTab(hw, container, irGraph) {
  if (!hw) { container.innerHTML = '<div class="empty-state">No hardware mapping data</div>'; return; }

  let html = `
    <div class="grid-3 hw-top-metrics" style="margin-bottom:20px">
      <div class="card"><div class="big-metric"><div class="value">${hw.total_cores}</div><div class="label">Total HW Cores</div></div></div>
      <div class="card"><div class="big-metric"><div class="value">${(hw.mean_utilization * 100).toFixed(1)}%</div><div class="label">Mean Utilization</div></div></div>
      <div class="card"><div class="big-metric"><div class="value">${hw.num_neural_segments} / ${hw.num_compute_ops}</div><div class="label">Neural Segs / Ops</div></div></div>
    </div>
    <div class="card hw-workbench-card" style="margin-bottom:20px">
      <div class="card-header">
        <span>Stage Execution Flow</span>
        <span class="hw-workbench-hint text-muted">Rail → click stage · canvas → click core · overlay → click span</span>
      </div>
      <div class="card-body no-pad">
        <div class="hw-workbench" id="hw-workbench"></div>
      </div>
    </div>
    <div class="grid-2">
      <div class="card"><div class="card-header">Core Utilization Distribution</div><div class="card-body"><div id="hw-util" style="min-height:200px"></div></div></div>
      <div class="card"><div class="card-header">Per-Segment Utilization</div><div class="card-body"><div id="hw-seg-util" style="min-height:200px"></div></div></div>
    </div>`;

  if (hw.global_core_layout?.length > 0) {
    html += '<div class="card" style="margin-top:20px"><div class="card-header">Hardware Core Layout (Minimum Requirement)</div><div class="card-body">';
    html += '<table class="data-table"><thead><tr><th>Core Type (Axons × Neurons)</th><th>Count</th></tr></thead><tbody>';
    for (const cl of hw.global_core_layout)
      html += `<tr><td>${cl.axons_per_core} × ${cl.neurons_per_core}</td><td>${cl.count}</td></tr>`;
    html += '</tbody></table></div></div>';
  }

  container.innerHTML = html;
  mountWorkbench(hw, irGraph);
  if (hw.utilization_histogram) plotHistogram('hw-util', hw.utilization_histogram, 'Utilization', '#4caf50');
  renderSegUtilChart(hw);
}

function renderSegUtilChart(hw) {
  const neuralSegs = hw.stages.filter(s => s.kind === 'neural' && s.cores);
  if (neuralSegs.length === 0) return;
  const colors = ['#5b8af5', '#4caf50', '#ff9800', '#f44336', '#9c27b0', '#00bcd4', '#ff5722', '#8bc34a'];
  safeReact('hw-seg-util', neuralSegs.map((s, si) => {
    const utils = s.cores.map(c => c.utilization);
    const avg = utils.reduce((a, b) => a + b, 0) / utils.length;
    return {
      y: [`Seg ${s.segment_index ?? si}`], x: [avg], name: s.name, type: 'bar', orientation: 'h',
      marker: { color: colors[si % colors.length] },
      text: [(avg * 100).toFixed(1) + '%'], textposition: 'outside', textfont: { size: 10, color: '#9a9daa' },
    };
  }), {
    barmode: 'group', xaxis: { title: 'Mean Utilization', range: [0, 1] },
    yaxis: { autorange: 'reversed' }, height: Math.max(160, neuralSegs.length * 40 + 60), showlegend: false,
  });
}

// ════════════════════════════════════════════════════════════════════════
// Workbench
// ════════════════════════════════════════════════════════════════════════
function mountWorkbench(hw, irGraph) {
  const root = document.getElementById('hw-workbench');
  if (!root) return;

  const state = {
    expanded: new Set(),
    selection: {
      kind: null,        // 'core' | 'soft' | 'span' | 'buffer' | null
      segIdx: null,
      coreIndex: null,
      softNodeId: null,
      softOrigin: null,  // { segIdx, coreIndex, placement }
      spanKey: null,
      spanData: null,    // resolved span object (for inspector)
      buffer: null,      // { segIdx, dir, bmIdx }
    },
    inspectorTab: 'core',
    inspectorPinned: false,
    focusedSegIdx: null,
  };
  const globalLayout = computeGlobalLayout(hw);

  function applyTrace() {
    if (window._hwTraceSoftId == null) return;
    const traceId = parseInt(window._hwTraceSoftId, 10);
    for (let si = 0; si < hw.stages.length; si++) {
      const stage = hw.stages[si];
      if (stage.kind === 'neural' && stage.cores) {
        const hit = stage.cores.some(c => (c.mapped_placements || []).some(p => p.ir_node_id === traceId));
        if (hit) state.expanded.add(segKey(stage, si));
      }
    }
  }

  // ── Scaffold ──────────────────────────────────────────────────────────
  root.innerHTML = `
    <div class="hw-rail" id="hw-rail"></div>
    <div class="hw-canvas-wrap">
      <div class="hw-breadcrumb" id="hw-breadcrumb"></div>
      <div class="hw-canvas" id="hw-canvas"></div>
    </div>
    <div class="hw-inspector" id="hw-inspector" data-open="false"></div>
  `;
  const railEl = root.querySelector('#hw-rail');
  const canvasEl = root.querySelector('#hw-canvas');
  const breadcrumbEl = root.querySelector('#hw-breadcrumb');
  const inspectorEl = root.querySelector('#hw-inspector');

  // ── Rail ──────────────────────────────────────────────────────────────
  function renderRail() {
    let html = '';
    for (let si = 0; si < hw.stages.length; si++) {
      const stage = hw.stages[si];
      const sk = segKey(stage, si);
      const focusKey = stage.kind === 'neural' ? sk : `s_${si}`;
      const isFocused = state.focusedSegIdx === focusKey;
      if (stage.kind === 'neural') {
        const cores = stage.cores || [];
        const avg = cores.length ? cores.reduce((s, c) => s + c.utilization, 0) / cores.length : 0;
        const minU = cores.length ? Math.min(...cores.map(c => c.utilization)) : 0;
        const maxU = cores.length ? Math.max(...cores.map(c => c.utilization)) : 0;
        const spark = renderRailSparkline(cores);
        html += `<button class="hw-rail-item is-neural ${isFocused ? 'is-focused' : ''}"
          data-action="rail-jump" data-stage-si="${si}"
          title="Segment ${sk} — ${esc(stage.name)} · ${cores.length} cores · util ${(minU*100).toFixed(0)}–${(maxU*100).toFixed(0)}%">
          <span class="hw-rail-spine"></span>
          <span class="hw-rail-body">
            <span class="hw-rail-titlerow"><span class="hw-rail-title">S${sk}</span><span class="hw-rail-meta-num">${(avg*100).toFixed(0)}%</span></span>
            <span class="hw-rail-name">${esc(stage.name)}</span>
            <span class="hw-rail-spark">${spark}</span>
            <span class="hw-rail-meta"><span>${cores.length} cores</span></span>
          </span>
        </button>`;
      } else if (stage.kind === 'compute_group') {
        const ops = stage.ops || [];
        const summary = (stage.op_types || []).slice(0, 2).join(', ') || 'Compute Group';
        html += `<button class="hw-rail-item is-compute ${isFocused ? 'is-focused' : ''}"
          data-action="rail-jump" data-stage-si="${si}"
          title="Compute Group · ${ops.length} ops">
          <span class="hw-rail-spine"></span>
          <span class="hw-rail-body">
            <span class="hw-rail-titlerow"><span class="hw-rail-title">≣</span><span class="hw-rail-meta-num">${ops.length}</span></span>
            <span class="hw-rail-name">${esc(summary)}</span>
            <span class="hw-rail-meta"><span>ops</span></span>
          </span>
        </button>`;
      } else {
        html += `<button class="hw-rail-item is-compute ${isFocused ? 'is-focused' : ''}"
          data-action="rail-jump" data-stage-si="${si}"
          title="${esc(stage.op_type || stage.op_name || 'Barrier')}">
          <span class="hw-rail-spine"></span>
          <span class="hw-rail-body">
            <span class="hw-rail-titlerow"><span class="hw-rail-title">▣</span></span>
            <span class="hw-rail-name">${esc(stage.name || stage.op_name || 'Barrier')}</span>
            <span class="hw-rail-meta"><span>${esc(stage.op_type || '?')}</span></span>
          </span>
        </button>`;
      }
    }
    railEl.innerHTML = html;
  }

  function renderRailSparkline(cores) {
    if (!cores || cores.length === 0) return '';
    const n = cores.length;
    const w = 100, h = 18;
    const stepX = w / Math.max(n, 1);
    let bars = '';
    for (let i = 0; i < n; i++) {
      const u = cores[i].utilization || 0;
      const bh = Math.max(1, u * h);
      const x = i * stepX;
      bars += `<rect x="${x.toFixed(2)}" y="${(h-bh).toFixed(2)}" width="${Math.max(stepX-0.4, 0.6).toFixed(2)}" height="${bh.toFixed(2)}" fill="${utilColor(u)}" opacity="0.85"/>`;
    }
    return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" width="100%" height="${h}" class="hw-rail-spark-svg">${bars}</svg>`;
  }

  // Compute the per-segment column budget from the current canvas width.
  // We subtract: canvas inner padding (≈36px), the input buffer reserve
  // (≈42px including its label + gap to the grid), the segment-body
  // padding (≈28px), and a few px of safety margin so the grid never
  // overflows and triggers a horizontal scrollbar.
  function currentSegViewWidth() {
    const w = canvasEl?.clientWidth || 0;
    if (!w) return SEG_VIEW_WIDTH_FALLBACK;
    return Math.max(360, w - 36 - 42 - 28 - 6);
  }

  // ── Canvas ────────────────────────────────────────────────────────────
  function renderCanvas() {
    const segWidth = currentSegViewWidth();
    let html = '';
    for (let si = 0; si < hw.stages.length; si++) {
      const stage = hw.stages[si];
      if (stage.kind === 'neural') html += renderNeuralStage(stage, si, segWidth);
      else if (stage.kind === 'compute_group') html += renderComputeGroup(stage, si);
      else html += renderComputeBarrier(stage, si);
      if (si < hw.stages.length - 1) html += '<div class="hw-stage-connector" aria-hidden="true">↓</div>';
    }
    canvasEl.innerHTML = html;
    canvasEl.querySelectorAll('.hw-stage-card[data-stage-si]').forEach(el => scrollObserver.observe(el));
    drawAllOverlays();
  }

  function renderNeuralStage(stage, si, segWidth) {
    const sk = segKey(stage, si);
    const isExp = state.expanded.has(sk);
    const cores = stage.cores || [];
    const avg = cores.length ? cores.reduce((s, c) => s + c.utilization, 0) / cores.length : 0;
    const minU = cores.length ? Math.min(...cores.map(c => c.utilization)) : 0;
    const maxU = cores.length ? Math.max(...cores.map(c => c.utilization)) : 0;
    const softCount = cores.reduce((s, c) => s + (c.mapped_placements?.length || 0), 0);
    const traceId = window._hwTraceSoftId != null ? parseInt(window._hwTraceSoftId, 10) : null;
    const isTraced = traceId != null && cores.some(c => (c.mapped_placements || []).some(p => p.ir_node_id === traceId));

    let html = `<div class="hw-stage-card hw-stage-neural ${isExp ? 'is-expanded' : ''}" data-stage-id="${sk}" data-stage-si="${si}">`;
    html += `<button class="hw-stage-header" data-action="toggle-stage" data-stage="${sk}" title="${esc(stage.name)} · util ${(minU*100).toFixed(0)}–${(maxU*100).toFixed(0)}%">`;
    html += `<span class="hw-stage-chevron">${isExp ? '▾' : '▸'}</span>`;
    html += '<span class="hw-stage-headline">';
    html += `<span class="hw-stage-tag">SEG ${sk}</span>`;
    html += `<span class="hw-stage-name" title="${esc(stage.name)}">${esc(stage.name)}</span>`;
    html += '</span>';
    html += '<span class="hw-stage-badges">';
    html += `<span class="hw-stage-badge"><span class="hw-stage-badge-num">${cores.length}</span><span class="hw-stage-badge-lbl">cores</span></span>`;
    html += `<span class="hw-stage-badge"><span class="hw-stage-badge-num" style="color:${utilColor(avg)}">${(avg*100).toFixed(0)}%</span><span class="hw-stage-badge-lbl">util</span></span>`;
    if (softCount > 0) html += `<span class="hw-stage-badge"><span class="hw-stage-badge-num">${softCount}</span><span class="hw-stage-badge-lbl">soft</span></span>`;
    if (isTraced) html += `<span class="hw-stage-badge is-trace" title="Trace: n${traceId}">trace · n${traceId}</span>`;
    html += '</span>';
    html += '</button>';
    if (isExp) {
      html += `<div class="hw-stage-body" data-stage-id="${sk}">${buildSegmentDetail(stage, sk, segWidth)}</div>`;
    }
    html += '</div>';
    return html;
  }

  function renderComputeGroup(stage, si) {
    const cgKey = `cg_${si}`;
    const isExp = state.expanded.has(cgKey);
    const ops = stage.ops || [];
    const summary = (stage.op_types || ops.map(o => o.op_type)).slice(0, 3).join(', ');
    let html = `<div class="hw-stage-card hw-stage-compute hw-stage-compute-group ${isExp ? 'is-expanded' : ''}" data-stage-si="${si}">`;
    html += `<button class="hw-stage-header" data-action="toggle-stage" data-stage="${cgKey}">`;
    html += `<span class="hw-stage-chevron">${isExp ? '▾' : '▸'}</span>`;
    html += '<span class="hw-stage-headline">';
    html += '<span class="hw-stage-tag">COMPUTE GROUP</span>';
    html += `<span class="hw-stage-name">${esc(summary || '—')}</span>`;
    html += '</span>';
    html += `<span class="hw-stage-badges"><span class="hw-stage-badge"><span class="hw-stage-badge-num">${ops.length}</span><span class="hw-stage-badge-lbl">ops</span></span></span>`;
    html += '</button>';
    if (isExp) {
      html += '<div class="hw-stage-body">';
      html += '<table class="data-table compact"><thead><tr><th>Op Type</th><th>Name</th><th>Input</th><th>Output</th></tr></thead><tbody>';
      for (const op of ops) {
        const inS = op.input_shape ? `[${op.input_shape.join(', ')}]` : '—';
        const outS = op.output_shape ? `[${op.output_shape.join(', ')}]` : '—';
        html += `<tr><td>${esc(op.op_type || '?')}</td><td title="${esc(op.op_name || '')}">${esc((op.op_name || '—').substring(0, 40))}</td><td><code>${inS}</code></td><td><code>${outS}</code></td></tr>`;
      }
      html += '</tbody></table>';
      html += '</div>';
    }
    html += '</div>';
    return html;
  }

  function renderComputeBarrier(stage, si) {
    const inS = stage.input_shape ? `[${stage.input_shape.join(', ')}]` : null;
    const outS = stage.output_shape ? `[${stage.output_shape.join(', ')}]` : null;
    let html = `<div class="hw-stage-card hw-stage-compute hw-stage-barrier" data-stage-si="${si}">`;
    html += '<div class="hw-stage-header is-static">';
    html += '<span class="hw-stage-chevron is-mute">▣</span>';
    html += '<span class="hw-stage-headline">';
    html += '<span class="hw-stage-tag">BARRIER</span>';
    html += `<span class="hw-stage-name" title="${esc(stage.name)}">${esc(stage.name)}</span>`;
    html += '</span>';
    html += '<span class="hw-stage-badges">';
    html += `<span class="hw-stage-badge"><span class="hw-stage-badge-num">${esc(stage.op_type || stage.op_name || '?')}</span><span class="hw-stage-badge-lbl">op</span></span>`;
    if (inS) html += `<span class="hw-stage-badge"><span class="hw-stage-badge-num"><code>${inS}</code></span><span class="hw-stage-badge-lbl">in</span></span>`;
    if (outS) html += `<span class="hw-stage-badge"><span class="hw-stage-badge-num"><code>${outS}</code></span><span class="hw-stage-badge-lbl">out</span></span>`;
    html += '</span>';
    html += '</div>';
    html += '</div>';
    return html;
  }

  // ── Segment detail (cores + buffers) — preserves pixel math verbatim ─
  function buildSegmentDetail(stage, segIdx, segWidth) {
    const cores = stage.cores;
    if (!cores || cores.length === 0) return '<div class="hw-empty">No cores in this segment</div>';
    const SEG_VIEW_WIDTH = segWidth || SEG_VIEW_WIDTH_FALLBACK;

    const hasInput = stage.input_map?.length > 0;
    const hasOutput = stage.output_map?.length > 0;
    const totalIn = hasInput ? stage.input_map.reduce((s, m) => s + m.size, 0) : 0;
    const totalOut = hasOutput ? stage.output_map.reduce((s, m) => s + m.size, 0) : 0;

    const maxAxons = globalLayout
      ? Math.max(...globalLayout.map(g => g.axons))
      : Math.max(...cores.map(c => c.axons_per_core));
    const maxNeurons = globalLayout
      ? Math.max(...globalLayout.map(g => g.neurons))
      : Math.max(...cores.map(c => c.neurons_per_core));
    const maxDimension = Math.max(maxAxons, maxNeurons, 1);

    let html = '<div class="hw-segment-col" id="hw-layout-' + segIdx + '">';
    html += '<div class="hw-segment-row-inner">';

    if (hasInput) {
      const showLabels = stage.input_map.length <= 12;
      html += `<div class="hw-buffer-line hw-buffer-input vertical" id="hw-in-${segIdx}" title="Input buffer: ${totalIn} elements">`;
      html += `<span class="hw-buffer-label">IN ${compactCount(totalIn)}</span>`;
      html += `<div class="hw-buffer-track" id="hw-in-track-${segIdx}">`;
      for (let mi = 0; mi < stage.input_map.length; mi++) {
        const m = stage.input_map[mi];
        const pct = Math.max(1, (m.size / Math.max(totalIn, 1)) * 100);
        const label = nodeTag(m.node_id);
        html += `<span class="hw-buffer-segment" data-action="select-buffer" data-seg="${segIdx}" data-bm-idx="${mi}" data-dir="in" style="height:${pct}%" data-offset="${m.offset}" data-end="${m.offset + m.size}" data-node-id="${m.node_id}" title="${label} [${m.offset}:${m.offset + m.size}]">${showLabels ? esc(label) : ''}</span>`;
      }
      html += '</div></div>';
    }

    const coresByDim = new Map();
    for (const c of cores) {
      const key = `${c.axons_per_core}x${c.neurons_per_core}`;
      if (!coresByDim.has(key)) coresByDim.set(key, []);
      coresByDim.get(key).push(c);
    }
    const groupOrder = globalLayout
      ? globalLayout.map(g => `${g.axons}x${g.neurons}`)
      : [...coresByDim.keys()];

    html += `<div class="hw-core-grid-wrap" id="hw-grid-wrap-${segIdx}">`;
    html += `<div class="hw-core-grid" id="hw-grid-${segIdx}">`;

    for (const key of groupOrder) {
      const [axStr, nStr] = key.split('x');
      const ax = parseInt(axStr, 10);
      const n = parseInt(nStr, 10);
      const pool = coresByDim.get(key) || [];
      const grpCount = globalLayout ? (globalLayout.find(g => `${g.axons}x${g.neurons}` === key)?.count ?? pool.length) : pool.length;
      const longPx = Math.max(MIN_CORE_DISPLAY_PX, Math.round((Math.max(ax, n) / maxDimension) * MAX_CORE_DISPLAY_PX));
      const actualW = ax >= n ? Math.round(longPx * (n / ax)) : longPx;
      const actualH = ax >= n ? longPx : Math.round(longPx * (ax / n));
      const cellW = ID_WIDTH + 4 + actualW;
      const cellH = actualH + UTIL_HEIGHT;
      const numCols = Math.max(1, Math.floor((SEG_VIEW_WIDTH + GRID_GAP) / (cellW + GRID_GAP)));

      html += '<div class="hw-core-group">';
      html += `<div class="hw-core-group-label"><span class="hw-core-group-key">${ax}×${n}</span><span class="hw-core-group-count">${pool.length}/${grpCount}</span></div>`;
      html += `<div class="hw-core-group-cores" style="grid-template-columns:repeat(${numCols},${cellW}px);grid-auto-rows:${cellH}px">`;
      for (let i = 0; i < grpCount; i++) {
        const actualCore = i < pool.length ? pool[i] : null;
        html += emitCoreCell(actualCore, ax, n, actualW, actualH, !actualCore, segIdx);
      }
      html += '</div></div>';
    }

    html += '</div></div>'; // hw-core-grid, hw-core-grid-wrap
    html += '</div>';       // hw-segment-row-inner

    if (hasOutput) {
      const showLabels = stage.output_map.length <= 12;
      html += `<div class="hw-buffer-line hw-buffer-output thick" id="hw-out-${segIdx}" title="Output buffer: ${totalOut} elements">`;
      html += `<span class="hw-buffer-label">OUT ${compactCount(totalOut)}</span>`;
      html += `<div class="hw-buffer-track" id="hw-out-track-${segIdx}">`;
      for (let mi = 0; mi < stage.output_map.length; mi++) {
        const m = stage.output_map[mi];
        const pct = Math.max(2, (m.size / Math.max(totalOut, 1)) * 100);
        const label = nodeTag(m.node_id);
        html += `<span class="hw-buffer-segment" data-action="select-buffer" data-seg="${segIdx}" data-bm-idx="${mi}" data-dir="out" style="width:${pct}%" data-offset="${m.offset}" data-end="${m.offset + m.size}" data-node-id="${m.node_id}" title="${label} [${m.offset}:${m.offset + m.size}]">${showLabels ? esc(label) : ''}</span>`;
      }
      html += '</div></div>';
    }

    html += '</div>'; // hw-segment-col
    return html;
  }

  function emitCoreCell(core, ax, n, actualW, actualH, isPlaceholder, segIdx) {
    const cellW = ID_WIDTH + 4 + actualW;
    const cellH = actualH + UTIL_HEIGHT;
    const cellStyle = `width:${cellW}px;height:${cellH}px`;
    const ar = Math.max(0.01, n / ax);
    const coreStyle = ax >= n
      ? `height:${actualH}px;width:auto;aspect-ratio:${ar}`
      : `width:${actualW}px;height:auto;aspect-ratio:${ar}`;
    if (isPlaceholder) {
      return `<div class="hw-core-cell" style="${cellStyle}" title="Unused slot">
        <span class="hw-core-id">—</span>
        <div class="hw-core-cell-main"><div class="hw-core hw-core-empty" style="${coreStyle}"></div><span class="hw-core-util"><span class="hw-core-util-bar" style="width:0%"></span><span class="hw-core-util-num">—</span></span></div>
      </div>`;
    }
    const pct = (core.utilization * 100).toFixed(0);
    const traceId = window._hwTraceSoftId != null ? parseInt(window._hwTraceSoftId, 10) : null;
    const isTraced = traceId != null && (core.mapped_placements || []).some(p => p.ir_node_id === traceId);
    const isSelected = state.selection.kind && state.selection.segIdx === segIdx && state.selection.coreIndex === core.core_index;
    const selCls = (isSelected ? ' hw-core-selected' : '') + (isTraced ? ' hw-core-traced' : '');

    let cell = `<div class="hw-core-cell" data-action="select-core" data-seg="${segIdx}" data-core="${core.core_index}" style="${cellStyle}" title="Core ${core.core_index} · ${core.used_axons}/${core.axons_per_core}ax × ${core.used_neurons}/${core.neurons_per_core}n · ${pct}%">`;
    cell += `<span class="hw-core-id">${core.core_index}</span>`;
    cell += '<div class="hw-core-cell-main">';
    cell += `<div class="hw-core${selCls}" id="hc-${segIdx}-${core.core_index}" style="${coreStyle};position:relative">`;
    const cellHeatmapUrl = imgSrcAttr(core.heatmap_resource);
    if (cellHeatmapUrl) {
      cell += `<img class="hw-core-canvas" src="${cellHeatmapUrl}" loading="lazy" decoding="async" style="width:100%;height:100%;display:block;object-fit:fill" draggable="false">`;
    }
    const placements = core.mapped_placements || [];
    const aTotal = Math.max(1, core.heatmap_axons || core.axons_per_core);
    const nTotal = Math.max(1, core.heatmap_neurons || core.neurons_per_core);
    const fusedBoundaries = core.fused_axon_boundaries;
    const fusedCount = core.fused_component_count != null ? core.fused_component_count : (fusedBoundaries && fusedBoundaries.length > 1 ? fusedBoundaries.length - 1 : 0);
    if (placements.length > 1 || (fusedBoundaries && fusedBoundaries.length >= 2)) {
      cell += '<div class="hw-core-constituent-overlay">';
      if (placements.length > 1) {
        for (const pl of placements) {
          const top = (pl.axon_offset / aTotal) * 100;
          const left = (pl.neuron_offset / nTotal) * 100;
          const width = (pl.neurons / nTotal) * 100;
          const height = (pl.axons / aTotal) * 100;
          cell += `<div class="hw-core-constituent-boundary" style="top:${top}%;left:${left}%;width:${width}%;height:${height}%"></div>`;
        }
      }
      if (fusedBoundaries && fusedBoundaries.length >= 2) {
        for (let i = 1; i < fusedBoundaries.length - 1; i++) {
          const topPct = (fusedBoundaries[i] / aTotal) * 100;
          cell += `<div class="hw-core-fused-boundary-line" style="top:${topPct}%;left:0;right:0;height:1px"></div>`;
        }
      }
      cell += '</div>';
    }
    if (fusedCount > 1) {
      cell += `<div class="hw-core-fused-badge" title="Fused: ${fusedCount} physical HW cores">${fusedCount}▣</div>`;
    }
    if (placements.length > 1) {
      cell += `<div class="hw-core-soft-badge" title="${placements.length} soft cores">${placements.length}◌</div>`;
    }
    cell += '</div>'; // .hw-core
    cell += `<span class="hw-core-util"><span class="hw-core-util-bar" style="width:${pct}%;background:${utilColor(core.utilization)}"></span><span class="hw-core-util-num">${pct}%</span></span>`;
    cell += '</div></div>'; // .hw-core-cell-main, .hw-core-cell
    return cell;
  }

  // ── Inspector ─────────────────────────────────────────────────────────
  function inspectorOpen() {
    return state.inspectorPinned || state.selection.kind != null;
  }

  function renderInspector() {
    const open = inspectorOpen();
    inspectorEl.dataset.open = open ? 'true' : 'false';
    root.dataset.inspectorClosed = open ? 'false' : 'true';
    if (!open) { inspectorEl.innerHTML = ''; return; }

    const tabs = ['core', 'soft', 'span', 'buffer'];
    const labels = { core: 'Hard Core', soft: 'Soft Core', span: 'Span', buffer: 'Buffer' };
    const has = {
      core: state.selection.coreIndex != null,
      soft: state.selection.softNodeId != null,
      span: state.selection.spanData != null,
      buffer: state.selection.buffer != null,
    };
    let tab = state.inspectorTab;
    if (!has[tab]) {
      const fallback = tabs.find(t => has[t]);
      if (fallback) tab = fallback;
    }

    let html = '<div class="hw-inspector-bar">';
    html += '<div class="hw-inspector-tabs">';
    for (const t of tabs) {
      const cls = (t === tab ? 'is-active' : '') + (has[t] ? '' : ' is-empty');
      html += `<button class="hw-inspector-tab ${cls}" data-action="tab-switch" data-tab="${t}">${labels[t]}</button>`;
    }
    html += '</div>';
    html += '<div class="hw-inspector-actions">';
    html += `<button class="hw-inspector-icon ${state.inspectorPinned ? 'is-on' : ''}" data-action="inspector-pin" title="Pin inspector open">${state.inspectorPinned ? '📌' : '📍'}</button>`;
    html += '<button class="hw-inspector-icon" data-action="inspector-close" title="Close">✕</button>';
    html += '</div></div>';

    html += '<div class="hw-inspector-body">';
    if (tab === 'core') html += renderInspectorCore();
    else if (tab === 'soft') html += renderInspectorSoft();
    else if (tab === 'span') html += renderInspectorSpan();
    else if (tab === 'buffer') html += renderInspectorBuffer();
    html += '</div>';

    inspectorEl.innerHTML = html;
  }

  function renderInspectorCore() {
    const sel = state.selection;
    if (sel.coreIndex == null || sel.segIdx == null) {
      return '<div class="hw-empty">Select a core in a neural segment.</div>';
    }
    const stage = hw.stages.find((s, idx) => s.kind === 'neural' && segKey(s, idx) === sel.segIdx);
    const core = stage?.cores?.find(c => c.core_index === sel.coreIndex);
    if (!core) return '<div class="hw-empty">Core not found.</div>';

    const pct = (core.utilization * 100).toFixed(1);
    const placements = core.mapped_placements || [];
    const aTotal = Math.max(1, core.heatmap_axons || core.axons_per_core);
    const nTotal = Math.max(1, core.heatmap_neurons || core.neurons_per_core);
    const fusedBoundaries = core.fused_axon_boundaries;

    let html = '<div class="hw-insp-section">';
    html += `<div class="hw-insp-title">Segment ${sel.segIdx} · Core ${core.core_index}</div>`;
    const heatUrl = imgSrcAttr(core.heatmap_resource);
    if (heatUrl) {
      // Compute the wrapper's exact pixel size to match the heatmap's
      // aspect, then fill the image into it. Without this, max-height
      // clamps the wrapper but the image (object-fit:contain) letterboxes
      // inside, so the absolutely-positioned region overlays drift away
      // from the visible heatmap pixels.
      const heatN = Math.max(1, core.heatmap_neurons || core.neurons_per_core);
      const heatA = Math.max(1, core.heatmap_axons || core.axons_per_core);
      const ar = heatN / heatA;                 // width / height
      const MAX_W = 320;                        // inspector body inner width budget
      const MAX_H = 240;
      let dispW, dispH;
      if (ar >= MAX_W / MAX_H) { dispW = MAX_W; dispH = Math.max(40, Math.round(MAX_W / ar)); }
      else                     { dispH = MAX_H; dispW = Math.max(40, Math.round(MAX_H * ar)); }
      html += `<div class="hw-insp-heatmap-wrap" style="width:${dispW}px;height:${dispH}px">`;
      html += `<img src="${heatUrl}" alt="Core ${core.core_index} heatmap" loading="lazy" decoding="async" class="hw-insp-heatmap" style="width:100%;height:100%;object-fit:fill">`;
      for (let pi = 0; pi < placements.length; pi++) {
        const pl = placements[pi];
        const top = (pl.axon_offset / aTotal) * 100;
        const left = (pl.neuron_offset / nTotal) * 100;
        const height = (pl.axons / aTotal) * 100;
        const width = (pl.neurons / nTotal) * 100;
        const isSplit = pl.split_group_id != null;
        const splitCls = isSplit ? ' is-split' : '';
        const titleStr = isSplit && pl.neuron_range_in_original
          ? `n${pl.ir_node_id} [${pl.neuron_range_in_original[0]}:${pl.neuron_range_in_original[1]}/${pl.split_original_neurons || '?'}]`
          : `Software core n${pl.ir_node_id}`;
        html += `<div class="hw-insp-heatmap-region${splitCls}" data-action="select-soft" data-seg="${sel.segIdx}" data-core="${core.core_index}" data-pl="${pi}" data-node="${pl.ir_node_id}" style="top:${top}%;left:${left}%;width:${width}%;height:${height}%" title="${titleStr}"></div>`;
      }
      if (fusedBoundaries && fusedBoundaries.length >= 2) {
        for (let i = 1; i < fusedBoundaries.length - 1; i++) {
          const topPct = (fusedBoundaries[i] / aTotal) * 100;
          html += `<div class="hw-insp-fused-line" style="top:${topPct}%"></div>`;
        }
      }
      html += '</div>';
    }
    html += '<table class="data-table compact hw-insp-meta">';
    html += `<tr><td>Utilization</td><td><span class="hw-insp-utilbar"><span style="width:${pct}%;background:${utilColor(core.utilization)}"></span></span> ${pct}%</td></tr>`;
    html += `<tr><td>Dimensions</td><td>${core.axons_per_core} ax × ${core.neurons_per_core} n</td></tr>`;
    html += `<tr><td>Used</td><td>${core.used_axons ?? '—'} / ${core.used_neurons ?? '—'}</td></tr>`;
    html += `<tr><td>Threshold</td><td>${core.threshold != null ? core.threshold.toFixed(4) : '—'}</td></tr>`;
    html += '</table>';
    html += '</div>';

    if (placements.length > 0) {
      html += '<div class="hw-insp-section">';
      html += `<div class="hw-insp-subtitle">Constituents · ${placements.length}</div>`;
      html += '<table class="data-table compact hw-constituents-table"><thead><tr><th>ID</th><th>Layer</th><th>Dims</th><th>Util</th><th>Split</th><th>Coalesce</th></tr></thead><tbody>';
      for (let pi = 0; pi < placements.length; pi++) {
        const pl = placements[pi];
        const utilPct = (pl.utilization_frac != null ? pl.utilization_frac * 100 : (pl.axons * pl.neurons) / (aTotal * nTotal) * 100).toFixed(1);
        const coalesce = pl.coalescing_role ? `${esc(pl.coalescing_role)}${pl.coalescing_group_id != null ? ' G' + pl.coalescing_group_id : ''}` : '—';
        const splitInfo = pl.split_group_id != null && pl.neuron_range_in_original
          ? `[${pl.neuron_range_in_original[0]}:${pl.neuron_range_in_original[1]}/${pl.split_original_neurons || '?'}]`
          : '—';
        const layerLabel = _resolveLayerLabel(pl.ir_node_id, irGraph);
        html += `<tr class="hw-constituent-row" data-action="select-soft" data-seg="${sel.segIdx}" data-core="${core.core_index}" data-pl="${pi}" data-node="${pl.ir_node_id}" title="Open soft-core detail">
          <td><span class="hw-constituent-id">n${pl.ir_node_id}</span></td>
          <td class="hw-trunc" title="${esc(layerLabel)}">${esc(layerLabel)}</td>
          <td>${pl.axons}×${pl.neurons}</td>
          <td>${utilPct}%</td>
          <td>${splitInfo}</td>
          <td>${coalesce}</td>
        </tr>`;
      }
      html += '</tbody></table>';
      html += '</div>';
    }
    return html;
  }

  function renderInspectorSoft() {
    const sel = state.selection;
    if (sel.softNodeId == null) return '<div class="hw-empty">Open a soft-core from the Hard Core tab.</div>';
    if (!irGraph || !irGraph.nodes) return '<div class="hw-empty">No IR graph available.</div>';
    const node = irGraph.nodes.find(n => n.id === sel.softNodeId || n.id === parseInt(sel.softNodeId, 10));
    if (!node) return '<div class="hw-empty">Unknown node.</div>';
    const isNeural = node.type === 'neural_core';
    const origin = sel.softOrigin;

    let html = '<div class="hw-insp-section">';
    html += `<div class="hw-insp-title">Software core n${node.id}</div>`;
    html += `<p class="hw-insp-name" title="${esc(node.name)}">${esc(node.name || node.layer_group || `n${node.id}`)}</p>`;
    if (origin && origin.placement) {
      const p = origin.placement;
      const aEnd = p.axon_offset + (p.axons ?? 0);
      const nEnd = p.neuron_offset + (p.neurons ?? 0);
      html += `<p class="hw-insp-located">Located in: <button class="hw-insp-located-link" data-action="select-core" data-seg="${origin.segIdx}" data-core="${origin.coreIndex}">Seg ${origin.segIdx} · Core ${origin.coreIndex}</button> · axons ${p.axon_offset}..${aEnd}, neurons ${p.neuron_offset}..${nEnd}</p>`;
      if (p.split_group_id != null && p.neuron_range_in_original) {
        const r = p.neuron_range_in_original;
        const total = p.split_original_neurons || '?';
        const fragIdx = p.split_fragment_index != null ? p.split_fragment_index : '?';
        html += `<p class="hw-insp-split">Split fragment ${fragIdx} — neurons ${r[0]}..${r[1]} of ${total}</p>`;
      }
    }
    if (isNeural) {
      html += '<table class="data-table compact hw-insp-meta">';
      html += `<tr><td>ID</td><td>${node.id}</td></tr>`;
      const preAx = node.pre_pruning_axons;
      const preNu = node.pre_pruning_neurons;
      const postAx = node.axons ?? '—';
      const postNu = node.neurons ?? '—';
      if (preAx != null && preNu != null) {
        html += `<tr><td>Pre-pruning</td><td>${preAx}×${preNu}</td></tr>`;
        html += `<tr><td>Post-pruning</td><td>${postAx}×${postNu}</td></tr>`;
      } else {
        html += `<tr><td>Axons</td><td>${postAx}</td></tr>`;
        html += `<tr><td>Neurons</td><td>${postNu}</td></tr>`;
      }
      html += `<tr><td>Threshold</td><td>${node.threshold != null ? Number(node.threshold).toFixed(4) : '—'}</td></tr>`;
      if (node.psum_role) html += `<tr><td>PSum</td><td>${esc(node.psum_role)} (G${node.psum_group_id})</td></tr>`;
      if (node.coalescing_role) html += `<tr><td>Coalesce</td><td>${esc(node.coalescing_role)} (G${node.coalescing_group_id})</td></tr>`;
      html += `<tr><td>Latency</td><td>${node.latency ?? '—'}</td></tr>`;
      html += `<tr><td>Activation scale</td><td>${node.activation_scale != null ? Number(node.activation_scale).toFixed(4) : '—'}</td></tr>`;
      html += `<tr><td>Parameter scale</td><td>${node.parameter_scale != null ? Number(node.parameter_scale).toFixed(4) : '—'}</td></tr>`;
      html += `<tr><td>Sparsity</td><td>${node.weight_stats ? (node.weight_stats.sparsity * 100).toFixed(1) + '%' : '—'}</td></tr>`;
      html += '</table>';
      const nodeHeatmapUrl = imgSrcAttr(node.heatmap_resource);
      const nodePreHeatmapUrl = imgSrcAttr(node.pre_pruning_resource);
      if (nodeHeatmapUrl || nodePreHeatmapUrl) {
        const maxLong = Math.max(preAx ?? 0, preNu ?? 0, node.axons ?? 0, node.neurons ?? 0, 1);
        function softHeatmapSizePre(axons, neurons) {
          if (!axons || !neurons || maxLong < 1) return { w: HW_SOFT_MAX, h: HW_SOFT_MAX };
          let w = Math.round((neurons / maxLong) * HW_SOFT_MAX);
          let h = Math.round((axons / maxLong) * HW_SOFT_MAX);
          if (w < MIN_HEATMAP_VIEW_WIDTH && w <= h) {
            w = MIN_HEATMAP_VIEW_WIDTH;
            h = Math.round((axons / neurons) * w);
          } else if (h < MIN_HEATMAP_VIEW_HEIGHT && h <= w) {
            h = MIN_HEATMAP_VIEW_HEIGHT;
            w = Math.round((neurons / axons) * h);
          }
          return { w: Math.max(2, w), h: Math.max(2, h) };
        }
        function softHeatmapSizePost(postAxons, postNeurons, preSz, preAxons, preNeurons) {
          if (preSz && preAxons && preNeurons) {
            const scaleW = preSz.w / preNeurons;
            const scaleH = preSz.h / preAxons;
            return {
              w: Math.max(2, Math.round(postNeurons * scaleW)),
              h: Math.max(2, Math.round(postAxons * scaleH)),
            };
          }
          let w = Math.round((postNeurons / maxLong) * HW_SOFT_MAX);
          let h = Math.round((postAxons / maxLong) * HW_SOFT_MAX);
          return { w: Math.max(2, w), h: Math.max(2, h) };
        }
        html += '<div class="hw-insp-subtitle">Weight heatmap</div>';
        html += '<div class="hw-softcore-heatmaps-scroll">';
        html += '<div class="hw-insp-heatmap-row">';
        let preSz = null;
        if (nodePreHeatmapUrl) {
          const preLabel = (preAx != null && preNu != null) ? ` (${preAx}×${preNu})` : '';
          preSz = (preAx != null && preNu != null) ? softHeatmapSizePre(preAx, preNu) : { w: HW_SOFT_MAX, h: HW_SOFT_MAX };
          html += `<div class="hw-insp-heatmap-tile"><div class="hw-insp-heatmap-tile-lbl">Pre-pruning${preLabel}</div><img src="${nodePreHeatmapUrl}" alt="Pre-pruning" loading="lazy" decoding="async" class="hw-softcore-heatmap" style="width:${preSz.w}px;height:${preSz.h}px;object-fit:fill"></div>`;
        }
        if (nodeHeatmapUrl) {
          const postAxNum = node.axons ?? 0;
          const postNuNum = node.neurons ?? 0;
          const postLabel = (postAxNum && postNuNum) ? ` (${postAxNum}×${postNuNum})` : '';
          const postSz = softHeatmapSizePost(postAxNum, postNuNum, preSz, preAx ?? 0, preNu ?? 0);
          html += `<div class="hw-insp-heatmap-tile"><div class="hw-insp-heatmap-tile-lbl">Post-pruning${postLabel}</div><img src="${nodeHeatmapUrl}" alt="Post-pruning" loading="lazy" decoding="async" class="hw-softcore-heatmap" style="width:${postSz.w}px;height:${postSz.h}px;object-fit:fill"></div>`;
        }
        html += '</div></div>';
      }
    } else {
      html += '<table class="data-table compact hw-insp-meta">';
      html += `<tr><td>ID</td><td>${node.id}</td></tr>`;
      html += `<tr><td>Type</td><td>${esc(node.op_type || '—')}</td></tr>`;
      html += `<tr><td>Input shape</td><td><code>${node.input_shape ? node.input_shape.join('×') : '—'}</code></td></tr>`;
      html += `<tr><td>Output shape</td><td><code>${node.output_shape ? node.output_shape.join('×') : '—'}</code></td></tr>`;
      html += '</table>';
    }
    html += '</div>';
    return html;
  }

  function renderInspectorSpan() {
    const sp = state.selection.spanData;
    if (!sp) return '<div class="hw-empty">Click a connectivity span on a selected core.</div>';
    const segIdx = state.selection.segIdx;
    const stage = hw.stages.find((s, idx) => s.kind === 'neural' && segKey(s, idx) === segIdx);
    const coreByIdx = new Map();
    if (stage?.cores) for (const c of stage.cores) coreByIdx.set(c.core_index, c);
    const srcLabel = sp.src_core === -2 ? 'Input Buffer' : `Core ${sp.src_core}`;
    const dstLabel = sp.dst_core === -3 ? 'Output Buffer' : `Core ${sp.dst_core}`;
    let html = '<div class="hw-insp-section">';
    html += `<div class="hw-insp-title">${srcLabel} → ${dstLabel}</div>`;
    html += '<table class="data-table compact hw-insp-meta">';
    html += `<tr><td>Direction</td><td>${sp.dir === 'in' ? '← incoming' : '→ outgoing'}</td></tr>`;
    html += `<tr><td>Span length</td><td>${sp.length}</td></tr>`;
    if (sp.src_core >= 0) {
      html += `<tr><td>Source neurons</td><td>[${sp.src_start}:${sp.src_end}]</td></tr>`;
      const sc = coreByIdx.get(sp.src_core);
      if (sc) {
        html += `<tr><td>Source dims</td><td>${sc.axons_per_core}ax × ${sc.neurons_per_core}n</td></tr>`;
        html += `<tr><td>Source util</td><td>${(sc.utilization * 100).toFixed(1)}%</td></tr>`;
      }
    } else {
      html += `<tr><td>Source input range</td><td>[${sp.src_start}:${sp.src_end}]</td></tr>`;
    }
    html += `<tr><td>Dest axons</td><td>[${sp.dst_start}:${sp.dst_end}]</td></tr>`;
    if (sp.dst_core >= 0) {
      const dc = coreByIdx.get(sp.dst_core);
      if (dc) {
        html += `<tr><td>Dest dims</td><td>${dc.axons_per_core}ax × ${dc.neurons_per_core}n</td></tr>`;
        html += `<tr><td>Dest util</td><td>${(dc.utilization * 100).toFixed(1)}%</td></tr>`;
        html += `<tr><td>Dest threshold</td><td>${dc.threshold != null ? dc.threshold.toFixed(3) : '—'}</td></tr>`;
      }
    } else if (sp.dst_core === -3) {
      html += `<tr><td>Output range</td><td>[${sp.dst_start}:${sp.dst_end}]</td></tr>`;
    }
    html += '</table></div>';
    return html;
  }

  function renderInspectorBuffer() {
    const buf = state.selection.buffer;
    if (!buf) return '<div class="hw-empty">Click a slice of an input or output buffer.</div>';
    const stage = hw.stages.find((s, idx) => s.kind === 'neural' && segKey(s, idx) === buf.segIdx);
    if (!stage) return '<div class="hw-empty">Buffer source not found.</div>';
    const map = (buf.dir === 'in' ? stage.input_map : stage.output_map) || [];
    const m = map[buf.bmIdx];
    if (!m) return '<div class="hw-empty">Buffer slice not found.</div>';
    const total = map.reduce((s, x) => s + x.size, 0) || 1;
    const pct = (m.size / total * 100).toFixed(1);
    const tag = nodeTag(m.node_id);
    const layer = _resolveLayerLabel(m.node_id, irGraph);
    let html = '<div class="hw-insp-section">';
    html += `<div class="hw-insp-title">${buf.dir === 'in' ? 'Input' : 'Output'} buffer · ${tag}</div>`;
    html += '<table class="data-table compact hw-insp-meta">';
    html += `<tr><td>Segment</td><td>${buf.segIdx}</td></tr>`;
    html += `<tr><td>Direction</td><td>${buf.dir === 'in' ? 'IN' : 'OUT'}</td></tr>`;
    html += `<tr><td>Node</td><td>${tag} · ${esc(layer)}</td></tr>`;
    html += `<tr><td>Range</td><td>[${m.offset}:${m.offset + m.size}]</td></tr>`;
    html += `<tr><td>Size</td><td>${m.size} (${pct}% of buffer)</td></tr>`;
    html += '</table>';
    if (m.node_id >= 0) {
      html += `<button class="hw-insp-link" data-action="select-soft-from-buffer" data-node="${m.node_id}">Open soft-core n${m.node_id}</button>`;
    }
    html += '</div>';
    return html;
  }

  // ── Breadcrumb ────────────────────────────────────────────────────────
  function renderBreadcrumb() {
    const focus = state.focusedSegIdx;
    const sel = state.selection;
    const tokens = [];
    tokens.push({ label: 'Pipeline', staticOnly: true });
    if (typeof focus === 'string' && focus.startsWith('s_')) {
      const si = parseInt(focus.substring(2), 10);
      const stage = hw.stages[si];
      tokens.push({ label: `Stage ${si}`, jumpSi: si });
      if (stage) tokens.push({ label: stage.name || stage.op_type || 'compute', staticOnly: true });
    } else if (focus != null) {
      const idx = hw.stages.findIndex((s, i) => s.kind === 'neural' && segKey(s, i) === focus);
      tokens.push({ label: `Segment ${focus}`, jumpSi: idx });
      const stage = idx >= 0 ? hw.stages[idx] : null;
      if (stage) tokens.push({ label: stage.name, staticOnly: true });
      if (sel.coreIndex != null && sel.segIdx === focus) {
        tokens.push({ label: `Core ${sel.coreIndex}`, staticOnly: true });
      }
    }
    let html = '';
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i];
      if (i > 0) html += '<span class="hw-breadcrumb-sep">›</span>';
      if (t.jumpSi != null && t.jumpSi >= 0) {
        html += `<button class="hw-breadcrumb-token" data-action="rail-jump" data-stage-si="${t.jumpSi}">${esc(t.label)}</button>`;
      } else {
        html += `<span class="hw-breadcrumb-token is-static">${esc(t.label)}</span>`;
      }
    }
    breadcrumbEl.innerHTML = html;
  }

  // ── Connectivity overlay ──────────────────────────────────────────────
  function drawAllOverlays() {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        for (const sk of state.expanded) {
          if (state.selection.kind && state.selection.segIdx === sk && state.selection.coreIndex != null) {
            drawConnOverlay(sk, state.selection.coreIndex,
              state.selection.kind === 'span' ? state.selection.spanKey : null);
          }
        }
      });
    });
  }

  function drawConnOverlay(segIdx, selCoreIdx, selSpanKey) {
    if (!canvasEl) return;
    // Single canvas-wide overlay, appended LAST so tree-order paints it
    // above every card / connector / buffer. canvasEl has position:relative
    // so the SVG's absolute positioning is in canvas content-space.
    canvasEl.querySelectorAll('.hw-conn-overlay').forEach(n => n.remove());
    canvasEl.querySelectorAll('.hw-buffer-segment-active-in, .hw-buffer-segment-active-out').forEach(n => {
      n.classList.remove('hw-buffer-segment-active-in', 'hw-buffer-segment-active-out');
    });

    const stage = hw.stages.find((s, idx) => s.kind === 'neural' && segKey(s, idx) === segIdx);
    if (!stage) return;

    const spans = getConnectivitySpans(stage, selCoreIdx, () => {
      drawConnOverlay(segIdx, selCoreIdx, selSpanKey);
    });
    if (!spans || spans.length === 0) return;
    const incoming = spans.filter(sp => sp.dst_core === selCoreIdx && sp.kind !== 'off');
    const outgoing = spans.filter(sp => sp.src_core === selCoreIdx && sp.kind !== 'off');
    if (incoming.length === 0 && outgoing.length === 0) return;

    // Coordinate origin = canvas's content box (top-left of scrollable area).
    // child.getBoundingClientRect() returns viewport coords; subtract canvas's
    // viewport rect and add canvas scroll offsets to land in content space.
    const canvasViewport = canvasEl.getBoundingClientRect();
    if (canvasViewport.width === 0 || canvasViewport.height === 0) return;
    const sx = canvasEl.scrollLeft;
    const sy = canvasEl.scrollTop;
    const coreByIdx = new Map();
    if (stage.cores) for (const c of stage.cores) coreByIdx.set(c.core_index, c);
    const totalInput = Math.max(stage.input_map?.reduce((s, m) => s + m.size, 0) || 0, 1);
    const totalOutput = Math.max(stage.output_map?.reduce((s, m) => s + m.size, 0) || 0, 1);

    function getRect(el) {
      if (!el) return null;
      const r = el.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) return null;
      const left = r.left - canvasViewport.left + sx;
      const top  = r.top  - canvasViewport.top  + sy;
      return {
        left, top,
        width: r.width, height: r.height,
        right: left + r.width,
        bottom: top + r.height,
      };
    }
    function coreRect(ci) {
      if (ci === -2) return getRect(document.getElementById(`hw-in-track-${segIdx}`));
      if (ci === -3) return getRect(document.getElementById(`hw-out-track-${segIdx}`));
      return getRect(document.getElementById(`hc-${segIdx}-${ci}`));
    }

    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.classList.add('hw-conn-overlay');
    const layoutW = Math.max(1, Math.round(canvasEl.scrollWidth));
    const layoutH = Math.max(1, Math.round(canvasEl.scrollHeight));
    svg.setAttribute('width', layoutW);
    svg.setAttribute('height', layoutH);
    svg.setAttribute('viewBox', `0 0 ${layoutW} ${layoutH}`);
    svg.innerHTML = `<defs>
      <filter id="hw-conn-glow-${segIdx}" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="2.4" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>`;

    const selRect = coreRect(selCoreIdx);
    if (!selRect) return;

    function highlightBufferSegments(trackId, spanStart, spanEnd, isInput) {
      const track = document.getElementById(trackId);
      if (!track) return;
      const segs = track.querySelectorAll('.hw-buffer-segment');
      segs.forEach(seg => {
        const off = parseInt(seg.dataset.offset, 10);
        const end = parseInt(seg.dataset.end, 10);
        if (!isNaN(off) && !isNaN(end) && off < spanEnd && end > spanStart) {
          seg.classList.add(isInput ? 'hw-buffer-segment-active-in' : 'hw-buffer-segment-active-out');
        }
      });
    }
    function bezierMid(x1, y1, cx1, cy1, cx2, cy2, x2, y2) {
      const t = 0.5, mt = 1 - t;
      return {
        x: mt*mt*mt*x1 + 3*mt*mt*t*cx1 + 3*mt*t*t*cx2 + t*t*t*x2,
        y: mt*mt*mt*y1 + 3*mt*mt*t*cy1 + 3*mt*t*t*cy2 + t*t*t*y2,
      };
    }
    function svgArrowAt(mx, my, dx, dy, color, spanId) {
      const len = Math.sqrt(dx * dx + dy * dy) || 1;
      const ux = dx / len, uy = dy / len;
      const sz = 5;
      const px = -uy * sz, py = ux * sz;
      const tri = document.createElementNS(svgNS, 'polygon');
      tri.setAttribute('points',
        `${mx + ux * sz},${my + uy * sz} ${mx + px},${my + py} ${mx - px},${my - py}`);
      tri.setAttribute('fill', color);
      tri.setAttribute('opacity', '0.85');
      if (spanId) tri.setAttribute('data-span-id', spanId);
      return tri;
    }
    const spanElements = {};
    function registerSpanEl(key, el) {
      if (!spanElements[key]) spanElements[key] = [];
      spanElements[key].push(el);
    }

    for (const sp of incoming) {
      const spanKey = `in_${sp.src_core}_${sp.dst_start}_${sp.length}`;
      const isActive = selSpanKey === spanKey;
      const dstCore = coreByIdx.get(sp.dst_core);
      const dstTotalAxons = Math.max(1, dstCore ? dstCore.axons_per_core : 1);
      const dstY0 = selRect.top + (sp.dst_start / dstTotalAxons) * selRect.height;
      const dstY1 = selRect.top + (sp.dst_end / dstTotalAxons) * selRect.height;
      const dstX = selRect.left;

      const hlDst = document.createElementNS(svgNS, 'line');
      hlDst.setAttribute('x1', dstX); hlDst.setAttribute('y1', dstY0);
      hlDst.setAttribute('x2', dstX); hlDst.setAttribute('y2', dstY1);
      hlDst.setAttribute('stroke', '#34d399'); hlDst.setAttribute('stroke-width', isActive ? '4' : '3');
      hlDst.setAttribute('opacity', isActive ? '1' : '0.7'); hlDst.setAttribute('stroke-linecap', 'round');
      hlDst.setAttribute('data-span-id', spanKey);
      svg.appendChild(hlDst); registerSpanEl(spanKey, hlDst);

      let srcMidX, srcMidY;
      if (sp.src_core === -2) {
        const inRect = coreRect(-2);
        if (!inRect) continue;
        const sy0 = inRect.top + (sp.src_start / totalInput) * inRect.height;
        const sy1 = inRect.top + (sp.src_end / totalInput) * inRect.height;
        const sx = inRect.right;
        const hlSrc = document.createElementNS(svgNS, 'line');
        hlSrc.setAttribute('x1', sx); hlSrc.setAttribute('y1', sy0);
        hlSrc.setAttribute('x2', sx); hlSrc.setAttribute('y2', sy1);
        hlSrc.setAttribute('stroke', '#64b5f6'); hlSrc.setAttribute('stroke-width', '3');
        hlSrc.setAttribute('opacity', '0.7'); hlSrc.setAttribute('stroke-linecap', 'round');
        hlSrc.setAttribute('data-span-id', spanKey);
        svg.appendChild(hlSrc); registerSpanEl(spanKey, hlSrc);
        srcMidX = sx; srcMidY = (sy0 + sy1) / 2;
        highlightBufferSegments(`hw-in-track-${segIdx}`, sp.src_start, sp.src_end, true);
      } else {
        const srcRect = coreRect(sp.src_core);
        if (!srcRect) continue;
        const srcCore = coreByIdx.get(sp.src_core);
        const srcTotalNeurons = Math.max(1, srcCore ? srcCore.neurons_per_core : 1);
        const sx0 = srcRect.left + (sp.src_start / srcTotalNeurons) * srcRect.width;
        const sx1 = srcRect.left + (sp.src_end / srcTotalNeurons) * srcRect.width;
        const sy = srcRect.bottom;
        const hlSrc = document.createElementNS(svgNS, 'line');
        hlSrc.setAttribute('x1', sx0); hlSrc.setAttribute('y1', sy);
        hlSrc.setAttribute('x2', sx1); hlSrc.setAttribute('y2', sy);
        hlSrc.setAttribute('stroke', '#34d399'); hlSrc.setAttribute('stroke-width', isActive ? '4' : '3');
        hlSrc.setAttribute('opacity', isActive ? '1' : '0.7'); hlSrc.setAttribute('stroke-linecap', 'round');
        hlSrc.setAttribute('data-span-id', spanKey);
        svg.appendChild(hlSrc); registerSpanEl(spanKey, hlSrc);
        srcMidX = (sx0 + sx1) / 2; srcMidY = sy;
      }
      const dstMidY = (dstY0 + dstY1) / 2;
      const dxAbs = Math.abs(dstX - srcMidX);
      const cpOff = Math.max(40, dxAbs * 0.5);
      const cpx1 = srcMidX, cpy1 = srcMidY + cpOff * 0.4;
      const cpx2 = dstX - cpOff, cpy2 = dstMidY;
      const path = document.createElementNS(svgNS, 'path');
      path.setAttribute('d', `M${srcMidX},${srcMidY} C${cpx1},${cpy1} ${cpx2},${cpy2} ${dstX},${dstMidY}`);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', '#34d399');
      path.setAttribute('stroke-width', isActive ? '2.5' : '1.5');
      path.setAttribute('opacity', isActive ? '0.95' : '0.45');
      path.setAttribute('stroke-linecap', 'round');
      if (isActive) path.setAttribute('filter', `url(#hw-conn-glow-${segIdx})`);
      path.setAttribute('data-span-id', spanKey);
      path.setAttribute('data-action', 'select-span');
      path.setAttribute('data-seg', String(segIdx));
      path.setAttribute('data-key', spanKey);
      path.style.cursor = 'pointer';
      path.style.pointerEvents = 'auto';
      svg.appendChild(path); registerSpanEl(spanKey, path);
      const mid = bezierMid(srcMidX, srcMidY, cpx1, cpy1, cpx2, cpy2, dstX, dstMidY);
      const arrow = svgArrowAt(mid.x, mid.y, dstX - srcMidX, dstMidY - srcMidY, '#34d399', spanKey);
      svg.appendChild(arrow); registerSpanEl(spanKey, arrow);
    }

    for (const sp of outgoing) {
      const spanKey = `out_${sp.dst_core}_${sp.src_start}_${sp.length}`;
      const isActive = selSpanKey === spanKey;
      const srcCore = coreByIdx.get(sp.src_core);
      const srcTotalNeurons = Math.max(1, srcCore ? srcCore.neurons_per_core : 1);
      const srcX0 = selRect.left + (sp.src_start / srcTotalNeurons) * selRect.width;
      const srcX1 = selRect.left + (sp.src_end / srcTotalNeurons) * selRect.width;
      const srcY = selRect.bottom;
      const srcMidX = (srcX0 + srcX1) / 2;
      const srcMidY = srcY;

      const hlSrc = document.createElementNS(svgNS, 'line');
      hlSrc.setAttribute('x1', srcX0); hlSrc.setAttribute('y1', srcY);
      hlSrc.setAttribute('x2', srcX1); hlSrc.setAttribute('y2', srcY);
      hlSrc.setAttribute('stroke', '#fb923c'); hlSrc.setAttribute('stroke-width', isActive ? '4' : '3');
      hlSrc.setAttribute('opacity', isActive ? '1' : '0.75'); hlSrc.setAttribute('stroke-linecap', 'round');
      hlSrc.setAttribute('data-span-id', spanKey);
      svg.appendChild(hlSrc); registerSpanEl(spanKey, hlSrc);

      let dstMidX, dstMidY;
      if (sp.dst_core === -3) {
        const outRect = coreRect(-3);
        if (!outRect) continue;
        const dx0 = outRect.left + (sp.dst_start / totalOutput) * outRect.width;
        const dx1 = outRect.left + (sp.dst_end / totalOutput) * outRect.width;
        const dy = outRect.top;
        const hlDst = document.createElementNS(svgNS, 'line');
        hlDst.setAttribute('x1', dx0); hlDst.setAttribute('y1', dy);
        hlDst.setAttribute('x2', dx1); hlDst.setAttribute('y2', dy);
        hlDst.setAttribute('stroke', '#ce93d8'); hlDst.setAttribute('stroke-width', '3');
        hlDst.setAttribute('opacity', '0.7'); hlDst.setAttribute('stroke-linecap', 'round');
        hlDst.setAttribute('data-span-id', spanKey);
        svg.appendChild(hlDst); registerSpanEl(spanKey, hlDst);
        dstMidX = (dx0 + dx1) / 2; dstMidY = dy;
        highlightBufferSegments(`hw-out-track-${segIdx}`, sp.dst_start, sp.dst_end, false);
      } else {
        const dstRect = coreRect(sp.dst_core);
        if (!dstRect) continue;
        const dstCore = coreByIdx.get(sp.dst_core);
        const dstTotalAxons = Math.max(1, dstCore ? dstCore.axons_per_core : 1);
        const dstXPos = dstRect.left;
        const dstY0 = dstRect.top + (sp.dst_start / dstTotalAxons) * dstRect.height;
        const dstY1 = dstRect.top + (sp.dst_end / dstTotalAxons) * dstRect.height;
        const hlDst = document.createElementNS(svgNS, 'line');
        hlDst.setAttribute('x1', dstXPos); hlDst.setAttribute('y1', dstY0);
        hlDst.setAttribute('x2', dstXPos); hlDst.setAttribute('y2', dstY1);
        hlDst.setAttribute('stroke', '#fb923c'); hlDst.setAttribute('stroke-width', isActive ? '4' : '3');
        hlDst.setAttribute('opacity', isActive ? '1' : '0.75'); hlDst.setAttribute('stroke-linecap', 'round');
        hlDst.setAttribute('data-span-id', spanKey);
        svg.appendChild(hlDst); registerSpanEl(spanKey, hlDst);
        dstMidX = dstXPos; dstMidY = (dstY0 + dstY1) / 2;
      }

      const dyAbs = Math.abs(dstMidY - srcMidY) + Math.abs(dstMidX - srcMidX);
      const cpOff = Math.max(36, dyAbs * 0.35);
      const cpx1 = srcMidX, cpy1 = srcMidY + cpOff;
      const cpx2 = dstMidX - cpOff * 0.6, cpy2 = dstMidY;
      const path = document.createElementNS(svgNS, 'path');
      path.setAttribute('d', `M${srcMidX},${srcMidY} C${cpx1},${cpy1} ${cpx2},${cpy2} ${dstMidX},${dstMidY}`);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', '#fb923c');
      path.setAttribute('stroke-width', isActive ? '2.5' : '1.5');
      path.setAttribute('opacity', isActive ? '0.95' : '0.5');
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('stroke-dasharray', '6,3');
      path.classList.add('hw-conn-flow');
      if (isActive) path.setAttribute('filter', `url(#hw-conn-glow-${segIdx})`);
      path.setAttribute('data-span-id', spanKey);
      path.setAttribute('data-action', 'select-span');
      path.setAttribute('data-seg', String(segIdx));
      path.setAttribute('data-key', spanKey);
      path.style.cursor = 'pointer';
      path.style.pointerEvents = 'auto';
      svg.appendChild(path); registerSpanEl(spanKey, path);
      const mid = bezierMid(srcMidX, srcMidY, cpx1, cpy1, cpx2, cpy2, dstMidX, dstMidY);
      const arrow = svgArrowAt(mid.x, mid.y, dstMidX - srcMidX, dstMidY - srcMidY, '#fb923c', spanKey);
      svg.appendChild(arrow); registerSpanEl(spanKey, arrow);
    }

    for (const [key, elements] of Object.entries(spanElements)) {
      for (const el of elements) {
        if (el.tagName === 'path' || el.tagName === 'line') {
          el.addEventListener('mouseenter', () => {
            for (const sib of elements) {
              sib.setAttribute('opacity', '1');
              if (sib.tagName === 'line') sib.setAttribute('stroke-width', '5');
            }
          });
          el.addEventListener('mouseleave', () => {
            const isAct = selSpanKey === key;
            for (const sib of elements) {
              if (sib.tagName === 'path') {
                sib.setAttribute('opacity', isAct ? '0.95' : '0.45');
                sib.setAttribute('stroke-width', isAct ? '2.5' : '1.5');
              } else if (sib.tagName === 'line') {
                sib.setAttribute('opacity', isAct ? '1' : '0.7');
                sib.setAttribute('stroke-width', isAct ? '4' : '3');
              } else if (sib.tagName === 'polygon') {
                sib.setAttribute('opacity', '0.85');
              }
            }
          });
        }
      }
    }
    canvasEl.appendChild(svg);
  }

  // ── Span resolver ─────────────────────────────────────────────────────
  function findSpanByKey(segIdx, coreIdx, spanKey) {
    const stage = hw.stages.find((s, idx) => s.kind === 'neural' && segKey(s, idx) === segIdx);
    if (!stage) return null;
    const spans = getConnectivitySpans(stage, coreIdx, () => render());
    if (!spans) return null;
    for (const sp of spans) {
      if (sp.dst_core === coreIdx && sp.kind !== 'off' && `in_${sp.src_core}_${sp.dst_start}_${sp.length}` === spanKey) return { ...sp, dir: 'in' };
      if (sp.src_core === coreIdx && sp.kind !== 'off' && `out_${sp.dst_core}_${sp.src_start}_${sp.length}` === spanKey) return { ...sp, dir: 'out' };
    }
    return null;
  }

  // ── Delegated event dispatch ──────────────────────────────────────────
  function dispatch(action, target) {
    switch (action) {
      case 'toggle-stage': {
        const raw = target.dataset.stage;
        const sk = isNaN(parseInt(raw, 10)) ? raw : parseInt(raw, 10);
        if (state.expanded.has(sk)) state.expanded.delete(sk);
        else state.expanded.add(sk);
        if (!state.expanded.has(sk) && state.selection.segIdx === sk) clearSelection();
        if (typeof sk === 'number') state.focusedSegIdx = sk;
        else if (typeof sk === 'string' && sk.startsWith('cg_')) {
          state.focusedSegIdx = `s_${parseInt(sk.substring(3), 10)}`;
        }
        render();
        break;
      }
      case 'select-core': {
        const segIdx = parseInt(target.dataset.seg, 10);
        const coreIdx = parseInt(target.dataset.core, 10);
        state.expanded.add(segIdx);
        const same = state.selection.kind === 'core' && state.selection.segIdx === segIdx && state.selection.coreIndex === coreIdx;
        if (same) {
          clearSelection();
        } else {
          state.selection = {
            kind: 'core', segIdx, coreIndex: coreIdx,
            softNodeId: null, softOrigin: null,
            spanKey: null, spanData: null, buffer: null,
          };
          state.inspectorTab = 'core';
          state.focusedSegIdx = segIdx;
        }
        render();
        break;
      }
      case 'select-soft': {
        const segIdx = parseInt(target.dataset.seg, 10);
        const coreIdx = parseInt(target.dataset.core, 10);
        const plIdx = target.dataset.pl != null ? parseInt(target.dataset.pl, 10) : null;
        const nodeId = parseInt(target.dataset.node, 10);
        let origin = null;
        if (!isNaN(coreIdx) && plIdx != null && !isNaN(plIdx)) {
          const stage = hw.stages.find((s, idx) => s.kind === 'neural' && segKey(s, idx) === segIdx);
          const core = stage?.cores?.find(c => c.core_index === coreIdx);
          const pl = core?.mapped_placements?.[plIdx];
          if (pl) origin = { segIdx, coreIndex: coreIdx, placement: pl };
        }
        state.selection = {
          ...state.selection,
          kind: 'soft', segIdx, coreIndex: coreIdx,
          softNodeId: nodeId, softOrigin: origin,
          spanKey: null, spanData: null, buffer: null,
        };
        state.inspectorTab = 'soft';
        state.focusedSegIdx = segIdx;
        render();
        break;
      }
      case 'select-soft-from-buffer': {
        const nodeId = parseInt(target.dataset.node, 10);
        state.selection = {
          ...state.selection,
          kind: 'soft', softNodeId: nodeId, softOrigin: null,
        };
        state.inspectorTab = 'soft';
        renderInspector();
        break;
      }
      case 'select-span': {
        const segIdx = parseInt(target.dataset.seg, 10);
        const key = target.dataset.key;
        if (state.selection.spanKey === key) {
          state.selection = { ...state.selection, kind: 'core', spanKey: null, spanData: null };
          state.inspectorTab = 'core';
        } else {
          const sp = findSpanByKey(segIdx, state.selection.coreIndex, key);
          state.selection = {
            ...state.selection, kind: 'span',
            spanKey: key, spanData: sp,
          };
          state.inspectorTab = 'span';
        }
        render();
        break;
      }
      case 'select-buffer': {
        const segIdx = parseInt(target.dataset.seg, 10);
        const bmIdx = parseInt(target.dataset.bmIdx, 10);
        const dir = target.dataset.dir;
        state.selection = {
          ...state.selection,
          kind: 'buffer',
          segIdx,
          buffer: { segIdx, dir, bmIdx },
          spanKey: null, spanData: null,
        };
        state.inspectorTab = 'buffer';
        renderInspector();
        break;
      }
      case 'tab-switch': {
        state.inspectorTab = target.dataset.tab;
        renderInspector();
        break;
      }
      case 'inspector-pin': {
        state.inspectorPinned = !state.inspectorPinned;
        renderInspector();
        break;
      }
      case 'inspector-close': {
        state.inspectorPinned = false;
        clearSelection();
        renderInspector();
        renderCanvas();
        renderRail();
        renderBreadcrumb();
        break;
      }
      case 'rail-jump': {
        const si = parseInt(target.dataset.stageSi, 10);
        if (isNaN(si)) break;
        const stage = hw.stages[si];
        if (!stage) break;
        if (stage.kind === 'neural') {
          const sk = segKey(stage, si);
          state.expanded.add(sk);
          state.focusedSegIdx = sk;
        } else {
          state.focusedSegIdx = `s_${si}`;
        }
        render();
        requestAnimationFrame(() => {
          const card = canvasEl.querySelectorAll('.hw-stage-card')[si];
          if (card) card.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
        break;
      }
    }
  }

  function clearSelection() {
    state.selection = {
      kind: null, segIdx: null, coreIndex: null,
      softNodeId: null, softOrigin: null,
      spanKey: null, spanData: null, buffer: null,
    };
  }

  root.addEventListener('click', (ev) => {
    const t = ev.target.closest('[data-action]');
    if (!t || !root.contains(t)) return;
    ev.stopPropagation();
    dispatch(t.dataset.action, t);
  });

  // Click-outside the inspector closes it (when not pinned).
  document.addEventListener('click', (ev) => {
    if (!state.selection.kind || state.inspectorPinned) return;
    if (inspectorEl.contains(ev.target)) return;
    if (root.querySelector('.hw-canvas')?.contains(ev.target)) return;
    if (railEl.contains(ev.target)) return;
    // Outside everything in the workbench → close.
    clearSelection();
    renderInspector();
    renderCanvas();
    renderRail();
    renderBreadcrumb();
  });

  // ── Scroll-driven focus tracking for breadcrumb / rail "you are here" ─
  const scrollObserver = new IntersectionObserver((entries) => {
    let topMost = null;
    let topMostY = Infinity;
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const r = e.boundingClientRect.top;
      if (r < topMostY) { topMostY = r; topMost = e.target; }
    }
    if (!topMost) return;
    const si = parseInt(topMost.dataset.stageSi, 10);
    if (isNaN(si)) return;
    const stage = hw.stages[si];
    const newFocus = stage.kind === 'neural' ? segKey(stage, si) : `s_${si}`;
    if (state.focusedSegIdx !== newFocus) {
      state.focusedSegIdx = newFocus;
      renderRail();
      renderBreadcrumb();
    }
  }, { root: null, rootMargin: '-100px 0px -60% 0px', threshold: 0 });

  // ── Top-level render ──────────────────────────────────────────────────
  function render() {
    renderRail();
    renderCanvas();
    renderInspector();
    renderBreadcrumb();
  }

  // ── Resize → re-render canvas when the centre column width changes ──
  // Required for the dynamic SEG_VIEW_WIDTH math: when the inspector
  // toggles open/closed, or the viewport resizes, the canvas width
  // changes and the per-segment grid should re-pack.
  let _lastCanvasW = 0;
  let _resizeRaf = 0;
  const resizeObserver = new ResizeObserver(() => {
    const w = canvasEl?.clientWidth || 0;
    if (Math.abs(w - _lastCanvasW) < 16) return;
    _lastCanvasW = w;
    if (_resizeRaf) cancelAnimationFrame(_resizeRaf);
    _resizeRaf = requestAnimationFrame(() => {
      _resizeRaf = 0;
      renderCanvas();
    });
  });
  resizeObserver.observe(canvasEl);

  // ── Boot ──────────────────────────────────────────────────────────────
  applyTrace();
  const firstNeural = hw.stages.findIndex(s => s.kind === 'neural');
  if (firstNeural >= 0) {
    state.focusedSegIdx = segKey(hw.stages[firstNeural], firstNeural);
  } else if (hw.stages.length > 0) {
    state.focusedSegIdx = `s_0`;
  }
  render();
}
