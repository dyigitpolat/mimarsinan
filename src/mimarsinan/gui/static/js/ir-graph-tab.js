/* IR Graph tab — hierarchical zoomable DAG visualization.
 *
 * Groups are ordered by topological position so that ComputeOps appear
 * at their correct location in the data flow (not appended at the end).
 *
 * 3-level hierarchy:
 *   Level 0: Tiers (collapsed) — consecutive same-latency neural groups
 *   Level 1: Groups within expanded tiers
 *   Level 2: Detail panel for a selected group
 *
 * Features: ctrl+wheel zoom, scroll both axes, auto-expand small tiers,
 * group-level edges when tiers are expanded, clickable nodes. */
import { esc, safeReact, plotHistogram } from './util.js';

window._irGraphState = window._irGraphState || { selectedTier: null, detailGroup: null, selectedEdge: null };

export function renderIRGraphTab(irGraph, container) {
  if (!irGraph) { container.innerHTML = '<div class="empty-state">No IR graph data</div>'; return; }

  // Only rewrite the outer HTML if the container is empty, to preserve scroll/zoom state
  if (!container.querySelector('.ir-dag-viewport')) {
    let html = `
      <div class="grid-3" style="margin-bottom:20px">
        <div class="card"><div class="big-metric"><div class="value" id="ir-val-cores">0</div><div class="label">Neural Cores</div></div></div>
        <div class="card"><div class="big-metric"><div class="value" id="ir-val-ops">0</div><div class="label">Compute Ops</div></div></div>
        <div class="card"><div class="big-metric"><div class="value" id="ir-val-lat">0</div><div class="label">Max Latency</div></div></div>
      </div>
      <div class="card" style="margin-bottom:20px">
        <div class="card-header">Data Flow Graph
          <div class="ir-zoom-controls" id="ir-zoom-bar">
            <button onclick="window._irZoom(-0.15)">−</button>
            <span id="ir-zoom-label">100%</span>
            <button onclick="window._irZoom(0.15)">+</button>
            <button onclick="window._irZoom(0)" style="margin-left:6px;font-size:9px">Reset</button>
          </div>
        </div>
        <div class="card-body no-pad">
          <div class="ir-dag-viewport" id="ir-viewport">
            <div class="ir-dag-canvas" id="ir-canvas"><div id="ir-dag-root"></div></div>
          </div>
        </div>
      </div>
      <div id="ir-detail-host"></div>
      <div class="grid-2">
        <div class="card"><div class="card-header">Core Dimensions (Axons × Neurons)</div><div class="card-body"><div id="ir-dims" style="min-height:220px"></div></div></div>
        <div class="card"><div class="card-header">Threshold Distribution</div><div class="card-body"><div id="ir-thresh" style="min-height:220px"></div></div></div>
      </div>
      <div class="grid-2">
        <div class="card"><div class="card-header">Latency Distribution</div><div class="card-body"><div id="ir-latency" style="min-height:220px"></div></div></div>
        <div class="card"><div class="card-header">Weight Sparsity per Core</div><div class="card-body"><div id="ir-sparsity" style="min-height:220px"></div></div></div>
      </div>`;
    container.innerHTML = html;
    setupZoom();
  }

  // Update metrics
  document.getElementById('ir-val-cores').textContent = irGraph.num_neural_cores;
  document.getElementById('ir-val-ops').textContent = irGraph.num_compute_ops;
  document.getElementById('ir-val-lat').textContent = irGraph.max_latency;

  renderTierDAG(irGraph);
  renderStatCharts(irGraph);
}

// ── Zoom support ─────────────────────────────────────────────────────────
function setupZoom() {
  const vp = document.getElementById('ir-viewport');
  const canvas = document.getElementById('ir-canvas');
  if (!vp || !canvas) return;
  let zoom = 1;
  const apply = () => {
    canvas.style.transform = `scale(${zoom})`;
    const label = document.getElementById('ir-zoom-label');
    if (label) label.textContent = Math.round(zoom * 100) + '%';
  };
  vp.addEventListener('wheel', (e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      zoom = Math.max(0.2, Math.min(3, zoom + (e.deltaY > 0 ? -0.1 : 0.1)));
      apply();
    }
  }, { passive: false });
  window._irZoom = (delta) => {
    if (delta === 0) zoom = 1; else zoom = Math.max(0.2, Math.min(3, zoom + delta));
    apply();
  };
}

// ── Tier computation ─────────────────────────────────────────────────────
// Groups arrive sorted by topological order from the backend.
// We merge consecutive groups that share the same latency into one tier,
// while compute-only groups become their own tier entry.
function buildTiers(groups) {
  const tiers = [];
  let currentTier = null;
  // Do NOT split same-latency groups into multiple tiers; keep one tier per latency.

  for (const g of groups) {
    let lat;
    if (g.type === 'virtual') {
      lat = (g.key === 'input' || g.key === 'const1') ? '__input__' : '__output__';
    } else if (g.type === 'neural' && g.latency_range) {
      lat = 'lat_' + g.latency_range[0];
    } else {
      lat = '__op_' + g.order;
    }

    if (currentTier && currentTier._latKey === lat) {
      currentTier.groups.push(g);
    } else {
      currentTier = {
        idx: tiers.length, _latKey: lat, groups: [g],
        latency: g.latency_range ? g.latency_range[0] : null,
      };
      tiers.push(currentTier);
    }
  }

  for (const t of tiers) {
    t.totalCores = t.groups.reduce((s, g) => s + (g.num_cores || 0), 0);
    t.totalOps = t.groups.reduce((s, g) => s + (g.num_ops || 0), 0);
    t.isVirtual = t.groups.every(g => g.type === 'virtual');
    t.isCompute = !t.isVirtual && t.totalCores === 0 && t.totalOps > 0;
    t.opTypes = [...new Set(t.groups.flatMap(g => g.op_types || []))];
  }

  const g2t = new Map();
  for (const t of tiers) for (const g of t.groups) g2t.set(g.key, t.idx);
  return { tiers, g2t };
}

// ── Color palette (PlotNeuralNet / NN-SVG inspired) ──────────────────────
const BLOCK_COLORS = {
  neural:  { front: '#4a7cf5', top: '#6b9cff', side: '#3560c8', stroke: '#2a4a9e' },
  compute: { front: '#9c5bdb', top: '#b87ae8', side: '#7a3cb8', stroke: '#5a2d8a' },
  virtual: { front: '#4caf50', top: '#6ec972', side: '#358a38', stroke: '#1e6e22' },
};

function blockColors(type) { return BLOCK_COLORS[type] || BLOCK_COLORS.neural; }

// ── 3D block SVG helper (NN-SVG LeNet stacked-rectangle style) ───────────
function svgBlock3D(x, y, w, h, dx, dy, c, opacity) {
  const o = opacity ?? 1;
  const ow = `opacity="${o}"`;
  let s = '';
  // Right face
  s += `<polygon points="${x+w},${y} ${x+w+dx},${y-dy} ${x+w+dx},${y+h-dy} ${x+w},${y+h}" fill="${c.side}" stroke="${c.stroke}" stroke-width="0.6" ${ow}/>`;
  // Top face
  s += `<polygon points="${x},${y} ${x+dx},${y-dy} ${x+w+dx},${y-dy} ${x+w},${y}" fill="${c.top}" stroke="${c.stroke}" stroke-width="0.6" ${ow}/>`;
  // Front face
  s += `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${c.front}" stroke="${c.stroke}" stroke-width="0.8" ${ow}/>`;
  return s;
}

function svgStackedBlock(x, y, w, h, stackCount, stackOff, dx, dy, c, highlight) {
  let s = '';
  for (let i = stackCount - 1; i >= 0; i--) {
    const bx = x + i * stackOff;
    const by = y - i * stackOff;
    const op = i === 0 ? (highlight ? 1 : 0.92) : Math.max(0.35, 0.7 - i * 0.08);
    s += svgBlock3D(bx, by, w, h, dx, dy, c, op);
  }
  return s;
}

// ── DAG renderer (PlotNeuralNet / NN-SVG style, left-to-right) ───────────
// All tiers are always shown as collapsed blocks in the SVG.
// Clicking a tier, group, or edge opens a detail panel below the graph.
function renderTierDAG(irGraph) {
  const root = document.getElementById('ir-dag-root');
  const detailHost = document.getElementById('ir-detail-host');
  if (!root) return;

  const groups = irGraph.groups || [];
  const groupEdges = irGraph.group_edges || [];
  const nodeById = {};
  for (const n of irGraph.nodes) nodeById[n.id] = n;
  if (groups.length === 0) { root.innerHTML = '<div class="empty-state">No topology data</div>'; return; }

  const { tiers, g2t } = buildTiers(groups);
  const st = window._irGraphState;
  if (st.selectedTier != null && (st.selectedTier < 0 || st.selectedTier >= tiers.length)) st.selectedTier = null;
  if (st.detailGroup && !groups.some(g => g.key === st.detailGroup)) st.detailGroup = null;

  // Compute global dimension range for proportional block sizing
  let gMaxAx = 1, gMaxN = 1;
  for (const g of groups) {
    if (g.axon_range) gMaxAx = Math.max(gMaxAx, g.axon_range[1] || g.axon_range[0]);
    if (g.neuron_range) gMaxN = Math.max(gMaxN, g.neuron_range[1] || g.neuron_range[0]);
  }

  function tierDims(tier) {
    const MIN_W = 44, MAX_W = 110, MIN_H = 36, MAX_H = 96;
    if (tier.isCompute) return { w: 36, h: 28 };
    if (tier.isVirtual) return { w: 44, h: 36 };
    let w = MIN_W, h = MIN_H;
    for (const g of tier.groups) {
      if (g.neuron_range) w = Math.max(w, MIN_W + ((g.neuron_range[1] || g.neuron_range[0]) / gMaxN) * (MAX_W - MIN_W));
      if (g.axon_range) h = Math.max(h, MIN_H + ((g.axon_range[1] || g.axon_range[0]) / gMaxAx) * (MAX_H - MIN_H));
    }
    return { w: Math.round(w), h: Math.round(h) };
  }

  function render() {
    const DX = 7, DY = 5, STACK_OFF = 3;
    const PAD = 36, TIER_GAP = 54, LABEL_H = 22;

    // ── 1. Layout: always collapsed tier blocks, left-to-right ──
    const tierCols = [];
    let curX = PAD;

    for (const tier of tiers) {
      const { w, h } = tierDims(tier);
      const sc = tier.isCompute || tier.isVirtual ? 1 : Math.min(tier.groups.length, 6);
      const fullW = w + (sc - 1) * STACK_OFF + DX;
      const fullH = h + (sc - 1) * STACK_OFF + DY;
      tierCols.push({ tier, x: curX, w, h, sc, fullW, fullH, colW: fullW, colH: fullH + LABEL_H });
      curX += fullW + TIER_GAP;
    }

    const svgW = curX + PAD;
    const maxColH = Math.max(...tierCols.map(tc => tc.colH));
    const svgH = maxColH + PAD * 2;

    for (const tc of tierCols) {
      const topY = PAD + (maxColH - tc.colH) / 2;
      tc.bx = tc.x + (tc.sc - 1) * STACK_OFF;
      tc.by = topY + LABEL_H + (tc.sc - 1) * STACK_OFF + DY;
      tc.labelY = topY + 14;
    }

    // Position map for edges
    const tierPos = new Map();
    for (const tc of tierCols) {
      const cx = tc.bx + tc.w / 2, cy = tc.by + tc.h / 2;
      tierPos.set(tc.tier.idx, { cx, cy, right: tc.bx + tc.w + DX, left: tc.bx, top: tc.by, bottom: tc.by + tc.h });
    }

    // ── 2. Aggregate tier-level edges ──
    const tierEdgeMap = new Map();
    for (const e of groupEdges) {
      const ft = g2t.get(e.from), tt = g2t.get(e.to);
      if (ft == null || tt == null || ft === tt) continue;
      const key = `${ft}->${tt}`;
      const prev = tierEdgeMap.get(key);
      if (prev) prev.count += (e.count || 1);
      else tierEdgeMap.set(key, { fromTier: ft, toTier: tt, count: (e.count || 1) });
    }
    let tierEdges = [...tierEdgeMap.values()];
    tierEdges.sort((a, b) => b.count - a.count);
    if (tierEdges.length > 200) tierEdges = tierEdges.slice(0, 200);

    const edgeAttr = (from, to, count) => {
      const f = String(from).replace(/"/g, '&quot;');
      const t = String(to).replace(/"/g, '&quot;');
      return `class="ir-edge" data-edge-from="${f}" data-edge-to="${t}" data-edge-count="${count || 1}"`;
    };
    const isEdgeSel = (from, to) => st.selectedEdge && st.selectedEdge.from === from && st.selectedEdge.to === to;

    // ── 3. Build SVG ──
    let svg = `<svg width="${svgW}" height="${svgH}" xmlns="http://www.w3.org/2000/svg" style="font-family:'Inter',sans-serif">`;
    svg += `<defs>
      <marker id="arr-fwd" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#8a8d9a" opacity="0.7"/></marker>
      <marker id="arr-skip" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#ff9800" opacity="0.7"/></marker>
      <marker id="arr-sel" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#fff"/></marker>
      <filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    </defs>`;

    // Layer 1: Edges
    svg += '<g class="layer-edges">';
    const hBezier = (fp, tp) => {
      const x1 = fp.right, y1 = fp.cy, x2 = tp.left, y2 = tp.cy;
      const cpOff = Math.max(20, Math.abs(x2 - x1) * 0.4);
      return `M${x1},${y1} C${x1 + cpOff},${y1} ${x2 - cpOff},${y2} ${x2},${y2}`;
    };

    for (const e of tierEdges) {
      const fp = tierPos.get(e.fromTier), tp = tierPos.get(e.toTier);
      if (!fp || !tp) continue;
      const skip = Math.abs(e.toTier - e.fromTier) > 1;
      const fromKey = `tier:${e.fromTier}`;
      const toKey = `tier:${e.toTier}`;
      const sel = isEdgeSel(fromKey, toKey);
      const involvesSel = st.selectedTier != null && (e.fromTier === st.selectedTier || e.toTier === st.selectedTier);
      const color = sel ? '#fff' : involvesSel ? 'rgba(91,138,245,0.7)' : (skip ? 'rgba(255,152,0,0.55)' : 'rgba(138,141,154,0.45)');
      const dash = skip ? ' stroke-dasharray="6,3"' : '';
      const marker = sel ? 'url(#arr-sel)' : (skip ? 'url(#arr-skip)' : 'url(#arr-fwd)');
      const sw = sel ? 3 : involvesSel ? 2.5 : Math.min(3.5, 1.0 + Math.log10((e.count || 1) + 1) * 1.2);
      const d = hBezier(fp, tp);
      svg += `<path ${edgeAttr(fromKey, toKey, e.count)} d="${d}" fill="none" stroke="${color}" stroke-width="${sw.toFixed(1)}"${dash} marker-end="${marker}"${sel ? ' filter="url(#glow)"' : ''}><title>${e.count} connection${e.count !== 1 ? 's' : ''}</title></path>`;
    }
    svg += '</g>';

    // Layer 2: Tier blocks
    svg += '<g class="layer-blocks">';
    for (const tc of tierCols) {
      const tier = tc.tier;
      const c = blockColors(tier.isCompute ? 'compute' : (tier.isVirtual ? 'virtual' : 'neural'));
      const isSel = st.selectedTier === tier.idx;
      const canExpand = !tier.isVirtual;

      svg += `<g data-tier-idx="${tier.idx}" data-tier-expandable="${canExpand ? '1' : '0'}" style="cursor:${canExpand ? 'pointer' : 'default'}"${isSel ? ' filter="url(#glow)"' : ''}>`;
      svg += svgStackedBlock(tc.bx, tc.by, tc.w, tc.h, tc.sc, STACK_OFF, DX, DY, c, isSel);

      if (isSel) {
        svg += `<rect x="${tc.bx - 2}" y="${tc.by - 2}" width="${tc.w + 4}" height="${tc.h + 4}" rx="2" fill="none" stroke="#fff" stroke-width="2" opacity="0.8"/>`;
      }

      const cx = tc.bx + tc.w / 2, fy = tc.by;
      if (tier.isVirtual) {
        const label = tier._latKey === '__input__' ? 'INPUT' : 'OUTPUT';
        svg += `<text x="${cx}" y="${fy + tc.h / 2 + 4}" text-anchor="middle" fill="#fff" font-size="11" font-weight="700">${label}</text>`;
      } else if (tier.isCompute) {
        const opLabel = tier.opTypes.slice(0, 2).join(', ') || 'op';
        svg += `<text x="${cx}" y="${fy + tc.h / 2 + 3}" text-anchor="middle" fill="#fff" font-size="9" font-weight="500">${esc(opLabel.substring(0, 20))}</text>`;
      } else {
        svg += `<text x="${cx}" y="${fy + Math.min(16, tc.h * 0.38)}" text-anchor="middle" fill="#fff" font-size="10" font-weight="700">Lat ${tier.latency ?? '?'}</text>`;
        const sub = `${tier.totalCores} core${tier.totalCores !== 1 ? 's' : ''}`;
        if (tc.h >= 42) {
          svg += `<text x="${cx}" y="${fy + Math.min(30, tc.h * 0.7)}" text-anchor="middle" fill="rgba(255,255,255,0.65)" font-size="8">${tier.groups.length} grp · ${sub}</text>`;
        }
      }
      svg += '</g>';

      // Column label
      const labelX = tc.x + tc.colW / 2;
      if (!tier.isVirtual) {
        const labelText = tier.isCompute ? (tier.opTypes[0] || 'compute') : `Latency ${tier.latency ?? '?'}`;
        svg += `<text x="${labelX}" y="${tc.labelY}" text-anchor="middle" fill="${isSel ? '#5b8af5' : '#6b6e7a'}" font-size="9" font-weight="600">${esc(labelText)}</text>`;
      }
    }
    svg += '</g>';
    svg += '</svg>';
    root.innerHTML = svg;

    // ── Event handlers ──
    root.querySelectorAll('g[data-tier-idx][data-tier-expandable="1"]').forEach(el => {
      el.addEventListener('click', (ev) => {
        ev.stopPropagation();
        const idx = Number(el.getAttribute('data-tier-idx'));
        if (!Number.isFinite(idx)) return;
        st.selectedTier = (st.selectedTier === idx) ? null : idx;
        st.detailGroup = null;
        st.selectedEdge = null;
        render();
      });
    });

    root.querySelectorAll('.ir-edge').forEach(path => {
      path.addEventListener('click', (ev) => {
        ev.stopPropagation();
        const from = path.getAttribute('data-edge-from');
        const to = path.getAttribute('data-edge-to');
        const count = path.getAttribute('data-edge-count') || '1';
        st.selectedEdge = (st.selectedEdge && st.selectedEdge.from === from && st.selectedEdge.to === to)
          ? null : { from, to, count: parseInt(count, 10) };
        st.selectedTier = null;
        st.detailGroup = null;
        render();
      });
    });

    // ── Detail panel: stack tier + group details when both are selected ──
    if (detailHost) {
      let detailHtml = '';
      if (st.selectedTier != null) {
        const tier = tiers[st.selectedTier];
        if (tier) detailHtml += buildTierDetail(tier, groups, groupEdges, nodeById, irGraph, g2t, tiers);
      }
      if (st.detailGroup) {
        const g = groups.find(x => x.key === st.detailGroup);
        if (g) detailHtml += buildGroupDetail(g, nodeById, irGraph);
      }
      if (!detailHtml && st.selectedEdge) {
        detailHtml = buildEdgeDetail(st.selectedEdge, groups, nodeById, irGraph, tiers);
      }
      detailHost.innerHTML = detailHtml;
    }
  }

  window._irGroupClick = (key) => { st.detailGroup = (st.detailGroup === key) ? null : key; st.selectedEdge = null; render(); };
  window._irEdgeClose = () => { st.selectedEdge = null; render(); };
  window._irTierClose = () => { st.selectedTier = null; st.detailGroup = null; render(); };
  render();
}

// ── Tier detail panel — shows groups within the selected tier ─────────────
function buildTierDetail(tier, allGroups, groupEdges, nodeById, irGraph, g2t, tiers) {
  const tierGroupKeys = new Set(tier.groups.map(g => g.key));

  // Find inter-tier edges touching this tier
  const incomingTiers = new Set(), outgoingTiers = new Set();
  for (const e of groupEdges) {
    const ft = g2t.get(e.from), tt = g2t.get(e.to);
    if (ft === tier.idx && tt !== tier.idx && tt != null) outgoingTiers.add(tt);
    if (tt === tier.idx && ft !== tier.idx && ft != null) incomingTiers.add(ft);
  }

  const tierLabel = tier.isCompute
    ? `Compute: ${tier.opTypes.join(', ') || '?'}`
    : `Latency ${tier.latency ?? '?'}`;

  let html = `<div class="ir-node-detail ir-tier-detail" style="margin-top:0">
    <div class="ir-node-detail-header">
      <span>${esc(tierLabel)} — ${tier.groups.length} group${tier.groups.length !== 1 ? 's' : ''}, ${tier.totalCores} core${tier.totalCores !== 1 ? 's' : ''}</span>
      <button class="ir-detail-close" onclick="window._irTierClose()">✕</button>
    </div>
    <div style="padding:12px 16px">`;

  // Connectivity summary
  if (incomingTiers.size > 0 || outgoingTiers.size > 0) {
    html += '<div style="margin-bottom:12px">';
    if (incomingTiers.size > 0) {
      const labels = [...incomingTiers].map(ti => {
        const t = tiers[ti];
        return t.isCompute ? (t.opTypes[0] || 'compute') : (t.isVirtual ? (t._latKey === '__input__' ? 'INPUT' : 'OUTPUT') : `Lat ${t.latency}`);
      });
      html += `<div class="conn-row" style="margin-bottom:4px"><span class="conn-label">Inputs from:</span>${labels.map(s => `<span class="conn-chip incoming">${esc(s)}</span>`).join(' ')}</div>`;
    }
    if (outgoingTiers.size > 0) {
      const labels = [...outgoingTiers].map(ti => {
        const t = tiers[ti];
        return t.isCompute ? (t.opTypes[0] || 'compute') : (t.isVirtual ? (t._latKey === '__input__' ? 'INPUT' : 'OUTPUT') : `Lat ${t.latency}`);
      });
      html += `<div class="conn-row"><span class="conn-label">Outputs to:</span>${labels.map(s => `<span class="conn-chip outgoing">${esc(s)}</span>`).join(' ')}</div>`;
    }
    html += '</div>';
  }

  // Group cards — clickable to drill into
  html += '<div class="section-label" style="margin-bottom:8px">Groups (click to expand)</div>';
  html += '<div class="ir-tier-group-list">';
  for (const g of tier.groups) {
    const safeKey = g.key.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
    const coreCount = g.num_cores || 0;
    const opCount = g.num_ops || 0;
    const dims = [];
    if (g.axon_range) dims.push(`${g.axon_range[0]}ax`);
    if (g.neuron_range) dims.push(`${g.neuron_range[0]}n`);
    const opTypes = (g.op_types || []).join(', ');

    const isSel = window._irGraphState.detailGroup === g.key;
    html += `<div class="ir-tier-group-card${isSel ? ' ir-tier-group-card-selected' : ''}" onclick="window._irGroupClick('${safeKey}')">`;
    html += `<div class="ir-tier-group-card-title">${esc(g.key)}</div>`;
    html += '<div class="ir-tier-group-card-meta">';
    if (g.type === 'neural') {
      html += `<span>${coreCount} core${coreCount !== 1 ? 's' : ''}</span>`;
      if (dims.length > 0) html += `<span>${dims.join(' × ')}</span>`;
    }
    if (g.type === 'compute' || opCount > 0) {
      html += `<span>${opCount} op${opCount !== 1 ? 's' : ''}</span>`;
      if (opTypes) html += `<span>${esc(opTypes.substring(0, 30))}</span>`;
    }
    html += '</div></div>';
  }
  html += '</div>';

  html += '</div></div>';
  return html;
}

// ── Group detail panel (Level 2) ─────────────────────────────────────────
function buildGroupDetail(g, nodeById, irGraph) {
  const members = (g.node_ids || []).map(id => nodeById[id]).filter(Boolean);
  const cores = members.filter(n => n.type === 'neural_core');
  const ops = members.filter(n => n.type === 'compute_op');
  const safeKey = g.key.replace(/\\/g, '\\\\').replace(/'/g, "\\'");

  const nodeIds = new Set((g.node_ids || []).map(String));
  const inEdges = (irGraph.edges || []).filter(e => nodeIds.has(String(e.to)) && !nodeIds.has(String(e.from)));
  const outEdges = (irGraph.edges || []).filter(e => nodeIds.has(String(e.from)) && !nodeIds.has(String(e.to)));

  let html = `<div class="ir-node-detail" style="margin-top:0">
    <div class="ir-node-detail-header">
      <span>${esc(g.key)} — ${cores.length} cores, ${ops.length} ops</span>
      <button class="ir-detail-close" onclick="window._irGroupClick('${safeKey}')">✕</button>
    </div><div style="padding:12px 16px">`;

  if (inEdges.length > 0 || outEdges.length > 0) {
    html += '<div style="margin-bottom:12px">';
    if (inEdges.length > 0) {
      const srcNames = [...new Set(inEdges.map(e => {
        if (e.from === 'input') return 'INPUT';
        if (e.from === 'const1') return 'CONST1';
        const n = nodeById[e.from];
        return n ? (n.layer_group || n.name) : `node ${e.from}`;
      }))];
      html += `<div class="conn-row" style="margin-bottom:4px"><span class="conn-label">Inputs from:</span>${srcNames.map(s => `<span class="conn-chip incoming">${esc(s)}</span>`).join(' ')}</div>`;
    }
    if (outEdges.length > 0) {
      const dstNames = [...new Set(outEdges.map(e => {
        if (e.to === 'output') return 'OUTPUT';
        const n = nodeById[e.to];
        return n ? (n.layer_group || n.name) : `node ${e.to}`;
      }))];
      html += `<div class="conn-row"><span class="conn-label">Outputs to:</span>${dstNames.map(s => `<span class="conn-chip outgoing">${esc(s)}</span>`).join(' ')}</div>`;
    }
    html += '</div>';
  }

  if (cores.length > 0) {
    html += '<table class="data-table compact"><thead><tr><th>ID</th><th>Name</th><th>Axons</th><th>Neurons</th><th>Norm</th><th>Activation</th><th>Threshold</th><th>Act Scale</th><th>Param Scale</th><th>InAct Scale</th><th>Latency</th><th>Sparsity</th></tr></thead><tbody>';
    for (const c of cores) {
      const sp = c.weight_stats ? (c.weight_stats.sparsity * 100).toFixed(1) + '%' : '-';
      const short = c.name.length > 30 ? c.name.substring(0, 28) + '..' : c.name;
      const norm = c.normalization_type || '-';
      const act = c.activation_type || '-';
      const normShort = norm.length > 16 ? norm.substring(0, 14) + '..' : norm;
      const actShort = act.length > 22 ? act.substring(0, 20) + '..' : act;
      html += `<tr><td>${c.id}</td><td title="${esc(c.name)}">${esc(short)}</td>
        <td>${c.axons}</td><td>${c.neurons}</td><td title="${esc(norm)}">${esc(normShort)}</td><td title="${esc(act)}">${esc(actShort)}</td><td>${c.threshold.toFixed(2)}</td>
        <td>${c.activation_scale != null ? c.activation_scale.toFixed(3) : '-'}</td>
        <td>${c.parameter_scale != null ? c.parameter_scale.toFixed(3) : '-'}</td>
        <td>${c.input_activation_scale != null ? c.input_activation_scale.toFixed(3) : '-'}</td>
        <td>${c.latency ?? '-'}</td><td>${sp}</td></tr>`;
    }
    html += '</tbody></table>';
  }
  if (ops.length > 0) {
    html += `<div style="margin-top:${cores.length > 0 ? '12px' : '0'}"><table class="data-table compact"><thead><tr><th>ID</th><th>Name</th><th>Type</th><th>Input</th><th>Output</th><th>Params</th></tr></thead><tbody>`;
    for (const o of ops) {
      html += `<tr><td>${o.id}</td><td>${esc(o.name)}</td><td>${esc(o.op_type)}</td>
        <td>${o.input_shape || '-'}</td><td>${o.output_shape || '-'}</td>
        <td title="${esc(o.params || '')}">${esc((o.params || '-').substring(0, 35))}</td></tr>`;
    }
    html += '</tbody></table></div>';
  }
  html += '</div></div>';
  return html;
}

function buildEdgeDetail(edge, groups, nodeById, irGraph, tiers) {
  const describeTierKey = (k) => {
    const m = k.match(/^tier:(\d+)$/);
    if (m && tiers) {
      const ti = parseInt(m[1], 10);
      const t = tiers[ti];
      if (t) {
        const label = t.isCompute ? (t.opTypes[0] || 'compute') : (t.isVirtual ? (t._latKey === '__input__' ? 'INPUT' : 'OUTPUT') : `Latency ${t.latency}`);
        return { label, cores: t.totalCores, ops: t.totalOps, latRange: t.latency != null ? [t.latency, t.latency] : null };
      }
    }
    if (k === 'input') return { label: 'INPUT', cores: 0, ops: 0 };
    if (k === 'const1') return { label: 'CONST1', cores: 0, ops: 0 };
    if (k === 'output') return { label: 'OUTPUT', cores: 0, ops: 0 };
    const g = groups.find(x => x.key === k);
    return g ? { label: esc(k), cores: g.num_cores || 0, ops: g.num_ops || 0, latRange: g.latency_range } : { label: esc(k), cores: 0, ops: 0 };
  };

  const fromDesc = describeTierKey(edge.from);
  const toDesc = describeTierKey(edge.to);

  let html = `<div class="ir-node-detail ir-edge-detail" style="margin-top:0">
    <div class="ir-node-detail-header">
      <span>Edge: ${fromDesc.label} → ${toDesc.label}</span>
      <button class="ir-detail-close" onclick="window._irEdgeClose && window._irEdgeClose()">✕</button>
    </div>
    <div style="padding:12px 16px">
      <table class="data-table compact">
        <tr><td>Source</td><td><span class="conn-chip incoming">${fromDesc.label}</span></td></tr>
        <tr><td>Destination</td><td><span class="conn-chip outgoing">${toDesc.label}</span></td></tr>
        <tr><td>Edge count</td><td>${edge.count}</td></tr>`;

  if (fromDesc.cores > 0) html += `<tr><td>Source cores</td><td>${fromDesc.cores}</td></tr>`;
  if (toDesc.cores > 0) html += `<tr><td>Dest cores</td><td>${toDesc.cores}</td></tr>`;
  if (fromDesc.latRange) html += `<tr><td>Source latency</td><td>${fromDesc.latRange[0]}${fromDesc.latRange[1] !== fromDesc.latRange[0] ? '–' + fromDesc.latRange[1] : ''}</td></tr>`;
  if (toDesc.latRange) html += `<tr><td>Dest latency</td><td>${toDesc.latRange[0]}${toDesc.latRange[1] !== toDesc.latRange[0] ? '–' + toDesc.latRange[1] : ''}</td></tr>`;

  html += '</table></div></div>';
  return html;
}

// ── Supplementary charts ─────────────────────────────────────────────────
function renderStatCharts(irGraph) {
  const cores = irGraph.nodes.filter(n => n.type === 'neural_core');
  if (cores.length === 0) return;

  const maxScatterPoints = 2500;
  const scatterStride = Math.max(1, Math.ceil(cores.length / maxScatterPoints));
  const scatterCores = scatterStride > 1 ? cores.filter((_, i) => i % scatterStride === 0) : cores;

  safeReact('ir-dims', [{
    x: scatterCores.map(c => c.axons), y: scatterCores.map(c => c.neurons),
    text: scatterCores.map(c => `Core ${c.id}: ${c.axons}×${c.neurons}`),
    mode: 'markers', type: 'scattergl',
    marker: { size: 8, color: scatterCores.map(c => c.latency || 0), colorscale: 'Viridis', showscale: true, colorbar: { title: 'Latency', titlefont: { size: 10 }, thickness: 14 } },
  }], { height: 280, xaxis: { title: 'Axons' }, yaxis: { title: 'Neurons' } });

  if (irGraph.threshold_distribution) plotHistogram('ir-thresh', irGraph.threshold_distribution, 'Threshold', '#ff9800');
  if (irGraph.latency_distribution) plotHistogram('ir-latency', irGraph.latency_distribution, 'Latency', '#4caf50');

  const rawSparsities = cores.filter(c => c.weight_stats).map(c => c.weight_stats.sparsity);
  let sparsities = rawSparsities;
  let labels = rawSparsities.map((_, i) => `C${i}`);
  const maxBars = 400;
  if (rawSparsities.length > maxBars) {
    const bucketSize = Math.ceil(rawSparsities.length / maxBars);
    sparsities = [];
    labels = [];
    for (let i = 0; i < rawSparsities.length; i += bucketSize) {
      const bucket = rawSparsities.slice(i, i + bucketSize);
      const mean = bucket.reduce((a, b) => a + b, 0) / bucket.length;
      sparsities.push(mean);
      labels.push(`C${i}-${Math.min(i + bucket.length - 1, rawSparsities.length - 1)}`);
    }
  }

  if (sparsities.length > 0) {
    safeReact('ir-sparsity', [{
      x: labels, y: sparsities, type: 'bar',
      marker: { color: sparsities.map(s => s > 0.5 ? '#f44336' : '#5b8af5') },
    }], { height: 240, yaxis: { title: 'Sparsity', range: [0, 1] } });
  }
}
