/* Hardware mapping tab — consistent layout, side-based connectivity, buffer lines. */
import { esc, safeReact, plotHistogram } from './util.js';

export function renderHardwareTab(hw, container) {
  if (!hw) { container.innerHTML = '<div class="empty-state">No hardware mapping data</div>'; return; }

  let html = `
    <div class="grid-3" style="margin-bottom:20px">
      <div class="card"><div class="big-metric"><div class="value">${hw.total_cores}</div><div class="label">Total HW Cores</div></div></div>
      <div class="card"><div class="big-metric"><div class="value">${(hw.mean_utilization * 100).toFixed(1)}%</div><div class="label">Mean Utilization</div></div></div>
      <div class="card"><div class="big-metric"><div class="value">${hw.num_neural_segments} / ${hw.num_compute_ops}</div><div class="label">Neural Segs / Ops</div></div></div>
    </div>
    <div class="card" style="margin-bottom:20px">
      <div class="card-header">Stage Execution Flow <span class="text-muted" style="font-size:10px;font-weight:400;text-transform:none">(expand segments, click cores for connectivity)</span></div>
      <div class="card-body"><div id="hw-stage-flow"></div></div>
    </div>
    <div class="grid-2">
      <div class="card"><div class="card-header">Core Utilization Distribution</div><div class="card-body"><div id="hw-util" style="min-height:200px"></div></div></div>
      <div class="card"><div class="card-header">Per-Segment Utilization</div><div class="card-body"><div id="hw-seg-util" style="min-height:200px"></div></div></div>
    </div>`;

  if (hw.global_core_layout?.length > 0) {
    html += '<div class="card" style="margin-bottom:20px"><div class="card-header">Hardware Core Layout (Minimum Requirement)</div><div class="card-body">';
    html += '<table class="data-table"><thead><tr><th>Core Type (Axons × Neurons)</th><th>Count</th></tr></thead><tbody>';
    for (const cl of hw.global_core_layout)
      html += `<tr><td>${cl.axons_per_core} × ${cl.neurons_per_core}</td><td>${cl.count}</td></tr>`;
    html += '</tbody></table></div></div>';
  }

  container.innerHTML = html;
  renderStageFlow(hw);
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

// ── Compute global layout (max cores of each type across all segments) ───
function computeGlobalLayout(hw) {
  const layout = hw.global_core_layout || [];
  if (layout.length === 0) return null;
  return layout.map(cl => ({
    axons: cl.axons_per_core, neurons: cl.neurons_per_core, count: cl.count,
    key: `${cl.axons_per_core}x${cl.neurons_per_core}`,
  }));
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

// ── Stage flow with consistent layout + side-based connectivity ──────────
function renderStageFlow(hw) {
  const el = document.getElementById('hw-stage-flow');
  if (!el) return;
  const expanded = new Set();
  const selectedCore = {};
  const selectedSpan = {};
  const globalLayout = computeGlobalLayout(hw);

  function render() {
    let html = '<div class="stage-flow">';
    for (let si = 0; si < hw.stages.length; si++) {
      const stage = hw.stages[si];
      if (stage.kind === 'neural') {
        const segIdx = stage.segment_index ?? stage.index;
        const isExp = expanded.has(segIdx);
        const avgUtil = stage.cores ? (stage.cores.reduce((s, c) => s + c.utilization, 0) / stage.cores.length * 100).toFixed(1) : '0.0';
        html += `<div class="stage-block neural" data-seg="${segIdx}">
          <div class="stage-block-header" onclick="window._hwToggle(${segIdx})">
            <span class="stage-expand">${isExp ? '&#9660;' : '&#9654;'}</span>
            <span class="stage-block-title">Segment ${segIdx}</span>
            <span class="stage-block-info">${stage.num_cores || 0} cores</span>
            <span class="stage-block-info">${avgUtil}% util</span>
          </div>
          <div class="stage-block-name" onclick="window._hwToggle(${segIdx})">${esc(stage.name)}</div>`;
        if (isExp && stage.cores) {
          html += buildSegmentDetail(stage, segIdx, selectedCore[segIdx], globalLayout);
        }
        html += '</div>';
      } else {
        const shapeStr = [];
        if (stage.input_shape) shapeStr.push(`In: [${stage.input_shape.join(', ')}]`);
        if (stage.output_shape) shapeStr.push(`Out: [${stage.output_shape.join(', ')}]`);
        
        html += `<div class="stage-block compute">
          <div class="stage-block-header"><span class="stage-block-title">Barrier</span><span class="stage-block-info">${esc(stage.op_type || stage.op_name || '?')}</span></div>
          <div class="stage-block-name">${esc(stage.name)}</div>
          ${shapeStr.length > 0 ? `<div class="stage-block-dims" style="font-size:10px; color:#9a9daa; margin-top:4px;">${shapeStr.join(' &nbsp; ')}</div>` : ''}
        </div>`;
      }
      if (si < hw.stages.length - 1) html += '<div class="stage-connector">&#8595;</div>';
    }
    html += '</div>';
    el.innerHTML = html;

    // Wait for all heatmap images to decode + double-rAF to ensure layout is settled
    const drawOverlays = () => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          for (const segIdx of expanded) {
            if (selectedCore[segIdx] != null) {
              drawConnOverlay(hw, segIdx, selectedCore[segIdx], selectedSpan[segIdx]);
            }
          }
        });
      });
    };
    const imgs = el.querySelectorAll('img.hw-core-canvas');
    if (imgs.length > 0) {
      Promise.all([...imgs].map(img => img.decode ? img.decode().catch(() => {}) : Promise.resolve()))
        .then(drawOverlays);
    } else {
      drawOverlays();
    }
  }

  window._hwToggle = (si) => { expanded.has(si) ? expanded.delete(si) : expanded.add(si); selectedCore[si] = null; selectedSpan[si] = null; render(); };
  window._hwCoreClick = (segIdx, ci) => {
    if (selectedCore[segIdx] === ci) { selectedCore[segIdx] = null; selectedSpan[segIdx] = null; }
    else { selectedCore[segIdx] = ci; selectedSpan[segIdx] = null; }
    render();
  };
  window._hwSpanClick = (segIdx, spanKey) => {
    if (selectedSpan[segIdx] === spanKey) selectedSpan[segIdx] = null;
    else selectedSpan[segIdx] = spanKey;
    render();
  };

  render();
}

// ── Segment detail with consistent global layout ─────────────────────────
function buildSegmentDetail(stage, segIdx, selCoreIdx, globalLayout) {
  const cores = stage.cores;
  if (!cores || cores.length === 0) return '';

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
  const maxDim = Math.max(maxAxons, maxNeurons, 1);
  const MAX_PX = 80, MIN_PX = 24;

  let html = '<div class="stage-detail" onclick="event.stopPropagation()">';
  html += `<div class="hw-segment-col" id="hw-layout-${segIdx}">`;
  html += '<div class="hw-segment-row-inner">';

  if (hasInput) {
    const showLabels = stage.input_map.length <= 12;
    html += `<div class="hw-buffer-line hw-buffer-input vertical" id="hw-in-${segIdx}" title="Input buffer: ${totalIn} elements">`;
    html += `<span class="hw-buffer-label">IN ${compactCount(totalIn)}</span>`;
    html += `<div class="hw-buffer-track" id="hw-in-track-${segIdx}">`;
    for (const m of stage.input_map) {
      const pct = Math.max(1, (m.size / Math.max(totalIn, 1)) * 100);
      const label = nodeTag(m.node_id);
      html += `<span class="hw-buffer-segment" style="height:${pct}%" data-offset="${m.offset}" data-end="${m.offset + m.size}" data-node-id="${m.node_id}" title="${label} [${m.offset}:${m.offset + m.size}]">${showLabels ? esc(label) : ''}</span>`;
    }
    html += '</div></div>';
  }

  html += `<div class="hw-core-grid" id="hw-grid-${segIdx}">`;

  if (globalLayout) {
    // Group actual segment cores by their dimension key
    const coresByDim = new Map();
    for (const c of cores) {
      const key = `${c.axons_per_core}x${c.neurons_per_core}`;
      if (!coresByDim.has(key)) coresByDim.set(key, []);
      coresByDim.get(key).push(c);
    }
    for (const grp of globalLayout) {
      const key = `${grp.axons}x${grp.neurons}`;
      const pool = coresByDim.get(key) || [];
      for (let i = 0; i < grp.count; i++) {
        const actualCore = i < pool.length ? pool[i] : null;

        const wPx = Math.max(MIN_PX, Math.round((grp.neurons / maxDim) * MAX_PX));
        const hPx = Math.max(MIN_PX, Math.round((grp.axons / maxDim) * MAX_PX));

        if (actualCore) {
          const pct = (actualCore.utilization * 100).toFixed(0);
          const selCls = selCoreIdx === actualCore.core_index ? ' hw-core-selected' : '';
          html += `<div class="hw-core${selCls}" id="hc-${segIdx}-${actualCore.core_index}" style="width:${wPx}px;height:${hPx}px" onclick="window._hwCoreClick(${segIdx},${actualCore.core_index})" title="Core ${actualCore.core_index}: ${actualCore.used_axons}/${actualCore.axons_per_core}ax × ${actualCore.used_neurons}/${actualCore.neurons_per_core}n, util=${pct}%">`;
          if (actualCore.heatmap) {
            const hmUrl = getHeatmapDataUrl(segIdx, actualCore);
            html += `<img class="hw-core-canvas" src="${hmUrl}" draggable="false">`;
          }
          html += `<span class="hw-core-label">${actualCore.core_index}</span>`;
          html += `<span class="hw-core-util">${pct}%</span>`;
          html += '</div>';
        } else {
          html += `<div class="hw-core hw-core-empty" style="width:${wPx}px;height:${hPx}px" title="Unused slot (${grp.axons}×${grp.neurons})">`;
          html += `<span class="hw-core-label" style="opacity:0.3">—</span>`;
          html += '</div>';
        }
      }
    }
  } else {
    for (const core of cores) {
      const wPx = Math.max(MIN_PX, Math.round((core.neurons_per_core / maxDim) * MAX_PX));
      const hPx = Math.max(MIN_PX, Math.round((core.axons_per_core / maxDim) * MAX_PX));
      const pct = (core.utilization * 100).toFixed(0);
      const selCls = selCoreIdx === core.core_index ? ' hw-core-selected' : '';
      html += `<div class="hw-core${selCls}" id="hc-${segIdx}-${core.core_index}" style="width:${wPx}px;height:${hPx}px" onclick="window._hwCoreClick(${segIdx},${core.core_index})" title="Core ${core.core_index}: ${core.used_axons}/${core.axons_per_core}ax × ${core.used_neurons}/${core.neurons_per_core}n, util=${pct}%">`;
      if (core.heatmap) {
        const hmUrl = getHeatmapDataUrl(segIdx, core);
        html += `<img class="hw-core-canvas" src="${hmUrl}" draggable="false">`;
      }
      html += `<span class="hw-core-label">${core.core_index}</span>`;
      html += `<span class="hw-core-util">${pct}%</span>`;
      html += '</div>';
    }
  }
  html += '</div>'; // end hw-core-grid

  html += '</div>'; // end hw-segment-row-inner

  if (hasOutput) {
    const showLabels = stage.output_map.length <= 12;
    html += `<div class="hw-buffer-line hw-buffer-output thick" id="hw-out-${segIdx}" title="Output buffer: ${totalOut} elements">`;
    html += `<span class="hw-buffer-label">OUT ${compactCount(totalOut)}</span>`;
    html += `<div class="hw-buffer-track" id="hw-out-track-${segIdx}">`;
    for (const m of stage.output_map) {
      const pct = Math.max(2, (m.size / Math.max(totalOut, 1)) * 100);
      const label = nodeTag(m.node_id);
      html += `<span class="hw-buffer-segment" style="width:${pct}%" data-offset="${m.offset}" data-end="${m.offset + m.size}" data-node-id="${m.node_id}" title="${label} [${m.offset}:${m.offset + m.size}]">${showLabels ? esc(label) : ''}</span>`;
    }
    html += '</div></div>';
  }

  html += '</div>'; // end hw-segment-col
  html += '</div>'; // end stage-detail
  return html;
}

// ── Connectivity overlay ─────────────────────────────────────────────────
// Input buffer: vertical left.  Output buffer: horizontal bottom.
// Incoming spans: drawn at left edge (axon side) of cores.
// Outgoing spans: drawn at bottom edge (neuron/output side) of cores.
// Arrowheads placed at path midpoint; hover brightens span slices.
function drawConnOverlay(hw, segIdx, selCoreIdx, selSpanKey) {
  const layout = document.getElementById(`hw-layout-${segIdx}`);
  const grid = document.getElementById(`hw-grid-${segIdx}`);
  if (!layout || !grid) return;
  layout.style.position = 'relative';
  layout.querySelectorAll('.hw-conn-overlay').forEach(n => n.remove());
  layout.querySelectorAll('.hw-edge-popover').forEach(n => n.remove());
  layout.querySelectorAll('.hw-buffer-segment-active-in, .hw-buffer-segment-active-out').forEach(n => {
    n.classList.remove('hw-buffer-segment-active-in', 'hw-buffer-segment-active-out');
  });

  const stage = hw.stages.find(s => s.kind === 'neural' && (s.segment_index ?? s.index) === segIdx);
  if (!stage || !stage.connectivity) return;

  const spans = stage.connectivity;
  const incoming = spans.filter(sp => sp.dst_core === selCoreIdx && sp.kind !== 'off');
  const outgoing = spans.filter(sp => sp.src_core === selCoreIdx && sp.kind !== 'off');
  if (incoming.length === 0 && outgoing.length === 0) return;

  const contRect = layout.getBoundingClientRect();
  if (contRect.width === 0 || contRect.height === 0) return;
  const coreByIdx = new Map();
  if (stage.cores) for (const c of stage.cores) coreByIdx.set(c.core_index, c);
  const totalInput = Math.max(stage.input_map?.reduce((s, m) => s + m.size, 0) || 0, 1);
  const totalOutput = Math.max(stage.output_map?.reduce((s, m) => s + m.size, 0) || 0, 1);

  function getRect(el) {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) return null;
    return {
      left: r.left - contRect.left, top: r.top - contRect.top,
      width: r.width, height: r.height,
      right: r.left - contRect.left + r.width,
      bottom: r.top - contRect.top + r.height,
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
  const layoutW = Math.max(1, layout.offsetWidth || contRect.width);
  const layoutH = Math.max(1, layout.offsetHeight || contRect.height);
  svg.setAttribute('width', layoutW);
  svg.setAttribute('height', layoutH);
  svg.innerHTML = '<defs></defs>';

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

  // Cubic bezier midpoint helper
  function bezierMid(x1, y1, cx1, cy1, cx2, cy2, x2, y2) {
    const t = 0.5;
    const mt = 1 - t;
    return {
      x: mt*mt*mt*x1 + 3*mt*mt*t*cx1 + 3*mt*t*t*cx2 + t*t*t*x2,
      y: mt*mt*mt*y1 + 3*mt*mt*t*cy1 + 3*mt*t*t*cy2 + t*t*t*y2,
    };
  }

  // Draw a small arrow triangle at a given point along a direction
  function svgArrowAt(mx, my, dx, dy, color, spanId) {
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const ux = dx / len, uy = dy / len;
    const sz = 5;
    const px = -uy * sz, py = ux * sz;
    const tri = document.createElementNS(svgNS, 'polygon');
    tri.setAttribute('points',
      `${mx + ux * sz},${my + uy * sz} ${mx + px},${my + py} ${mx - px},${my - py}`);
    tri.setAttribute('fill', color);
    tri.setAttribute('opacity', '0.8');
    if (spanId) tri.setAttribute('data-span-id', spanId);
    return tri;
  }

  // Collect all span elements by spanKey for hover interaction
  const spanElements = {};
  function registerSpanEl(key, el) {
    if (!spanElements[key]) spanElements[key] = [];
    spanElements[key].push(el);
  }

  // Incoming spans: source -> selected core left edge (axon side).
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
    hlDst.setAttribute('stroke', '#4caf50'); hlDst.setAttribute('stroke-width', isActive ? '4' : '3');
    hlDst.setAttribute('opacity', isActive ? '1' : '0.7');
    hlDst.setAttribute('data-span-id', spanKey);
    svg.appendChild(hlDst);
    registerSpanEl(spanKey, hlDst);

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
      hlSrc.setAttribute('opacity', '0.6');
      hlSrc.setAttribute('data-span-id', spanKey);
      svg.appendChild(hlSrc);
      registerSpanEl(spanKey, hlSrc);
      srcMidX = sx;
      srcMidY = (sy0 + sy1) / 2;

      highlightBufferSegments(`hw-in-track-${segIdx}`, sp.src_start, sp.src_end, true);
    } else {
      const srcRect = coreRect(sp.src_core);
      if (!srcRect) continue;
      const srcCore = coreByIdx.get(sp.src_core);
      const srcTotalNeurons = Math.max(1, srcCore ? srcCore.neurons_per_core : 1);
      // Source outputs from bottom of source core
      const sx0 = srcRect.left + (sp.src_start / srcTotalNeurons) * srcRect.width;
      const sx1 = srcRect.left + (sp.src_end / srcTotalNeurons) * srcRect.width;
      const sy = srcRect.bottom;

      const hlSrc = document.createElementNS(svgNS, 'line');
      hlSrc.setAttribute('x1', sx0); hlSrc.setAttribute('y1', sy);
      hlSrc.setAttribute('x2', sx1); hlSrc.setAttribute('y2', sy);
      hlSrc.setAttribute('stroke', '#4caf50'); hlSrc.setAttribute('stroke-width', isActive ? '4' : '3');
      hlSrc.setAttribute('opacity', isActive ? '1' : '0.7');
      hlSrc.setAttribute('data-span-id', spanKey);
      svg.appendChild(hlSrc);
      registerSpanEl(spanKey, hlSrc);
      srcMidX = (sx0 + sx1) / 2;
      srcMidY = sy;
    }

    const dstMidY = (dstY0 + dstY1) / 2;
    const cpx = (srcMidX + dstX) / 2;
    const cpy1 = srcMidY, cpy2 = dstMidY;
    const path = document.createElementNS(svgNS, 'path');
    path.setAttribute('d', `M${srcMidX},${srcMidY} C${cpx},${cpy1} ${cpx},${cpy2} ${dstX},${dstMidY}`);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', '#4caf50');
    path.setAttribute('stroke-width', isActive ? '2.5' : '1.5');
    path.setAttribute('opacity', isActive ? '0.9' : '0.4');
    path.setAttribute('data-span-id', spanKey);
    path.style.cursor = 'pointer';
    path.style.pointerEvents = 'auto';
    path.addEventListener('click', (ev) => { ev.stopPropagation(); window._hwSpanClick(segIdx, spanKey); });
    svg.appendChild(path);
    registerSpanEl(spanKey, path);

    // Midpoint arrow
    const mid = bezierMid(srcMidX, srcMidY, cpx, cpy1, cpx, cpy2, dstX, dstMidY);
    const arrow = svgArrowAt(mid.x, mid.y, dstX - srcMidX, dstMidY - srcMidY, '#4caf50', spanKey);
    svg.appendChild(arrow);
    registerSpanEl(spanKey, arrow);
  }

  // Outgoing spans: selected core bottom edge (neuron/output side) -> destination.
  for (const sp of outgoing) {
    const spanKey = `out_${sp.dst_core}_${sp.src_start}_${sp.length}`;
    const isActive = selSpanKey === spanKey;

    const srcCore = coreByIdx.get(sp.src_core);
    const srcTotalNeurons = Math.max(1, srcCore ? srcCore.neurons_per_core : 1);
    // Output spans drawn at bottom of selected core
    const srcX0 = selRect.left + (sp.src_start / srcTotalNeurons) * selRect.width;
    const srcX1 = selRect.left + (sp.src_end / srcTotalNeurons) * selRect.width;
    const srcY = selRect.bottom;
    const srcMidX = (srcX0 + srcX1) / 2;
    const srcMidY = srcY;

    const hlSrc = document.createElementNS(svgNS, 'line');
    hlSrc.setAttribute('x1', srcX0); hlSrc.setAttribute('y1', srcY);
    hlSrc.setAttribute('x2', srcX1); hlSrc.setAttribute('y2', srcY);
    hlSrc.setAttribute('stroke', '#ff9800'); hlSrc.setAttribute('stroke-width', isActive ? '4' : '3');
    hlSrc.setAttribute('opacity', isActive ? '1' : '0.7');
    hlSrc.setAttribute('data-span-id', spanKey);
    svg.appendChild(hlSrc);
    registerSpanEl(spanKey, hlSrc);

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
      hlDst.setAttribute('opacity', '0.6');
      hlDst.setAttribute('data-span-id', spanKey);
      svg.appendChild(hlDst);
      registerSpanEl(spanKey, hlDst);
      dstMidX = (dx0 + dx1) / 2;
      dstMidY = dy;

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
      hlDst.setAttribute('stroke', '#ff9800'); hlDst.setAttribute('stroke-width', isActive ? '4' : '3');
      hlDst.setAttribute('opacity', isActive ? '1' : '0.7');
      hlDst.setAttribute('data-span-id', spanKey);
      svg.appendChild(hlDst);
      registerSpanEl(spanKey, hlDst);
      dstMidX = dstXPos;
      dstMidY = (dstY0 + dstY1) / 2;
    }

    const cpx = (srcMidX + dstMidX) / 2;
    const cpy1 = srcMidY, cpy2 = dstMidY;
    const path = document.createElementNS(svgNS, 'path');
    path.setAttribute('d', `M${srcMidX},${srcMidY} C${cpx},${cpy1} ${cpx},${cpy2} ${dstMidX},${dstMidY}`);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', '#ff9800');
    path.setAttribute('stroke-width', isActive ? '2.5' : '1.5');
    path.setAttribute('opacity', isActive ? '0.9' : '0.4');
    path.setAttribute('stroke-dasharray', '6,3');
    path.setAttribute('data-span-id', spanKey);
    path.style.cursor = 'pointer';
    path.style.pointerEvents = 'auto';
    path.addEventListener('click', (ev) => { ev.stopPropagation(); window._hwSpanClick(segIdx, spanKey); });
    svg.appendChild(path);
    registerSpanEl(spanKey, path);

    // Midpoint arrow
    const mid = bezierMid(srcMidX, srcMidY, cpx, cpy1, cpx, cpy2, dstMidX, dstMidY);
    const arrow = svgArrowAt(mid.x, mid.y, dstMidX - srcMidX, dstMidY - srcMidY, '#ff9800', spanKey);
    svg.appendChild(arrow);
    registerSpanEl(spanKey, arrow);
  }

  // Hover interaction: brighten all span elements when any element of the span is hovered
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
              sib.setAttribute('opacity', isAct ? '0.9' : '0.4');
              sib.setAttribute('stroke-width', isAct ? '2.5' : '1.5');
            } else if (sib.tagName === 'line') {
              sib.setAttribute('opacity', isAct ? '1' : '0.7');
              sib.setAttribute('stroke-width', isAct ? '4' : '3');
            } else if (sib.tagName === 'polygon') {
              sib.setAttribute('opacity', '0.8');
            }
          }
        });
      }
    }
  }

  layout.appendChild(svg);

  // Span detail popover
  if (selSpanKey) {
    let matchedSpan = null;
    for (const sp of incoming) {
      if (`in_${sp.src_core}_${sp.dst_start}_${sp.length}` === selSpanKey) { matchedSpan = { ...sp, dir: 'in' }; break; }
    }
    if (!matchedSpan) {
      for (const sp of outgoing) {
        if (`out_${sp.dst_core}_${sp.src_start}_${sp.length}` === selSpanKey) { matchedSpan = { ...sp, dir: 'out' }; break; }
      }
    }
    if (matchedSpan) {
      const sp = matchedSpan;
      const pop = document.createElement('div');
      pop.className = 'hw-edge-popover';
      const srcLabel = sp.src_core === -2 ? 'Input Buffer' : `Core ${sp.src_core}`;
      const dstLabel = sp.dst_core === -3 ? 'Output Buffer' : `Core ${sp.dst_core}`;
      let popHtml = `<div class="hw-edge-popover-header">${srcLabel} → ${dstLabel}
        <button class="ir-detail-close" onclick="window._hwSpanClick(${segIdx},'${selSpanKey}')">✕</button></div>`;
      popHtml += '<table class="data-table compact">';
      popHtml += `<tr><td>Direction</td><td>${sp.dir === 'in' ? '← incoming' : '→ outgoing'}</td></tr>`;
      popHtml += `<tr><td>Span length</td><td>${sp.length}</td></tr>`;
      if (sp.src_core >= 0) {
        popHtml += `<tr><td>Source neurons</td><td>[${sp.src_start}:${sp.src_end}]</td></tr>`;
        const sc = coreByIdx.get(sp.src_core);
        if (sc) {
          const sa = sc.axons_per_core;
          const sn = sc.neurons_per_core;
          popHtml += `<tr><td>Source dims</td><td>${sa}ax × ${sn}n</td></tr><tr><td>Source util</td><td>${(sc.utilization * 100).toFixed(1)}%</td></tr>`;
        }
      } else {
        popHtml += `<tr><td>Source input range</td><td>[${sp.src_start}:${sp.src_end}]</td></tr>`;
      }
      popHtml += `<tr><td>Dest axons</td><td>[${sp.dst_start}:${sp.dst_end}]</td></tr>`;
      if (sp.dst_core >= 0) {
        const dc = coreByIdx.get(sp.dst_core);
        if (dc) {
          const da = dc.axons_per_core;
          const dn = dc.neurons_per_core;
          popHtml += `<tr><td>Dest dims</td><td>${da}ax × ${dn}n</td></tr><tr><td>Dest util</td><td>${(dc.utilization * 100).toFixed(1)}%</td></tr><tr><td>Dest threshold</td><td>${dc.threshold != null ? dc.threshold.toFixed(3) : '-'}</td></tr>`;
        }
      } else if (sp.dst_core === -3) {
        popHtml += `<tr><td>Output range</td><td>[${sp.dst_start}:${sp.dst_end}]</td></tr>`;
      }
      popHtml += '</table>';
      pop.innerHTML = popHtml;
      layout.appendChild(pop);
    }
  }
}

// ── Pre-rendered heatmap data URLs ───────────────────────────────────────
const _heatmapUrlCache = new Map();

function getHeatmapDataUrl(segIdx, core) {
  const key = `${segIdx}-${core.core_index}`;
  if (_heatmapUrlCache.has(key)) return _heatmapUrlCache.get(key);
  const url = renderHeatmapDataUrl(core.heatmap, core);
  _heatmapUrlCache.set(key, url);
  return url;
}

function renderHeatmapDataUrl(heatmap, core) {
  const rows = heatmap.length, cols = heatmap[0]?.length || 0;
  if (rows === 0 || cols === 0) return '';

  let maxAbs = 0;
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      maxAbs = Math.max(maxAbs, Math.abs(Number(heatmap[r][c]) || 0));

  const canvas = document.createElement('canvas');
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext('2d');

  if (maxAbs < 1e-12) {
    ctx.fillStyle = '#2b303c';
    ctx.fillRect(0, 0, cols, rows);
    ctx.fillStyle = '#6b6e7a';
    ctx.font = `${Math.max(8, Math.round(rows / 6))}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const hasUsed = (core.used_axons || 0) > 0 && (core.used_neurons || 0) > 0;
    ctx.fillText(hasUsed ? 'all-zero' : 'empty', cols / 2, rows / 2);
    return canvas.toDataURL('image/png');
  }

  const allAbs = [];
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      allAbs.push(Math.abs(Number(heatmap[r][c]) || 0));
  allAbs.sort((a, b) => a - b);
  const p98Idx = Math.max(0, Math.floor(allAbs.length * 0.98));
  const scaleAbs = Math.max(allAbs[p98Idx] || maxAbs, maxAbs * 0.05, 1e-12);

  const img = ctx.createImageData(cols, rows);
  const data = img.data;
  let k = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const raw = Number(heatmap[r][c]) || 0;
      const v = raw / scaleAbs;
      const [rr, gg, bb] = heatColor(Math.max(-1, Math.min(1, v)));
      data[k++] = rr;
      data[k++] = gg;
      data[k++] = bb;
      data[k++] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  // Draw used sub-region boundary directly onto the heatmap image.
  const totalAxons = Math.max(1, core.axons_per_core || rows);
  const totalNeurons = Math.max(1, core.neurons_per_core || cols);
  const usedAxons = Math.max(0, Math.min(core.used_axons ?? totalAxons, totalAxons));
  const usedNeurons = Math.max(0, Math.min(core.used_neurons ?? totalNeurons, totalNeurons));
  const usedW = (usedNeurons / totalNeurons) * cols;
  const usedH = (usedAxons / totalAxons) * rows;
  if (usedW > 0 && usedH > 0 && (usedW < cols || usedH < rows)) {
    ctx.strokeStyle = 'rgba(231, 235, 244, 0.5)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, Math.max(0, usedW - 1), Math.max(0, usedH - 1));
  }

  return canvas.toDataURL('image/png');
}

function heatColor(v) {
  // Modern gradient: dark blue -> cyan -> yellow (positive), dark blue -> purple (negative). Gamma for mid-tones.
  const t = Math.max(0, Math.min(1, Math.abs(v)));
  const tt = Math.pow(t, 0.7);
  if (v >= 0) {
    return [Math.round(90 + tt * 165), Math.round(96 + tt * 140), Math.round(110 - tt * 110)];
  } else {
    return [Math.round(90 - tt * 60), Math.round(96 - tt * 40), Math.round(110 + tt * 145)];
  }
}