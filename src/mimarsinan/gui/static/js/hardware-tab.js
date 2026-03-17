/* Hardware mapping tab — consistent layout, side-based connectivity, buffer lines. */
import { esc, safeReact, plotHistogram } from './util.js';

export function renderHardwareTab(hw, container, irGraph) {
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
  renderStageFlow(hw, irGraph);
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

function buildCoreDetailPanelHtml(segIdx, core, irGraph) {
  const pct = (core.utilization * 100).toFixed(1);
  const placements = core.mapped_placements || [];
  const aTotal = Math.max(1, core.axons_per_core);
  const nTotal = Math.max(1, core.neurons_per_core);
  const fusedBoundaries = core.fused_axon_boundaries;
  let html = '<div class="hw-core-detail-panel">';
  if (core.heatmap_image) {
    const ar = Math.max(1, core.neurons_per_core) / Math.max(1, core.axons_per_core);
    html += `<div class="hw-core-detail-heatmap-wrap" style="aspect-ratio: ${ar}; max-height: 240px; position:relative">`;
    html += `<img src="${core.heatmap_image.replace(/"/g, '&quot;')}" alt="Core ${core.core_index} heatmap" class="hw-core-detail-heatmap">`;
    for (let pi = 0; pi < placements.length; pi++) {
      const pl = placements[pi];
      const top = (pl.axon_offset / aTotal) * 100;
      const left = (pl.neuron_offset / nTotal) * 100;
      const height = (pl.axons / aTotal) * 100;
      const width = (pl.neurons / nTotal) * 100;
      const nodeId = pl.ir_node_id;
      const isSplit = pl.split_group_id != null;
      const splitCls = isSplit ? ' hw-split-region' : '';
      const splitRange = isSplit && pl.neuron_range_in_original ? pl.neuron_range_in_original : null;
      const splitTitle = isSplit && splitRange
        ? `n${nodeId} [${splitRange[0]}:${splitRange[1]}/${pl.split_original_neurons || '?'}]`
        : `Software core n${nodeId}`;
      html += `<div class="hw-core-detail-heatmap-region${splitCls}" style="top:${top}%;left:${left}%;width:${width}%;height:${height}%" data-ir-node-id="${nodeId}" onclick="event.stopPropagation(); window._hwSoftCoreClick(${segIdx}, ${nodeId}, ${core.core_index}, ${pi})" title="${splitTitle}"> </div>`;
    }
    if (fusedBoundaries && fusedBoundaries.length >= 2) {
      for (let i = 1; i < fusedBoundaries.length - 1; i++) {
        const topPct = (fusedBoundaries[i] / aTotal) * 100;
        html += `<div class="hw-core-detail-fused-line" style="top:${topPct}%;left:0;right:0;height:2px"></div>`;
      }
    }
    html += '</div>';
  }
  html += '<table class="data-table compact">';
  html += `<tr><td>Segment</td><td>${segIdx}</td></tr>`;
  html += `<tr><td>Core index</td><td>${core.core_index}</td></tr>`;
  html += `<tr><td>Utilization</td><td>${pct}%</td></tr>`;
  html += `<tr><td>Dimensions</td><td>${core.axons_per_core} axons × ${core.neurons_per_core} neurons</td></tr>`;
  html += `<tr><td>Used</td><td>${core.used_axons ?? '—'} / ${core.used_neurons ?? '—'}</td></tr>`;
  html += `<tr><td>Threshold</td><td>${core.threshold != null ? core.threshold.toFixed(4) : '—'}</td></tr>`;
  html += '</table>';
  if (placements.length > 0) {
    html += `<div class="hw-constituents-section"><div class="section-label">Constituents (${placements.length})</div>`;
    html += '<table class="data-table compact hw-constituents-table"><thead><tr><th>ID</th><th>Dimensions</th><th>Util.</th><th>Split</th><th>Coalesce</th></tr></thead><tbody>';
    for (let pi = 0; pi < placements.length; pi++) {
      const pl = placements[pi];
      const utilPct = (pl.utilization_frac != null ? pl.utilization_frac * 100 : (pl.axons * pl.neurons) / (aTotal * nTotal) * 100).toFixed(1);
      const coalesce = pl.coalescing_role ? `${esc(pl.coalescing_role)}${pl.coalescing_group_id != null ? ' G' + pl.coalescing_group_id : ''}` : '\u2014';
      const splitInfo = pl.split_group_id != null && pl.neuron_range_in_original
        ? `[${pl.neuron_range_in_original[0]}:${pl.neuron_range_in_original[1]}/${pl.split_original_neurons || '?'}]`
        : '\u2014';
      html += `<tr class="hw-constituent-row" onclick="event.stopPropagation(); window._hwSoftCoreClick(${segIdx}, ${pl.ir_node_id}, ${core.core_index}, ${pi})" title="Click for soft-core detail">
        <td><span class="hw-constituent-id">n${pl.ir_node_id}</span></td>
        <td>${pl.axons}\u00d7${pl.neurons}</td>
        <td>${utilPct}%</td>
        <td>${splitInfo}</td>
        <td>${coalesce}</td>
      </tr>`;
    }
    html += '</tbody></table></div>';
  }
  html += '</div>';
  return html;
}

function buildSoftCoreDetailPanelHtml(nodeId, irGraph, origin) {
  if (!irGraph || !irGraph.nodes) return '';
  const node = irGraph.nodes.find(n => n.id === nodeId || n.id === parseInt(nodeId, 10));
  if (!node) return '<div class="hw-softcore-detail-panel"><div class="hw-softcore-detail-header">Unknown node</div></div>';
  const isNeural = node.type === 'neural_core';
  let html = '<div class="hw-softcore-detail-panel">';
  html += '<div class="hw-softcore-detail-header">';
  html += `<span>Software core n${node.id}</span>`;
  html += '<button class="ir-detail-close" onclick="window._hwSoftCoreClose()" title="Close">✕</button>';
  html += '</div><div class="hw-softcore-detail-body">';
  html += `<p class="hw-softcore-name" title="${esc(node.name)}">${esc(node.name || node.layer_group || `n${node.id}`)}</p>`;
  if (origin && origin.placement) {
    const p = origin.placement;
    const aEnd = p.axon_offset + (p.axons ?? 0);
    const nEnd = p.neuron_offset + (p.neurons ?? 0);
    html += `<p class="hw-softcore-located-in">Located in: Segment ${origin.segIdx}, Hard core ${origin.coreIndex}, region axons ${p.axon_offset}..${aEnd}, neurons ${p.neuron_offset}..${nEnd}</p>`;
    if (p.split_group_id != null && p.neuron_range_in_original) {
      const r = p.neuron_range_in_original;
      const total = p.split_original_neurons || '?';
      const fragIdx = p.split_fragment_index != null ? p.split_fragment_index : '?';
      html += `<p class="hw-softcore-split-info">Split fragment ${fragIdx} \u2014 neurons ${r[0]}..${r[1]} of ${total}</p>`;
    }
  }
  if (isNeural) {
    html += '<table class="data-table compact">';
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
    if (node.psum_role) {
      html += `<tr><td>PSum</td><td>${esc(node.psum_role)} (G${node.psum_group_id})</td></tr>`;
    }
    if (node.coalescing_role) {
      html += `<tr><td>Coalesce</td><td>${esc(node.coalescing_role)} (G${node.coalescing_group_id})</td></tr>`;
    }
    html += `<tr><td>Latency</td><td>${node.latency ?? '—'}</td></tr>`;
    html += `<tr><td>Activation scale</td><td>${node.activation_scale != null ? Number(node.activation_scale).toFixed(4) : '—'}</td></tr>`;
    html += `<tr><td>Parameter scale</td><td>${node.parameter_scale != null ? Number(node.parameter_scale).toFixed(4) : '—'}</td></tr>`;
    html += `<tr><td>Sparsity</td><td>${node.weight_stats ? (node.weight_stats.sparsity * 100).toFixed(1) + '%' : '—'}</td></tr>`;
    html += '</table>';
    if (node.heatmap_image || node.pre_pruning_heatmap_image) {
      const HW_SOFT_MAX = 200;
      const MIN_HEATMAP_VIEW_WIDTH = 80;
      const MIN_HEATMAP_VIEW_HEIGHT = 80;
      const maxLong = Math.max(
        preAx ?? 0, preNu ?? 0,
        node.axons ?? 0, node.neurons ?? 0,
        1
      );
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
        w = Math.max(2, w);
        h = Math.max(2, h);
        return { w, h };
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
        w = Math.max(2, w);
        h = Math.max(2, h);
        return { w, h };
      }
      html += '<div class="hw-softcore-heatmaps" style="margin-top:10px">';
      html += '<div class="section-label" style="font-size:10px;margin-bottom:6px;color:var(--text-muted)">Weight heatmap</div>';
      html += '<div class="hw-softcore-heatmaps-scroll" style="max-height:320px;max-width:100%;overflow:auto">';
      html += '<div style="display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start">';
      let preSz = null;
      if (node.pre_pruning_heatmap_image) {
        const preLabel = (preAx != null && preNu != null) ? ` (${preAx}×${preNu})` : '';
        preSz = (preAx != null && preNu != null) ? softHeatmapSizePre(preAx, preNu) : { w: HW_SOFT_MAX, h: HW_SOFT_MAX };
        html += `<div class="hw-softcore-heatmap-wrap"><div class="section-label" style="font-size:9px;margin-bottom:2px">Pre-pruning${preLabel}</div><img src="${node.pre_pruning_heatmap_image.replace(/"/g, '&quot;')}" alt="Pre-pruning" class="hw-softcore-heatmap" style="width:${preSz.w}px;height:${preSz.h}px;object-fit:fill"></div>`;
      }
      if (node.heatmap_image) {
        const postAxNum = node.axons ?? 0;
        const postNuNum = node.neurons ?? 0;
        const postLabel = (postAxNum && postNuNum) ? ` (${postAxNum}×${postNuNum})` : '';
        const postSz = softHeatmapSizePost(postAxNum, postNuNum, preSz, preAx ?? 0, preNu ?? 0);
        html += `<div class="hw-softcore-heatmap-wrap"><div class="section-label" style="font-size:9px;margin-bottom:2px">Post-pruning${postLabel}</div><img src="${node.heatmap_image.replace(/"/g, '&quot;')}" alt="Post-pruning" class="hw-softcore-heatmap" style="width:${postSz.w}px;height:${postSz.h}px;object-fit:fill"></div>`;
      }
      html += '</div></div></div>';
    }
  } else {
    html += '<table class="data-table compact">';
    html += `<tr><td>ID</td><td>${node.id}</td></tr>`;
    html += `<tr><td>Type</td><td>${esc(node.op_type || '—')}</td></tr>`;
    html += `<tr><td>Input shape</td><td>${node.input_shape ? node.input_shape.join('×') : '—'}</td></tr>`;
    html += `<tr><td>Output shape</td><td>${node.output_shape ? node.output_shape.join('×') : '—'}</td></tr>`;
    html += '</table>';
  }
  html += '</div></div>';
  return html;
}

// ── Stage flow with consistent layout + side-based connectivity ──────────
function renderStageFlow(hw, irGraph) {
  const el = document.getElementById('hw-stage-flow');
  if (!el) return;
  const expanded = new Set();
  const selectedCore = {};
  const selectedSpan = {};
  let selectedSoftCore = null;
  let selectedSoftCoreOrigin = null; // { segIdx, coreIndex, placement } when soft core opened from HW tab
  const globalLayout = computeGlobalLayout(hw);

  function render() {
    // If a soft-core trace is active, auto-expand segments that contain it.
    if (window._hwTraceSoftId != null) {
      const traceId = parseInt(window._hwTraceSoftId, 10);
      for (const stage of hw.stages) {
        if (stage.kind === 'neural' && stage.cores) {
          const hasTrace = stage.cores.some(c => (c.mapped_placements || []).some(p => p.ir_node_id === traceId));
          if (hasTrace) expanded.add(stage.segment_index ?? stage.index);
        }
      }
    }

    let html = '<div class="stage-flow">';
    html += '<div class="stage-flow-row">';
    html += '<div class="stage-flow-column">';
    for (let si = 0; si < hw.stages.length; si++) {
      const stage = hw.stages[si];
      if (stage.kind === 'neural') {
        const segIdx = stage.segment_index ?? stage.index;
        const isExp = expanded.has(segIdx);
        const avgUtil = stage.cores ? (stage.cores.reduce((s, c) => s + c.utilization, 0) / stage.cores.length * 100).toFixed(1) : '0.0';
        let blockHtml = `<div class="stage-block neural" data-seg="${segIdx}">
          <div class="stage-block-header" onclick="window._hwToggle(${segIdx})">
            <span class="stage-expand">${isExp ? '&#9660;' : '&#9654;'}</span>
            <span class="stage-block-title">Segment ${segIdx}</span>
            <span class="stage-block-info">${stage.num_cores || 0} cores</span>
            <span class="stage-block-info">${avgUtil}% util</span>
          </div>
          <div class="stage-block-name" onclick="window._hwToggle(${segIdx})">${esc(stage.name)}</div>`;
        if (isExp && stage.cores) {
          blockHtml += buildSegmentDetail(stage, segIdx, selectedCore[segIdx], globalLayout);
        }
        blockHtml += '</div>';
        if (selectedCore[segIdx] != null && stage.cores) {
          const core = stage.cores.find(c => c.core_index === selectedCore[segIdx]);
          if (core) {
            let detailHtml = buildCoreDetailPanelHtml(segIdx, core, irGraph);
            if (selectedSoftCore != null) {
              detailHtml += buildSoftCoreDetailPanelHtml(selectedSoftCore, irGraph, selectedSoftCoreOrigin);
            }
            blockHtml = '<div class="stage-block-with-detail">' + blockHtml + detailHtml + '</div>';
          }
        }
        html += blockHtml;
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
    html += '</div>'; // end stage-flow-column
    html += '</div>'; // end stage-flow-row
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

  window._hwToggle = (si) => { expanded.has(si) ? expanded.delete(si) : expanded.add(si); selectedCore[si] = null; selectedSpan[si] = null; selectedSoftCore = null; selectedSoftCoreOrigin = null; render(); };
  window._hwCoreClick = (segIdx, ci) => {
    if (selectedCore[segIdx] === ci) {
      selectedCore[segIdx] = null;
      selectedSpan[segIdx] = null;
      selectedSoftCore = null;
      selectedSoftCoreOrigin = null;
    } else {
      // Only one segment has a selection at a time so the detail panel shows this segment's core
      for (const k of Object.keys(selectedCore)) selectedCore[k] = null;
      for (const k of Object.keys(selectedSpan)) selectedSpan[k] = null;
      selectedCore[segIdx] = ci;
      selectedSpan[segIdx] = null;
      selectedSoftCore = null;
      selectedSoftCoreOrigin = null;
    }
    render();
  };
  window._hwSoftCoreClick = (segIdx, nodeId, coreIndex, placementIndex) => {
    selectedSoftCore = nodeId;
    selectedSoftCoreOrigin = null;
    if (coreIndex != null && placementIndex != null && hw && hw.stages) {
      const stage = hw.stages.find(s => s.kind === 'neural' && (s.segment_index ?? s.index) === segIdx);
      const core = stage?.cores?.find(c => c.core_index === coreIndex);
      const pl = core?.mapped_placements?.[placementIndex];
      if (pl) selectedSoftCoreOrigin = { segIdx, coreIndex, placement: pl };
    }
    render();
  };
  window._hwSoftCoreClose = () => {
    selectedSoftCore = null;
    selectedSoftCoreOrigin = null;
    render();
  };
  window._hwSpanClick = (segIdx, spanKey) => {
    if (selectedSpan[segIdx] === spanKey) selectedSpan[segIdx] = null;
    else selectedSpan[segIdx] = spanKey;
    render();
  };

  render();
}

// Maximum pixel length for the longest dimension among all core types in the segment.
// Every core is scaled relative to this: the core type with the largest dimension
// (max over all axons_per_core and neurons_per_core) gets this size on that dimension.
const MAX_CORE_DISPLAY_PX = 200;  /* ~2.5× previous (80) so mini-view heatmaps are larger */
// Minimum pixel size per dimension so very small cores remain visible; relative sizes are preserved.
const MIN_CORE_DISPLAY_PX = 8;

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
  // Longest dimension across all core types; the core with this dimension gets MAX_CORE_DISPLAY_PX.
  const maxDimension = Math.max(maxAxons, maxNeurons, 1);

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

  // Group cores by type (axons × neurons)
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

  const ID_WIDTH = 20;
  const UTIL_HEIGHT = 14;
  const GRID_GAP = 6;
  // Conservative: stage block 700px − padding − input buffer − scrollbar/margin so grid doesn't overflow
  const SEG_VIEW_WIDTH = 580;
  // Core cell sizes: long axis = (maxDimension scale) * MAX_CORE_DISPLAY_PX; other axis from exact aspect n/ax
  // so the mini-view frame matches the backend heatmap dimensions exactly (no size mismatch).

  function emitCoreCell(core, ax, n, actualW, actualH, isPlaceholder) {
    const cellW = ID_WIDTH + 4 + actualW;
    const cellH = actualH + UTIL_HEIGHT;
    const cellStyle = `width:${cellW}px;height:${cellH}px`;
    // Exact aspect width/height = neurons/axons to match backend heatmap; one dimension fixed, other from aspect-ratio.
    const ar = Math.max(0.01, n / ax);
    const coreStyle = ax >= n
      ? `height:${actualH}px;width:auto;aspect-ratio:${ar}`
      : `width:${actualW}px;height:auto;aspect-ratio:${ar}`;
    if (isPlaceholder) {
      return `<div class="hw-core-cell" style="${cellStyle}" title="Unused slot">
        <span class="hw-core-id">—</span>
        <div class="hw-core-cell-main"><div class="hw-core hw-core-empty" style="${coreStyle}"></div><span class="hw-core-util">—</span></div>
      </div>`;
    }
    const pct = (core.utilization * 100).toFixed(0);
    const isTraced = window._hwTraceSoftId != null && (core.mapped_placements || []).some(p => p.ir_node_id === parseInt(window._hwTraceSoftId, 10));
    const selCls = (selCoreIdx === core.core_index ? ' hw-core-selected' : '') + (isTraced ? ' hw-core-traced' : '');
    let cell = `<div class="hw-core-cell" style="${cellStyle}" onclick="window._hwCoreClick(${segIdx},${core.core_index})" title="Core ${core.core_index}: ${core.used_axons}/${core.axons_per_core}ax × ${core.used_neurons}/${core.neurons_per_core}n, util=${pct}%">`;
    cell += `<span class="hw-core-id">${core.core_index}</span>`;
    cell += '<div class="hw-core-cell-main">';
    cell += `<div class="hw-core${selCls}" id="hc-${segIdx}-${core.core_index}" style="${coreStyle};position:relative">`;
    if (core.heatmap_image) {
      cell += `<img class="hw-core-canvas" src="${core.heatmap_image}" style="width:100%;height:100%;display:block;object-fit:fill" draggable="false">`;
    }
    const placements = core.mapped_placements || [];
    const aTotal = Math.max(1, core.axons_per_core);
    const nTotal = Math.max(1, core.neurons_per_core);
    const fusedBoundaries = core.fused_axon_boundaries;
    const fusedCount = core.fused_component_count != null ? core.fused_component_count : (fusedBoundaries && fusedBoundaries.length > 1 ? fusedBoundaries.length - 1 : 0);
    if (placements.length > 1 || (fusedBoundaries && fusedBoundaries.length >= 2)) {
      cell += '<div class="hw-core-constituent-overlay" style="position:absolute;inset:0;pointer-events:none">';
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
      cell += `<div class="hw-core-fused-badge" title="Fused: ${fusedCount} physical HW cores">${fusedCount} HW</div>`;
    }
    if (placements.length > 1) {
      cell += `<div class="hw-core-soft-badge" title="${placements.length} soft cores">${placements.length} soft</div>`;
    }
    cell += '</div>';
    cell += `<span class="hw-core-util">${pct}%</span>`;
    cell += '</div></div>';
    return cell;
  }

  for (const key of groupOrder) {
    const [axStr, nStr] = key.split('x');
    const ax = parseInt(axStr, 10);
    const n = parseInt(nStr, 10);
    const pool = coresByDim.get(key) || [];
    const grpCount = globalLayout ? (globalLayout.find(g => `${g.axons}x${g.neurons}` === key)?.count ?? pool.length) : pool.length;
    // Scale long axis to MAX_CORE_DISPLAY_PX; derive the other from exact aspect n/ax so frame matches heatmap.
    const longPx = Math.max(MIN_CORE_DISPLAY_PX, Math.round((Math.max(ax, n) / maxDimension) * MAX_CORE_DISPLAY_PX));
    const actualW = ax >= n ? Math.round(longPx * (n / ax)) : longPx;
    const actualH = ax >= n ? longPx : Math.round(longPx * (ax / n));
    const cellW = ID_WIDTH + 4 + actualW;
    const cellH = actualH + UTIL_HEIGHT;
    const numCols = Math.max(1, Math.floor((SEG_VIEW_WIDTH + GRID_GAP) / (cellW + GRID_GAP)));

    html += '<div class="hw-core-group">';
    html += `<div class="hw-core-group-label">${ax}×${n}</div>`;
    html += `<div class="hw-core-group-cores" style="grid-template-columns:repeat(${numCols},${cellW}px);grid-auto-rows:${cellH}px">`;
    for (let i = 0; i < grpCount; i++) {
      const actualCore = i < pool.length ? pool[i] : null;
      html += emitCoreCell(actualCore, ax, n, actualW, actualH, !actualCore);
    }
    html += '</div></div>';
  }

  html += '</div>'; // end hw-core-grid
  html += '</div>'; // end hw-core-grid-wrap

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
  const rowInner = layout.querySelector('.hw-segment-row-inner');
  if (!rowInner) return;
  rowInner.style.position = 'relative';
  layout.querySelectorAll('.hw-conn-overlay').forEach(n => n.remove());
  layout.querySelectorAll('.hw-edge-popover').forEach(n => n.remove());
  layout.querySelectorAll('.hw-buffer-segment-active-in, .hw-buffer-segment-active-out').forEach(n => {
    n.classList.remove('hw-buffer-segment-active-in', 'hw-buffer-segment-active-out');
  });

  const stage = hw.stages.find(s => s.kind === 'neural' && (s.segment_index ?? s.index) === segIdx);
  if (!stage) return;

  if (!stage.connectivity) return;

  const spans = stage.connectivity;
  const incoming = spans.filter(sp => sp.dst_core === selCoreIdx && sp.kind !== 'off');
  const outgoing = spans.filter(sp => sp.src_core === selCoreIdx && sp.kind !== 'off');
  if (incoming.length === 0 && outgoing.length === 0) return;

  // Use the row that contains cores/buffers as the coordinate frame so the overlay aligns when scrolling.
  const contRect = rowInner.getBoundingClientRect();
  if (contRect.width === 0 || contRect.height === 0) return;
  const coreByIdx = new Map();
  if (stage.cores) for (const c of stage.cores) coreByIdx.set(c.core_index, c);
  const totalInput = Math.max(stage.input_map?.reduce((s, m) => s + m.size, 0) || 0, 1);
  const totalOutput = Math.max(stage.output_map?.reduce((s, m) => s + m.size, 0) || 0, 1);

  // All overlay coordinates relative to row-inner so SVG and core positions share the same origin.
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
  const layoutW = Math.max(1, Math.round(contRect.width));
  const layoutH = Math.max(1, Math.round(contRect.height));
  svg.setAttribute('width', layoutW);
  svg.setAttribute('height', layoutH);
  svg.setAttribute('viewBox', `0 0 ${layoutW} ${layoutH}`);
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

  rowInner.appendChild(svg);

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

// Heatmap images are generated on the backend; frontend displays heatmap_image (data URI) only.