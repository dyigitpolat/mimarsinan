/* Activations + Adaptation tabs. */
import { esc, safeReact } from './util.js';

// ── Activations tab ──────────────────────────────────────────────────────
export function renderActivationsTab(scales, model, container) {
  const hasScales = scales?.length > 0;
  const layers = model?.layers || [];
  const hasAct = layers.some(l => l.activation_scale != null);
  const hasParam = layers.some(l => l.parameter_scale != null);

  if (!hasScales && !hasAct && !hasParam) {
    container.innerHTML = '<div class="empty-state">No activation or scale data available.<br><span style="font-size:12px;color:var(--text-muted)">Scales are computed during the Activation Analysis step.</span></div>';
    return;
  }

  let html = '';
  if (hasScales) html += '<div class="card" style="margin-bottom:20px"><div class="card-header">Activation Scales (from Analysis)</div><div class="card-body"><div id="act-s" style="min-height:200px"></div></div></div>';
  if (hasAct || hasParam) {
    html += '<div class="grid-2" style="margin-bottom:20px">';
    if (hasAct) html += '<div class="card"><div class="card-header">Layer Activation Scales</div><div class="card-body"><div id="act-la" style="min-height:200px"></div></div></div>';
    if (hasParam) html += '<div class="card"><div class="card-header">Layer Parameter Scales</div><div class="card-body"><div id="act-lp" style="min-height:200px"></div></div></div>';
    html += '</div>';
  }
  if (layers.length > 0 && (hasAct || hasParam)) {
    html += '<div class="card"><div class="card-header">Scale Values Table</div><div class="card-body scrollable"><table class="data-table"><thead><tr><th>Layer</th><th>Act Scale</th><th>Param Scale</th></tr></thead><tbody>';
    for (const l of layers) {
      if (l.activation_scale == null && l.parameter_scale == null) continue;
      html += `<tr><td>L${l.index}</td><td>${l.activation_scale != null ? l.activation_scale.toFixed(4) : '-'}</td><td>${l.parameter_scale != null ? l.parameter_scale.toFixed(4) : '-'}</td></tr>`;
    }
    html += '</tbody></table></div></div>';
  }
  container.innerHTML = html;

  if (hasScales) safeReact('act-s', [{ x: scales.map((_, i) => `L${i}`), y: scales, type: 'bar', marker: { color: '#ff9800' } }], { height: 260, yaxis: { title: 'Scale' } });
  if (hasAct) { const f = layers.filter(l => l.activation_scale != null); safeReact('act-la', [{ x: f.map(l => `L${l.index}`), y: f.map(l => l.activation_scale), type: 'bar', marker: { color: '#4caf50' } }], { height: 260, yaxis: { title: 'Activation Scale' } }); }
  if (hasParam) { const f = layers.filter(l => l.parameter_scale != null); safeReact('act-lp', [{ x: f.map(l => `L${l.index}`), y: f.map(l => l.parameter_scale), type: 'bar', marker: { color: '#9c27b0' } }], { height: 260, yaxis: { title: 'Param Scale' } }); }
}

// ── Adaptation tab ───────────────────────────────────────────────────────
export function renderAdaptationTab(adaptMgr, model, metrics, container) {
  if (!adaptMgr && !model) { container.innerHTML = '<div class="empty-state">No adaptation data</div>'; return; }
  const layers = model?.layers || [];
  const hasAct = layers.some(l => l.activation_scale != null);
  const hasParam = layers.some(l => l.parameter_scale != null);

  let html = '';

  if (adaptMgr) {
    const rates = [
      { k: 'clamp_rate', label: 'Clamp', color: '#5b8af5' },
      { k: 'shift_rate', label: 'Shift', color: '#4caf50' },
      { k: 'quantization_rate', label: 'Quantization', color: '#ff9800' },
      { k: 'scale_rate', label: 'Scale', color: '#9c27b0' },
      { k: 'noise_rate', label: 'Noise', color: '#00bcd4' },
    ].filter(r => adaptMgr[r.k] != null);
    if (rates.length > 0) {
      html += '<div class="grid-3" style="margin-bottom:20px">';
      for (const r of rates.slice(0, 3)) {
        const val = typeof adaptMgr[r.k] === 'number' ? adaptMgr[r.k].toFixed(4) : String(adaptMgr[r.k]);
        html += `<div class="card"><div class="big-metric"><div class="value" style="color:${r.color};font-size:32px">${esc(val)}</div><div class="label">${r.label} Rate</div></div></div>`;
      }
      html += '</div>';
      if (rates.length > 3) {
        html += '<div class="card" style="margin-bottom:20px"><div class="card-header">All Rates</div><div class="card-body"><table class="config-table">';
        for (const r of rates) html += `<tr><td>${r.label}</td><td>${typeof adaptMgr[r.k] === 'number' ? adaptMgr[r.k].toFixed(6) : String(adaptMgr[r.k])}</td></tr>`;
        html += '</table></div></div>';
      }
    }
  }

  const adaptMetrics = Object.keys(metrics || {}).filter(n => { const l = n.toLowerCase(); return l.includes('tuning rate') || l.includes('adaptation'); });
  if (adaptMetrics.length > 0) html += '<div class="card" style="margin-bottom:20px"><div class="card-header">Adaptation Rate Timeline</div><div class="card-body"><div id="ad-timeline" style="min-height:220px"></div></div></div>';
  if (hasAct || hasParam) html += '<div class="card" style="margin-bottom:20px"><div class="card-header">Per-Layer Scale Comparison</div><div class="card-body"><div id="ad-scales" style="min-height:220px"></div></div></div>';
  if (layers.some(l => l.weight?.histogram)) html += '<div class="card" style="margin-bottom:20px"><div class="card-header">Weight Distributions</div><div class="card-body"><div id="ad-dist" style="min-height:220px"></div></div></div>';
  if (hasParam) html += '<div class="card"><div class="card-header">Quantization Staircase</div><div class="card-body"><div id="ad-quant" style="min-height:220px"></div></div></div>';

  container.innerHTML = html;

  if (adaptMetrics.length > 0) {
    safeReact('ad-timeline', adaptMetrics.map(name => ({
      y: (metrics[name] || []).map(p => p.value), x: (metrics[name] || []).map((_, i) => i),
      name, type: 'scatter', mode: 'lines', line: { width: 2 },
    })), { height: 260, showlegend: true, legend: { font: { size: 10 } }, xaxis: { title: 'Step' }, yaxis: { title: 'Rate' } });
  }

  if (hasAct || hasParam) {
    const traces = [];
    if (hasAct) { const f = layers.filter(l => l.activation_scale != null); traces.push({ x: f.map(l => `L${l.index}`), y: f.map(l => l.activation_scale), name: 'Activation', type: 'bar', marker: { color: '#4caf50' } }); }
    if (hasParam) { const f = layers.filter(l => l.parameter_scale != null); traces.push({ x: f.map(l => `L${l.index}`), y: f.map(l => l.parameter_scale), name: 'Parameter', type: 'bar', marker: { color: '#9c27b0' } }); }
    safeReact('ad-scales', traces, { barmode: 'group', height: 280, showlegend: true, legend: { font: { size: 10 } }, xaxis: { title: 'Layer' }, yaxis: { title: 'Scale Value' } });
  }

  const dLayers = layers.filter(l => l.weight?.histogram);
  if (dLayers.length > 0) {
    safeReact('ad-dist', dLayers.map(l => {
      const h = l.weight.histogram;
      const mids = h.bin_edges.slice(0, -1).map((e, j) => (e + h.bin_edges[j + 1]) / 2);
      return { x: mids, y: h.counts, name: `L${l.index}`, type: 'bar', opacity: 0.6 };
    }), { barmode: 'overlay', height: 300, showlegend: true, legend: { font: { size: 10 } }, xaxis: { title: 'Weight Value' }, yaxis: { title: 'Count' } });
  }

  if (hasParam) {
    const f = layers.filter(l => l.parameter_scale != null && l.parameter_scale > 0);
    if (f.length > 0) {
      safeReact('ad-quant', f.map(l => {
        const s = l.parameter_scale, levels = Math.round(2 * s);
        const xs = [], ys = [];
        for (let v = -levels; v <= levels; v++) { xs.push(v / s); ys.push(v); }
        return { x: xs, y: ys, name: `L${l.index} (s=${s.toFixed(2)})`, type: 'scatter', mode: 'lines+markers', line: { width: 1, shape: 'hv' }, marker: { size: 3 } };
      }), { height: 280, showlegend: true, legend: { font: { size: 10 } }, xaxis: { title: 'Continuous Weight' }, yaxis: { title: 'Quantized Level' } });
    }
  }
}
