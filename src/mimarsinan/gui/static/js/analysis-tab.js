/* Analysis tab: D-hat ratchet staircase, step timeline (Gantt), A6 gauges.
   Renders the /api/analysis view-model payload — no data reshaping here. */
import { esc, safeReact, emptyAnnotation } from './util.js';

const GROUP_COLORS = {
  pretraining: '#4ade80', configuration: '#8494a7', model_building: '#8494a7',
  torch_mapping: '#5b8af5', pruning: '#f472b6', activation: '#a78bfa',
  activation_quantization: '#8b5cf6', normalization: '#67e8f9',
  weight_quantization: '#f59e0b', soft_mapping: '#22d3ee',
  core_verification: '#94a3b8', hardware: '#f97316', simulation: '#f87171',
  other: '#8494a7',
};

let _lastFetch = 0;

export async function renderAnalysisTab(state, apiUrl, fetchJSON) {
  // Events are low-rate; a 2 s floor keeps WS-triggered re-renders cheap.
  const now = Date.now();
  if (now - _lastFetch < 2000) return;
  _lastFetch = now;
  let analysis;
  try {
    analysis = await fetchJSON(apiUrl('/analysis'));
  } catch (e) {
    return;
  }
  renderStaircase(analysis.staircase, analysis.highwater);
  renderGantt(analysis.gantt);
  renderA6(analysis.a6);
}

// ── D-hat ratchet staircase ──────────────────────────────────────────────
function renderStaircase(staircase, highwater) {
  const host = document.getElementById('chart-staircase');
  if (!host) return;
  const lanes = (staircase && staircase.tuners) || [];
  const traces = [];
  const annotations = [];
  for (const lane of lanes) {
    const accepted = lane.probes.filter(p => p.accepted);
    const rejected = lane.probes.filter(p => !p.accepted);
    traces.push({
      x: lane.staircase.map(p => p.i), y: lane.staircase.map(p => p.best),
      type: 'scatter', mode: 'lines', line: { shape: 'hv', width: 2 },
      name: `${lane.tuner} best-D̂${lane.stalled ? ' (stalled)' : ''}`,
    });
    if (accepted.length) traces.push({
      x: accepted.map(p => p.i + 1), y: accepted.map(p => p.full_acc),
      type: 'scatter', mode: 'markers', name: `${lane.tuner} accepted`,
      marker: { symbol: 'triangle-up', size: 9, color: '#4ade80' },
    });
    if (rejected.length) traces.push({
      x: rejected.map(p => p.i + 1), y: rejected.map(p => p.full_acc),
      type: 'scatter', mode: 'markers', name: `${lane.tuner} rejected`,
      marker: { symbol: 'triangle-down-open', size: 9, color: '#fbbf24' },
    });
  }
  const shapes = [];
  if (highwater != null) {
    shapes.push({
      type: 'line', xref: 'paper', x0: 0, x1: 1, y0: highwater, y1: highwater,
      line: { color: '#22d3ee', width: 1, dash: 'dash' },
    });
    annotations.push({
      xref: 'paper', x: 1, y: highwater, xanchor: 'right', yanchor: 'bottom',
      text: `D̂ high-water ${highwater.toFixed(4)}`, showarrow: false,
      font: { size: 10, color: '#22d3ee' },
    });
  }
  if (staircase && staircase.invariant_violation) {
    annotations.push({
      xref: 'paper', yref: 'paper', x: 0.5, y: 0.5, showarrow: false,
      text: '⚠ RATCHET VIOLATION: ' + staircase.invariant_violation,
      font: { size: 12, color: '#f87171' },
    });
  }
  if (!traces.length) annotations.push(...emptyAnnotation('No gate events yet'));
  safeReact(host, traces, {
    margin: { t: 30, r: 30, b: 40, l: 60 },
    xaxis: { title: 'gate probe →', dtick: 1 },
    yaxis: { title: 'deployed full-transform D̂', automargin: true },
    height: 300, annotations, shapes,
    legend: { orientation: 'h', y: -0.2, font: { size: 10 } },
  });
}

// ── Step timeline (Gantt) with the endpoint step-budget ledger ───────────
function renderGantt(gantt) {
  const host = document.getElementById('chart-gantt');
  if (!host) return;
  const rows = (gantt && gantt.rows) || [];
  const summary = document.getElementById('gantt-wall-summary');
  if (summary && gantt) {
    const budget = gantt.endpoint_budget || {};
    const budgetText = budget.budget_steps != null
      ? ` · endpoint steps ${budget.consumed_steps}/${budget.budget_steps}` : '';
    summary.textContent =
      `artifact wall ${gantt.artifact_wall_s.toFixed(1)}s · ` +
      `total ${gantt.total_wall_s.toFixed(1)}s (simulators hatched)${budgetText}`;
  }
  const traces = rows.length ? [{
    y: rows.map(r => r.step),
    x: rows.map(r => r.wall_s),
    base: rows.map(r => r.offset_s),
    type: 'bar', orientation: 'h',
    marker: {
      color: rows.map(r => GROUP_COLORS[r.group] || GROUP_COLORS.other),
      pattern: { shape: rows.map(r => (r.simulator ? '/' : '')) },
    },
    text: rows.map(r => `${r.wall_s.toFixed(1)}s${r.simulator ? ' (sim)' : ''}`),
    textposition: 'outside', textfont: { size: 9 },
    hovertext: rows.map(r => `${r.step} — ${r.wall_s.toFixed(1)}s (${r.group})`),
  }] : [];
  safeReact(host, traces, {
    margin: { t: 20, r: 60, b: 40, l: 200 },
    xaxis: { title: 'wall clock (s from run start)', automargin: true },
    yaxis: { autorange: 'reversed', automargin: true, tickfont: { size: 10 } },
    height: Math.max(220, rows.length * 26 + 60),
    annotations: rows.length ? [] : emptyAnnotation('No timed steps yet'),
    showlegend: false,
  });
}

// ── A6 gauge cards ───────────────────────────────────────────────────────
function renderA6(a6) {
  const host = document.getElementById('a6-gauges');
  if (!host) return;
  const cards = (a6 && a6.cards) || [];
  if (!cards.length) {
    host.innerHTML = '<div class="empty-state">No [MBH-A6] gauge events (pre-flight gauges emit on quantized installs)</div>';
    return;
  }
  host.innerHTML = '<div class="a6-card-row">' + cards.map(card => {
    const pass = card.verdict === 'PASS';
    const rows = Object.entries(card.detail || {})
      .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(JSON.stringify(v))}</td></tr>`)
      .join('');
    return `<div class="a6-card a6-${pass ? 'pass' : 'warn'}">
      <div class="a6-card-head">
        <span class="a6-kind">${esc(card.gauge)}</span>
        <span class="a6-verdict">${pass ? '✓ PASS' : '⚠ ' + esc(card.verdict || '')}</span>
      </div>
      <div class="a6-context">${esc(card.context)}</div>
      <table class="config-table">${rows}</table>
    </div>`;
  }).join('') + '</div>';
}
