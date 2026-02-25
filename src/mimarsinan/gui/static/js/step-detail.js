/* Step detail panel: tab routing, metrics tab, and simple tabs. */
import { esc, fmtDuration, cssId, safeReact } from './util.js';
import { renderModelTab } from './model-tab.js';
import { renderIRGraphTab } from './ir-graph-tab.js';
import { renderHardwareTab } from './hardware-tab.js';
import { renderSearchTab } from './search-tab.js';
import { renderActivationsTab, renderAdaptationTab } from './scales-tab.js';

// ── Public API ───────────────────────────────────────────────────────────
export async function refreshStepDetail(stepName, state, fetchJSON) {
  const panel = document.getElementById('step-detail');
  if (!panel) return;

  let detail;
  try { detail = await fetchJSON(`/api/steps/${encodeURIComponent(stepName)}`); }
  catch (e) { panel.innerHTML = '<div class="empty-state">Failed to load step detail</div>'; return; }
  if (detail.error) { panel.innerHTML = `<div class="empty-state">${esc(detail.error)}</div>`; return; }

  ingestServerMetrics(stepName, detail.metrics || [], state);

  // Only rebuild DOM when structure changes (NOT when duration updates)
  const sig = JSON.stringify({
    name: detail.name,
    status: detail.status,
    snapKeys: detail.snapshot ? Object.keys(detail.snapshot).sort() : [],
    metricNames: Object.keys(getStepMetrics(stepName, state)).sort(),
  });

  if (state.lastDetailJSON === sig && panel.querySelector('.step-detail-header')) return;
  state.lastDetailJSON = sig;

  panel.innerHTML = `
    <div class="step-detail-header">
      <h2>${esc(detail.name)}</h2>
      <span class="badge ${detail.status}">${detail.status}</span>
      ${detail.duration ? `<span class="detail-meta">${fmtDuration(detail.duration)}</span>` : ''}
      ${detail.target_metric != null ? `<span class="detail-metric">Metric: ${detail.target_metric.toFixed(4)}</span>` : ''}
    </div>
    <div class="tabs" id="step-tabs"></div>
    <div id="step-tab-content"></div>`;

  const metrics = getStepMetrics(stepName, state);
  const tabs = determineTabs(detail, metrics);
  if (!state.activeTab || !tabs.includes(state.activeTab)) state.activeTab = tabs[0];
  renderTabs(tabs, detail, metrics, state);
}

export function updateLiveCharts(stepName, state) {
  const activeTab = document.querySelector('#step-tabs .tab-btn.active');
  if (!activeTab || activeTab.dataset.tab !== 'metrics') return;
  const metrics = getStepMetrics(stepName, state);
  if (Object.keys(metrics).length === 0) return;
  const groups = groupMetricsByCategory(Object.keys(metrics));
  for (const [group, metricNames] of Object.entries(groups)) {
    const el = document.getElementById(`mc-${cssId(group)}`);
    if (!el || !el.data) return;
    Plotly.react(el, metricNames.map(name => ({
      y: (metrics[name] || []).map(p => p.value), x: (metrics[name] || []).map((_, i) => i),
      name, type: 'scatter', mode: 'lines', line: { width: 1.5 },
    })), el.layout, { displayModeBar: false, responsive: true });
  }
}

// ── Metrics ingestion ────────────────────────────────────────────────────
function ingestServerMetrics(stepName, serverMetrics, state) {
  if (!state.metricBuffers[stepName]) state.metricBuffers[stepName] = {};
  if (!state.seenSeqs[stepName]) state.seenSeqs[stepName] = new Set();
  for (const m of serverMetrics) {
    if (m.seq != null && state.seenSeqs[stepName].has(m.seq)) continue;
    if (m.seq != null) state.seenSeqs[stepName].add(m.seq);
    if (!state.metricBuffers[stepName][m.name]) state.metricBuffers[stepName][m.name] = [];
    state.metricBuffers[stepName][m.name].push({ seq: m.seq, timestamp: m.timestamp, value: parseFloat(m.value) });
  }
}

function getStepMetrics(stepName, state) { return state.metricBuffers[stepName] || {}; }

// ── Tab system ───────────────────────────────────────────────────────────
function determineTabs(detail, metrics) {
  const tabs = [];
  if (Object.keys(metrics).length > 0) tabs.push('metrics');
  const snap = detail.snapshot || {};
  if (snap.model) tabs.push('model');
  if (snap.ir_graph) tabs.push('ir_graph');
  if (snap.hard_core_mapping) tabs.push('hardware');
  if (snap.search_result) tabs.push('search');
  const hasScaleData = snap.activation_scales || (snap.model?.layers?.some(l => l.activation_scale != null || l.parameter_scale != null));
  if (hasScaleData) tabs.push('activations');
  if (snap.adaptation_manager) tabs.push('adaptation');
  if (snap.platform_constraints) tabs.push('constraints');
  if (snap.step_summary) tabs.push('summary');
  if (tabs.length === 0) tabs.push('metrics');
  return tabs;
}

const TAB_LABELS = {
  metrics: 'Metrics', model: 'Model', ir_graph: 'IR Graph', hardware: 'Hardware',
  search: 'Search', activations: 'Activations', adaptation: 'Adaptation',
  constraints: 'Constraints', summary: 'Summary',
};

function renderTabs(tabs, detail, metrics, state) {
  const tabBar = document.getElementById('step-tabs');
  const content = document.getElementById('step-tab-content');
  if (!tabBar || !content) return;

  tabBar.innerHTML = tabs.map(t =>
    `<button class="tab-btn ${state.activeTab === t ? 'active' : ''}" data-tab="${t}">${TAB_LABELS[t] || t}</button>`
  ).join('');

  tabBar.onclick = (e) => {
    const btn = e.target.closest('.tab-btn');
    if (!btn) return;
    state.activeTab = btn.dataset.tab;
    tabBar.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === state.activeTab));
    renderTabContent(state.activeTab, detail, metrics, content);
  };

  renderTabContent(state.activeTab, detail, metrics, content);
}

function renderTabContent(tab, detail, metrics, container) {
  const snap = detail.snapshot || {};
  switch (tab) {
    case 'metrics': renderMetricsTab(metrics, container); break;
    case 'model': renderModelTab(snap.model, container); break;
    case 'ir_graph': renderIRGraphTab(snap.ir_graph, container); break;
    case 'hardware': renderHardwareTab(snap.hard_core_mapping, container); break;
    case 'search': renderSearchTab(snap.search_result, container); break;
    case 'activations': renderActivationsTab(snap.activation_scales, snap.model, container); break;
    case 'adaptation': renderAdaptationTab(snap.adaptation_manager, snap.model, metrics, container); break;
    case 'constraints': renderConstraintsTab(snap.platform_constraints, container); break;
    case 'summary': renderSummaryTab(snap.step_summary, container); break;
    default: container.innerHTML = '<div class="empty-state">No data</div>';
  }
}

// ── Metrics tab ──────────────────────────────────────────────────────────
function renderMetricsTab(metrics, container) {
  const names = Object.keys(metrics);
  if (names.length === 0) { container.innerHTML = '<div class="empty-state">No metrics recorded</div>'; return; }
  const groups = groupMetricsByCategory(names);
  let html = '<div class="grid-2">';
  for (const group of Object.keys(groups))
    html += `<div class="card"><div class="card-header">${esc(group)}</div><div class="card-body"><div id="mc-${cssId(group)}" style="min-height:200px"></div></div></div>`;
  html += '</div>';
  container.innerHTML = html;
  for (const [group, metricNames] of Object.entries(groups)) plotMetricGroup(group, metricNames, metrics);
}

function plotMetricGroup(group, metricNames, metrics) {
  const el = document.getElementById(`mc-${cssId(group)}`);
  if (!el) return;
  safeReact(el, metricNames.map(name => ({
    y: (metrics[name] || []).map(p => p.value), x: (metrics[name] || []).map((_, i) => i),
    name, type: 'scatter', mode: 'lines', line: { width: 1.5 },
  })), { showlegend: metricNames.length > 1, legend: { font: { size: 10 }, x: 0, y: 1 }, height: 240 });
}

function groupMetricsByCategory(names) {
  const groups = {};
  const add = (cat, name) => { if (!groups[cat]) groups[cat] = []; groups[cat].push(name); };
  for (const name of names) {
    const l = name.toLowerCase();
    if (l.includes('loss')) add('Loss', name);
    else if (l.includes('accuracy') || l.includes('acc')) add('Accuracy', name);
    else if (l === 'lr' || l.includes('learning rate')) add('Learning Rate', name);
    else if (l.includes('adaptation') || l.includes('tuning rate')) add('Adaptation', name);
    else if (l.includes('search')) add('Search', name);
    else add('Other', name);
  }
  return groups;
}

// ── Simple tabs ──────────────────────────────────────────────────────────
function renderConstraintsTab(constraints, container) {
  if (!constraints) { container.innerHTML = '<div class="empty-state">No constraint data</div>'; return; }
  let html = '<div class="card"><div class="card-header">Resolved Platform Constraints</div><div class="card-body"><table class="config-table">';
  for (const [k, v] of Object.entries(constraints)) {
    if (k === 'cores' && Array.isArray(v)) {
      html += `<tr><td>cores</td><td>${v.map(c => `[${c.count || '?'}×] ${c.max_axons || '?'}a × ${c.max_neurons || '?'}n`).join(' &nbsp; ')}</td></tr>`;
    } else html += `<tr><td>${esc(String(k))}</td><td>${esc(String(v))}</td></tr>`;
  }
  html += '</table></div></div>';
  container.innerHTML = html;
}

function renderSummaryTab(summary, container) {
  if (!summary) { container.innerHTML = '<div class="empty-state">No summary data</div>'; return; }
  let html = '<div class="card"><div class="card-header">Step Summary</div><div class="card-body">';
  if (typeof summary === 'string') html += `<p style="color:var(--text-secondary)">${esc(summary)}</p>`;
  else {
    html += '<table class="config-table">';
    for (const [k, v] of Object.entries(summary)) html += `<tr><td>${esc(String(k))}</td><td>${esc(String(v))}</td></tr>`;
    html += '</table>';
  }
  html += '</div></div>';
  container.innerHTML = html;
}
