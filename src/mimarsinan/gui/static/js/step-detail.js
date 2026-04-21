/* Step detail panel: tab routing, metrics tab, and simple tabs. */
import { esc, fmtDuration, cssId, safeReact } from './util.js';
import { renderModelTab } from './model-tab.js';
import { renderIRGraphTab } from './ir-graph-tab.js';
import { renderHardwareTab } from './hardware-tab.js';
import { renderSearchTab } from './search-tab.js';
import { renderActivationsTab, renderAdaptationTab } from './scales-tab.js';
import { renderPruningTab } from './pruning-tab.js';
import { initSearchLive, detachSearchLive, replaySearchEvents, syncSearchEventsFromState } from './search-live.js';
import { setResourceContext } from './resource-urls.js';

// ── Public API ───────────────────────────────────────────────────────────
// Per-step ETag + metric-cursor cache shared across refreshStepDetail
// calls. `_etagCache[step]` is the last ``If-None-Match`` value the
// server handed us; `_sinceSeqCache[step]` is the largest metric seq
// we've ingested so the server only has to ship new points.
const _etagCache = {};
const _sinceSeqCache = {};
const _detailCache = {};

export async function refreshStepDetail(stepName, state, fetchJSON) {
  const panel = document.getElementById('step-detail');
  if (!panel) return;

  let stepsBase;
  if (state.historicalRunId && state.isActiveRun) {
    stepsBase = `/api/active_runs/${encodeURIComponent(state.historicalRunId)}/steps`;
  } else if (state.historicalRunId) {
    stepsBase = `/api/runs/${encodeURIComponent(state.historicalRunId)}/steps`;
  } else {
    stepsBase = '/api/steps';
  }

  const sinceSeq = _sinceSeqCache[stepName] || 0;
  const url = `${stepsBase}/${encodeURIComponent(stepName)}?since_seq=${sinceSeq}`;

  // Only the primary /api/steps endpoint currently supports ETag/304.
  // Historical and active-run mirrors keep the plain path for now.
  // Skip the conditional request when the panel was cleared (step
  // switch / tab reset): we need the full body to rebuild DOM.
  const canUseEtag = stepsBase === '/api/steps' && state.lastDetailJSON != null;
  const ifNoneMatch = canUseEtag ? _etagCache[stepName] : null;

  let detail;
  try {
    if (canUseEtag) {
      const headers = {};
      if (ifNoneMatch) headers['If-None-Match'] = ifNoneMatch;
      const res = await fetch(url, { headers });
      if (res.status === 304) {
        // Snapshot is unchanged; keep the cached detail, still ingest
        // nothing (server returned no body). But metrics may have
        // advanced via WS in the meantime — trigger a chart refresh.
        const cached = _detailCache[stepName];
        if (cached) updateLiveCharts(stepName, state);
        return;
      }
      if (!res.ok) {
        panel.innerHTML = '<div class="empty-state">Failed to load step detail</div>';
        return;
      }
      detail = await res.json();
      const newEtag = res.headers.get('etag') || res.headers.get('ETag');
      if (newEtag) _etagCache[stepName] = newEtag;
    } else {
      detail = await fetchJSON(url);
    }
  } catch (e) {
    panel.innerHTML = '<div class="empty-state">Failed to load step detail</div>';
    return;
  }
  if (detail.error) { panel.innerHTML = `<div class="empty-state">${esc(detail.error)}</div>`; return; }
  _detailCache[stepName] = detail;

  // Advance the cursor so the next poll asks only for newer metrics.
  if (typeof detail.latest_metric_seq === 'number') {
    _sinceSeqCache[stepName] = detail.latest_metric_seq;
  } else if (Array.isArray(detail.metrics) && detail.metrics.length > 0) {
    const maxSeq = detail.metrics.reduce((m, x) => Math.max(m, x.seq || 0), _sinceSeqCache[stepName] || 0);
    _sinceSeqCache[stepName] = maxSeq;
  }

  const prevCount = _totalMetricPoints(stepName, state);
  const prevSearchLen = (state.searchEvents && state.searchEvents[stepName])
    ? state.searchEvents[stepName].length : 0;
  ingestServerMetrics(stepName, detail.metrics || [], state);
  const newCount = _totalMetricPoints(stepName, state);
  const newSearchLen = (state.searchEvents && state.searchEvents[stepName])
    ? state.searchEvents[stepName].length : 0;

  // Only rebuild DOM when structure changes (NOT when duration updates)
  const sig = JSON.stringify({
    name: detail.name,
    status: detail.status,
    snapKeys: detail.snapshot ? Object.keys(detail.snapshot).sort() : [],
    snapshotKeyKinds: detail.snapshot_key_kinds ? Object.keys(detail.snapshot_key_kinds).sort() : [],
    metricNames: Object.keys(getStepMetrics(stepName, state)).sort(),
  });

  if (state.lastDetailJSON === sig && panel.querySelector('.step-detail-header')) {
    if (newCount > prevCount) updateLiveCharts(stepName, state);
    if (newSearchLen > prevSearchLen && state.activeTab === 'live_search' && state.selectedStep === stepName) {
      syncSearchEventsFromState(stepName, state);
    }
    return;
  }
  state.lastDetailJSON = sig;

  const stepIdx = state.pipeline && state.pipeline.steps
    ? state.pipeline.steps.findIndex(s => s.name === stepName) : -1;
  const stepCountLabel = stepIdx >= 0 && state.pipeline
    ? `<span class="detail-meta" style="font-size:11px;color:var(--text-muted)">Step ${stepIdx + 1} of ${state.pipeline.steps.length}</span>` : '';

  panel.innerHTML = `
    <div class="step-detail-header">
      <h2>${esc(detail.name)}</h2>
      <button class="step-copy-name" title="Copy step name" style="background:none;border:1px solid var(--border);color:var(--text-muted);padding:2px 8px;border-radius:4px;font-size:11px;cursor:pointer">Copy</button>
      <span class="badge ${detail.status}">${detail.status}</span>
      ${stepCountLabel}
      ${detail.duration ? `<span class="detail-meta">${fmtDuration(detail.duration)}</span>` : ''}
      ${detail.target_metric != null ? `<span class="detail-metric">Metric: ${detail.target_metric.toFixed(4)}</span>` : ''}
    </div>
    <div class="tabs" id="step-tabs"></div>
    <div id="step-tab-content"></div>`;

  const copyBtn = panel.querySelector('.step-copy-name');
  if (copyBtn) copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(stepName).then(() => { copyBtn.textContent = 'Copied!'; setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1200); });
  });

  const metrics = getStepMetrics(stepName, state);
  const searchEvts = (state.searchEvents && state.searchEvents[stepName]) || [];
  if (searchEvts.length > 0) detail._hasSearchEvents = true;
  detail._searchEvents = searchEvts;
  const tabs = determineTabs(detail, metrics);
  if (!state.activeTab || !tabs.includes(state.activeTab)) state.activeTab = tabs[0];
  renderTabs(stepName, tabs, detail, metrics, state);
}

export function updateLiveCharts(stepName, state) {
  const activeTab = document.querySelector('#step-tabs .tab-btn.active');
  if (!activeTab || activeTab.dataset.tab !== 'metrics') return;
  const metrics = getStepMetrics(stepName, state);
  const plottableNames = Object.keys(metrics).filter(n => n !== 'search_event');
  if (plottableNames.length === 0) return;
  const stepStartTime = state.pipeline?.steps?.find(s => s.name === stepName)?.start_time;
  const stepStartSec = stepStartTime != null ? (stepStartTime > 1e12 ? stepStartTime / 1000 : stepStartTime) : null;
  const groups = groupMetricsByCategory(plottableNames);

  // First-metric scaffold: when the metrics tab was previously rendered
  // with an empty buffer, ``renderMetricsTab`` emitted a "No metrics
  // recorded" empty-state and *no* chart containers. The same applies
  // when a brand-new metric group shows up that wasn't present at the
  // last full render. In either case, ``Plotly.extendTraces`` has no
  // DOM node to write to and silently does nothing — which is what
  // made the Metrics tab appear frozen until the user manually clicked
  // the tab. Detect the missing scaffold and rebuild the tab from the
  // live buffers. This is a pure client-side recovery; no REST fetch.
  const groupNames = Object.keys(groups);
  const anyMissing = groupNames.some(
    g => document.getElementById(`mc-${cssId(g)}`) == null
  );
  if (anyMissing) {
    const content = document.getElementById('step-tab-content');
    if (content) {
      renderMetricsTab(metrics, content, stepStartSec);
      // ``renderMetricsTab`` freshly plots every group, so bookkeeping
      // on the previous chart DOMs is already reset inside
      // ``plotMetricGroup``. Nothing more to do this tick.
      return;
    }
  }

  for (const [group, metricNames] of Object.entries(groups)) {
    const el = document.getElementById(`mc-${cssId(group)}`);
    if (!el || !el.data) continue;

    // Incremental fast path: if this chart has the same trace layout
    // we last drew, we can call Plotly.extendTraces with only the new
    // points per trace. This is orders of magnitude cheaper than
    // Plotly.react for high-frequency metrics (training loss at
    // batch granularity can flood at hundreds of Hz).
    const trackedNames = el._mimarsinanTraceNames;
    const structureMatches =
      Array.isArray(trackedNames) &&
      trackedNames.length === metricNames.length &&
      trackedNames.every((n, i) => n === metricNames[i]);

    if (structureMatches) {
      const lastCounts = el._mimarsinanTraceCounts || [];
      const newXs = [];
      const newYs = [];
      const traceIndices = [];
      let grew = false;
      for (let i = 0; i < metricNames.length; i++) {
        const name = metricNames[i];
        let points = metrics[name] || [];
        if (stepStartSec != null) points = points.filter(p => p.timestamp >= stepStartSec);
        const prev = lastCounts[i] || 0;
        if (points.length > prev) {
          grew = true;
          const slice = points.slice(prev);
          newXs.push(stepStartSec != null
            ? slice.map(p => p.timestamp - stepStartSec)
            : slice.map((_, j) => prev + j));
          newYs.push(slice.map(p => p.value));
          traceIndices.push(i);
          lastCounts[i] = points.length;
        }
      }
      if (grew) {
        try {
          Plotly.extendTraces(el, { x: newXs, y: newYs }, traceIndices);
        } catch (_) {
          // Fall through to full redraw if Plotly state desynced.
          _fullRedraw(el, group, metricNames, metrics, stepStartSec);
          continue;
        }
        el._mimarsinanTraceCounts = lastCounts;
      }
      continue;
    }

    _fullRedraw(el, group, metricNames, metrics, stepStartSec);
  }
}

function _fullRedraw(el, group, metricNames, metrics, stepStartSec) {
  const allTraces = metricNames.map(name => {
    let points = metrics[name] || [];
    if (stepStartSec != null) points = points.filter(p => p.timestamp >= stepStartSec);
    const x = stepStartSec != null ? points.map(p => (p.timestamp - stepStartSec)) : points.map((_, i) => i);
    const y = points.map(p => p.value);
    return { x, y, name, pointCount: points.length };
  });
  const xMax = computeGroupXMax(allTraces);
  const traces = allTraces.map(({ x, y, name }) => {
    let xOut = x, yOut = y;
    if (x.length === 1 && xMax != null) {
      xOut = [x[0], xMax];
      yOut = [y[0], y[0]];
    }
    return { x: xOut, y: yOut, name, type: 'scatter', mode: 'lines', line: { width: 1.5 } };
  });
  const layout = { ...el.layout };
  if (group === 'Accuracy' || group === 'Adaptation') layout.yaxis = { ...(layout.yaxis || {}), range: [0, 1] };
  Plotly.react(el, traces, layout, { displayModeBar: false, responsive: true });
  el._mimarsinanTraceNames = metricNames.slice();
  el._mimarsinanTraceCounts = allTraces.map(t => t.pointCount);
}

// ── Metrics ingestion ────────────────────────────────────────────────────
function _totalMetricPoints(stepName, state) {
  const buf = state.metricBuffers[stepName];
  if (!buf) return 0;
  let n = 0;
  for (const arr of Object.values(buf)) n += arr.length;
  return n;
}

function ingestServerMetrics(stepName, serverMetrics, state) {
  if (!state.metricBuffers[stepName]) state.metricBuffers[stepName] = {};
  if (!state.seenSeqs[stepName]) state.seenSeqs[stepName] = new Set();
  if (!state.searchEvents) state.searchEvents = {};
  if (!state.searchEvents[stepName]) state.searchEvents[stepName] = [];
  for (const m of serverMetrics) {
    if (m.seq != null && state.seenSeqs[stepName].has(m.seq)) continue;
    if (m.seq != null) state.seenSeqs[stepName].add(m.seq);
    if (m.name === 'search_event') {
      try {
        const parsed = typeof m.value === 'string' ? JSON.parse(m.value) : m.value;
        state.searchEvents[stepName].push(parsed);
      } catch (_) { /* skip malformed */ }
      if (!state.metricBuffers[stepName][m.name]) state.metricBuffers[stepName][m.name] = [];
      state.metricBuffers[stepName][m.name].push({ seq: m.seq, timestamp: m.timestamp, value: m.value });
      continue;
    }
    if (!state.metricBuffers[stepName][m.name]) state.metricBuffers[stepName][m.name] = [];
    state.metricBuffers[stepName][m.name].push({ seq: m.seq, timestamp: m.timestamp, value: parseFloat(m.value) });
  }
}

// Return the *live* metric buffer for ``stepName`` — always lazily
// creating the entry in ``state.metricBuffers`` so every caller sees
// the same object reference. Without this, a caller that hit an
// undefined bucket would get a throwaway ``{}``; a later
// ``bufferMetric`` would then assign a *new* ``{}`` to state and any
// reference cached by a closure (e.g. the tab ``onclick`` handler or
// ``renderTabs``/``renderMetricsTab``) would stay stale and keep
// showing "No metrics recorded" even though metrics were arriving.
function getStepMetrics(stepName, state) {
  if (!state.metricBuffers[stepName]) state.metricBuffers[stepName] = {};
  return state.metricBuffers[stepName];
}

// ── Tab system ───────────────────────────────────────────────────────────
function determineTabs(detail, metrics) {
  const tabs = [];
  if (Object.keys(metrics).length > 0) tabs.push('metrics');
  if (metrics['search_event'] || detail._hasSearchEvents) tabs.push('live_search');
  const snap = detail.snapshot || {};
  if (snap.model) tabs.push('model');
  if (snap.ir_graph) tabs.push('ir_graph');
  if (snap.hard_core_mapping) tabs.push('hardware');
  if (snap.search_result) tabs.push('search');
  const hasScaleData = snap.activation_scales || (snap.model?.layers?.some(l => l.activation_scale != null || l.parameter_scale != null));
  if (hasScaleData) tabs.push('activations');
  if (snap.adaptation_manager) tabs.push('adaptation');
  if (snap.platform_constraints) tabs.push('constraints');
  if (snap.pruning_layers && Array.isArray(snap.pruning_layers.layers)) tabs.push('pruning');
  if (snap.step_summary) tabs.push('summary');
  if (tabs.length === 0) tabs.push('metrics');
  return tabs;
}

const TAB_LABELS = {
  metrics: 'Metrics', live_search: 'Live Search', model: 'Model', ir_graph: 'IR Graph',
  hardware: 'Hardware', search: 'Search', activations: 'Activations', adaptation: 'Adaptation',
  constraints: 'Constraints', pruning: 'Pruning', summary: 'Summary',
};

function renderTabs(stepName, tabs, detail, metrics, state) {
  const tabBar = document.getElementById('step-tabs');
  const content = document.getElementById('step-tab-content');
  if (!tabBar || !content) return;

  const kinds = detail.snapshot_key_kinds || {};
  tabBar.innerHTML = tabs.map(t => {
    const kindClass = kinds[t] === 'new' ? ' new-tab' : (kinds[t] === 'edited' ? ' edited-tab' : '');
    return `<button class="tab-btn${kindClass} ${state.activeTab === t ? 'active' : ''}" data-tab="${t}">${TAB_LABELS[t] || t}</button>`;
  }).join('');

  tabBar.onclick = (e) => {
    const btn = e.target.closest('.tab-btn');
    if (!btn) return;
    state.activeTab = btn.dataset.tab;
    tabBar.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === state.activeTab));
    renderTabContent(stepName, state.activeTab, detail, metrics, content, state);
  };

  renderTabContent(stepName, state.activeTab, detail, metrics, content, state);
}

function renderTabContent(stepName, tab, detail, metrics, container, state) {
  if (tab !== 'live_search') detachSearchLive();
  // Install resource context for any lazy <img>/fetch inside the tab
  // renderers. Tabs themselves stay stateless and import
  // resourceUrl() from resource-urls.js.
  setResourceContext({
    stepName,
    historicalRunId: state.historicalRunId || null,
    isActiveRun: !!state.isActiveRun,
  });
  const snap = detail.snapshot || {};
  const stepStartTime = detail.start_time != null ? (detail.start_time > 1e12 ? detail.start_time / 1000 : detail.start_time) : null;
  switch (tab) {
    case 'metrics': renderMetricsTab(metrics, container, stepStartTime); break;
    case 'live_search': renderLiveSearchTab(stepName, detail, container, state); break;
    case 'model': renderModelTab(snap.model, container); break;
    case 'ir_graph': renderIRGraphTab(snap.ir_graph, container); break;
    case 'hardware': renderHardwareTab(snap.hard_core_mapping, container, snap.ir_graph); break;
    case 'search': renderSearchTab(snap.search_result, container); break;
    case 'activations': renderActivationsTab(snap.activation_scales, snap.model, container); break;
    case 'adaptation': renderAdaptationTab(snap.adaptation_manager, snap.model, metrics, container); break;
    case 'constraints': renderConstraintsTab(snap.platform_constraints, container); break;
    case 'pruning': renderPruningTab(snap.pruning_layers, container); break;
    case 'summary': renderSummaryTab(snap.step_summary, container); break;
    default: container.innerHTML = '<div class="empty-state">No data</div>';
  }
}

// ── Live search tab ──────────────────────────────────────────────────────
function renderLiveSearchTab(stepName, detail, container, state) {
  initSearchLive(container);
  const fromState = (state.searchEvents && state.searchEvents[stepName]) || [];
  const events = fromState.length > 0 ? fromState : (detail._searchEvents || []);
  if (events.length > 0) replaySearchEvents(events);
}

// ── Metrics tab ──────────────────────────────────────────────────────────
function renderMetricsTab(metrics, container, stepStartTime) {
  const names = Object.keys(metrics).filter(n => n !== 'search_event');
  if (names.length === 0) { container.innerHTML = '<div class="empty-state">No metrics recorded</div>'; return; }
  const groups = groupMetricsByCategory(names);
  let html = '<div class="grid-2">';
  for (const group of Object.keys(groups))
    html += `<div class="card"><div class="card-header">${esc(group)}</div><div class="card-body"><div id="mc-${cssId(group)}" style="min-height:200px"></div></div></div>`;
  html += '</div>';
  container.innerHTML = html;
  for (const [group, metricNames] of Object.entries(groups)) plotMetricGroup(group, metricNames, metrics, stepStartTime);
}

function plotMetricGroup(group, metricNames, metrics, stepStartTime) {
  const el = document.getElementById(`mc-${cssId(group)}`);
  if (!el) return;
  const allTraces = metricNames.map(name => {
    let points = metrics[name] || [];
    if (stepStartTime != null) {
      points = points.filter(p => p.timestamp >= stepStartTime);
    }
    const x = stepStartTime != null ? points.map(p => (p.timestamp - stepStartTime)) : points.map((_, i) => i);
    const y = points.map(p => p.value);
    return { x, y, name, pointCount: points.length };
  });
  const xMax = computeGroupXMax(allTraces);
  const traces = allTraces.map(({ x, y, name }) => {
    let xOut = x, yOut = y;
    if (x.length === 1 && xMax != null) {
      xOut = [x[0], xMax];
      yOut = [y[0], y[0]];
    }
    return { x: xOut, y: yOut, name, type: 'scatter', mode: 'lines', line: { width: 1.5 } };
  });
  const layoutOpts = {
    showlegend: metricNames.length > 1,
    legend: { x: 1.02, y: 1, xanchor: 'left', orientation: 'v', font: { size: 10 } },
    margin: { r: 100 },
    height: 240,
    xaxis: { title: stepStartTime != null ? 'Elapsed (s)' : 'Index' },
  };
  if (group === 'Accuracy' || group === 'Adaptation') {
    layoutOpts.yaxis = { range: [0, 1] };
  }
  safeReact(el, traces, layoutOpts);
  // Record trace layout so the WS-driven fast path (updateLiveCharts)
  // can extendTraces incrementally instead of rebuilding the chart.
  el._mimarsinanTraceNames = metricNames.slice();
  el._mimarsinanTraceCounts = allTraces.map(t => t.pointCount);
}

function computeGroupXMax(traces) {
  let max = null;
  for (const t of traces) {
    if (t.x.length > 0) {
      const m = Math.max(...t.x);
      if (max == null || m > max) max = m;
    }
  }
  if (max != null && typeof max === 'number') {
    return max + (max === 0 ? 1 : Math.max(1, max * 0.1));
  }
  return null;
}

function groupMetricsByCategory(names) {
  const groups = {};
  const add = (cat, name) => { if (!groups[cat]) groups[cat] = []; groups[cat].push(name); };
  for (const name of names) {
    const l = name.toLowerCase();
    if (l.includes('loss')) add('Loss', name);
    else if (l.includes('adaptation target')) add('Accuracy', name);
    else if (l.includes('accuracy') || l.includes('acc')) add('Accuracy', name);
    else if (l === 'lr' || l.includes('learning rate')) add('Learning Rate', name);
    else if (l.includes('adaptation') || l.includes('tuning rate')) add('Adaptation', name);
    else if (l.includes('search')) add(`Search: ${name}`, name);  // one plot per search metric (different scales)
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
      html += `<tr><td>cores</td><td>${v.map(c => {
        const base = `[${c.count || '?'}×] ${c.max_axons || '?'}a × ${c.max_neurons || '?'}n`;
        return c.has_bias ? base + ' (bias)' : base;
      }).join(' &nbsp; ')}</td></tr>`;
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
