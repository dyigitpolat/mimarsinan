/* Mimarsinan Pipeline Monitor — Entry point.
 * State management, WebSocket, refresh loop, pipeline bar. */
import { esc, fmtDuration, elapsedFromStepStart } from './util.js';
import { renderPipelineBar, renderOverviewCards, renderConfig } from './overview.js';
import { refreshStepDetail, updateLiveCharts } from './step-detail.js';

// ── Historical run mode ──────────────────────────────────────────────────
const _params = new URLSearchParams(window.location.search);
const _historicalRunId = _params.get('run_id') || null;

// ── Global state ─────────────────────────────────────────────────────────
const state = {
  pipeline: null,
  selectedStep: null,
  activeTab: null,
  activeMainTab: 'overview',
  autoFollow: !_historicalRunId,
  ws: null,
  metricBuffers: {},
  seenSeqs: {},
  connected: false,
  lastDetailJSON: null,
  historicalRunId: _historicalRunId,
  isActiveRun: false,
};

let _isActiveRun = false;

function apiUrl(path) {
  if (state.historicalRunId) {
    if (_isActiveRun) return '/api/active_runs/' + encodeURIComponent(state.historicalRunId) + path;
    return '/api/runs/' + encodeURIComponent(state.historicalRunId) + path;
  }
  return '/api' + path;
}

async function fetchJSON(url) { return (await fetch(url)).json(); }

// ── Init ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  setupPipelineBarClicks();
  setupMainTabs();
  document.getElementById('auto-follow-btn').addEventListener('click', toggleAutoFollow);

  if (state.historicalRunId) {
    const activeCheck = await fetch('/api/active_runs/' + encodeURIComponent(state.historicalRunId) + '/pipeline').then(r => r.ok).catch(() => false);
    if (activeCheck) {
      _isActiveRun = true;
      state.isActiveRun = true;
      state.autoFollow = true;
    }
    setupHistoricalBanner();
  }

  await refreshPipeline();
  if (!state.historicalRunId) {
    connectWebSocket();
    setInterval(refreshPipeline, 5000);
  } else if (_isActiveRun) {
    setInterval(refreshPipeline, 3000);
  }
  setInterval(updateElapsedTimer, 1000);
});

// ── Refresh loop ─────────────────────────────────────────────────────────
let _refreshTimer = null;
function scheduleRefresh() {
  if (_refreshTimer) return;
  _refreshTimer = setTimeout(() => { _refreshTimer = null; refreshPipeline(); }, 200);
}

async function refreshPipeline() {
  try {
    state.pipeline = await fetchJSON(apiUrl('/pipeline'));
    renderPipelineBar(state.pipeline, state.selectedStep);
    renderOverviewCards(state.pipeline);
    if (state.activeMainTab === 'config') renderConfig(state.pipeline?.config);

    if (state.autoFollow && state.pipeline.current_step) {
      const cur = state.pipeline.current_step;
      if (state.selectedStep !== cur) {
        state.selectedStep = cur;
        state.activeTab = null;
        state.lastDetailJSON = null;
      }
    }
    if (state.selectedStep) await refreshStepDetail(state.selectedStep, state, fetchJSON);
  } catch (e) {
    console.error('Refresh failed:', e);
  }
}

// ── WebSocket ────────────────────────────────────────────────────────────
function connectWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => { state.connected = true; state.ws = ws; updateConnectionDot(); };
  ws.onclose = () => { state.connected = false; updateConnectionDot(); setTimeout(connectWebSocket, 3000); };
  ws.onmessage = (evt) => { try { handleWSMessage(JSON.parse(evt.data)); } catch (e) { /* ignore parse errors */ } };
}

function handleWSMessage(msg) {
  if (msg.type === 'step_started') {
    delete state.metricBuffers[msg.step];
    delete state.seenSeqs[msg.step];
    if (state.autoFollow) { state.selectedStep = msg.step; state.activeTab = null; state.lastDetailJSON = null; }
    scheduleRefresh();
  }
  if (msg.type === 'step_completed' || msg.type === 'step_failed') scheduleRefresh();
  if (msg.type === 'metric') {
    bufferMetric(msg.step, msg.name, msg.value, msg.seq, msg.timestamp);
    if (state.selectedStep === msg.step) scheduleLiveChartUpdate(msg.step);
  }
}

function bufferMetric(step, name, value, seq, timestamp) {
  if (!state.metricBuffers[step]) state.metricBuffers[step] = {};
  if (!state.seenSeqs[step]) state.seenSeqs[step] = new Set();
  if (!state.metricBuffers[step][name]) state.metricBuffers[step][name] = [];
  if (seq != null && state.seenSeqs[step].has(seq)) return;
  if (seq != null) state.seenSeqs[step].add(seq);
  state.metricBuffers[step][name].push({ seq, timestamp: timestamp || Date.now() / 1000, value: parseFloat(value) });
}

let _liveTimer = null;
function scheduleLiveChartUpdate(stepName) {
  if (_liveTimer) return;
  _liveTimer = setTimeout(() => { _liveTimer = null; updateLiveCharts(stepName, state); }, 500);
}

// ── UI helpers ───────────────────────────────────────────────────────────
function updateConnectionDot() {
  const dot = document.getElementById('conn-dot');
  if (dot) dot.className = 'status-dot ' + (state.connected ? 'connected' : 'disconnected');
}

function updateElapsedTimer() {
  if (!state.pipeline) return;
  const running = (state.pipeline.steps || []).find(s => s.status === 'running');
  const el = document.getElementById('elapsed-time');
  if (!el) return;
  if (running?.start_time != null) {
    const elapsed = elapsedFromStepStart(running.start_time);
    el.textContent = fmtDuration(elapsed);
    el.style.display = 'inline';
  } else el.style.display = 'none';
}

function setupMainTabs() {
  const tabBar = document.getElementById('main-tabs');
  const overviewPane = document.getElementById('main-tab-overview');
  const configPane = document.getElementById('main-tab-config');
  if (!tabBar || !overviewPane || !configPane) return;
  tabBar.addEventListener('click', (e) => {
    const btn = e.target.closest('.tab-btn[data-main-tab]');
    if (!btn) return;
    const tab = btn.dataset.mainTab;
    if (tab !== 'overview' && tab !== 'config') return;
    state.activeMainTab = tab;
    tabBar.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.mainTab === tab);
    });
    overviewPane.classList.toggle('active', tab === 'overview');
    configPane.classList.toggle('active', tab === 'config');
    if (tab === 'config') renderConfig(state.pipeline?.config);
  });
}

function setupPipelineBarClicks() {
  document.getElementById('pipeline-bar').addEventListener('click', (e) => {
    const block = e.target.closest('.step-block');
    if (!block) return;
    const stepName = block.dataset.step;
    if (!stepName) return;
    state.autoFollow = false;
    updateAutoFollowBtn();
    if (state.selectedStep === stepName) return;
    state.selectedStep = stepName;
    state.activeTab = null;
    state.lastDetailJSON = null;
    renderPipelineBar(state.pipeline, state.selectedStep);
    refreshStepDetail(stepName, state, fetchJSON);
  });
}

function toggleAutoFollow() {
  state.autoFollow = !state.autoFollow;
  updateAutoFollowBtn();
  if (state.autoFollow && state.pipeline?.current_step) {
    state.selectedStep = state.pipeline.current_step;
    state.activeTab = null;
    state.lastDetailJSON = null;
    renderPipelineBar(state.pipeline, state.selectedStep);
    refreshStepDetail(state.selectedStep, state, fetchJSON);
  }
}

function updateAutoFollowBtn() {
  const btn = document.getElementById('auto-follow-btn');
  if (btn) { btn.classList.toggle('active', state.autoFollow); btn.textContent = state.autoFollow ? 'Following' : 'Follow'; }
}

function setupHistoricalBanner() {
  const header = document.querySelector('.header');
  if (!header) return;
  const banner = document.createElement('div');
  if (_isActiveRun) {
    banner.style.cssText = 'padding:8px 32px;background:rgba(34,211,238,0.08);border-bottom:1px solid rgba(34,211,238,0.25);font-size:0.82rem;color:#22d3ee;display:flex;align-items:center;gap:12px;';
    banner.innerHTML = `<span style="font-weight:600">Active run:</span> ${esc(state.historicalRunId)} <span style="margin-left:auto;display:flex;gap:12px"><a href="/monitor" style="color:var(--accent-blue);font-size:0.78rem;text-decoration:none;">Default monitor</a> <a href="/" style="color:var(--text-dim);font-size:0.78rem;text-decoration:none;">Home</a></span>`;
  } else {
    banner.style.cssText = 'padding:8px 32px;background:rgba(139,92,246,0.1);border-bottom:1px solid rgba(139,92,246,0.3);font-size:0.82rem;color:#8b5cf6;display:flex;align-items:center;gap:12px;';
    banner.innerHTML = `<span style="font-weight:600">Historical run:</span> ${esc(state.historicalRunId)} <span style="margin-left:auto;display:flex;gap:12px"><a href="/monitor" style="color:var(--accent-blue);font-size:0.78rem;text-decoration:none;">Live monitor</a> <a href="/" style="color:var(--text-dim);font-size:0.78rem;text-decoration:none;">Home</a></span>`;
  }
  header.parentNode.insertBefore(banner, header.nextSibling);
}
