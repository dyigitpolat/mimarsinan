/* Mimarsinan Pipeline Monitor — Entry point.
 * State management, WebSocket, refresh loop, pipeline bar. */
import { esc, fmtDuration, elapsedFromStepStart } from './util.js';
import { renderPipelineBar, renderOverviewCards, renderConfig } from './overview.js';
import { refreshStepDetail, updateLiveCharts } from './step-detail.js';
import { appendConsoleLogs, clearConsoleLogs } from './console-tab.js';

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
  consoleOffset: 0,
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
  document.getElementById('console-clear-btn')?.addEventListener('click', () => {
    clearConsoleLogs();
    state.consoleOffset = 0;
  });

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
    setInterval(() => { if (state.activeMainTab === 'console') refreshConsoleLogs(); }, 2000);
  } else if (_isActiveRun) {
    setInterval(refreshPipeline, 3000);
    setInterval(() => { if (state.activeMainTab === 'console') refreshConsoleLogs(); }, 2000);
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
  if (msg.type === 'console_log') {
    state.consoleOffset++;
    if (state.activeMainTab === 'console') appendConsoleLogs([msg]);
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
  const consolePane = document.getElementById('main-tab-console');
  if (!tabBar || !overviewPane || !configPane) return;
  tabBar.addEventListener('click', (e) => {
    const btn = e.target.closest('.tab-btn[data-main-tab]');
    if (!btn) return;
    const tab = btn.dataset.mainTab;
    if (tab !== 'overview' && tab !== 'config' && tab !== 'console') return;
    state.activeMainTab = tab;
    tabBar.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.mainTab === tab);
    });
    overviewPane.classList.toggle('active', tab === 'overview');
    configPane.classList.toggle('active', tab === 'config');
    if (consolePane) consolePane.classList.toggle('active', tab === 'console');
    if (tab === 'config') renderConfig(state.pipeline?.config);
    if (tab === 'console') refreshConsoleLogs();
  });
}

async function refreshConsoleLogs() {
  try {
    const url = apiUrl('/console') + '?offset=' + state.consoleOffset;
    const entries = await fetchJSON(url);
    if (entries && entries.length > 0) {
      state.consoleOffset += entries.length;
      appendConsoleLogs(entries);
    }
  } catch (e) {
    // silent — console polling is best-effort
  }
}

function setupPipelineBarClicks() {
  document.getElementById('pipeline-bar').addEventListener('click', (e) => {
    const block = e.target.closest('.psb-col');
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
  const rid = state.historicalRunId;
  const eRid = esc(rid);
  const navLinks = `<a href="/monitor" style="color:var(--accent);font-size:0.78rem;text-decoration:none">Live monitor</a>
    <a href="/" style="color:var(--text-secondary);font-size:0.78rem;text-decoration:none">Home</a>`;
  if (_isActiveRun) {
    banner.style.cssText = 'padding:8px 32px;background:rgba(34,211,238,0.06);border-bottom:1px solid rgba(34,211,238,0.2);font-size:0.82rem;color:var(--accent-cyan);display:flex;align-items:center;gap:12px;';
    banner.innerHTML = `<span style="font-weight:600">Active run:</span> ${eRid}
      <span style="margin-left:auto;display:flex;gap:12px;align-items:center">
        <button id="banner-stop-btn" style="background:rgba(248,113,113,0.12);color:var(--error);border:1px solid rgba(248,113,113,0.3);padding:3px 12px;border-radius:6px;font-size:0.75rem;font-weight:600;cursor:pointer">Stop</button>
        ${navLinks}
      </span>`;
  } else {
    banner.style.cssText = 'padding:8px 32px;background:rgba(139,92,246,0.06);border-bottom:1px solid rgba(139,92,246,0.2);font-size:0.82rem;color:var(--accent-purple);display:flex;align-items:center;gap:12px;';
    banner.innerHTML = `<span style="font-weight:600">Historical run:</span> ${eRid}
      <span style="margin-left:auto;display:flex;gap:12px;align-items:center">
        <button id="banner-save-tpl-btn" style="background:var(--bg-hover);color:var(--text-primary);border:1px solid var(--border);padding:3px 12px;border-radius:6px;font-size:0.75rem;font-weight:500;cursor:pointer">Save as Template</button>
        ${navLinks}
      </span>`;
  }
  header.parentNode.insertBefore(banner, header.nextSibling);

  const stopBtn = document.getElementById('banner-stop-btn');
  if (stopBtn) stopBtn.addEventListener('click', async () => {
    if (!confirm('Stop this run?')) return;
    await fetch('/api/active_runs/' + encodeURIComponent(rid), { method: 'DELETE' });
    stopBtn.textContent = 'Stopped';
    stopBtn.disabled = true;
  });

  const saveTplBtn = document.getElementById('banner-save-tpl-btn');
  if (saveTplBtn) saveTplBtn.addEventListener('click', async () => {
    try {
      const cfg = await fetchJSON('/api/runs/' + encodeURIComponent(rid) + '/config');
      if (!cfg || cfg.error) { alert('Cannot load config'); return; }
      const name = prompt('Template name:', cfg.experiment_name || rid);
      if (!name) return;
      await fetch('/api/templates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, config: cfg }),
      });
      alert('Template saved!');
    } catch (e) { alert('Failed: ' + e.message); }
  });
}
