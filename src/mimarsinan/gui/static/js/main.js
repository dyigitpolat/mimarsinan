/* Mimarsinan Pipeline Monitor — Entry point.
 * State management, WebSocket, refresh loop, pipeline bar. */
import { esc, fmtDuration, elapsedFromStepStart } from './util.js';
import { renderPipelineBar, renderOverviewCards, renderConfig } from './overview.js';
import { refreshStepDetail, updateLiveCharts } from './step-detail.js';
import { syncSearchEventsFromState } from './search-live.js';
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
  pollOk: false,
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
    // 30 s watchdog — the server pushes pipeline_overview on every step
    // lifecycle event via WS, so this poll is purely a safety net for
    // cases where the WS connection drops silently.
    setInterval(refreshPipeline, 30000);
    setInterval(() => { if (state.activeMainTab === 'console') refreshConsoleLogs(); }, 2000);
  } else if (_isActiveRun) {
    // Active (subprocess) runs stream metrics + pipeline_overview via
    // /ws/active_runs/{rid} instead of polling, so the charts update at
    // ~20 Hz instead of 3-second batches. The 30 s watchdog still polls
    // the overview in case the WS drops silently.
    connectActiveRunWebSocket(state.historicalRunId);
    setInterval(refreshPipeline, 30000);
    setInterval(() => { if (state.activeMainTab === 'console') refreshConsoleLogs(); }, 2000);
  }
  setInterval(updateElapsedTimer, 1000);
});

// ── Refresh loop ─────────────────────────────────────────────────────────
// Leading-edge throttle for step-detail REST refetches. We *fire
// immediately* on the first call so step_started -> DOM scaffold latency
// is dominated by the REST RTT (~50 ms) rather than the old trailing
// 200 ms debounce. Subsequent calls inside the cooldown window coalesce
// into at most one trailing refresh — still enough to collapse a burst
// of WS lifecycle events (step_started -> step_completed) into a single
// follow-up fetch when metrics keep flooding.
const _DETAIL_COOLDOWN_MS = 200;
let _detailLastRunAt = 0;
let _detailTrailingTimer = null;
function scheduleStepDetailRefresh() {
  if (!state.selectedStep) return;
  const now = performance.now();
  const since = now - _detailLastRunAt;
  if (since >= _DETAIL_COOLDOWN_MS && !_detailTrailingTimer) {
    _detailLastRunAt = now;
    refreshStepDetail(state.selectedStep, state, fetchJSON).catch(() => {});
    return;
  }
  // Inside cooldown — queue exactly one trailing refresh so we don't
  // miss the last update in a burst.
  if (_detailTrailingTimer) return;
  const wait = Math.max(_DETAIL_COOLDOWN_MS - since, 0);
  _detailTrailingTimer = setTimeout(() => {
    _detailTrailingTimer = null;
    _detailLastRunAt = performance.now();
    if (state.selectedStep) {
      refreshStepDetail(state.selectedStep, state, fetchJSON).catch(() => {});
    }
  }, wait);
}

function applyPipelineOverviewFromWS(overview) {
  // Match the /api/pipeline payload shape so downstream renderers don't
  // care whether the data came over WS or HTTP.
  state.pipeline = {
    steps: overview.steps || [],
    current_step: overview.current_step,
    config: overview.config ?? state.pipeline?.config,
    is_alive: true,
  };
  renderPipelineBar(state.pipeline, state.selectedStep);
  renderOverviewCards(state.pipeline);
  if (state.activeMainTab === 'config') renderConfig(state.pipeline.config);

  if (state.autoFollow && state.pipeline.current_step) {
    const cur = state.pipeline.current_step;
    if (state.selectedStep !== cur) {
      state.selectedStep = cur;
      state.activeTab = null;
      state.lastDetailJSON = null;
      // Give the user immediate visual feedback that we've switched —
      // without this, the panel keeps showing the *previous* step's
      // data until the REST fetch returns (one RTT + any throttle
      // window), which reads as "the step switch is lagging by several
      // seconds". A lightweight placeholder is far better UX than
      // stale data.
      showStepDetailLoading(cur);
      scheduleStepDetailRefresh();
    }
  }
  updateErrorBanner(state.pipeline);
  if (!state.pollOk) { state.pollOk = true; updateConnectionDot(); }
}

// Render a short "loading" placeholder in the step-detail panel so the
// UI never appears frozen on the previous step during a transition.
// ``refreshStepDetail`` overwrites this the moment its REST response
// arrives.
function showStepDetailLoading(stepName) {
  const panel = document.getElementById('step-detail');
  if (!panel) return;
  const safe = stepName == null ? '' : String(stepName).replace(/[&<>"]/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
  ));
  panel.innerHTML = `
    <div class="step-detail-header">
      <h2>${safe}</h2>
      <span class="badge running">loading…</span>
    </div>
    <div class="tabs" id="step-tabs"></div>
    <div id="step-tab-content"><div class="empty-state">Loading step detail…</div></div>`;
}

let _prevAlive = true;

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

    updateErrorBanner(state.pipeline);

    if (_prevAlive && state.pipeline.is_alive === false) {
      await refreshConsoleLogs();
    }
    _prevAlive = !!state.pipeline.is_alive;

    if (!state.pollOk) { state.pollOk = true; updateConnectionDot(); }
  } catch (e) {
    console.error('Refresh failed:', e);
    if (state.pollOk) { state.pollOk = false; updateConnectionDot(); }
  }
}

function updateErrorBanner(pipeline) {
  let banner = document.getElementById('pipeline-error-banner');
  if (!pipeline || pipeline.is_alive || !pipeline.error) {
    if (banner) banner.remove();
    return;
  }
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'pipeline-error-banner';
    banner.style.cssText = 'padding:10px 32px;background:rgba(248,113,113,0.10);border-bottom:2px solid rgba(248,113,113,0.4);color:var(--error);font-size:0.85rem;font-weight:600;';
    const header = document.querySelector('.header');
    if (header) header.parentNode.insertBefore(banner, header.nextSibling);
    else document.body.prepend(banner);
  }
  banner.textContent = 'Pipeline failed: ' + pipeline.error;
}

// ── WebSocket ────────────────────────────────────────────────────────────
function connectWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => { state.connected = true; state.ws = ws; updateConnectionDot(); };
  ws.onclose = () => { state.connected = false; updateConnectionDot(); setTimeout(connectWebSocket, 3000); };
  ws.onmessage = (evt) => { try { handleWSMessage(JSON.parse(evt.data)); } catch (e) { /* ignore parse errors */ } };
}

// Subprocess-spawned active runs use a dedicated WS endpoint that tails
// the child's live_metrics.jsonl and steps.json for per-event push,
// eliminating the 3-second polling jank. The reconnect loop is identical
// to the in-process /ws channel so a momentary disconnect auto-recovers.
function connectActiveRunWebSocket(runId) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${location.host}/ws/active_runs/${encodeURIComponent(runId)}`;
  const ws = new WebSocket(url);
  ws.onopen = () => { state.connected = true; state.ws = ws; updateConnectionDot(); };
  ws.onclose = () => {
    state.connected = false;
    updateConnectionDot();
    setTimeout(() => connectActiveRunWebSocket(runId), 3000);
  };
  ws.onmessage = (evt) => { try { handleWSMessage(JSON.parse(evt.data)); } catch (e) { /* ignore parse errors */ } };
}

function handleWSMessage(msg) {
  if (msg.type === 'pipeline_overview') {
    // The collector piggy-backs a full overview on every step lifecycle
    // event, so we can update the bar + cards synchronously without
    // touching the network. This is what lets us drop the 5 s poll to
    // a 30 s watchdog.
    applyPipelineOverviewFromWS(msg);
    return;
  }
  if (msg.type === 'step_started') {
    delete state.metricBuffers[msg.step];
    delete state.seenSeqs[msg.step];
    if (state.searchEvents) delete state.searchEvents[msg.step];
    if (state.autoFollow) {
      state.selectedStep = msg.step;
      state.activeTab = null;
      state.lastDetailJSON = null;
      // Swap the panel to a loading placeholder right now — otherwise
      // the previous step's DOM (and its stale empty-state "No metrics
      // recorded") would remain visible until the REST response lands,
      // which users perceive as the monitor being frozen for seconds.
      showStepDetailLoading(msg.step);
    }
    // Step detail (metrics, snapshot) still needs a REST fetch — the
    // WS overview only carries step lifecycle state, not the heavy
    // per-step payload.
    scheduleStepDetailRefresh();
  }
  if (msg.type === 'step_completed' || msg.type === 'step_failed') scheduleStepDetailRefresh();
  if (msg.type === 'metric') {
    bufferMetric(msg.step, msg.name, msg.value, msg.seq, msg.timestamp);
    if (state.selectedStep === msg.step) {
      scheduleLiveChartUpdate(msg.step);
      // If the step-detail panel hasn't rendered yet (i.e. the metrics
      // tab DOM scaffold isn't in place), buffered metrics would be
      // invisible until the next refresh. Kick a leading-edge refresh
      // right now so the user sees points the moment they arrive,
      // instead of waiting for a step_started debounce window.
      const panel = document.getElementById('step-detail');
      if (!panel || !panel.querySelector('.step-detail-header')) {
        scheduleStepDetailRefresh();
      }
    }
  }
  if (msg.type === 'console_log') {
    state.consoleOffset++;
    if (state.activeMainTab === 'console') appendConsoleLogs([msg]);
  }
}

function bufferMetric(step, name, value, seq, timestamp) {
  if (!state.metricBuffers[step]) state.metricBuffers[step] = {};
  if (!state.seenSeqs[step]) state.seenSeqs[step] = new Set();
  if (!state.searchEvents) state.searchEvents = {};
  if (seq != null && state.seenSeqs[step].has(seq)) return;
  if (seq != null) state.seenSeqs[step].add(seq);

  if (name === 'search_event') {
    if (!state.searchEvents[step]) state.searchEvents[step] = [];
    try {
      const parsed = typeof value === 'string' ? JSON.parse(value) : value;
      state.searchEvents[step].push(parsed);
      if (state.selectedStep === step) syncSearchEventsFromState(step, state);
    } catch (_) { /* skip malformed */ }
    if (!state.metricBuffers[step][name]) state.metricBuffers[step][name] = [];
    state.metricBuffers[step][name].push({ seq, timestamp: timestamp || Date.now() / 1000, value });
    return;
  }

  if (!state.metricBuffers[step][name]) state.metricBuffers[step][name] = [];
  state.metricBuffers[step][name].push({ seq, timestamp: timestamp || Date.now() / 1000, value: parseFloat(value) });
}

// Coalesce bursts of metric messages into a single repaint per animation
// frame. rAF is the natural rate for smooth UI updates, replaces the old
// 500 ms debounce which produced visibly jerky lines during heavy
// training metric rates.
let _liveRaf = 0;
let _liveDirtyStep = null;
function scheduleLiveChartUpdate(stepName) {
  _liveDirtyStep = stepName;
  if (_liveRaf) return;
  _liveRaf = requestAnimationFrame(() => {
    _liveRaf = 0;
    const target = _liveDirtyStep;
    _liveDirtyStep = null;
    if (target) updateLiveCharts(target, state);
  });
}

// ── UI helpers ───────────────────────────────────────────────────────────
function updateConnectionDot() {
  const dot = document.getElementById('conn-dot');
  if (!dot) return;
  // Live runs and active subprocess runs both stream over WS now;
  // only finished/historical runs rely on HTTP polling.
  const usesWebSocket = !state.historicalRunId || _isActiveRun;
  const healthy = usesWebSocket ? state.connected : state.pollOk;
  dot.className = 'status-dot ' + (healthy ? 'connected' : 'disconnected');
  dot.title = usesWebSocket ? 'WebSocket' : 'HTTP polling';
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
  const navLinks = `<a href="/" style="color:var(--text-secondary);font-size:0.78rem;text-decoration:none">Home</a>`;
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
