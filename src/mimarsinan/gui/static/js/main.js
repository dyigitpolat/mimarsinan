/* Mimarsinan Pipeline Monitor — Entry point.
 * State management, WebSocket, refresh loop, pipeline bar. */
import { esc, fmtDuration } from './util.js';
import { renderPipelineBar, renderOverviewCards } from './overview.js';
import { refreshStepDetail, updateLiveCharts } from './step-detail.js';

// ── Global state ─────────────────────────────────────────────────────────
const state = {
  pipeline: null,
  selectedStep: null,
  activeTab: null,
  autoFollow: true,
  ws: null,
  metricBuffers: {},
  seenSeqs: {},
  connected: false,
  lastDetailJSON: null,
};

async function fetchJSON(url) { return (await fetch(url)).json(); }

// ── Init ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  setupPipelineBarClicks();
  document.getElementById('auto-follow-btn').addEventListener('click', toggleAutoFollow);
  await refreshPipeline();
  connectWebSocket();
  setInterval(refreshPipeline, 5000);
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
    state.pipeline = await fetchJSON('/api/pipeline');
    renderPipelineBar(state.pipeline, state.selectedStep);
    renderOverviewCards(state.pipeline);

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
  if (running?.start_time) { el.textContent = fmtDuration(Date.now() / 1000 - running.start_time); el.style.display = 'inline'; }
  else el.style.display = 'none';
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
