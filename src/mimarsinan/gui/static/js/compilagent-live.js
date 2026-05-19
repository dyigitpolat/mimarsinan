/* Live monitor for the compilagent optimizer.
 *
 * Consumes the `compilagent_*` event vocabulary emitted by
 * `mimarsinan.search.optimizers.compilagent.sink.MultiObjectiveSink`. The
 * AgentEvolve live view (search-live.js) is intentionally untouched —
 * the two monitors do not share a DOM tree or state.
 *
 * Layout (top → bottom, single column):
 *   1. COMPILAGENT header (model, harness, candidate budget, elapsed)
 *   2. Live Pareto strip (one chip per non-dominated objective tuple)
 *   3. Candidate grid (one tile per agent-proposed plan)
 *   4. Activity feed (chronological tool calls + agent thinking + compile phases)
 *
 * Every event handler is idempotent: replaying the captured event list
 * after a tab switch produces the exact same DOM.
 */

import { esc, renderMarkdown } from './util.js';

// ── Module state ─────────────────────────────────────────────────────────
let _container = null;
let _candidateGridEl = null;
let _activityFeedEl = null;
let _paretoStripEl = null;
let _headerEl = null;
let _agentStreamEl = null;
let _leadersStripEl = null;

let _candidates = {};       // candidate_id -> {tile, idx, ...}
let _candidateOrder = [];   // insertion order
let _activityCounter = 0;
let _appliedSearchEventCount = 0;
let _sessionMeta = null;
let _objectiveSpecs = [];
let _totals = { proposed: 0, succeeded: 0, failed: 0, pareto: 0, max: 0 };
let _startTime = null;
let _timerInterval = null;
let _agentTextBuffer = '';
let _agentThinkingBuffer = '';
let _activeAgentPart = null;     // 'thinking' | 'text' | null
// Bulk-replay flag: when set, expensive per-event re-renders (markdown,
// per-metric ranks, leader strip) are deferred until the replay finishes
// so opening the tab is O(N) instead of O(N²).
let _replayBusy = false;
let _agentStreamDirty = false;
let _metricRanksDirty = false;
let _paretoDirty = false;

// ── Public API ───────────────────────────────────────────────────────────

export function initCompilagentLive(container) {
  _container = container;
  _candidates = {};
  _candidateOrder = [];
  _activityCounter = 0;
  _appliedSearchEventCount = 0;
  _sessionMeta = null;
  _objectiveSpecs = [];
  _totals = { proposed: 0, succeeded: 0, failed: 0, pareto: 0, max: 0 };
  _startTime = Date.now();
  _agentTextBuffer = '';
  _agentThinkingBuffer = '';
  _activeAgentPart = null;

  container.innerHTML = `
    <div class="cl-tracker">
      <div class="cl-header" id="cl-header">
        <div class="cl-header-left">
          <span class="cl-logo">⬢</span>
          <span class="cl-title">COMPILAGENT</span>
          <span class="cl-subtitle" id="cl-subtitle">awaiting session…</span>
        </div>
        <div class="cl-header-stats">
          <div class="cl-stat"><span class="cl-stat-label">PROPOSED</span><span class="cl-stat-val" id="cl-stat-proposed">0</span></div>
          <div class="cl-stat"><span class="cl-stat-label">VALID</span><span class="cl-stat-val cl-val-ok" id="cl-stat-succeeded">0</span></div>
          <div class="cl-stat"><span class="cl-stat-label">FAILED</span><span class="cl-stat-val cl-val-fail" id="cl-stat-failed">0</span></div>
          <div class="cl-stat"><span class="cl-stat-label">PARETO</span><span class="cl-stat-val cl-val-pareto" id="cl-stat-pareto">0</span></div>
          <div class="cl-stat"><span class="cl-stat-label">BUDGET</span><span class="cl-stat-val" id="cl-stat-budget">0/0</span></div>
          <div class="cl-stat"><span class="cl-stat-label">ELAPSED</span><span class="cl-stat-val" id="cl-stat-elapsed">0s</span></div>
        </div>
      </div>

      <div class="cl-pareto-strip" id="cl-pareto-strip">
        <div class="cl-pareto-title">Live Pareto front</div>
        <div class="cl-pareto-inner" id="cl-pareto-inner">
          <div class="cl-pareto-placeholder">No non-dominated candidates yet — waiting for the first compile to succeed.</div>
        </div>
      </div>

      <div class="cl-leaders-strip" id="cl-leaders-strip">
        <div class="cl-leaders-title">Metric leaders (best so far per axis)</div>
        <div class="cl-leaders-inner" id="cl-leaders-inner">
          <div class="cl-leaders-placeholder">No scored candidates yet.</div>
        </div>
      </div>

      <div class="cl-twocol">
        <div class="cl-col cl-col-main">
          <div class="cl-section-title">Candidates</div>
          <div class="cl-candidates" id="cl-candidates"></div>
          <div class="cl-empty" id="cl-candidates-empty">Waiting for the agent to propose its first candidate…</div>
        </div>
        <div class="cl-col cl-col-side">
          <div class="cl-section-title">Agent stream</div>
          <div class="cl-agent-stream" id="cl-agent-stream">
            <div class="cl-agent-empty">No tokens streamed yet.</div>
          </div>
          <div class="cl-section-title cl-section-title-spaced">Activity</div>
          <div class="cl-activity-feed" id="cl-activity-feed"></div>
        </div>
      </div>
    </div>`;

  _headerEl = container.querySelector('#cl-header');
  _candidateGridEl = container.querySelector('#cl-candidates');
  _activityFeedEl = container.querySelector('#cl-activity-feed');
  _paretoStripEl = container.querySelector('#cl-pareto-inner');
  _agentStreamEl = container.querySelector('#cl-agent-stream');
  _leadersStripEl = container.querySelector('#cl-leaders-inner');

  if (_timerInterval) clearInterval(_timerInterval);
  _timerInterval = setInterval(_updateElapsed, 1000);
}

export function isCompilagentLiveActive() {
  return _container !== null && _container.isConnected;
}

export function detachCompilagentLive() {
  _container = null;
  _headerEl = null;
  _candidateGridEl = null;
  _activityFeedEl = null;
  _paretoStripEl = null;
  _agentStreamEl = null;
  _leadersStripEl = null;
  _appliedSearchEventCount = 0;
  if (_timerInterval) { clearInterval(_timerInterval); _timerInterval = null; }
}

export function syncCompilagentEventsFromState(stepName, state) {
  if (!_container || !_container.isConnected) {
    _container = null;
    return;
  }
  const evs = (state.searchEvents && state.searchEvents[stepName]) || [];
  if (evs.length < _appliedSearchEventCount) _appliedSearchEventCount = 0;
  for (let i = _appliedSearchEventCount; i < evs.length; i++) {
    _handle(evs[i]);
    if (!_container || !_container.isConnected) {
      _appliedSearchEventCount = i + 1;
      return;
    }
  }
  _appliedSearchEventCount = evs.length;
}

export function replayCompilagentEvents(events) {
  // Bulk replay: suppress heavy re-renders that get triggered per-event
  // (agent stream markdown, per-metric rank chips, leaders strip).
  // We do one flush at the end. Without this, opening the Live Search
  // tab on a session with hundreds of candidates / thousands of agent
  // tokens stalls the main thread for several seconds.
  _replayBusy = true;
  _agentStreamDirty = false;
  _metricRanksDirty = false;
  _paretoDirty = false;
  try {
    for (const ev of events) _handle(ev);
  } finally {
    _replayBusy = false;
  }
  if (_agentStreamDirty) {
    _agentStreamDirty = false;
    _renderAgentStream();
  }
  if (_metricRanksDirty) {
    _metricRanksDirty = false;
    _refreshMetricLeaders();
    _refreshPerMetricRanks();
  }
  if (_paretoDirty) {
    _paretoDirty = false;
    _refreshLivePareto();
  }
  _appliedSearchEventCount = events.length;
}

/** Detect whether an event belongs to the compilagent vocabulary. */
export function isCompilagentEvent(ev) {
  if (!ev || typeof ev !== 'object') return false;
  return typeof ev.type === 'string' && ev.type.startsWith('compilagent_');
}

// ── Event dispatch ───────────────────────────────────────────────────────

function _handle(ev) {
  if (!_container || !_container.isConnected) {
    _container = null;
    return;
  }
  const handler = _handlers[ev.type];
  if (handler) handler(ev);
}

const _handlers = {
  compilagent_session_start(ev) {
    _sessionMeta = ev;
    _objectiveSpecs = ev.objectives || [];
    _totals.max = ev.max_candidates || 0;
    _updateHeader();
    _appendActivity({
      kind: 'session',
      label: 'Session started',
      detail: `model=${ev.model}, harness=${ev.harness}, primary=${ev.primary_objective}, max_candidates=${ev.max_candidates}`,
    });
  },

  compilagent_session_observed(ev) {
    _appendActivity({
      kind: 'session',
      label: 'Session bootstrapped',
      detail: `run_id=${ev.run_id || '?'}`,
    });
  },

  compilagent_search_space_derived(ev) {
    _appendActivity({
      kind: 'session',
      label: 'Search space derived',
      detail: `${ev.lever_count} lever${ev.lever_count === 1 ? '' : 's'}`,
    });
  },

  compilagent_candidate_proposed(ev) {
    _totals.proposed = Math.max(_totals.proposed, (ev.idx || 0) + 1);
    _ensureCandidate(ev.candidate_id, ev.idx, {
      description: ev.description,
      expected_effect: ev.expected_effect,
      interventions: ev.interventions || [],
    });
    _updateHeader();
    _hideEmptyCandidates();
    _appendActivity({
      kind: 'candidate',
      label: `Candidate #${ev.idx + 1} proposed`,
      detail: ev.description || '(no description)',
      candidateId: ev.candidate_id,
    });
  },

  compilagent_candidate_compiling(ev) {
    const c = _candidates[ev.candidate_id];
    if (c) {
      c.tile.classList.add('cl-cand-compiling');
      _setStatus(c, 'compiling…');
    }
    _appendActivity({
      kind: 'compile',
      label: `Candidate #${(ev.idx || 0) + 1} compile started`,
      candidateId: ev.candidate_id,
    });
  },

  compilagent_candidate_compiled(ev) {
    const c = _ensureCandidate(ev.candidate_id, ev.idx, {});
    c.tile.classList.remove('cl-cand-compiling');
    c.compiled = true;
    c.compile_ok = ev.ok;
    c.diagnostics = ev.diagnostics;
    if (!ev.ok) {
      _setStatus(c, 'compile failed');
      c.tile.classList.add('cl-cand-fail');
      const diagBlock = c.tile.querySelector('.cl-cand-diag');
      if (diagBlock) {
        diagBlock.classList.remove('cl-hidden');
        diagBlock.textContent = ev.diagnostics || 'no diagnostics';
      }
    } else {
      _setStatus(c, 'compiled');
    }
    _appendActivity({
      kind: ev.ok ? 'compile' : 'compile-fail',
      label: `Candidate #${(ev.idx || 0) + 1} compile ${ev.ok ? 'ok' : 'failed'}`,
      detail: ev.diagnostics || (ev.elapsed_ms != null ? `${ev.elapsed_ms.toFixed(1)} ms` : ''),
      candidateId: ev.candidate_id,
    });
  },

  compilagent_candidate_benchmarked(ev) {
    const c = _candidates[ev.candidate_id];
    if (c) {
      c.median_ms = ev.median_ms;
      _setStatus(c, `primary=${(ev.median_ms ?? 0).toFixed(2)}`);
    }
  },

  compilagent_candidate_objectives(ev) {
    const c = _ensureCandidate(ev.candidate_id, ev.idx, {});
    c.objectives = ev.objectives || {};
    c.metadata = ev.metadata || {};
    c.tile.classList.add('cl-cand-ok');
    _renderCandidateObjectives(c);
    _setStatus(c, 'evaluated');
    _totals.succeeded = Object.values(_candidates).filter(
      x => x.compile_ok !== false && x.objectives && Object.keys(x.objectives).length > 0,
    ).length;
    _updateHeader();
    // Recompute per-metric ranks and the live Pareto front across the
    // population. During bulk replay we mark dirty and flush once.
    if (_replayBusy) {
      _metricRanksDirty = true;
      _paretoDirty = true;
    } else {
      _refreshMetricLeaders();
      _refreshPerMetricRanks();
      _refreshLivePareto();
    }
    _appendActivity({
      kind: 'objectives',
      label: `Candidate #${(ev.idx || 0) + 1} objectives recorded`,
      detail: Object.entries(ev.objectives || {})
        .slice(0, 3)
        .map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`)
        .join(', '),
      candidateId: ev.candidate_id,
    });
  },

  compilagent_candidate_rejected(ev) {
    const c = _ensureCandidate(ev.candidate_id, ev.idx, {});
    c.tile.classList.remove('cl-cand-compiling');
    c.tile.classList.add('cl-cand-fail');
    c.rejected = true;
    c.reject_reason = ev.reason;
    _setStatus(c, ev.reason || 'rejected');
    const diagBlock = c.tile.querySelector('.cl-cand-diag');
    if (diagBlock && (ev.diagnostics || ev.reason)) {
      diagBlock.classList.remove('cl-hidden');
      diagBlock.textContent = ev.diagnostics || ev.reason;
    }
    _totals.failed = Object.values(_candidates).filter(x => x.rejected).length;
    _updateHeader();
    _appendActivity({
      kind: 'rejected',
      label: `Candidate #${(ev.idx || 0) + 1} rejected`,
      detail: ev.diagnostics || ev.reason,
      candidateId: ev.candidate_id,
    });
  },

  compilagent_compile_phase(ev) {
    _appendActivity({
      kind: 'compile-phase',
      label: `Compile phase: ${ev.stage || ev.name}`,
      detail: `${ev.duration_ms != null ? ev.duration_ms.toFixed(2) + ' ms' : ''}`,
      candidateId: ev.candidate_id,
    });
  },

  compilagent_tool_call(ev) {
    _appendActivity({
      kind: 'tool',
      label: `Tool ${ev.phase}: ${ev.tool_name}`,
      detail: ev.candidate_id ? `→ ${ev.candidate_id}` : '',
      candidateId: ev.candidate_id,
    });
  },

  compilagent_run_progress(ev) {
    _totals.succeeded = ev.successful_count ?? _totals.succeeded;
    _totals.failed = ev.failed_attempts ?? _totals.failed;
    _totals.max = ev.max_candidates ?? _totals.max;
    _updateHeader();
  },

  compilagent_continuation(ev) {
    _appendActivity({
      kind: 'session',
      label: `Continuation #${ev.iteration}`,
      detail: `${ev.successful_count} ok / ${ev.slots_remaining} slots remaining — ${ev.reason_to_continue}`,
    });
  },

  compilagent_leaderboard() {
    /* No-op for now — we render Pareto from compilagent_pareto_update. */
  },

  compilagent_agent_thinking(ev) {
    if (_activeAgentPart !== 'thinking') {
      _agentThinkingBuffer = '';
      _activeAgentPart = 'thinking';
    }
    _agentThinkingBuffer += ev.text || '';
    if (_replayBusy) _agentStreamDirty = true; else _renderAgentStream();
  },

  compilagent_agent_text(ev) {
    if (_activeAgentPart !== 'text') {
      _agentTextBuffer = '';
      _activeAgentPart = 'text';
    }
    _agentTextBuffer += ev.text || '';
    if (_replayBusy) _agentStreamDirty = true; else _renderAgentStream();
  },

  compilagent_guidance(ev) {
    // The GuidedToolset injects [GUIDANCE] / [BASELINE FOOTPRINT]
    // blocks into tool results that the agent reads. Mirror the
    // same text into the activity feed so the operator sees what the
    // agent was told.
    _appendActivity({
      kind: 'guidance',
      label: ev.target_tool === 'inspect_workload'
        ? 'Baseline footprint injected'
        : 'Guidance injected',
      detail: ev.text || '',
      preformatted: true,
      candidateId: null,
    });
  },

  compilagent_pareto_update(ev) {
    const front = ev.pareto_front || [];
    _totals.pareto = front.length;
    _updateHeader();
    _renderParetoStrip(front);
    // Mark per-candidate tiles that appear on the front
    const frontIds = new Set(front.map(p => p.candidate_id).filter(Boolean));
    for (const cid of Object.keys(_candidates)) {
      _candidates[cid].tile.classList.toggle('cl-cand-pareto', frontIds.has(cid));
    }
  },

  compilagent_session_complete(ev) {
    _totals.succeeded = ev.total_valid;
    _totals.failed = ev.total_failed;
    _totals.pareto = ev.pareto_size;
    _updateHeader();
    if (_headerEl) _headerEl.classList.add('cl-session-done');
    if (_timerInterval) { clearInterval(_timerInterval); _timerInterval = null; }
    _appendActivity({
      kind: 'session',
      label: 'Session complete',
      detail: `valid=${ev.total_valid}, failed=${ev.total_failed}, pareto=${ev.pareto_size}, elapsed=${(ev.elapsed_ms / 1000).toFixed(1)}s`,
    });
  },
};

// ── Header / stats ───────────────────────────────────────────────────────

function _updateHeader() {
  if (!_container) return;
  const subtitle = _container.querySelector('#cl-subtitle');
  if (subtitle) {
    if (_sessionMeta) {
      subtitle.textContent = `${_sessionMeta.model} via ${_sessionMeta.harness} · primary: ${_sessionMeta.primary_objective || '?'}`;
    } else {
      subtitle.textContent = 'awaiting session…';
    }
  }
  const setText = (id, val) => {
    const el = _container.querySelector(`#${id}`);
    if (el) el.textContent = val;
  };
  setText('cl-stat-proposed', _totals.proposed);
  setText('cl-stat-succeeded', _totals.succeeded);
  setText('cl-stat-failed', _totals.failed);
  setText('cl-stat-pareto', _totals.pareto);
  setText('cl-stat-budget', `${_totals.succeeded}/${_totals.max}`);
}

function _updateElapsed() {
  if (!_container || !_startTime) return;
  const el = _container.querySelector('#cl-stat-elapsed');
  if (!el) return;
  const seconds = Math.floor((Date.now() - _startTime) / 1000);
  el.textContent = seconds < 60
    ? `${seconds}s`
    : `${Math.floor(seconds / 60)}m${(seconds % 60).toString().padStart(2, '0')}s`;
}

// ── Candidates ───────────────────────────────────────────────────────────

function _ensureCandidate(candidateId, idx, payload) {
  if (_candidates[candidateId]) return _candidates[candidateId];
  const tile = document.createElement('div');
  tile.className = 'cl-cand';
  tile.dataset.candidateId = candidateId;
  tile.innerHTML = `
    <div class="cl-cand-header">
      <span class="cl-cand-idx">#${(idx || 0) + 1}</span>
      <span class="cl-cand-id">${esc(candidateId.substring(0, 12))}</span>
      <span class="cl-cand-status">proposed</span>
    </div>
    <div class="cl-cand-desc">${esc(payload.description || '')}</div>
    <div class="cl-cand-bars"></div>
    <div class="cl-cand-ranks"></div>
    <div class="cl-cand-diag cl-hidden"></div>
  `;
  _candidateGridEl.appendChild(tile);
  const c = {
    candidateId,
    idx: idx || 0,
    tile,
    description: payload.description || '',
    interventions: payload.interventions || [],
    objectives: {},
    metadata: {},
    compiled: false,
    compile_ok: null,
    rejected: false,
  };
  _candidates[candidateId] = c;
  _candidateOrder.push(candidateId);
  return c;
}

function _setStatus(c, text) {
  const s = c.tile.querySelector('.cl-cand-status');
  if (s) s.textContent = text;
}

function _renderCandidateObjectives(c) {
  const wrap = c.tile.querySelector('.cl-cand-bars');
  if (!wrap) return;
  let html = '';
  for (const [k, v] of Object.entries(c.objectives)) {
    const meta = (c.metadata || {})[k] || {};
    const goal = meta.goal || 'min';
    const unit = meta.unit || '';
    const fmt = typeof v === 'number'
      ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3))
      : String(v);
    const colorClass = goal === 'max' ? 'cl-bar--max' : 'cl-bar--min';
    html += `
      <div class="cl-bar ${colorClass}" title="${esc(k)}: ${fmt}${unit ? ' ' + esc(unit) : ''}">
        <span class="cl-bar-name">${esc(k)}</span>
        <span class="cl-bar-val">${fmt}${unit ? esc(unit) : ''}</span>
      </div>`;
  }
  wrap.innerHTML = html;
}

function _hideEmptyCandidates() {
  if (!_container) return;
  const empty = _container.querySelector('#cl-candidates-empty');
  if (empty) empty.classList.add('cl-hidden');
}

// ── Per-metric ranks (chips on each tile) + metric leaders panel ───────

/** For each metric, return a {candidate_id: {rank, total, value, goal}}
 *  ranking. Lower rank = better; goal direction is read from metadata. */
function _computeMetricRanks() {
  const scored = Object.values(_candidates).filter(c => c.objectives && Object.keys(c.objectives).length > 0);
  if (scored.length === 0) return {};
  const metrics = new Set();
  for (const c of scored) for (const k of Object.keys(c.objectives)) metrics.add(k);
  const ranking = {};
  for (const m of metrics) {
    const goal = (scored.find(c => (c.metadata || {})[m])?.metadata[m]?.goal) || 'min';
    const sorted = scored
      .filter(c => typeof c.objectives[m] === 'number')
      .map(c => ({ id: c.candidateId, val: c.objectives[m] }))
      .sort((a, b) => (goal === 'max' ? b.val - a.val : a.val - b.val));
    const total = sorted.length;
    const perCandidate = {};
    sorted.forEach((entry, i) => {
      perCandidate[entry.id] = { rank: i + 1, total, value: entry.val, goal };
    });
    ranking[m] = perCandidate;
  }
  return ranking;
}

function _refreshMetricLeaders() {
  if (!_leadersStripEl) return;
  const ranks = _computeMetricRanks();
  const metrics = Object.keys(ranks);
  if (metrics.length === 0) {
    _leadersStripEl.innerHTML = `<div class="cl-leaders-placeholder">No scored candidates yet.</div>`;
    return;
  }
  let html = '';
  for (const m of metrics) {
    const perCand = ranks[m];
    const leaderEntry = Object.entries(perCand).find(([_, info]) => info.rank === 1);
    if (!leaderEntry) continue;
    const [cid, info] = leaderEntry;
    const cand = _candidates[cid];
    const meta = (cand?.metadata || {})[m] || {};
    const unit = meta.unit || '';
    const fmt = typeof info.value === 'number'
      ? (Math.abs(info.value) >= 100 ? info.value.toFixed(0) : info.value.toFixed(3))
      : String(info.value);
    html += `
      <div class="cl-leader" title="${esc(m)}: best so far">
        <div class="cl-leader-head">
          <span class="cl-leader-metric">${esc(m)}</span>
          <span class="cl-leader-goal">${esc(info.goal)}</span>
        </div>
        <div class="cl-leader-val">${fmt}${unit ? esc(unit) : ''}</div>
        <div class="cl-leader-cid">${esc(cid.substring(0, 12))}</div>
      </div>`;
  }
  _leadersStripEl.innerHTML = html;
}

function _refreshPerMetricRanks() {
  const ranks = _computeMetricRanks();
  for (const c of Object.values(_candidates)) {
    if (!c.objectives || Object.keys(c.objectives).length === 0) continue;
    const ranksEl = c.tile.querySelector('.cl-cand-ranks');
    if (!ranksEl) continue;
    let html = '';
    for (const [m, perCand] of Object.entries(ranks)) {
      const info = perCand[c.candidateId];
      if (!info) continue;
      const cls = info.rank === 1 ? 'cl-rank-chip cl-rank-best' : 'cl-rank-chip';
      const short = m.length > 18 ? m.substring(0, 16) + '…' : m;
      html += `<span class="${cls}" title="${esc(m)}: rank ${info.rank}/${info.total}">${esc(short)}: ${info.rank}/${info.total}</span>`;
    }
    ranksEl.innerHTML = html;
  }
}

// ── Pareto strip ─────────────────────────────────────────────────────────

function _commonMetricsForPareto(scored) {
  const seen = new Set();
  for (const c of scored) {
    for (const k of Object.keys(c.objectives || {})) seen.add(k);
  }
  return [...seen];
}

function _goalForParetoMetric(scored, metric) {
  for (const c of scored) {
    const g = (c.metadata || {})[metric]?.goal;
    if (g === 'min' || g === 'max') return g;
  }
  return 'min';
}

function _dominatesObjectives(aObjs, bObjs, metrics, goalByMetric) {
  let betterInOne = false;
  for (const m of metrics) {
    const av = aObjs[m];
    const bv = bObjs[m];
    if (typeof av !== 'number' || typeof bv !== 'number') return false;
    const goal = goalByMetric[m] || 'min';
    if (goal === 'max') {
      if (av < bv) return false;
      if (av > bv) betterInOne = true;
    } else {
      if (av > bv) return false;
      if (av < bv) betterInOne = true;
    }
  }
  return betterInOne;
}

/** Non-dominated front from in-memory candidates (mirrors compilagent pareto_front). */
function _computeParetoFrontFromCandidates() {
  const rows = Object.values(_candidates).filter(
    c => c.objectives && Object.keys(c.objectives).length > 0,
  );
  if (rows.length <= 1) {
    return rows.map(c => ({ candidate_id: c.candidateId, objectives: { ...c.objectives } }));
  }
  const metrics = _commonMetricsForPareto(rows);
  if (metrics.length === 0) {
    return rows.map(c => ({ candidate_id: c.candidateId, objectives: { ...c.objectives } }));
  }
  const goalByMetric = Object.fromEntries(
    metrics.map(m => [m, _goalForParetoMetric(rows, m)]),
  );
  const front = [];
  for (let i = 0; i < rows.length; i++) {
    if (metrics.some(m => typeof rows[i].objectives[m] !== 'number')) continue;
    let dominated = false;
    for (let j = 0; j < rows.length; j++) {
      if (i === j) continue;
      if (metrics.some(m => typeof rows[j].objectives[m] !== 'number')) continue;
      if (_dominatesObjectives(rows[j].objectives, rows[i].objectives, metrics, goalByMetric)) {
        dominated = true;
        break;
      }
    }
    if (!dominated) {
      front.push({ candidate_id: rows[i].candidateId, objectives: { ...rows[i].objectives } });
    }
  }
  return front;
}

function _refreshLivePareto() {
  const front = _computeParetoFrontFromCandidates();
  _totals.pareto = front.length;
  _updateHeader();
  _renderParetoStrip(front);
  const frontIds = new Set(front.map(p => p.candidate_id).filter(Boolean));
  for (const cid of Object.keys(_candidates)) {
    _candidates[cid].tile.classList.toggle('cl-cand-pareto', frontIds.has(cid));
  }
}

function _renderParetoStrip(front) {
  if (!_paretoStripEl) return;
  if (!front || front.length === 0) {
    _paretoStripEl.innerHTML = `<div class="cl-pareto-placeholder">No non-dominated candidates yet — waiting for the first compile to succeed.</div>`;
    return;
  }
  let html = '';
  for (let i = 0; i < front.length; i++) {
    const p = front[i];
    const objs = p.objectives || {};
    const cells = Object.entries(objs)
      .map(([k, v]) => `<span class="cl-chip-cell"><span class="cl-chip-k">${esc(k)}</span><span class="cl-chip-v">${typeof v === 'number' ? v.toFixed(3) : esc(String(v))}</span></span>`)
      .join('');
    html += `<div class="cl-chip" title="${esc(p.candidate_id || '')}"><span class="cl-chip-rank">#${i + 1}</span>${cells}</div>`;
  }
  _paretoStripEl.innerHTML = html;
}

// ── Agent stream ─────────────────────────────────────────────────────────

function _renderAgentStream() {
  if (!_agentStreamEl) return;
  const blocks = [];
  if (_agentThinkingBuffer) {
    blocks.push(`<div class="cl-agent-block cl-agent-thinking"><span class="cl-agent-label">thinking</span><pre class="cl-agent-text">${esc(_agentThinkingBuffer)}</pre></div>`);
  }
  if (_agentTextBuffer) {
    blocks.push(`<div class="cl-agent-block cl-agent-text"><span class="cl-agent-label">text</span><div class="cl-agent-md">${renderMarkdown(_agentTextBuffer)}</div></div>`);
  }
  if (blocks.length === 0) {
    _agentStreamEl.innerHTML = `<div class="cl-agent-empty">No tokens streamed yet.</div>`;
    return;
  }
  _agentStreamEl.innerHTML = blocks.join('');
  _agentStreamEl.scrollTop = _agentStreamEl.scrollHeight;
}

// ── Activity feed ────────────────────────────────────────────────────────

function _appendActivity({ kind, label, detail, candidateId, preformatted }) {
  if (!_activityFeedEl) return;
  _activityCounter++;
  const row = document.createElement('div');
  row.className = `cl-activity-row cl-activity-${kind}`;
  let detailHtml = '';
  if (detail) {
    if (preformatted) {
      // Long multi-line guidance text — collapsible <details>.
      detailHtml = `<details class="cl-activity-pre"><summary>show details</summary><pre>${esc(detail)}</pre></details>`;
    } else {
      detailHtml = `<div class="cl-activity-detail">${esc(detail)}</div>`;
    }
  }
  row.innerHTML = `
    <span class="cl-activity-counter">${_activityCounter}</span>
    <div class="cl-activity-body">
      <div class="cl-activity-label">${esc(label || '')}</div>
      ${detailHtml}
      ${candidateId ? `<div class="cl-activity-cid">${esc(candidateId.substring(0, 12))}</div>` : ''}
    </div>`;
  _activityFeedEl.appendChild(row);
  // Cap at 200 rows so the DOM doesn't grow unbounded during long sessions.
  while (_activityFeedEl.children.length > 200) {
    _activityFeedEl.removeChild(_activityFeedEl.firstChild);
  }
  // Forcing scrollTop = scrollHeight triggers a synchronous layout. During
  // replay we'd pay this cost per event; skip it and rely on the
  // post-replay flush (last row is naturally visible after the final append
  // since the feed is short).
  if (!_replayBusy) _activityFeedEl.scrollTop = _activityFeedEl.scrollHeight;
}
