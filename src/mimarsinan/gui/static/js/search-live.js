/* Live cyberpunk search tracker for AgentEvolve architecture search.
 * Renders generation cards, candidate tiles with health bars, and
 * reasoning/insight panels — all streamed via WebSocket events. */

import { esc } from './util.js';

// ── Module state ─────────────────────────────────────────────────────────
let _container = null;
let _headerEl = null;
let _genContainer = null;
let _generations = {};          // gen -> { card, candidatesGrid, ... }
let _totals = { valid: 0, failed: 0, elapsed: 0, gen: 0, totalGens: 0, pareto: 0 };
let _startTime = null;
let _timerInterval = null;

// ── Public API ───────────────────────────────────────────────────────────

export function initSearchLive(container) {
  _container = container;
  _generations = {};
  _totals = { valid: 0, failed: 0, elapsed: 0, gen: 0, totalGens: 0, pareto: 0 };
  _startTime = Date.now();

  container.innerHTML = `
    <div class="sl-tracker">
      <div class="sl-header" id="sl-header">
        <div class="sl-header-left">
          <span class="sl-logo">⬡</span>
          <span class="sl-title">AGENTIC EVOLUTION</span>
        </div>
        <div class="sl-header-stats">
          <div class="sl-stat"><span class="sl-stat-label">GEN</span><span class="sl-stat-val" id="sl-gen">0/0</span></div>
          <div class="sl-stat"><span class="sl-stat-label">VALID</span><span class="sl-stat-val sl-val-ok" id="sl-valid">0</span></div>
          <div class="sl-stat"><span class="sl-stat-label">FAILED</span><span class="sl-stat-val sl-val-fail" id="sl-failed">0</span></div>
          <div class="sl-stat"><span class="sl-stat-label">PARETO</span><span class="sl-stat-val sl-val-pareto" id="sl-pareto">0</span></div>
          <div class="sl-stat"><span class="sl-stat-label">ELAPSED</span><span class="sl-stat-val" id="sl-elapsed">0s</span></div>
        </div>
      </div>
      <div class="sl-generations" id="sl-generations"></div>
    </div>`;

  _headerEl = container.querySelector('#sl-header');
  _genContainer = container.querySelector('#sl-generations');

  if (_timerInterval) clearInterval(_timerInterval);
  _timerInterval = setInterval(_updateElapsed, 1000);
}

export function handleSearchEvent(ev) {
  if (!_container) return;
  const handler = _handlers[ev.type];
  if (handler) handler(ev);
}

export function isSearchLiveActive() {
  return _container !== null && Object.keys(_generations).length > 0;
}

export function replaySearchEvents(events) {
  for (const ev of events) handleSearchEvent(ev);
}

// ── Event handlers ───────────────────────────────────────────────────────

const _handlers = {
  generation_start(ev) {
    _totals.gen = ev.gen;
    _totals.totalGens = ev.total_gens;
    _updateHeader();
    _ensureGenCard(ev.gen, ev.phase);
  },

  candidates_generated(ev) {
    const gc = _ensureGenCard(ev.gen, null);
    if (ev.reasoning) {
      gc.reasoningEl.innerHTML = `<div class="sl-reasoning-text">${esc(ev.reasoning)}</div>`;
      gc.reasoningEl.classList.add('sl-visible');
    }
    gc.countEl.textContent = `${ev.count} candidates`;
  },

  candidate_result(ev) {
    const gc = _ensureGenCard(ev.gen, null);
    if (ev.is_valid) _totals.valid++; else _totals.failed++;
    _updateHeader();
    _addCandidateCard(gc, ev);
    _updateGenCounts(gc);
  },

  batch_summary(ev) {
    const gc = _ensureGenCard(ev.gen, null);
    _updateGenCounts(gc);
  },

  generation_complete(ev) {
    const gc = _ensureGenCard(ev.gen, null);
    _totals.pareto = ev.pareto_size;
    gc.validCount = ev.valid_count;
    gc.failedCount = ev.failed_count;
    _updateGenCounts(gc);
    _updateHeader();

    if (ev.pareto_front && ev.pareto_front.length > 0) {
      gc.paretoEl.innerHTML = _renderParetoSummary(ev.pareto_front);
      gc.paretoEl.classList.add('sl-visible');
    }
    if (ev.constraint_instruction || ev.performance_insights) {
      let html = '';
      if (ev.constraint_instruction)
        html += `<div class="sl-insight-block"><span class="sl-insight-label">CONSTRAINTS</span>${esc(ev.constraint_instruction)}</div>`;
      if (ev.performance_insights)
        html += `<div class="sl-insight-block"><span class="sl-insight-label">INSIGHTS</span>${esc(ev.performance_insights)}</div>`;
      gc.insightsEl.innerHTML = html;
      gc.insightsEl.classList.add('sl-visible');
    }
    gc.card.classList.add('sl-gen-complete');
  },

  search_complete(ev) {
    _totals.valid = ev.total_valid;
    _totals.failed = ev.total_failed;
    _totals.pareto = ev.final_pareto_size;
    _updateHeader();
    if (_timerInterval) { clearInterval(_timerInterval); _timerInterval = null; }
    if (_headerEl) _headerEl.classList.add('sl-search-done');
  },
};

// ── Generation card management ───────────────────────────────────────────

function _ensureGenCard(gen, phase) {
  if (_generations[gen]) return _generations[gen];

  const phaseLabel = phase === 'initial' ? 'INITIAL' : (phase === 'evolution' ? 'EVOLUTION' : '');
  const card = document.createElement('div');
  card.className = 'sl-gen-card sl-fadein';
  card.innerHTML = `
    <div class="sl-gen-header">
      <span class="sl-gen-num">GEN ${gen}</span>
      ${phaseLabel ? `<span class="sl-phase-pill sl-phase-${phase || 'initial'}">${phaseLabel}</span>` : ''}
      <span class="sl-gen-count" id="sl-gc-${gen}">—</span>
      <span class="sl-gen-stats"><span class="sl-gs-ok" id="sl-gv-${gen}">0</span> ✓ <span class="sl-gs-fail" id="sl-gf-${gen}">0</span> ✗</span>
      <button class="sl-collapse-btn" title="Toggle details">▾</button>
    </div>
    <div class="sl-gen-body">
      <div class="sl-reasoning sl-collapsible" id="sl-reason-${gen}"></div>
      <div class="sl-candidates-grid" id="sl-cands-${gen}"></div>
      <div class="sl-pareto-summary sl-collapsible" id="sl-pareto-${gen}"></div>
      <div class="sl-insights sl-collapsible" id="sl-insights-${gen}"></div>
    </div>`;

  const collapseBtn = card.querySelector('.sl-collapse-btn');
  const body = card.querySelector('.sl-gen-body');
  collapseBtn.addEventListener('click', () => {
    body.classList.toggle('sl-collapsed');
    collapseBtn.textContent = body.classList.contains('sl-collapsed') ? '▸' : '▾';
  });

  if (_genContainer.firstChild) {
    _genContainer.insertBefore(card, _genContainer.firstChild);
  } else {
    _genContainer.appendChild(card);
  }

  const gc = {
    card,
    countEl: card.querySelector(`#sl-gc-${gen}`),
    validEl: card.querySelector(`#sl-gv-${gen}`),
    failedEl: card.querySelector(`#sl-gf-${gen}`),
    reasoningEl: card.querySelector(`#sl-reason-${gen}`),
    candidatesGrid: card.querySelector(`#sl-cands-${gen}`),
    paretoEl: card.querySelector(`#sl-pareto-${gen}`),
    insightsEl: card.querySelector(`#sl-insights-${gen}`),
    validCount: 0,
    failedCount: 0,
    candidates: [],
  };
  _generations[gen] = gc;
  return gc;
}

// ── Candidate cards ──────────────────────────────────────────────────────

function _addCandidateCard(gc, ev) {
  const tile = document.createElement('div');
  tile.className = `sl-cand-tile sl-fadein ${ev.is_valid ? 'sl-cand-ok' : 'sl-cand-fail'}`;

  let inner = `<div class="sl-cand-idx">#${ev.idx + 1}</div>`;

  if (ev.is_valid && ev.objectives) {
    inner += '<div class="sl-cand-bars">';
    for (const [key, val] of Object.entries(ev.objectives)) {
      const pct = _objToPercent(key, val);
      const color = _objColor(key);
      const label = _shortLabel(key);
      inner += `<div class="sl-bar-row" title="${esc(key)}: ${typeof val === 'number' ? val.toFixed(4) : val}">
        <span class="sl-bar-label">${esc(label)}</span>
        <div class="sl-bar-track"><div class="sl-bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <span class="sl-bar-val">${typeof val === 'number' ? (val < 1 ? (val * 100).toFixed(1) + '%' : val.toFixed(2)) : val}</span>
      </div>`;
    }
    inner += '</div>';
  } else {
    if (ev.failure_phase)
      inner += `<span class="sl-fail-phase">${esc(ev.failure_phase)}</span>`;
    if (ev.error_message)
      inner += `<div class="sl-fail-msg" title="${esc(ev.error_message)}">${esc(ev.error_message.substring(0, 80))}${ev.error_message.length > 80 ? '…' : ''}</div>`;
  }

  tile.innerHTML = inner;
  gc.candidatesGrid.appendChild(tile);
  gc.candidates.push(ev);
  if (ev.is_valid) gc.validCount++; else gc.failedCount++;
}

function _updateGenCounts(gc) {
  gc.validEl.textContent = gc.validCount;
  gc.failedEl.textContent = gc.failedCount;
}

// ── Health bar helpers ───────────────────────────────────────────────────

function _objToPercent(key, val) {
  const k = key.toLowerCase();
  if (k.includes('accuracy') || k.includes('acc')) return Math.min(100, Math.max(0, val * 100));
  if (k.includes('wastage') || k.includes('waste')) return Math.min(100, Math.max(0, val * 100));
  if (k.includes('utilization') || k.includes('util')) return Math.min(100, Math.max(0, val * 100));
  if (val >= 0 && val <= 1) return val * 100;
  if (val >= 0 && val <= 100) return val;
  return Math.min(100, Math.max(0, val));
}

function _objColor(key) {
  const k = key.toLowerCase();
  if (k.includes('accuracy') || k.includes('acc')) return 'var(--accent-cyan)';
  if (k.includes('wastage') || k.includes('waste')) return 'var(--accent-rose, #f43f5e)';
  if (k.includes('utilization') || k.includes('util')) return 'var(--success)';
  if (k.includes('power') || k.includes('energy')) return 'var(--accent-amber, #f59e0b)';
  return 'var(--accent, #5b8af5)';
}

function _shortLabel(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()).substring(0, 12);
}

// ── Pareto summary ───────────────────────────────────────────────────────

function _renderParetoSummary(front) {
  if (!front || front.length === 0) return '';
  const keys = Object.keys(front[0]);
  let html = '<div class="sl-pareto-title">PARETO FRONT (top ' + front.length + ')</div>';
  html += '<div class="sl-pareto-grid">';
  for (let i = 0; i < Math.min(front.length, 5); i++) {
    html += `<div class="sl-pareto-entry">`;
    for (const k of keys) {
      const v = front[i][k];
      const pct = _objToPercent(k, v);
      const color = _objColor(k);
      html += `<div class="sl-bar-row sl-bar-mini" title="${esc(k)}: ${typeof v === 'number' ? v.toFixed(4) : v}">
        <span class="sl-bar-label">${esc(_shortLabel(k))}</span>
        <div class="sl-bar-track"><div class="sl-bar-fill" style="width:${pct}%;background:${color}"></div></div>
      </div>`;
    }
    html += '</div>';
  }
  html += '</div>';
  return html;
}

// ── Header updates ───────────────────────────────────────────────────────

function _updateHeader() {
  const setTxt = (id, txt) => {
    const el = _container?.querySelector('#' + id);
    if (el) el.textContent = txt;
  };
  setTxt('sl-gen', `${_totals.gen}/${_totals.totalGens}`);
  setTxt('sl-valid', String(_totals.valid));
  setTxt('sl-failed', String(_totals.failed));
  setTxt('sl-pareto', String(_totals.pareto));
}

function _updateElapsed() {
  if (!_startTime || !_container) return;
  const sec = Math.floor((Date.now() - _startTime) / 1000);
  const el = _container.querySelector('#sl-elapsed');
  if (el) el.textContent = sec < 60 ? `${sec}s` : `${Math.floor(sec / 60)}m ${sec % 60}s`;
}
