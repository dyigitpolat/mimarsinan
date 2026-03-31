/* Live cyberpunk search tracker for AgentEvolve architecture search.
 * Generation layout: main column (subpanels) + diagnostics column (LLM trace). */

import { esc, renderMarkdown } from './util.js';

const CALL_KIND_LABELS = {
  initial_candidates: 'Initial candidates',
  regenerate_candidates: 'Regenerate candidates',
  offspring: 'Offspring',
  regenerate_offspring: 'Regenerate offspring',
  failure_insights: 'Failure insights',
  constraint_instruction: 'Constraint instruction',
  update_constraint: 'Update constraints',
  performance_insights: 'Performance insights',
  update_performance_insights: 'Update performance insights',
  unknown: 'LLM call',
};

// ── Module state ─────────────────────────────────────────────────────────
let _container = null;
let _headerEl = null;
let _genContainer = null;
let _paretoStripInner = null;
let _generations = {};
let _lastObjSpecs = null;
let _totals = { valid: 0, failed: 0, gen: 0, totalGens: 0, pareto: 0 };
let _startTime = null;
let _timerInterval = null;
/** Events from state.searchEvents already applied to the live DOM (WS + poll tail). */
let _appliedSearchEventCount = 0;

// ── Public API ───────────────────────────────────────────────────────────

export function initSearchLive(container) {
  _container = container;
  _generations = {};
  _lastObjSpecs = null;
  _appliedSearchEventCount = 0;
  _totals = { valid: 0, failed: 0, gen: 0, totalGens: 0, pareto: 0 };
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
      <div class="sl-pareto-strip" id="sl-pareto-strip">
        <div class="sl-pareto-strip-title">Current Pareto front</div>
        <div class="sl-pareto-strip-inner" id="sl-pareto-strip-inner">
          <div class="sl-pareto-strip-placeholder">Pareto summary updates after each generation completes.</div>
        </div>
      </div>
      <div class="sl-generations" id="sl-generations"></div>
    </div>`;

  _headerEl = container.querySelector('#sl-header');
  _paretoStripInner = container.querySelector('#sl-pareto-strip-inner');
  _genContainer = container.querySelector('#sl-generations');

  if (_timerInterval) clearInterval(_timerInterval);
  _timerInterval = setInterval(_updateElapsed, 1000);
}

export function detachSearchLive() {
  _container = null;
  _paretoStripInner = null;
  _headerEl = null;
  _genContainer = null;
  _appliedSearchEventCount = 0;
}

/**
 * Apply search_event payloads from state that are not yet reflected in the DOM.
 * WebSocket and HTTP poll both append to state.searchEvents; this drains the tail.
 */
export function syncSearchEventsFromState(stepName, state) {
  if (!_container || !_container.isConnected) {
    _container = null;
    return;
  }
  const evs = (state.searchEvents && state.searchEvents[stepName]) || [];
  if (evs.length < _appliedSearchEventCount) _appliedSearchEventCount = 0;
  for (let i = _appliedSearchEventCount; i < evs.length; i++) {
    handleSearchEvent(evs[i]);
    if (!_container || !_container.isConnected) {
      _appliedSearchEventCount = i + 1;
      return;
    }
  }
  _appliedSearchEventCount = evs.length;
}

export function handleSearchEvent(ev) {
  if (!_container || !_container.isConnected) {
    _container = null;
    return;
  }
  const handler = _handlers[ev.type];
  if (handler) handler(ev);
}

export function isSearchLiveActive() {
  return _container !== null && Object.keys(_generations).length > 0;
}

export function replaySearchEvents(events) {
  for (const ev of events) handleSearchEvent(ev);
  _appliedSearchEventCount = events.length;
}

// ── Event handlers ───────────────────────────────────────────────────────

const _handlers = {
  generation_start(ev) {
    _totals.gen = ev.gen;
    _totals.totalGens = ev.total_gens;
    _updateHeader();
    const gc = _ensureGenCard(ev.gen, ev.phase);
    if (ev.objectives) {
      gc.objSpecs = ev.objectives;
      _lastObjSpecs = ev.objectives;
    }
  },

  llm_trace(ev) {
    const gc = _ensureGenCard(ev.gen, null);
    if (!gc.traces) gc.traces = [];
    gc.traces.push(ev);
    gc.traces.sort((a, b) => (a.ordinal || 0) - (b.ordinal || 0));
    _appendTraceRow(gc, ev);
  },

  candidates_generated(ev) {
    const gc = _ensureGenCard(ev.gen, null);
    if (ev.reasoning) {
      gc.reasoningBody.innerHTML = `<div class="sl-reasoning-text">${esc(ev.reasoning)}</div>`;
      gc.reasoningPanel.classList.add('sl-subpanel--filled');
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

    _sortCandidatesByRank(gc);

    if (ev.pareto_front && ev.pareto_front.length > 0) {
      _updateGlobalPareto(ev.pareto_front, gc.objSpecs || _lastObjSpecs);
    }
    if (ev.constraint_instruction) {
      gc.constraintsBody.innerHTML = `<div class="sl-prose sl-md">${renderMarkdown(ev.constraint_instruction)}</div>`;
      gc.constraintsBlock.classList.add('sl-collapse-filled', 'sl-collapse-open');
      _setCollapseChevron(gc.constraintsBlock, true);
    }
    if (ev.performance_insights) {
      gc.perfBody.innerHTML = `<div class="sl-prose sl-md">${renderMarkdown(ev.performance_insights)}</div>`;
      gc.perfBlock.classList.add('sl-collapse-filled', 'sl-collapse-open');
      _setCollapseChevron(gc.perfBlock, true);
    }
    gc.card.classList.add('sl-gen-complete');
  },

  search_complete(ev) {
    _totals.valid = ev.total_valid;
    _totals.failed = ev.total_failed;
    _totals.pareto = ev.final_pareto_size;
    _updateHeader();
    if (_paretoStripInner && ev.final_pareto_size != null) {
      const ph = _paretoStripInner.querySelector('.sl-pareto-strip-placeholder');
      if (ph) ph.remove();
    }
    if (_timerInterval) { clearInterval(_timerInterval); _timerInterval = null; }
    if (_headerEl) _headerEl.classList.add('sl-search-done');
  },
};

// ── Generation card management ───────────────────────────────────────────

function _wireSubpanelToggle(panel, headerBtn) {
  headerBtn.addEventListener('click', () => {
    panel.classList.toggle('sl-subpanel--open');
    const ch = headerBtn.querySelector('.sl-subpanel-chev');
    if (ch) ch.textContent = panel.classList.contains('sl-subpanel--open') ? '▾' : '▸';
  });
}

function _wireCollapseBlock(block, headerBtn) {
  headerBtn.addEventListener('click', () => {
    block.classList.toggle('sl-collapse-open');
    const ch = headerBtn.querySelector('.sl-collapse-chev');
    if (ch) ch.textContent = block.classList.contains('sl-collapse-open') ? '▾' : '▸';
  });
}

function _setCollapseChevron(block, open) {
  const ch = block.querySelector('.sl-collapse-chev');
  if (ch) ch.textContent = open ? '▾' : '▸';
}

function _updateGlobalPareto(front, objSpecs) {
  if (!_paretoStripInner || !front || front.length === 0) return;
  const ph = _paretoStripInner.querySelector('.sl-pareto-strip-placeholder');
  if (ph) ph.remove();
  _paretoStripInner.innerHTML = _renderParetoSummary(front, objSpecs);
}

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
      <button class="sl-collapse-btn" title="Toggle all">▾</button>
    </div>
    <div class="sl-gen-body">
      <div class="sl-gen-layout">
        <div class="sl-gen-main">
          <div class="sl-subpanel sl-subpanel--reasoning" id="sl-sp-reason-${gen}">
            <button type="button" class="sl-subpanel-h" aria-expanded="false">
              <span class="sl-subpanel-title">Reasoning</span>
              <span class="sl-subpanel-chev">▸</span>
            </button>
            <div class="sl-subpanel-c"><div class="sl-subpanel-inner" id="sl-reason-body-${gen}"></div></div>
          </div>
          <div class="sl-subpanel sl-subpanel--candidates sl-subpanel--open">
            <div class="sl-subpanel-h sl-subpanel-h--static">
              <span class="sl-subpanel-title">Candidates</span>
            </div>
            <div class="sl-subpanel-c"><div class="sl-candidates-grid" id="sl-cands-${gen}"></div></div>
          </div>
          <div class="sl-subpanel sl-subpanel--summary">
            <div class="sl-subpanel-h sl-subpanel-h--static">
              <span class="sl-subpanel-title">Constraints &amp; insights</span>
            </div>
            <div class="sl-subpanel-c">
              <div class="sl-collapse-block" id="sl-constr-${gen}">
                <button type="button" class="sl-collapse-h">
                  <span class="sl-collapse-title">Constraint instructions</span>
                  <span class="sl-collapse-chev">▸</span>
                </button>
                <div class="sl-collapse-c"><div class="sl-collapse-inner" id="sl-constr-body-${gen}"></div></div>
              </div>
              <div class="sl-collapse-block" id="sl-perf-${gen}">
                <button type="button" class="sl-collapse-h">
                  <span class="sl-collapse-title">Performance insights</span>
                  <span class="sl-collapse-chev">▸</span>
                </button>
                <div class="sl-collapse-c"><div class="sl-collapse-inner" id="sl-perf-body-${gen}"></div></div>
              </div>
            </div>
          </div>
        </div>
        <aside class="sl-gen-diagnostics">
          <div class="sl-diag-head">LLM trace</div>
          <div class="sl-trace-list" id="sl-trace-list-${gen}"></div>
          <div class="sl-trace-detail" id="sl-trace-detail-${gen}">
            <div class="sl-trace-detail-placeholder">Select a call to inspect request and response.</div>
          </div>
        </aside>
      </div>
    </div>`;

  const collapseBtn = card.querySelector('.sl-collapse-btn');
  const body = card.querySelector('.sl-gen-body');
  collapseBtn.addEventListener('click', () => {
    body.classList.toggle('sl-collapsed');
    collapseBtn.textContent = body.classList.contains('sl-collapsed') ? '▸' : '▾';
  });

  const reasoningPanel = card.querySelector(`#sl-sp-reason-${gen}`);
  _wireSubpanelToggle(reasoningPanel, reasoningPanel.querySelector('.sl-subpanel-h'));

  const constrBlock = card.querySelector(`#sl-constr-${gen}`);
  _wireCollapseBlock(constrBlock, constrBlock.querySelector('.sl-collapse-h'));
  const perfBlock = card.querySelector(`#sl-perf-${gen}`);
  _wireCollapseBlock(perfBlock, perfBlock.querySelector('.sl-collapse-h'));

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
    reasoningPanel,
    reasoningBody: card.querySelector(`#sl-reason-body-${gen}`),
    candidatesGrid: card.querySelector(`#sl-cands-${gen}`),
    constraintsBlock: constrBlock,
    constraintsBody: card.querySelector(`#sl-constr-body-${gen}`),
    perfBlock,
    perfBody: card.querySelector(`#sl-perf-body-${gen}`),
    traceListEl: card.querySelector(`#sl-trace-list-${gen}`),
    traceDetailEl: card.querySelector(`#sl-trace-detail-${gen}`),
    validCount: 0,
    failedCount: 0,
    candidates: [],
    objSpecs: null,
    traces: [],
    selectedTraceOrdinal: null,
  };
  _generations[gen] = gc;
  return gc;
}

function _appendTraceRow(gc, ev) {
  const label = CALL_KIND_LABELS[ev.call_kind] || ev.call_kind || 'LLM';
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'sl-trace-row';
  btn.dataset.ordinal = String(ev.ordinal);
  btn.innerHTML = `<span class="sl-trace-ord">${ev.ordinal}</span><span class="sl-trace-name">${esc(label)}</span>`;
  btn.addEventListener('click', () => {
    gc.traceListEl.querySelectorAll('.sl-trace-row').forEach(b => b.classList.remove('sl-trace-row--active'));
    btn.classList.add('sl-trace-row--active');
    gc.selectedTraceOrdinal = ev.ordinal;
    gc.traceDetailEl.innerHTML = _renderTraceDetail(ev);
  });
  gc.traceListEl.appendChild(btn);
  if (gc.traceListEl.children.length === 1) {
    btn.classList.add('sl-trace-row--active');
    gc.traceDetailEl.innerHTML = _renderTraceDetail(ev);
  }
}

function _renderResponseStructured(resp, schemaKeys) {
  let html = '';
  if (resp.reasoning_preview) {
    html += `<div class="sl-trace-kv"><span class="sl-trace-k">Reasoning</span><div class="sl-trace-v sl-prose sl-md">${renderMarkdown(resp.reasoning_preview)}</div></div>`;
  }
  if (resp.candidate_count != null) {
    html += `<div class="sl-trace-kv"><span class="sl-trace-k">Candidates</span><span class="sl-trace-v">${resp.candidate_count}</span></div>`;
  }
  if (resp.candidate_previews && resp.candidate_previews.length) {
    html += '<div class="sl-trace-kv"><span class="sl-trace-k">Previews</span><div class="sl-trace-v">';
    for (const p of resp.candidate_previews) {
      html += `<div class="sl-trace-preview"><span class="sl-trace-preview-idx">#${p.index + 1}</span><pre class="sl-trace-pre sl-trace-pre--sm">${esc(p.summary)}</pre></div>`;
    }
    html += '</div></div>';
  }
  if (resp.insights && resp.insights.length) {
    html += '<div class="sl-trace-kv"><span class="sl-trace-k">Insights</span><div class="sl-trace-v">';
    for (const it of resp.insights) {
      html += `<div class="sl-trace-insight"><span class="sl-trace-insight-idx">${it.index + 1}.</span> ${esc(it.text)}</div>`;
    }
    html += '</div></div>';
  }
  if (resp.insight_count != null && !resp.insights) {
    html += `<div class="sl-trace-kv"><span class="sl-trace-k">Count</span><span class="sl-trace-v">${resp.insight_count}</span></div>`;
  }
  if (resp.text_preview) {
    html += `<div class="sl-trace-kv"><span class="sl-trace-k">Text</span><div class="sl-trace-v sl-prose sl-md">${renderMarkdown(resp.text_preview)}</div></div>`;
  }
  if (resp.text_preview_truncated) {
    const n = resp.text_preview_full_len != null ? String(resp.text_preview_full_len) : '?';
    html += `<div class="sl-trace-meta sl-trace-meta--warn">Showing truncated preview (${n} characters total).</div>`;
  }
  if (!html) {
    html = '<div class="sl-trace-empty">No structured fields</div>';
  }
  if (schemaKeys && schemaKeys.length) {
    html += `<div class="sl-trace-meta">Output keys: ${esc(schemaKeys.join(', '))}</div>`;
  }
  return html;
}

function _renderTraceDetail(ev) {
  const req = ev.request || {};
  const sec = req.sections || [];
  let requestBody = '';
  if (req.total_chars != null) {
    requestBody += `<div class="sl-trace-meta">${req.truncated ? 'Truncated · ' : ''}${req.total_chars.toLocaleString()} characters</div>`;
  }
  if (sec.length === 0) {
    requestBody += '<div class="sl-trace-empty">No sections</div>';
  } else {
    for (const s of sec) {
      requestBody += `<div class="sl-trace-section"><div class="sl-trace-section-label">${esc(s.label || 'Section')}</div>`;
      requestBody += `<pre class="sl-trace-pre">${esc(s.text || '')}</pre></div>`;
    }
  }
  const nSec = sec.length;
  const secLabel = nSec === 1 ? '1 section' : `${nSec} sections`;
  let html = '<div class="sl-trace-columns">';
  html += `<div class="sl-trace-col"><details class="sl-trace-req"><summary class="sl-trace-req-summary">Request (${secLabel})</summary><div class="sl-trace-req-body">`;
  html += requestBody;
  html += '</div></details></div>';
  html += '<div class="sl-trace-col"><div class="sl-trace-col-title">Response</div>';
  html += _renderResponseStructured(ev.response || {}, ev.output_schema_keys || []);
  html += '</div></div>';
  return html;
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
      const display = _fmtVal(key, val);
      inner += `<div class="sl-bar-row" title="${esc(key)}: ${typeof val === 'number' ? val.toFixed(6) : val}">
        <span class="sl-bar-label">${esc(label)}</span>
        <div class="sl-bar-track"><div class="sl-bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <span class="sl-bar-val">${esc(display)}</span>
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
  gc.candidates.push({ tile, ev });
  if (ev.is_valid) gc.validCount++; else gc.failedCount++;
}

function _updateGenCounts(gc) {
  gc.validEl.textContent = gc.validCount;
  gc.failedEl.textContent = gc.failedCount;
}

// ── Minimax-rank sort (mirrors select_minimax_rank in search/results.py) ──

function _computeRanks(entries, objSpecs) {
  const n = entries.length;
  const ranks = Array.from({ length: n }, () => Array(objSpecs.length).fill(1));

  for (let j = 0; j < objSpecs.length; j++) {
    const spec = objSpecs[j];
    const values = entries.map(e => (e.ev.objectives && e.ev.objectives[spec.name] != null)
      ? e.ev.objectives[spec.name] : (spec.goal === 'max' ? -Infinity : Infinity));

    const order = [...Array(n).keys()].sort((a, b) =>
      spec.goal === 'max' ? values[b] - values[a] : values[a] - values[b]);

    let currentRank = 1;
    for (let pos = 0; pos < order.length; pos++) {
      const idx = order[pos];
      if (pos > 0 && values[order[pos]] !== values[order[pos - 1]]) currentRank = pos + 1;
      ranks[idx][j] = currentRank;
    }
  }
  return ranks;
}

function _sortCandidatesByRank(gc) {
  const objSpecs = gc.objSpecs;
  const valid = gc.candidates.filter(c => c.ev.is_valid);
  const invalid = gc.candidates.filter(c => !c.ev.is_valid);

  if (valid.length > 0 && objSpecs && objSpecs.length > 0) {
    const ranks = _computeRanks(valid, objSpecs);
    const worstRanks = ranks.map(r => Math.max(...r));
    const rankSums = ranks.map(r => r.reduce((a, b) => a + b, 0));
    const sortedValid = valid
      .map((c, i) => ({ c, worst: worstRanks[i], sum: rankSums[i] }))
      .sort((a, b) => a.worst !== b.worst ? a.worst - b.worst : a.sum - b.sum)
      .map(({ c }) => c);

    const grid = gc.candidatesGrid;
    for (const { tile } of sortedValid) grid.appendChild(tile);
    for (const { tile } of invalid) grid.appendChild(tile);
  }
}

// ── Health bar helpers ───────────────────────────────────────────────────

function _isPercent(key) {
  const k = key.toLowerCase();
  return k.includes('pct') || k.includes('percent') || k.includes('utilization')
    || k.includes('wastage') || k.includes('waste')
    || k.includes('accuracy') || k.includes('acc');
}

function _objToPercent(key, val) {
  if (typeof val !== 'number') return 0;
  if (_isPercent(key)) return Math.min(100, Math.max(0, val <= 1 ? val * 100 : val));
  if (val <= 0) return 0;
  const logVal = Math.log10(Math.max(1, val));
  return Math.min(100, Math.max(0, (logVal / 7) * 100));
}

function _objColor(key) {
  const k = key.toLowerCase();
  if (k.includes('accuracy') || k.includes('acc')) return 'var(--sl-accent-cyan)';
  if (k.includes('wastage') || k.includes('waste')) return 'var(--sl-accent-rose)';
  if (k.includes('utilization') || k.includes('util')) return 'var(--sl-accent-green)';
  if (k.includes('power') || k.includes('energy')) return 'var(--sl-accent-amber)';
  if (k.includes('param')) return 'var(--sl-accent-blue)';
  return 'var(--sl-accent-blue)';
}

function _shortLabel(key) {
  const abbrev = {
    estimated_accuracy: 'Accuracy',
    total_params: 'Params',
    total_param_capacity: 'Capacity',
    total_sync_barriers: 'Barriers',
    param_utilization_pct: 'Utiliz',
    neuron_wastage_pct: 'N-Waste',
    axon_wastage_pct: 'A-Waste',
  };
  return abbrev[key] || key.replace(/_/g, ' ').replace(/\b(\w)/g, c => c.toUpperCase()).substring(0, 9);
}

function _fmtVal(key, val) {
  if (typeof val !== 'number') return String(val);
  if (_isPercent(key)) {
    const pv = val <= 1 ? val * 100 : val;
    return pv.toFixed(1) + '%';
  }
  const abs = Math.abs(val);
  if (abs >= 1e9) return (val / 1e9).toFixed(1) + 'B';
  if (abs >= 1e6) return (val / 1e6).toFixed(2) + 'M';
  if (abs >= 1e3) return (val / 1e3).toFixed(1) + 'K';
  if (Number.isInteger(val)) return String(val);
  return val.toFixed(3);
}

// ── Pareto summary ───────────────────────────────────────────────────────

function _renderParetoSummary(front, objSpecs) {
  if (!front || front.length === 0) return '';
  const keys = Object.keys(front[0]);
  let html = '<div class="sl-pareto-title">PARETO FRONT (top ' + front.length + ')</div>';
  html += '<div class="sl-pareto-grid">';
  for (let i = 0; i < Math.min(front.length, 5); i++) {
    html += '<div class="sl-pareto-entry">';
    for (const k of keys) {
      const v = front[i][k];
      const pct = _objToPercent(k, v);
      const color = _objColor(k);
      const display = _fmtVal(k, v);
      html += `<div class="sl-bar-row sl-bar-mini" title="${esc(k)}: ${typeof v === 'number' ? v.toFixed(6) : v}">
        <span class="sl-bar-label">${esc(_shortLabel(k))}</span>
        <div class="sl-bar-track"><div class="sl-bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <span class="sl-bar-val">${esc(display)}</span>
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
