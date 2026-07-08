/* Per-step insight panels: the events timeline strip + wall context on every
   step page, the MBH gate story for tuner steps, the integer-grid
   quantization report, activation-scale distributions, and the Wilson
   binomial interval for sampled gate verdicts. Data: the step detail's
   annotations (every step event, with payload) and snapshot. */
import { esc, fmtDuration, safeReact, emptyAnnotation } from './util.js';

const TONE_COLORS = { good: '#4ade80', warn: '#fbbf24', bad: '#f87171', neutral: '#8494a7' };

// ── Context strip: wall share + events timeline (every step page) ────────
export function renderStepContextStrip(container, detail, annotations, pipeline) {
  if (!container) return;
  const chips = [];
  if (detail.duration != null && pipeline && pipeline.steps) {
    let total = 0;
    for (const s of pipeline.steps) total += s.duration || 0;
    const share = total > 0 ? (detail.duration / total) * 100 : null;
    chips.push(`<span class="tk-chip mono" title="This step's wall time and its share of the run's total wall">
      wall ${fmtDuration(detail.duration)}${share != null ? ` · ${share.toFixed(1)}% of run` : ''}</span>`);
  }
  const events = annotations || [];
  if (events.length) {
    chips.push(`<span class="tk-chip dim">${events.length} events</span>`);
  }
  const strip = timelineStripHtml(events, detail.duration);
  container.innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px">
      ${chips.join('')}
      ${strip ? `<div style="flex:1 1 260px;min-width:200px">${strip}</div>` : ''}
    </div>`;
}

function timelineStripHtml(events, duration) {
  const timed = events.filter(e => e.x != null);
  if (!timed.length) return '';
  const span = Math.max(duration || 0, ...timed.map(e => e.x)) || 1;
  const W = 100; // percent-based positions inside a fixed-height strip
  const marks = timed.map(e => {
    const x = Math.min(100, (e.x / span) * W);
    const color = TONE_COLORS[e.tone] || TONE_COLORS.neutral;
    return `<span title="${esc(`${e.label} — t=${e.x.toFixed(1)}s`)}"
      style="position:absolute;left:${x.toFixed(2)}%;top:2px;bottom:2px;width:2.5px;border-radius:1px;background:${color};cursor:default"></span>`;
  }).join('');
  return `<div title="Step events over elapsed time"
    style="position:relative;height:16px;border:1px solid var(--border);border-radius:6px;background:var(--bg-input);overflow:hidden">${marks}</div>`;
}

// ── Gate story (MBH rung progression for tuner steps) ────────────────────
export function hasGateStory(annotations) {
  return (annotations || []).some(e => e.kind === 'mbh_gate');
}

export function renderGateStoryTab(annotations, container) {
  const events = (annotations || []).filter(e => e.payload);
  const gates = events.filter(e => e.kind === 'mbh_gate');
  if (!gates.length) {
    container.innerHTML = '<div class="empty-state">No gate events for this step</div>';
    return;
  }
  const endpoint = events.find(e => e.kind === 'mbh_endpoint');
  const refusals = events.filter(e => e.kind === 'lr_refusal');

  // Rebuild the per-step staircase from the gate events (mirrors
  // viewmodel/staircase_vm.py lanes, scoped to this step).
  const probes = [];
  const staircase = [];
  let stalled = false;
  for (const g of gates) {
    const p = g.payload;
    if (p.action === 'entry') {
      staircase.push({ i: probes.length, best: p.best_full_acc });
    } else if (p.action === 'accept' || p.action === 'reject') {
      probes.push({
        i: probes.length, rung: p.rung, rate: p.rate,
        full_acc: p.full_acc, accepted: p.action === 'accept',
      });
      staircase.push({ i: probes.length, best: p.best_full_acc });
    } else if (p.action === 'stall') stalled = true;
  }

  const chips = [];
  const accepted = probes.filter(p => p.accepted).length;
  chips.push(`<span class="tk-chip good">▲ ${accepted} accepted</span>`);
  chips.push(`<span class="tk-chip warn">▽ ${probes.length - accepted} refused</span>`);
  if (stalled) chips.push('<span class="tk-chip bad">constructive stall</span>');
  if (refusals.length) chips.push(`<span class="tk-chip bad">${refusals.length} LR refusals</span>`);
  if (endpoint) {
    const p = endpoint.payload;
    chips.push(`<span class="tk-chip ${p.reached ? 'good' : 'warn'}" title="endpoint train-to-D-hat">
      endpoint ${p.reached ? 'reached' : 'exhausted'} · ${p.steps_used ?? '?'} / ${p.budget_steps ?? '?'} steps</span>`);
    if (p.entry != null && p.exit != null) {
      chips.push(`<span class="tk-chip mono">D̂ ${Number(p.entry).toFixed(4)} → ${Number(p.exit).toFixed(4)}</span>`);
    }
  }

  container.innerHTML = `
    <div class="card" style="margin-bottom:16px">
      <div class="card-header"><span>MBH gate story</span>
        <span class="note">accepted best-D&#770; only ever rises — probes vs the ratchet</span></div>
      <div class="card-body">
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">${chips.join('')}</div>
        <div id="gate-story-chart" style="min-height:280px"></div>
      </div>
    </div>`;

  const traces = [{
    x: staircase.map(p => p.i), y: staircase.map(p => p.best),
    type: 'scatter', mode: 'lines', line: { shape: 'hv', width: 2, color: '#22d3ee' },
    name: 'accepted best-D̂',
  }];
  const acceptedProbes = probes.filter(p => p.accepted);
  const rejectedProbes = probes.filter(p => !p.accepted);
  if (acceptedProbes.length) traces.push({
    x: acceptedProbes.map(p => p.i + 1), y: acceptedProbes.map(p => p.full_acc),
    type: 'scatter', mode: 'markers', name: 'accepted',
    marker: { symbol: 'triangle-up', size: 10, color: '#4ade80' },
    text: acceptedProbes.map(p => `rung ${p.rung} · rate ${p.rate != null ? p.rate.toFixed(3) : '?'}`),
  });
  if (rejectedProbes.length) traces.push({
    x: rejectedProbes.map(p => p.i + 1), y: rejectedProbes.map(p => p.full_acc),
    type: 'scatter', mode: 'markers', name: 'refused',
    marker: { symbol: 'triangle-down-open', size: 10, color: '#fbbf24' },
    text: rejectedProbes.map(p => `rung ${p.rung} · rate ${p.rate != null ? p.rate.toFixed(3) : '?'}`),
  });
  safeReact('gate-story-chart', traces, {
    height: 300,
    xaxis: { title: 'gate probe →', dtick: 1 },
    yaxis: { title: 'deployed full-transform D̂', automargin: true },
    legend: { orientation: 'h', y: -0.25, font: { size: 10 } },
  });
}

// ── Quantization report (integer-grid diagnostics) ───────────────────────
export function quantizationReport(annotations) {
  const e = (annotations || []).find(a => a.kind === 'quantization_report');
  return e && e.payload && Array.isArray(e.payload.layers) ? e.payload : null;
}

export function renderQuantizationTab(report, container) {
  const layers = report.layers;
  const maxLevels = 2 * report.q_max + 1;
  container.innerHTML = `
    <div class="card" style="margin-bottom:16px">
      <div class="card-header"><span>Integer weight grid</span>
        <span class="note">${report.bits}-bit · levels on the ±${report.q_max} grid — the deployed reality</span></div>
      <div class="card-body">
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">
          <span class="tk-chip mono">bits ${report.bits}</span>
          <span class="tk-chip mono">grid ±${report.q_max} (${maxLevels} levels)</span>
          <span class="tk-chip dim">${layers.length} layers</span>
        </div>
        <div id="quant-levels-chart" style="min-height:260px"></div>
      </div>
    </div>
    <div class="card">
      <div class="card-header">Per-layer grid diagnostics</div>
      <div class="card-body no-pad">
        <table class="data-table compact">
          <thead><tr><th>Layer</th><th>Weights</th><th>Levels used</th><th>Zero %</th><th>At clip %</th><th>Scale</th></tr></thead>
          <tbody>${layers.map(l => `<tr>
            <td>${esc(l.name)}</td>
            <td class="num">${l.n_weights}</td>
            <td class="num">${l.effective_levels} / ${maxLevels}</td>
            <td class="num">${(l.zero_frac * 100).toFixed(1)}</td>
            <td class="num">${(l.clip_frac * 100).toFixed(1)}</td>
            <td class="num">${l.parameter_scale != null ? l.parameter_scale.toFixed(3) : '—'}</td>
          </tr>`).join('')}</tbody>
        </table>
      </div>
    </div>`;

  safeReact('quant-levels-chart', [
    {
      x: layers.map(l => l.name), y: layers.map(l => l.effective_levels / maxLevels),
      type: 'bar', name: 'grid utilization', marker: { color: '#22d3ee' },
    },
    {
      x: layers.map(l => l.name), y: layers.map(l => l.clip_frac),
      type: 'bar', name: 'clip fraction', marker: { color: '#f43f5e' },
    },
    {
      x: layers.map(l => l.name), y: layers.map(l => l.zero_frac),
      type: 'bar', name: 'zero fraction', marker: { color: '#55637a' },
    },
  ], {
    height: 280, barmode: 'group',
    yaxis: { title: 'fraction', range: [0, 1.05] },
    xaxis: { tickangle: -30, automargin: true, tickfont: { size: 10 } },
    legend: { orientation: 'h', y: 1.12, font: { size: 10 } },
  });
}

// ── Activation-scale distributions (Activation Analysis) ─────────────────
export function renderActivationScaleStats(stats, container) {
  const layers = (stats && stats.layers) || [];
  if (!layers.length) return;
  const host = document.createElement('div');
  host.className = 'card';
  host.style.marginBottom = '16px';
  host.innerHTML = `
    <div class="card-header"><span>Per-layer activation-scale distributions</span>
      <span class="note">q${((stats.quantile ?? 0.99) * 100).toFixed(0)} chosen scale vs the sampled activation range · ${stats.num_batches ?? '?'} batches</span></div>
    <div class="card-body">
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">
        <span class="tk-chip mono" title="chosen scales across layers">scales ${fmtSig(stats.summary?.min_scale)} … ${fmtSig(stats.summary?.max_scale)}</span>
        <span class="tk-chip mono">median ${fmtSig(stats.summary?.median_scale)}</span>
      </div>
      <div id="act-scale-chart" style="min-height:${Math.max(220, layers.length * 26 + 90)}px"></div>
    </div>`;
  container.prepend(host);

  const names = layers.map(l => l.name);
  const traces = [
    // Sampled activation range as horizontal bars (min → max).
    {
      x: layers.map(l => (l.sample_max ?? 0) - (l.sample_min ?? 0)),
      base: layers.map(l => l.sample_min ?? 0),
      y: names, type: 'bar', orientation: 'h', name: 'sampled range',
      marker: { color: 'rgba(59,130,246,0.35)' },
      hovertext: layers.map(l =>
        `${l.name}: samples ${l.active_sample_count}/${l.sample_count} active · median ${fmtSig(l.sample_median)}`),
    },
    {
      x: layers.map(l => l.sample_median ?? 0), y: names,
      type: 'scatter', mode: 'markers', name: 'sample median',
      marker: { symbol: 'line-ns-open', size: 12, color: '#8494a7' },
    },
    {
      x: layers.map(l => l.scale), y: names,
      type: 'scatter', mode: 'markers', name: 'chosen scale',
      marker: { symbol: 'diamond', size: 9, color: '#22d3ee' },
    },
  ];
  safeReact('act-scale-chart', traces, {
    height: Math.max(220, layers.length * 26 + 90),
    xaxis: { title: 'activation value', automargin: true },
    yaxis: { autorange: 'reversed', automargin: true, tickfont: { size: 10 } },
    legend: { orientation: 'h', y: 1.1, font: { size: 10 } },
    showlegend: true,
  });
}

function fmtSig(v) {
  return v == null ? '—' : Number(v).toPrecision(3);
}

// ── Wilson interval for sampled gate verdicts ─────────────────────────────
// detail = verdict.detail; renders when it carries a sample count and a
// rate-like value in [0,1] (parity/accuracy reads are binomial).
export function wilsonRowHtml(detail) {
  if (!detail) return '';
  const n = Number(detail.samples ?? detail.sample_count ?? detail.n_samples);
  if (!Number.isFinite(n) || n <= 0) return '';
  const rateKey = ['agreement', 'parity', 'accuracy', 'metric'].find(
    k => detail[k] != null && Number(detail[k]) >= 0 && Number(detail[k]) <= 1);
  if (!rateKey) return '';
  const p = Number(detail[rateKey]);
  const z = 1.96;
  const denom = 1 + z * z / n;
  const center = (p + z * z / (2 * n)) / denom;
  const half = (z / denom) * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  const lo = Math.max(0, center - half), hi = Math.min(1, center + half);
  return `<tr><td>95% Wilson CI (${esc(rateKey)}, n=${n})</td>
    <td>[${lo.toFixed(4)}, ${hi.toFixed(4)}]</td></tr>`;
}
