/* Overview charts (measured points, verdict/carried markers, timing). */
import { esc, fmtDuration, elapsedFromStepStart, safeReact, emptyAnnotation, groupAccent } from './util.js';
import {
  planTraceExtension,
  applyStreamPlan,
  readStreamState,
  markStreamState,
} from './stream-plot.js';

const TONE_COLORS = { pass: '#4ade80', fail: '#f87171', carried: '#8494a7' };

// ── Overview cards ───────────────────────────────────────────────────────
export function renderOverviewCards(pipeline) {
  if (!pipeline) return;
  const steps = pipeline.steps || [];
  const completed = steps.filter(s => s.status === 'completed');
  const running = steps.find(s => s.status === 'running');

  const curEl = document.getElementById('current-step-name');
  if (curEl) curEl.textContent = running ? running.name : (completed.length === steps.length && steps.length > 0 ? 'Pipeline Complete' : 'Waiting...');
  const progEl = document.getElementById('progress-text');
  if (progEl) progEl.textContent = `${completed.length} / ${steps.length}`;

  renderMetricProgression(steps, pipeline.overview_chart);
  renderStepTiming(steps);
}

function renderMetricProgression(steps, chart) {
  const points = (chart && chart.points) || [];
  const markers = (chart && chart.markers) || [];
  // The x axis is the FULL step sequence so no step is ever invisible.
  const categories = steps.map(s => s.name);
  const { shapes, annotations } = _progressionOverlays(points, markers);
  const values = points.map(p => p.value);
  const maxY = values.length > 0 ? Math.max(1, ...values) * 1.05 : 1;

  // The measured line is append-only: a completed measured step adds one real
  // point and never mutates a prior one. Extend just the new tail and relayout
  // the event lines/labels so the line GROWS smoothly instead of the whole
  // plot flashing on every step-completion frame. A structural change or a
  // reconcile-shrink (REST snapshot after a reconnect) falls back to a redraw.
  const el = document.getElementById('chart-metric-progression');
  const nextTraces = points.length > 0
    ? [{ name: 'measured', x: points.map(p => p.step), y: values }]
    : [];
  if (el && el.data) {
    const plan = planTraceExtension(readStreamState(el), nextTraces);
    if (applyStreamPlan(el, plan)) {
      Plotly.relayout(el, {
        shapes, annotations,
        'yaxis.range': [0, maxY],
        'xaxis.categoryarray': categories,
      });
      return;
    }
  }

  const traces = points.length > 0 ? [{
    x: points.map(p => p.step), y: values,
    type: 'scatter', mode: 'lines+markers',
    marker: { size: 8, color: '#22d3ee' }, line: { width: 2, color: '#5b8af5' },
    name: 'measured',
  }] : [];
  safeReact('chart-metric-progression', traces, {
    margin: { t: 40, r: 30, b: 100, l: 60 },
    xaxis: {
      tickangle: -45, automargin: true, tickfont: { size: 10 },
      type: 'category', categoryorder: 'array', categoryarray: categories,
    },
    yaxis: { title: 'Measured Metric', automargin: true, range: [0, maxY] },
    height: 300, annotations, shapes,
  });
  markStreamState(document.getElementById('chart-metric-progression'),
    { names: nextTraces.map(t => t.name), counts: nextTraces.map(t => t.x.length) });
}

// Event lines (verdict/failed/carried) + the last-point value label. Carried
// steps become a neutral labeled line — NEVER a fabricated data point at the
// carried metric (the server already split points vs markers honestly).
function _progressionOverlays(points, markers) {
  const shapes = [];
  const annotations = (points.length === 0 && markers.length === 0)
    ? emptyAnnotation('No measured metrics yet') : [];
  const lastPt = points.length > 0 ? points[points.length - 1] : null;
  if (lastPt) {
    annotations.push({
      x: lastPt.step, y: lastPt.value,
      text: lastPt.value.toFixed(4),
      showarrow: true, arrowhead: 2, ax: 30, ay: -25,
      font: { size: 11, color: '#22d3ee' },
      bgcolor: 'rgba(15,17,23,0.85)', bordercolor: '#22d3ee', borderwidth: 1,
    });
  }
  for (const marker of markers) {
    const color = TONE_COLORS[marker.status] || TONE_COLORS.pass;
    shapes.push({
      type: 'line', xref: 'x', yref: 'paper',
      x0: marker.step, x1: marker.step, y0: 0, y1: 1,
      line: { color, width: 1.5, dash: 'dot' },
    });
    annotations.push({
      x: marker.step, y: 1.04, yref: 'paper',
      text: marker.glyph, showarrow: false,
      font: { size: 13, color },
      hovertext: `${marker.glyph} ${marker.label || ''}`,
    });
  }
  return { shapes, annotations };
}

function renderStepTiming(steps) {
  const now = Date.now() / 1000;
  const timed = [];
  for (const s of steps) {
    if (s.duration != null) timed.push({ ...s, duration: s.duration, running: false });
    else if (s.status === 'running' && s.start_time != null) timed.push({ ...s, duration: elapsedFromStepStart(s.start_time, now), running: true });
  }
  const total = timed.reduce((acc, s) => acc + s.duration, 0);
  const pct = d => total > 0 ? ` · ${((d / total) * 100).toFixed(1)}% of run` : '';
  const durations = timed.map(s => s.duration);
  const colors = timed.map(s => s.running ? '#ff9800' : `rgba(${groupAccent(s.semantic_group)},0.85)`);
  const texts = timed.map(s => fmtDuration(s.duration) + (s.running ? ' ●' : ''));
  const hover = timed.map(s => `${s.name} — ${fmtDuration(s.duration)}${s.running ? ' (running)' : pct(s.duration)}`);

  // Drop-to-latest fast path: when the bar SET is unchanged (only the running
  // bar's length + total grew) restyle the single trace's values in place —
  // no flash, no re-layout. A new or just-finished bar changes the set (and
  // the chart height), so that falls through to a full redraw.
  const el = document.getElementById('chart-step-timing');
  const key = timed.map(s => s.name).join('|');
  if (el && el.data && el._timingKey === key && timed.length > 0) {
    Plotly.restyle(el, {
      x: [durations], text: [texts], hovertext: [hover], 'marker.color': [colors],
    }, [0]);
    Plotly.relayout(el, { 'xaxis.autorange': true });
    return;
  }

  const anno = timed.length === 0 ? emptyAnnotation('No timing data yet') : [];
  const traces = timed.length > 0 ? [{
    x: durations, y: timed.map(s => s.name),
    type: 'bar', orientation: 'h',
    marker: { color: colors },
    text: texts,
    textposition: 'auto', textfont: { size: 10, color: '#e8eaed' },
    hovertext: hover,
    hoverinfo: 'text',
  }] : [];
  safeReact('chart-step-timing', traces, {
    margin: { t: 16, r: 30, b: 50, l: 8 },
    xaxis: { title: 'Duration (s)', automargin: true },
    yaxis: { autorange: 'reversed', automargin: true, tickfont: { size: 11 } },
    height: Math.max(200, timed.length * 26 + 70), annotations: anno,
  });
  if (el) el._timingKey = key;
}

export function renderConfig(config) {
  const el = document.getElementById('config-body');
  if (!el || !config) return;
  const priority = ['spiking_mode', 'pipeline_mode', 'model_config_mode', 'hw_config_mode', 'activation_quantization',
    'weight_quantization', 'max_axons', 'max_neurons', 'weight_bits', 'target_tq',
    'training_epochs', 'lr', 'simulation_steps', 'max_simulation_samples', 'input_shape', 'num_classes', 'device'];
  const all = Object.keys(config);
  const sorted = [...priority.filter(k => k in config), ...all.filter(k => !priority.includes(k)).sort()];
  let html = '<div class="card"><div class="card-header">Deployment parameters'
    + '<span class="note">flat fallback — no structured view for this run</span></div>'
    + '<div class="card-body no-pad"><table class="config-table">';
  for (const key of sorted) {
    const val = config[key];
    const display = typeof val === 'object' ? JSON.stringify(val) : String(val);
    html += `<tr><td>${esc(key)}</td><td>${esc(display)}</td></tr>`;
  }
  html += '</table></div></div>';
  el.innerHTML = html;
}
