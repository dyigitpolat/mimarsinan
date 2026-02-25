/* Architecture search results tab. */
import { esc, safeReact } from './util.js';

export function renderSearchTab(search, container) {
  if (!search) { container.innerHTML = '<div class="empty-state">No search data</div>'; return; }

  let html = '<div class="grid-2" style="margin-bottom:20px">';
  if (search.best) {
    html += '<div class="card"><div class="card-header">Best Candidate</div><div class="card-body">';
    if (search.best.objectives && Object.keys(search.best.objectives).length > 0) {
      html += '<div class="section-label">Objectives</div><table class="config-table">';
      for (const [k, v] of Object.entries(search.best.objectives))
        html += `<tr><td>${esc(k)}</td><td>${typeof v === 'number' ? v.toFixed(4) : esc(String(v))}</td></tr>`;
      html += '</table>';
    }
    if (search.best.config && Object.keys(search.best.config).length > 0) {
      html += '<div class="section-label" style="margin-top:12px">Configuration</div><table class="config-table">';
      for (const [k, v] of Object.entries(search.best.config)) {
        const d = formatValue(v);
        html += `<tr><td>${esc(k)}</td><td title="${esc(d)}">${esc(d.length > 60 ? d.substring(0, 57) + '...' : d)}</td></tr>`;
      }
      html += '</table>';
    }
    html += '</div></div>';
  }

  html += `<div class="card"><div class="big-metric"><div class="value">${search.num_candidates}</div><div class="label">Candidates Evaluated</div></div>`;
  if (search.objectives?.length > 0) {
    html += '<div style="padding:0 18px 18px"><table class="config-table">';
    for (const o of search.objectives) html += `<tr><td>${esc(o.name)}</td><td>${esc(o.goal)}</td></tr>`;
    html += '</table></div>';
  }
  html += '</div></div>';

  html += '<div class="card" style="margin-bottom:20px"><div class="card-header">Pareto Front</div><div class="card-body"><div id="s-pareto" style="min-height:220px"></div></div></div>';

  if (search.pareto_front?.length > 0) {
    html += '<div class="card" style="margin-bottom:20px"><div class="card-header">Pareto Candidates</div><div class="card-body scrollable">';
    html += buildParetoTable(search.pareto_front);
    html += '</div></div>';
  }

  if (search.history?.length > 0)
    html += '<div class="card"><div class="card-header">Search History</div><div class="card-body"><div id="s-hist" style="min-height:200px"></div></div></div>';

  container.innerHTML = html;
  renderParetoChart(search);
  renderHistoryChart(search);
}

function buildParetoTable(pareto) {
  const objKeys = Object.keys(pareto[0].objectives || {});
  const cfgKeys = Object.keys(pareto[0].config || {}).slice(0, 5);
  let html = '<table class="data-table"><thead><tr><th>#</th>';
  for (const k of objKeys) html += `<th>${esc(k)}</th>`;
  for (const k of cfgKeys) html += `<th>${esc(k)}</th>`;
  html += '</tr></thead><tbody>';
  for (let ci = 0; ci < pareto.length; ci++) {
    const c = pareto[ci];
    html += `<tr><td>${ci}</td>`;
    for (const k of objKeys) {
      const v = (c.objectives || {})[k];
      html += `<td>${typeof v === 'number' ? v.toFixed(4) : esc(String(v ?? '-'))}</td>`;
    }
    for (const k of cfgKeys) {
      const d = formatValue((c.config || {})[k]);
      html += `<td title="${esc(d)}">${esc(d.length > 40 ? d.substring(0, 37) + '...' : d)}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  return html;
}

function renderParetoChart(search) {
  if (!search.pareto_front?.length) return;
  const objs = search.objectives || [];
  const firstObj = search.pareto_front[0].objectives || {};
  const ok = Object.keys(firstObj);
  const xKey = objs.length > 0 ? objs[0].name : ok[0];
  const yKey = objs.length > 1 ? objs[1].name : (ok[1] || xKey);
  if (!xKey || !yKey) return;
  safeReact('s-pareto', [{
    x: search.pareto_front.map(c => (c.objectives || {})[xKey]),
    y: search.pareto_front.map(c => (c.objectives || {})[yKey]),
    text: search.pareto_front.map((_, i) => `Candidate ${i}`),
    mode: 'markers', type: 'scatter', marker: { size: 10, color: '#5b8af5' },
  }], { height: 300, xaxis: { title: xKey }, yaxis: { title: yKey } });
}

function renderHistoryChart(search) {
  if (!search.history?.length) return;
  const histKeys = Object.keys(search.history[0] || {}).filter(k => typeof search.history[0][k] === 'number');
  if (histKeys.length === 0) return;
  safeReact('s-hist', histKeys.map(k => ({
    x: search.history.map((_, i) => i), y: search.history.map(h => h[k]),
    name: k, type: 'scatter', mode: 'lines+markers', line: { width: 1.5 }, marker: { size: 4 },
  })), { height: 260, xaxis: { title: 'Generation' }, showlegend: true, legend: { font: { size: 10 } } });
}

function formatValue(v) {
  if (v == null) return '-';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}
