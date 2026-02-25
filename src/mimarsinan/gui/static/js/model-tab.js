/* Model visualization tab. */
import { esc, fmtNum, safeReact } from './util.js';

export function renderModelTab(model, container) {
  if (!model) { container.innerHTML = '<div class="empty-state">No model data</div>'; return; }

  const layers = model.layers || [];
  const firstShape = layers.length > 0 && layers[0].weight ? layers[0].weight.shape.join(' × ') : '-';

  let html = `
    <div class="grid-3" style="margin-bottom:20px">
      <div class="card"><div class="big-metric"><div class="value">${fmtNum(model.total_params)}</div><div class="label">Total Parameters</div></div></div>
      <div class="card"><div class="big-metric"><div class="value">${model.num_layers}</div><div class="label">Layers</div></div></div>
      <div class="card"><div class="big-metric"><div class="value">${firstShape}</div><div class="label">First Layer Shape</div></div></div>
    </div>
    <div class="grid-2">
      <div class="card"><div class="card-header">Parameter Count per Layer</div><div class="card-body"><div id="m-param" style="min-height:180px"></div></div></div>
      <div class="card"><div class="card-header">Weight Statistics per Layer</div><div class="card-body"><div id="m-weight" style="min-height:180px"></div></div></div>
    </div>
    <div class="card" style="margin-bottom:20px"><div class="card-header">Layer Details</div><div class="card-body scrollable">
      <table class="data-table"><thead><tr><th>Idx</th><th>Shape</th><th>Params</th><th>W Mean</th><th>W Std</th><th>Act Scale</th><th>Param Scale</th></tr></thead><tbody>`;

  for (const l of layers) {
    const w = l.weight;
    const params = w ? w.shape.reduce((a, b) => a * b, 1) : 0;
    html += `<tr><td>${l.index}</td><td>${w ? w.shape.join(' × ') : '-'}</td><td>${fmtNum(params)}</td>
      <td>${w ? w.mean.toFixed(4) : '-'}</td><td>${w ? w.std.toFixed(4) : '-'}</td>
      <td>${l.activation_scale != null ? l.activation_scale.toFixed(3) : '-'}</td>
      <td>${l.parameter_scale != null ? l.parameter_scale.toFixed(3) : '-'}</td></tr>`;
  }

  html += `</tbody></table></div></div>
    <div class="card"><div class="card-header">Weight Distributions</div><div class="card-body"><div id="m-dist" style="min-height:200px"></div></div></div>`;
  container.innerHTML = html;

  const h = Math.max(180, layers.length * 22 + 60);
  safeReact('m-param', [{
    y: layers.map(l => `L${l.index}`), x: layers.map(l => l.weight ? l.weight.shape.reduce((a, b) => a * b, 1) : 0),
    type: 'bar', orientation: 'h', marker: { color: '#5b8af5' },
  }], { height: h, yaxis: { autorange: 'reversed' } });

  safeReact('m-weight', [
    { y: layers.map(l => `L${l.index}`), x: layers.map(l => l.weight?.mean || 0), name: 'Mean', type: 'bar', orientation: 'h', marker: { color: '#5b8af5' } },
    { y: layers.map(l => `L${l.index}`), x: layers.map(l => l.weight?.std || 0), name: 'Std', type: 'bar', orientation: 'h', marker: { color: '#ff9800' } },
  ], { barmode: 'group', height: h, yaxis: { autorange: 'reversed' } });

  const distTraces = layers.filter(l => l.weight?.histogram).map(l => {
    const hist = l.weight.histogram;
    const mids = hist.bin_edges.slice(0, -1).map((e, j) => (e + hist.bin_edges[j + 1]) / 2);
    return { x: mids, y: hist.counts, name: `L${l.index}`, type: 'bar', opacity: 0.7 };
  });
  if (distTraces.length > 0)
    safeReact('m-dist', distTraces, { barmode: 'overlay', height: 300, xaxis: { title: 'Weight Value' }, yaxis: { title: 'Count' } });
}
