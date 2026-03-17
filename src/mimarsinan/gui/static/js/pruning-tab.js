/* Pruning tab: per-layer weight heatmaps with pruning masks (red lines).
 * Shown for the Pruning Adaptation step; same red-line convention as IR Graph and Hardware. */
import { esc } from './util.js';

const MAX_HEATMAP_PX = 400;

export function renderPruningTab(pruningData, container) {
  if (!pruningData || !pruningData.layers || pruningData.layers.length === 0) {
    container.innerHTML = '<div class="empty-state">No pruning data for this step. Pruning masks are available after Pruning Adaptation runs.</div>';
    return;
  }

  const layers = pruningData.layers;
  const first = layers[0];

  let html = `
    <div class="pruning-browse">
      <div class="pruning-layer-list card">
        <div class="card-header">Layers</div>
        <div class="card-body scrollable">
          <ul class="pruning-layer-ul">`;

  for (const L of layers) {
    const shapeStr = L.shape ? `${L.shape[0]} × ${L.shape[1]}` : '—';
    const prunedStr = `${L.pruned_rows ?? 0} rows, ${L.pruned_cols ?? 0} cols pruned`;
    const name = L.layer_name != null ? esc(String(L.layer_name)) : `Layer ${L.layer_index}`;
    const activeClass = L.layer_index === first.layer_index ? ' active' : '';
    html += `
            <li class="pruning-layer-item${activeClass}" data-idx="${L.layer_index}">
              <span class="pruning-layer-name">${name}</span>
              <span class="pruning-layer-meta">${shapeStr}</span>
              <span class="pruning-layer-pruned">${prunedStr}</span>
            </li>`;
  }

  html += `
          </ul>
        </div>
      </div>
      <div class="pruning-panels">
        <div class="pruning-detail card">
          <div class="card-header">Layer details</div>
          <div class="card-body">
            <div class="pruning-detail-meta" id="pruning-detail-meta"></div>
          </div>
        </div>
        <div class="pruning-heatmap-panel card">
          <div class="card-header">Weight matrix (pruning masks)</div>
          <div class="card-body">
            <div class="pruning-detail-heatmap-wrap" id="pruning-detail-heatmap"></div>
          </div>
        </div>
      </div>
    </div>`;

  container.innerHTML = html;

  const metaEl = document.getElementById('pruning-detail-meta');
  const heatmapEl = document.getElementById('pruning-detail-heatmap');

  function showLayer(layer) {
    if (!layer) return;
    const name = layer.layer_name != null ? esc(String(layer.layer_name)) : `Layer ${layer.layer_index}`;
    const shapeStr = layer.shape ? `${layer.shape[0]} × ${layer.shape[1]}` : '—';
    metaEl.innerHTML = `
      <table class="config-table">
        <tr><td>Layer</td><td>${name} (index ${layer.layer_index})</td></tr>
        <tr><td>Shape</td><td>${shapeStr} (rows × cols)</td></tr>
        <tr><td>Pruned rows</td><td>${layer.pruned_rows ?? 0}</td></tr>
        <tr><td>Pruned cols</td><td>${layer.pruned_cols ?? 0}</td></tr>
      </table>`;
    const uri = (layer.heatmap_image || '').replace(/"/g, '&quot;');
    heatmapEl.innerHTML = uri
      ? `<img src="${uri}" alt="Weight heatmap with pruning masks" class="pruning-detail-heatmap" style="border:1px solid var(--border-color, #2e3140);border-radius:4px">`
      : '<div class="empty-state">No heatmap</div>';
  }

  showLayer(first);

  container.querySelectorAll('.pruning-layer-item').forEach((li) => {
    li.addEventListener('click', () => {
      const idx = parseInt(li.dataset.idx, 10);
      const layer = layers.find((L) => L.layer_index === idx);
      if (!layer) return;
      container.querySelectorAll('.pruning-layer-item').forEach((el) => el.classList.remove('active'));
      li.classList.add('active');
      showLayer(layer);
    });
  });
}
