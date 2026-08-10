/* Pruning tab: per-layer pre→post dimensions, achieved vs configured sparsity,
 * mask map + weight heatmap (same red-line convention as IR Graph and Hardware),
 * and visible skip diagnostics — a skipped layer is never silently absent. */
import { esc } from './util.js';
import { imgSrcAttr } from './resource-urls.js';

function fmtPct(x) {
  if (x == null || Number.isNaN(Number(x))) return '—';
  return `${(Number(x) * 100).toFixed(1)}%`;
}

// Older persisted snapshots carry only shape/pruned counts; derive pre/post.
function layerDims(L) {
  const preN = L.pre_neurons ?? (L.shape ? L.shape[0] : null);
  const preA = L.pre_axons ?? (L.shape ? L.shape[1] : null);
  const postN = L.post_neurons ?? (preN != null ? preN - (L.pruned_rows ?? 0) : null);
  const postA = L.post_axons ?? (preA != null ? preA - (L.pruned_cols ?? 0) : null);
  return { preN, preA, postN, postA };
}

function dimArrow(pre, post, unit) {
  if (pre == null) return '—';
  return `${pre} → ${post ?? pre} ${unit}`;
}

function skippedBlock(skipped) {
  if (!skipped || skipped.length === 0) return '';
  const rows = skipped.map((s) => `
        <li class="pruning-skip-item">
          <span class="pruning-skip-layer">${esc(String(s.layer ?? '?'))}</span>
          <span class="pruning-skip-reason">${esc(String(s.reason ?? 'skipped'))}</span>
        </li>`).join('');
  return `
    <div class="pruning-skips card">
      <div class="card-header">Skipped layers (${skipped.length})</div>
      <div class="card-body">
        <ul class="pruning-skip-ul">${rows}</ul>
      </div>
    </div>`;
}

// Visible failure chip instead of a blank <img> when a resource 404s.
function wireImgFallbacks(rootEl) {
  rootEl.querySelectorAll('img[data-fallback-label]').forEach((img) => {
    img.addEventListener('error', () => {
      const chip = document.createElement('div');
      chip.className = 'img-fallback-chip';
      chip.textContent = `${img.dataset.fallbackLabel || 'image'} failed to load`;
      img.replaceWith(chip);
    });
  });
}

function imgPanelHtml(resourceRef, label, alt) {
  const uri = imgSrcAttr(resourceRef);
  if (!uri) return `<div class="empty-state">No ${esc(label)}</div>`;
  return `<img src="${uri}" alt="${esc(alt)}" loading="lazy" decoding="async"
    class="pruning-detail-heatmap" data-fallback-label="${esc(label)}"
    style="border:1px solid var(--border-color, #2e3140);border-radius:4px">`;
}

export function renderPruningTab(pruningData, container) {
  const layers = (pruningData && pruningData.layers) || [];
  const skipped = (pruningData && pruningData.skipped) || [];
  const configured = pruningData ? pruningData.configured_fraction : null;

  if (layers.length === 0) {
    // All-skipped or model-missing: show the diagnostic, never a bare empty tab.
    container.innerHTML = skipped.length > 0
      ? `<div class="empty-state">No pruning layer data — every layer was skipped.</div>${skippedBlock(skipped)}`
      : '<div class="empty-state">No pruning data for this step. Pruning masks are available after the pruning adaptation runs.</div>';
    return;
  }

  const first = layers[0];

  let html = `
    <div class="pruning-browse">
      <div class="pruning-layer-list card">
        <div class="card-header">Layers</div>
        <div class="card-body scrollable">
          <ul class="pruning-layer-ul">`;

  for (const L of layers) {
    const d = layerDims(L);
    const name = L.layer_name != null ? esc(String(L.layer_name)) : `Layer ${L.layer_index}`;
    const activeClass = L.layer_index === first.layer_index ? ' active' : '';
    html += `
            <li class="pruning-layer-item${activeClass}" data-idx="${L.layer_index}">
              <span class="pruning-layer-name">${name}</span>
              <span class="pruning-layer-meta">${dimArrow(d.preN, d.postN, 'neurons')}</span>
              <span class="pruning-layer-meta">${dimArrow(d.preA, d.postA, 'axons')}</span>
              <span class="pruning-layer-pruned">${L.pruned_rows ?? 0} rows, ${L.pruned_cols ?? 0} cols pruned</span>
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
          <div class="card-header">Mask map (kept vs pruned)</div>
          <div class="card-body">
            <div class="pruning-detail-heatmap-wrap" id="pruning-detail-maskmap"></div>
          </div>
        </div>
        <div class="pruning-heatmap-panel card">
          <div class="card-header">Weight matrix (pruning masks)</div>
          <div class="card-body">
            <div class="pruning-detail-heatmap-wrap" id="pruning-detail-heatmap"></div>
          </div>
        </div>
      </div>
    </div>
    ${skippedBlock(skipped)}`;

  container.innerHTML = html;

  const metaEl = document.getElementById('pruning-detail-meta');
  const maskMapEl = document.getElementById('pruning-detail-maskmap');
  const heatmapEl = document.getElementById('pruning-detail-heatmap');

  function showLayer(layer) {
    if (!layer) return;
    const d = layerDims(layer);
    const name = layer.layer_name != null ? esc(String(layer.layer_name)) : `Layer ${layer.layer_index}`;
    metaEl.innerHTML = `
      <table class="config-table">
        <tr><td>Layer</td><td>${name} (index ${layer.layer_index})</td></tr>
        <tr><td>Neurons</td><td>${dimArrow(d.preN, d.postN, '')}</td></tr>
        <tr><td>Axons</td><td>${dimArrow(d.preA, d.postA, '')}</td></tr>
        <tr><td>Pruned rows</td><td>${layer.pruned_rows ?? 0}</td></tr>
        <tr><td>Pruned cols</td><td>${layer.pruned_cols ?? 0}</td></tr>
        <tr><td>Achieved sparsity (neurons)</td><td>${fmtPct(layer.achieved_sparsity_neurons)}</td></tr>
        <tr><td>Achieved sparsity (axons)</td><td>${fmtPct(layer.achieved_sparsity_axons)}</td></tr>
        <tr><td>Configured fraction</td><td>${fmtPct(configured)}</td></tr>
      </table>`;
    maskMapEl.innerHTML = imgPanelHtml(
      layer.mask_map_resource, 'mask map', 'Row/col mask map: kept vs pruned',
    );
    heatmapEl.innerHTML = imgPanelHtml(
      layer.heatmap_resource, 'weight heatmap', 'Weight heatmap with pruning masks',
    );
    wireImgFallbacks(maskMapEl);
    wireImgFallbacks(heatmapEl);
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
