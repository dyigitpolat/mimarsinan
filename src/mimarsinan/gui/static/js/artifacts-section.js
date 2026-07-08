/* Artifacts section: the run directory's inventory, grouped and downloadable. */
import { esc } from './util.js';

const GROUP_ORDER = ['step cache', 'run outputs', 'segments', 'monitor state', 'config'];

function fmtBytes(n) {
  if (n == null) return '';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + ' GB';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + ' MB';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + ' KB';
  return n + ' B';
}

function downloadUrl(state, path) {
  if (state.historicalRunId && !state.isActiveRun) {
    return `/api/runs/${encodeURIComponent(state.historicalRunId)}/artifact_file?path=${encodeURIComponent(path)}`;
  }
  if (!state.historicalRunId) {
    return `/api/artifact_file?path=${encodeURIComponent(path)}`;
  }
  return null; // active subprocess runs: listing only
}

export async function renderArtifactsSection(state, apiUrl, fetchJSON) {
  const host = document.getElementById('artifacts-root');
  if (!host) return;
  let entries;
  try {
    entries = await fetchJSON(apiUrl('/artifacts'));
  } catch (e) {
    host.innerHTML = '<div class="empty-state">Failed to load artifacts</div>';
    return;
  }
  if (!Array.isArray(entries) || entries.length === 0) {
    host.innerHTML = '<div class="empty-state">No artifacts written yet</div>';
    return;
  }

  const groups = {};
  for (const entry of entries) {
    const group = entry.group || 'run outputs';
    if (!groups[group]) groups[group] = [];
    groups[group].push(entry);
  }
  const order = [...GROUP_ORDER.filter(g => groups[g]),
    ...Object.keys(groups).filter(g => !GROUP_ORDER.includes(g)).sort()];

  host.innerHTML = order.map(group => {
    const rows = groups[group].map(entry => {
      const url = entry.kind !== 'dir' ? downloadUrl(state, entry.path) : null;
      const name = url
        ? `<a href="${url}" download style="color:var(--text-primary);text-decoration:none">${esc(entry.path)}</a>`
        : esc(entry.path);
      const meta = entry.kind === 'dir'
        ? `${entry.files} files`
        : (entry.step ? esc(entry.step) : '');
      return `<tr class="artifact-row">
        <td>${name}</td>
        <td><span class="artifact-kind">${esc(entry.kind)}</span></td>
        <td class="num">${fmtBytes(entry.size)}</td>
        <td class="note">${meta}</td>
      </tr>`;
    }).join('');
    const total = groups[group].reduce((s, e) => s + (e.size || 0), 0);
    return `<div class="card artifact-group">
      <div class="card-header"><span>${esc(group)}</span>
        <span class="note">${groups[group].length} entries · ${fmtBytes(total)}</span></div>
      <div class="card-body no-pad">
        <table class="data-table compact">
          <thead><tr><th>Path</th><th>Kind</th><th>Size</th><th></th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>`;
  }).join('');
}
