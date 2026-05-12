/* SANA-FE Simulation tab — rich per-tile / per-core / per-segment stats.
 *
 * Consumes ``snap.sanafe_simulation`` (SanafeStepReport.to_snapshot_dict).
 * Lazy heatmap PNGs are fetched via ``resourceUrl({kind, rid})`` for the
 * three resource kinds emitted by ``snapshot_sanafe_simulation``:
 *   - sanafe_tile_energy
 *   - sanafe_core_energy
 *   - sanafe_core_spikes
 */
import { esc, safeReact } from './util.js';
import { imgSrcAttr, getResourceContext } from './resource-urls.js';

function fmtMj(j) { return (j * 1000).toFixed(3); }
function fmtSec(s) { return s.toExponential(3); }
function fmtInt(n) { return Number(n).toLocaleString(); }

function summaryCardsHtml(agg, archPreset) {
  return `
    <div class="grid-3" style="margin-bottom:20px">
      <div class="card"><div class="big-metric">
        <div class="value" style="color:#ff9800">${fmtMj(agg.total_energy_j ?? 0)}</div>
        <div class="label">Total Energy (mJ)</div>
      </div></div>
      <div class="card"><div class="big-metric">
        <div class="value" style="color:#5b8af5">${fmtSec(agg.max_sim_time_s ?? 0)}</div>
        <div class="label">Max Sim Time (s)</div>
      </div></div>
      <div class="card"><div class="big-metric">
        <div class="value" style="color:#4caf50">${fmtInt(agg.total_spikes ?? 0)}</div>
        <div class="label">Total Spikes</div>
      </div></div>
    </div>
    <div class="grid-3" style="margin-bottom:20px">
      <div class="card"><div class="big-metric">
        <div class="value" style="color:#9c27b0">${fmtInt(agg.total_packets ?? 0)}</div>
        <div class="label">NoC Packets</div>
      </div></div>
      <div class="card"><div class="big-metric">
        <div class="value" style="color:#00bcd4">${fmtInt(agg.sample_count ?? 0)}</div>
        <div class="label">Samples Run</div>
      </div></div>
      <div class="card"><div class="big-metric">
        <div class="value" style="color:#888;font-size:22px">${esc(archPreset || '')}</div>
        <div class="label">Arch Preset</div>
      </div></div>
    </div>`;
}

function energyBreakdownHtml(eb) {
  if (!eb) return '';
  return `
    <div class="card" style="margin-bottom:20px">
      <div class="card-header">Energy Decomposition (Joules)</div>
      <div class="card-body"><div id="sanafe-eb" style="min-height:280px"></div></div>
    </div>`;
}

function renderEnergyBreakdown(eb) {
  if (!eb) return;
  const components = ['synapse', 'dendrite', 'soma', 'network'];
  const colors = { synapse: '#5b8af5', dendrite: '#4caf50',
                   soma: '#ff9800', network: '#9c27b0' };
  const traces = components.map(c => ({
    x: [c.charAt(0).toUpperCase() + c.slice(1)],
    y: [eb[c] ?? 0],
    name: c,
    type: 'bar',
    marker: { color: colors[c] },
  }));
  safeReact('sanafe-eb', traces, {
    height: 280, barmode: 'group',
    yaxis: { title: 'Energy (J)', type: 'log' },
    showlegend: false,
  });
}

function renderPerSegmentCharts(perSample) {
  perSample.forEach((sampleEntry, si) => {
    sampleEntry.segments.forEach((seg, segIdx) => {
      // Per-core energy
      const ceId = `sanafe-core-energy-${si}-${segIdx}`;
      const csId = `sanafe-core-spikes-${si}-${segIdx}`;
      const coreLabels = seg.per_core.map(c => `c${c.core_index}`);
      const coreEnergies = seg.per_core.map(c => c.energy_j);
      const coreSpikes = seg.per_core.map(c => c.spikes_fired);
      safeReact(ceId, [{
        x: coreLabels, y: coreEnergies, type: 'bar',
        marker: { color: '#ff9800' },
      }], {
        height: 240, yaxis: { title: 'Energy (J)' },
        margin: { l: 50, r: 20, t: 10, b: 40 },
      });
      safeReact(csId, [{
        x: coreLabels, y: coreSpikes, type: 'bar',
        marker: { color: '#4caf50' },
      }], {
        height: 240, yaxis: { title: 'Spikes' },
        margin: { l: 50, r: 20, t: 10, b: 40 },
      });
    });
  });
}

function segmentHtml(sampleIdx, seg) {
  const ctx = getResourceContext();
  const tileRef = { kind: 'sanafe_tile_energy',
                    rid: `sample${sampleIdx}/seg${seg.stage_index}` };
  const tileSrc = imgSrcAttr(tileRef, ctx);
  return `
    <div class="card" style="margin-bottom:16px">
      <div class="card-header">Segment ${seg.stage_index} — <span style="font-weight:normal">${esc(seg.stage_name)}</span></div>
      <div class="card-body">
        <div class="grid-3" style="gap:12px;margin-bottom:12px">
          <div class="big-metric"><div class="value" style="font-size:18px;color:#ff9800">${fmtMj(seg.energy_j)}</div><div class="label">Energy (mJ)</div></div>
          <div class="big-metric"><div class="value" style="font-size:18px;color:#5b8af5">${fmtSec(seg.sim_time_s)}</div><div class="label">Sim Time (s)</div></div>
          <div class="big-metric"><div class="value" style="font-size:18px;color:#4caf50">${fmtInt(seg.spikes)}</div><div class="label">Spikes</div></div>
        </div>
        <div class="grid-3" style="gap:12px;margin-bottom:12px">
          <div class="big-metric"><div class="value" style="font-size:18px;color:#9c27b0">${fmtInt(seg.packets_sent)}</div><div class="label">NoC Packets</div></div>
          <div class="big-metric"><div class="value" style="font-size:18px;color:#00bcd4">${fmtInt(seg.neurons_fired)}</div><div class="label">Neurons Fired</div></div>
          <div class="big-metric"><div class="value" style="font-size:18px;color:#888">${seg.per_tile.length}</div><div class="label">Tiles</div></div>
        </div>
        <div style="margin-bottom:12px">
          <div style="font-size:12px;color:var(--text-muted);margin-bottom:6px">Per-Tile Energy</div>
          ${tileSrc ? `<img src="${tileSrc}" style="max-width:100%;border-radius:4px" alt="tile energy">` : ''}
        </div>
        <div class="grid-2" style="gap:12px">
          <div>
            <div style="font-size:12px;color:var(--text-muted);margin-bottom:6px">Per-Core Energy</div>
            <div id="sanafe-core-energy-${sampleIdx}-${seg.stage_index}" style="min-height:240px"></div>
          </div>
          <div>
            <div style="font-size:12px;color:var(--text-muted);margin-bottom:6px">Per-Core Spikes</div>
            <div id="sanafe-core-spikes-${sampleIdx}-${seg.stage_index}" style="min-height:240px"></div>
          </div>
        </div>
      </div>
    </div>`;
}

export function renderSanafeTab(snap, container) {
  if (!snap) {
    container.innerHTML =
      '<div class="empty-state">No SANA-FE simulation data available.<br>' +
      '<span style="font-size:12px;color:var(--text-muted)">Enable ' +
      '<code>enable_sanafe_simulation</code> in deployment parameters.</span></div>';
    return;
  }

  const agg = snap.aggregate || {};
  const archPreset = snap.arch_preset || '';
  const perSample = Array.isArray(snap.per_sample) ? snap.per_sample : [];
  const eb = agg.energy_breakdown_j || null;

  let html = summaryCardsHtml(agg, archPreset);
  html += energyBreakdownHtml(eb);

  if (perSample.length === 0) {
    html += '<div class="empty-state">No per-sample data.</div>';
    container.innerHTML = html;
    return;
  }

  // Sample selector (multi-sample runs only).
  if (perSample.length > 1) {
    html += '<div class="card" style="margin-bottom:16px"><div class="card-header">Sample</div><div class="card-body">';
    html += '<select id="sanafe-sample-select">';
    perSample.forEach((s, i) => {
      html += `<option value="${i}">Sample ${esc(String(s.sample_index))}</option>`;
    });
    html += '</select></div></div>';
  }

  // First sample's segments rendered initially.
  const initial = perSample[0];
  html += `<div id="sanafe-segments">`;
  initial.segments.forEach(seg => { html += segmentHtml(0, seg); });
  html += `</div>`;

  container.innerHTML = html;
  renderEnergyBreakdown(eb);
  // Only render charts for the active sample to keep DOM cost in check.
  renderPerSegmentCharts([perSample[0]]);

  const select = document.getElementById('sanafe-sample-select');
  if (select) {
    select.onchange = () => {
      const idx = Number(select.value) || 0;
      const segContainer = document.getElementById('sanafe-segments');
      const entry = perSample[idx];
      if (!segContainer || !entry) return;
      segContainer.innerHTML = entry.segments.map(seg => segmentHtml(idx, seg)).join('');
      renderPerSegmentCharts([entry]);
    };
  }
}
