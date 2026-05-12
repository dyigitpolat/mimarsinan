/* SANA-FE Simulation tab — chip-floorplan + per-core + NoC visualization.
 *
 * Consumes ``snap.sanafe_simulation`` (SanafeStepReport.to_snapshot_dict).
 * Every chart is a Plotly trace rendered inline from the JSON snapshot;
 * no lazy PNG resources are used by this tab (the old strip heatmaps
 * rendered as invisible 1-pixel images and have been retired).
 *
 * Per segment the tab shows:
 *   - Summary cards: energy / sim_time / spikes / packets / neurons fired.
 *   - **Chip Floorplan** — every core drawn as a cell on the actual NoC
 *     mesh (one column per tile, cores stacked vertically inside each
 *     tile), coloured by a user-selected metric.  Tile boundaries are
 *     drawn as red guides.  Hover gives full per-core stats.
 *   - **NoC Traffic** — directed edges between tiles weighted by packet /
 *     spike / hop count from ``message_trace``.  Hidden when the trace
 *     wasn't recorded (``log_message_trace=False``) or all traffic is
 *     intra-tile.
 *   - Per-core bar charts: energy + spike count side-by-side.
 *   - Energy decomposition: stacked synapse / dendrite / soma / network.
 *
 * Metric selector for the floorplan keys:
 *   energy_j         — total per-core energy in Joules (log10 cmap)
 *   spikes_fired     — output spike count
 *   core_latency     — depth in the cascade (categorical)
 *   has_hardware_bias — 0/1 flag
 *   n_always_on_axons — always-on axon count
 *   n_neurons        — live neuron count (geometry sanity check)
 *   n_axons_used     — live axon count (geometry sanity check)
 */
import { esc, safeReact } from './util.js';

function fmtMj(j) { return (j * 1000).toFixed(3); }
function fmtSec(s) { return s.toExponential(3); }
function fmtInt(n) { return Number(n).toLocaleString(); }

const METRIC_OPTIONS = [
  { key: 'energy_j',         label: 'Energy (J)',           log: true,  cmap: 'YlOrRd' },
  { key: 'spikes_fired',     label: 'Spikes Fired',         log: false, cmap: 'Viridis' },
  { key: 'core_latency',     label: 'Latency Layer',        log: false, cmap: 'Portland' },
  { key: 'has_hardware_bias',label: 'Has Hardware Bias',    log: false, cmap: 'Greys' },
  { key: 'n_always_on_axons',label: 'Always-On Axon Count', log: false, cmap: 'Cividis' },
  { key: 'n_neurons',        label: 'Live Neurons',         log: false, cmap: 'Plasma' },
  { key: 'n_axons_used',     label: 'Live Axons',           log: false, cmap: 'Plasma' },
];

const NOC_METRIC_OPTIONS = [
  { key: 'packet_count', label: 'Packet Count' },
  { key: 'spike_count',  label: 'Total Spikes' },
  { key: 'total_hops',   label: 'Total Hops'   },
];

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

function energyBreakdownHtml() {
  return `
    <div class="card" style="margin-bottom:20px">
      <div class="card-header">Energy Decomposition (Joules, log scale)</div>
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
    y: [Math.max(eb[c] ?? 0, 1e-18)],
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

/* --------------------------------------------------------------------------
 * Floorplan geometry helpers
 * --------------------------------------------------------------------------
 *
 * The arch synth currently lays tiles as ``width × 1`` (one row of N tiles)
 * but the snapshot carries explicit ``tiles_xy`` so we are robust to a
 * future 2D mesh.  Within each tile, cores are stacked vertically: we pick
 * a per-tile width = ``ceil(sqrt(max_cores_in_any_tile))`` so the cell grid
 * stays roughly square even for big tiles.
 * ------------------------------------------------------------------------ */

function tileGrid(perTile) {
  const maxCores = perTile.reduce((m, t) => Math.max(m, t.cores.length), 1);
  const w = Math.max(1, Math.ceil(Math.sqrt(maxCores)));
  const h = Math.max(1, Math.ceil(maxCores / w));
  return { tileW: w, tileH: h };
}

/* Build a (rows × cols) z-matrix where each cell is one core in its
 * physical position.  Cells outside any tile (the dead zone between
 * tiles in the arch grid, or empty slots in the last tile) are NaN so
 * Plotly renders them as transparent.  Returns the matrix plus a
 * parallel ``hover`` matrix for tooltips. */
function buildFloorplanGrid(seg, metric) {
  const perTile = Array.isArray(seg.per_tile) ? seg.per_tile : [];
  const perCore = Array.isArray(seg.per_core) ? seg.per_core : [];
  const geom = seg.arch_geometry || { width: 1, height: 1, tiles_xy: [[0, 0]] };
  const { tileW, tileH } = tileGrid(perTile);
  const coreById = new Map(perCore.map(c => [c.core_index, c]));

  // Total floorplan grid size (in cells)
  const cellRows = geom.height * tileH;
  const cellCols = geom.width  * tileW;

  // Initialise with NaN so empty cells render transparent
  const z = Array.from({ length: cellRows }, () => new Array(cellCols).fill(null));
  const hover = Array.from({ length: cellRows }, () => new Array(cellCols).fill(''));

  perTile.forEach((tile) => {
    const tx = tile.mesh_x >= 0 ? tile.mesh_x : 0;
    const ty = tile.mesh_y >= 0 ? tile.mesh_y : 0;
    tile.cores.forEach((coreIdx, pos) => {
      const innerX = pos % tileW;
      const innerY = Math.floor(pos / tileW);
      const col = tx * tileW + innerX;
      const row = ty * tileH + innerY;
      if (row < 0 || row >= cellRows || col < 0 || col >= cellCols) return;
      const c = coreById.get(coreIdx);
      if (!c) return;
      const v = c[metric] ?? 0;
      z[row][col] = (metric === 'energy_j' && v > 0) ? Math.log10(v) : v;
      hover[row][col] =
        `core ${c.core_index} (tile ${tile.tile_index})` +
        `<br>energy: ${fmtMj(c.energy_j ?? 0)} mJ` +
        `<br>spikes: ${fmtInt(c.spikes_fired ?? 0)}` +
        `<br>latency layer: ${c.core_latency ?? 0}` +
        `<br>neurons: ${fmtInt(c.n_neurons ?? 0)}` +
        `<br>axons used: ${fmtInt(c.n_axons_used ?? 0)}` +
        `<br>hw bias: ${c.has_hardware_bias ? 'yes' : 'no'}` +
        `<br>always-on axons: ${fmtInt(c.n_always_on_axons ?? 0)}`;
    });
  });

  // Tile boundary shapes — one rectangle per tile
  const shapes = perTile.map((tile) => {
    const tx = tile.mesh_x >= 0 ? tile.mesh_x : 0;
    const ty = tile.mesh_y >= 0 ? tile.mesh_y : 0;
    return {
      type: 'rect',
      x0: tx * tileW - 0.5,
      x1: (tx + 1) * tileW - 0.5,
      y0: ty * tileH - 0.5,
      y1: (ty + 1) * tileH - 0.5,
      xref: 'x', yref: 'y',
      line: { color: '#e53935', width: 1.5 },
      fillcolor: 'rgba(0,0,0,0)',
      layer: 'above',
    };
  });

  return { z, hover, shapes, geom, tileW, tileH, cellRows, cellCols };
}

function renderFloorplan(elId, seg, metricKey, nocOverlay, nocMetricKey) {
  const metric = METRIC_OPTIONS.find(m => m.key === metricKey) || METRIC_OPTIONS[0];
  const grid = buildFloorplanGrid(seg, metric.key);

  const heat = {
    z: grid.z,
    type: 'heatmap',
    colorscale: metric.cmap,
    showscale: true,
    hoverinfo: 'text',
    text: grid.hover,
    xgap: 1, ygap: 1,
    colorbar: {
      title: metric.log ? `${metric.label} log10` : metric.label,
      titleside: 'right',
    },
  };

  const traces = [heat];

  // NoC overlay — directed line segments tile centre → tile centre.
  if (nocOverlay && Array.isArray(seg.noc_links) && seg.noc_links.length > 0) {
    const links = seg.noc_links;
    const maxVal = Math.max(...links.map(L => L[nocMetricKey] || 0), 1);
    links.forEach((L) => {
      const sx = (L.src_x + 0.5) * grid.tileW - 0.5;
      const sy = (L.src_y + 0.5) * grid.tileH - 0.5;
      const dx = (L.dst_x + 0.5) * grid.tileW - 0.5;
      const dy = (L.dst_y + 0.5) * grid.tileH - 0.5;
      const w = 1 + 5 * Math.sqrt((L[nocMetricKey] || 0) / maxVal);
      traces.push({
        x: [sx, dx], y: [sy, dy],
        mode: 'lines+markers',
        line: { color: '#5b8af5', width: w },
        marker: {
          size: [6, 12],
          color: '#5b8af5',
          symbol: ['circle', 'triangle-up'],
        },
        hoverinfo: 'text',
        text: [
          `tile ${L.src_tile} → ${L.dst_tile}` +
          `<br>${nocMetricKey}: ${fmtInt(L[nocMetricKey])}`,
          `tile ${L.src_tile} → ${L.dst_tile}` +
          `<br>${nocMetricKey}: ${fmtInt(L[nocMetricKey])}`,
        ],
        showlegend: false,
      });
    });
  }

  safeReact(elId, traces, {
    height: Math.max(280, grid.cellRows * 12 + 80),
    shapes: grid.shapes,
    xaxis: {
      title: 'Mesh X (tile · core)',
      range: [-0.5, grid.cellCols - 0.5],
      showgrid: false, zeroline: false,
    },
    yaxis: {
      title: 'Mesh Y',
      range: [-0.5, grid.cellRows - 0.5],
      autorange: 'reversed',
      showgrid: false, zeroline: false,
      scaleanchor: 'x', scaleratio: 1,
    },
    margin: { l: 60, r: 40, t: 10, b: 50 },
  });
}

function renderPerCoreBars(sampleIdx, seg) {
  const ceId = `sanafe-core-energy-${sampleIdx}-${seg.stage_index}`;
  const csId = `sanafe-core-spikes-${sampleIdx}-${seg.stage_index}`;
  const coreLabels = seg.per_core.map(c => `c${c.core_index}`);
  const coreEnergies = seg.per_core.map(c => c.energy_j);
  const coreSpikes = seg.per_core.map(c => c.spikes_fired);
  safeReact(ceId, [{
    x: coreLabels, y: coreEnergies, type: 'bar',
    marker: { color: '#ff9800' },
    hovertemplate: '%{x}<br>%{y:.3e} J<extra></extra>',
  }], {
    height: 240, yaxis: { title: 'Energy (J)' },
    margin: { l: 60, r: 20, t: 10, b: 40 },
  });
  safeReact(csId, [{
    x: coreLabels, y: coreSpikes, type: 'bar',
    marker: { color: '#4caf50' },
    hovertemplate: '%{x}<br>%{y} spikes<extra></extra>',
  }], {
    height: 240, yaxis: { title: 'Spikes' },
    margin: { l: 60, r: 20, t: 10, b: 40 },
  });
}

function metricSelectorHtml(id, options, current) {
  return `<select id="${id}" style="margin-left:8px">` +
    options.map(o =>
      `<option value="${o.key}"${o.key === current ? ' selected' : ''}>${esc(o.label)}</option>`,
    ).join('') +
    `</select>`;
}

function segmentHtml(sampleIdx, seg) {
  const floorId = `sanafe-floor-${sampleIdx}-${seg.stage_index}`;
  const metricId = `sanafe-metric-${sampleIdx}-${seg.stage_index}`;
  const nocCheckId = `sanafe-noc-${sampleIdx}-${seg.stage_index}`;
  const nocMetricId = `sanafe-nocm-${sampleIdx}-${seg.stage_index}`;
  const hasNoc = Array.isArray(seg.noc_links) && seg.noc_links.length > 0;
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

        <div style="margin-bottom:6px;display:flex;align-items:center;gap:14px;flex-wrap:wrap">
          <span style="font-weight:600">Chip Floorplan — colour by:</span>
          ${metricSelectorHtml(metricId, METRIC_OPTIONS, 'energy_j')}
          ${hasNoc ? `
            <label style="display:flex;align-items:center;gap:4px;font-size:13px">
              <input type="checkbox" id="${nocCheckId}" checked> NoC traffic
            </label>
            <span style="font-size:13px;color:var(--text-muted)">edge width by:</span>
            ${metricSelectorHtml(nocMetricId, NOC_METRIC_OPTIONS, 'packet_count')}
          ` : '<span style="font-size:12px;color:var(--text-muted)">(NoC overlay disabled — no message_trace recorded or only intra-tile traffic)</span>'}
        </div>
        <div id="${floorId}" style="min-height:300px;margin-bottom:14px"></div>

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

function wireSegmentControls(sampleIdx, seg) {
  const floorId = `sanafe-floor-${sampleIdx}-${seg.stage_index}`;
  const metricId = `sanafe-metric-${sampleIdx}-${seg.stage_index}`;
  const nocCheckId = `sanafe-noc-${sampleIdx}-${seg.stage_index}`;
  const nocMetricId = `sanafe-nocm-${sampleIdx}-${seg.stage_index}`;
  const metricSel = document.getElementById(metricId);
  const nocCheck = document.getElementById(nocCheckId);
  const nocMetricSel = document.getElementById(nocMetricId);

  const draw = () => {
    const m = metricSel ? metricSel.value : 'energy_j';
    const overlay = nocCheck ? nocCheck.checked : false;
    const nm = nocMetricSel ? nocMetricSel.value : 'packet_count';
    renderFloorplan(floorId, seg, m, overlay, nm);
  };
  if (metricSel) metricSel.onchange = draw;
  if (nocCheck) nocCheck.onchange = draw;
  if (nocMetricSel) nocMetricSel.onchange = draw;
  draw();
  renderPerCoreBars(sampleIdx, seg);
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
  html += energyBreakdownHtml();

  if (perSample.length === 0) {
    html += '<div class="empty-state">No per-sample data.</div>';
    container.innerHTML = html;
    return;
  }

  if (perSample.length > 1) {
    html += '<div class="card" style="margin-bottom:16px"><div class="card-header">Sample</div><div class="card-body">';
    html += '<select id="sanafe-sample-select">';
    perSample.forEach((s, i) => {
      html += `<option value="${i}">Sample ${esc(String(s.sample_index))}</option>`;
    });
    html += '</select></div></div>';
  }

  const initial = perSample[0];
  html += `<div id="sanafe-segments">`;
  initial.segments.forEach(seg => { html += segmentHtml(0, seg); });
  html += `</div>`;

  container.innerHTML = html;
  renderEnergyBreakdown(eb);
  initial.segments.forEach(seg => wireSegmentControls(0, seg));

  const select = document.getElementById('sanafe-sample-select');
  if (select) {
    select.onchange = () => {
      const idx = Number(select.value) || 0;
      const segContainer = document.getElementById('sanafe-segments');
      const entry = perSample[idx];
      if (!segContainer || !entry) return;
      segContainer.innerHTML = entry.segments.map(seg => segmentHtml(idx, seg)).join('');
      entry.segments.forEach(seg => wireSegmentControls(idx, seg));
    };
  }
}
