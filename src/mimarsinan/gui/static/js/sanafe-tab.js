/* SANA-FE Simulation tab — chip-floorplan + per-cycle + per-core visuals.
 *
 * Renders everything inline from ``snap.sanafe_simulation`` — no lazy PNG
 * resources are used.  Each segment now exposes:
 *   1. **Chip Floorplan** with metric selector + NoC traffic overlay +
 *      NoC congestion (per-mesh-edge XY-routed) + live connectivity +
 *      critical-core border highlight + HCM-diff overlay layer.
 *   2. **Animated NoC playback** — play/pause/slider stepping through
 *      ``noc_traffic_per_cycle``.
 *   3. **Latency-cascade timeline** — stacked bar per cycle per
 *      ``core.latency`` depth.
 *   4. **Energy waterfall** — stacked area cumulative energy by
 *      category over cycles.
 *   5. **Efficiency scatter** — energy_j vs spikes_fired per core.
 *   6. **Per-core spike raster mini-view** on click of any core.
 *   7. Per-core energy + spike bars.
 *   8. Energy decomposition bar.
 */
import { esc, safeReact } from './util.js';

function fmtMj(j) { return (j * 1000).toFixed(3); }
function fmtSec(s) { return s.toExponential(3); }
function fmtInt(n) { return Number(n).toLocaleString(); }

const METRIC_OPTIONS = [
  { key: 'energy_j',         label: 'Energy (J, log10)',   log: true,  cmap: 'YlOrRd' },
  { key: 'spikes_fired',     label: 'Spikes Fired',         log: false, cmap: 'Viridis' },
  { key: 'core_latency',     label: 'Latency Layer',        log: false, cmap: 'Portland' },
  { key: 'has_hardware_bias',label: 'Has Hardware Bias',    log: false, cmap: 'Greys' },
  { key: 'n_always_on_axons',label: 'Always-On Axon Count', log: false, cmap: 'Cividis' },
  { key: 'n_neurons',        label: 'Live Neurons',         log: false, cmap: 'Plasma' },
  { key: 'n_axons_used',     label: 'Live Axons',           log: false, cmap: 'Plasma' },
  { key: 'hcm_diff_input',   label: 'HCM Δ input (SF−HCM)', log: false, cmap: 'RdBu',  diverging: true },
  { key: 'hcm_diff_output',  label: 'HCM Δ output (SF−HCM)',log: false, cmap: 'RdBu',  diverging: true },
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

/* -------------------------------------------------------------------------- */
/* Floorplan geometry helpers                                                  */
/* -------------------------------------------------------------------------- */

function tileGrid(perTile) {
  const maxCores = perTile.reduce((m, t) => Math.max(m, t.cores.length), 1);
  const w = Math.max(1, Math.ceil(Math.sqrt(maxCores)));
  const h = Math.max(1, Math.ceil(maxCores / w));
  return { tileW: w, tileH: h };
}

/* For a given core_index, return the (col, row) cell on the floorplan
 * grid.  Walks per_tile.cores to find which tile the core lives in,
 * then maps the within-tile position onto the per-tile sub-grid. */
function corePosition(coreIdx, perTile, tileW, tileH) {
  for (const tile of perTile) {
    const idx = tile.cores.indexOf(coreIdx);
    if (idx < 0) continue;
    const tx = tile.mesh_x >= 0 ? tile.mesh_x : 0;
    const ty = tile.mesh_y >= 0 ? tile.mesh_y : 0;
    const innerX = idx % tileW;
    const innerY = Math.floor(idx / tileW);
    return [tx * tileW + innerX, ty * tileH + innerY];
  }
  return null;
}

function buildFloorplanGrid(seg, metric) {
  const perTile = Array.isArray(seg.per_tile) ? seg.per_tile : [];
  const perCore = Array.isArray(seg.per_core) ? seg.per_core : [];
  const geom = seg.arch_geometry || { width: 1, height: 1, tiles_xy: [[0, 0]] };
  const { tileW, tileH } = tileGrid(perTile);
  const coreById = new Map(perCore.map(c => [c.core_index, c]));

  const cellRows = geom.height * tileH;
  const cellCols = geom.width  * tileW;
  const z = Array.from({ length: cellRows }, () => new Array(cellCols).fill(null));
  const hover = Array.from({ length: cellRows }, () => new Array(cellCols).fill(''));

  // HCM-diff lookup, keyed by core_index when present.
  const diffByCore = new Map(
    (Array.isArray(seg.hcm_diff) ? seg.hcm_diff : []).map(d => [d.core_index, d]),
  );

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
      let v;
      if (metric.key === 'hcm_diff_input') {
        v = diffByCore.get(coreIdx)?.input_delta_sum ?? 0;
      } else if (metric.key === 'hcm_diff_output') {
        v = diffByCore.get(coreIdx)?.output_delta_sum ?? 0;
      } else {
        v = c[metric.key] ?? 0;
        if (metric.log && v > 0) v = Math.log10(v);
        else if (metric.log) v = null;  // 0 in log space = transparent
      }
      z[row][col] = v;
      const d = diffByCore.get(coreIdx);
      hover[row][col] =
        `core ${c.core_index} (tile ${tile.tile_index})` +
        `<br>energy: ${fmtMj(c.energy_j ?? 0)} mJ` +
        `<br>spikes: ${fmtInt(c.spikes_fired ?? 0)}` +
        `<br>latency layer: ${c.core_latency ?? 0}` +
        `<br>neurons: ${fmtInt(c.n_neurons ?? 0)}` +
        `<br>axons used: ${fmtInt(c.n_axons_used ?? 0)}` +
        `<br>hw bias: ${c.has_hardware_bias ? 'yes' : 'no'}` +
        `<br>always-on axons: ${fmtInt(c.n_always_on_axons ?? 0)}` +
        (d ? `<br>HCM Δ in/out: ${d.input_delta_sum}/${d.output_delta_sum}` : '');
    });
  });

  const shapes = perTile.map((tile) => {
    const tx = tile.mesh_x >= 0 ? tile.mesh_x : 0;
    const ty = tile.mesh_y >= 0 ? tile.mesh_y : 0;
    return {
      type: 'rect',
      x0: tx * tileW - 0.5, x1: (tx + 1) * tileW - 0.5,
      y0: ty * tileH - 0.5, y1: (ty + 1) * tileH - 0.5,
      xref: 'x', yref: 'y',
      line: { color: '#e53935', width: 1.5 },
      fillcolor: 'rgba(0,0,0,0)',
      layer: 'above',
    };
  });

  return { z, hover, shapes, geom, tileW, tileH, cellRows, cellCols };
}

/* -------------------------------------------------------------------------- */
/* Floorplan render — handles all the overlays                                 */
/* -------------------------------------------------------------------------- */

function renderFloorplan(elId, seg, opts) {
  const metric = METRIC_OPTIONS.find(m => m.key === opts.metricKey) || METRIC_OPTIONS[0];
  const grid = buildFloorplanGrid(seg, metric);

  const heat = {
    z: grid.z,
    type: 'heatmap',
    colorscale: metric.cmap,
    showscale: true,
    hoverinfo: 'text',
    text: grid.hover,
    xgap: 1, ygap: 1,
    colorbar: { title: metric.label, titleside: 'right' },
  };
  if (metric.diverging) {
    // Symmetric colour range around zero so deltas read as red/blue.
    const flat = grid.z.flat().filter(v => v !== null && Number.isFinite(v));
    const absMax = flat.length ? Math.max(...flat.map(v => Math.abs(v)), 1) : 1;
    heat.zmin = -absMax; heat.zmax = absMax; heat.zmid = 0;
  }
  const traces = [heat];

  const perTile = Array.isArray(seg.per_tile) ? seg.per_tile : [];

  /* NoC traffic overlay — directed lines src tile → dst tile. */
  if (opts.nocOverlay && Array.isArray(seg.noc_links) && seg.noc_links.length > 0) {
    const maxVal = Math.max(...seg.noc_links.map(L => L[opts.nocMetricKey] || 0), 1);
    seg.noc_links.forEach((L) => {
      const sx = (L.src_x + 0.5) * grid.tileW - 0.5;
      const sy = (L.src_y + 0.5) * grid.tileH - 0.5;
      const dx = (L.dst_x + 0.5) * grid.tileW - 0.5;
      const dy = (L.dst_y + 0.5) * grid.tileH - 0.5;
      const w = 1 + 5 * Math.sqrt((L[opts.nocMetricKey] || 0) / maxVal);
      traces.push({
        x: [sx, dx], y: [sy, dy],
        mode: 'lines+markers',
        line: { color: '#5b8af5', width: w },
        marker: { size: [6, 12], color: '#5b8af5', symbol: ['circle', 'triangle-up'] },
        hoverinfo: 'text',
        text: [`tile ${L.src_tile} → ${L.dst_tile}<br>${opts.nocMetricKey}: ${fmtInt(L[opts.nocMetricKey])}`,
               `tile ${L.src_tile} → ${L.dst_tile}<br>${opts.nocMetricKey}: ${fmtInt(L[opts.nocMetricKey])}`],
        showlegend: false,
      });
    });
  }

  /* NoC per-mesh-edge congestion — colour each (tile→adjacent-tile) edge. */
  if (opts.nocCongestion && Array.isArray(seg.noc_link_load) && seg.noc_link_load.length > 0) {
    const maxLoad = Math.max(...seg.noc_link_load.map(L => L.packet_count), 1);
    seg.noc_link_load.forEach((L) => {
      const fx = (L.from_x + 0.5) * grid.tileW - 0.5;
      const fy = (L.from_y + 0.5) * grid.tileH - 0.5;
      const tx = (L.to_x   + 0.5) * grid.tileW - 0.5;
      const ty = (L.to_y   + 0.5) * grid.tileH - 0.5;
      const t = L.packet_count / maxLoad;
      const r = Math.round(255 * t), g = Math.round(140 * (1 - t)), b = 30;
      traces.push({
        x: [fx, tx], y: [fy, ty],
        mode: 'lines',
        line: { color: `rgb(${r},${g},${b})`, width: 2 + 5 * t },
        hoverinfo: 'text',
        text: [`mesh edge (${L.from_x},${L.from_y})→(${L.to_x},${L.to_y})<br>${fmtInt(L.packet_count)} pkts`,
               `mesh edge (${L.from_x},${L.from_y})→(${L.to_x},${L.to_y})<br>${fmtInt(L.packet_count)} pkts`],
        showlegend: false,
        opacity: 0.65,
      });
    });
  }

  /* Live-connectivity overlay — edges between cores with non-zero weights. */
  if (opts.connectivity && Array.isArray(seg.connectivity) && seg.connectivity.length > 0) {
    const maxW = Math.max(...seg.connectivity.map(e => e.weight_sum_abs), 1);
    seg.connectivity.forEach((e) => {
      const src = corePosition(e.src_core, perTile, grid.tileW, grid.tileH);
      const dst = corePosition(e.dst_core, perTile, grid.tileW, grid.tileH);
      if (!src || !dst) return;
      const t = e.weight_sum_abs / maxW;
      traces.push({
        x: [src[0], dst[0]], y: [src[1], dst[1]],
        mode: 'lines',
        line: { color: `rgba(76,175,80,${0.15 + 0.6 * t})`, width: 0.5 + 2 * t },
        hoverinfo: 'text',
        text: [`core ${e.src_core} → ${e.dst_core}<br>|w| sum: ${e.weight_sum_abs.toExponential(2)}`,
               `core ${e.src_core} → ${e.dst_core}<br>|w| sum: ${e.weight_sum_abs.toExponential(2)}`],
        showlegend: false,
      });
    });
  }

  /* Critical-core border highlight — black-outlined markers on cores
   * that ever appeared as the cycle's critical core. */
  if (opts.criticalCores && Array.isArray(seg.critical_cores) && seg.critical_cores.length > 0) {
    const freq = new Map();
    seg.critical_cores.forEach(p => freq.set(p.core_index, (freq.get(p.core_index) || 0) + 1));
    const maxFreq = Math.max(...freq.values(), 1);
    const xs = [], ys = [], texts = [], sizes = [];
    for (const [coreIdx, n] of freq.entries()) {
      const pos = corePosition(coreIdx, perTile, grid.tileW, grid.tileH);
      if (!pos) continue;
      xs.push(pos[0]); ys.push(pos[1]);
      sizes.push(6 + 12 * (n / maxFreq));
      texts.push(`critical on ${n} cycle(s)`);
    }
    if (xs.length) {
      traces.push({
        x: xs, y: ys, mode: 'markers',
        marker: { size: sizes, symbol: 'square-open', line: { color: '#000', width: 2 } },
        hoverinfo: 'text', text: texts, showlegend: false,
      });
    }
  }

  safeReact(elId, traces, {
    height: Math.max(280, grid.cellRows * 14 + 80),
    shapes: grid.shapes,
    xaxis: { title: 'Mesh X (tile · core)',
             range: [-0.5, grid.cellCols - 0.5],
             showgrid: false, zeroline: false },
    yaxis: { title: 'Mesh Y',
             range: [-0.5, grid.cellRows - 0.5],
             autorange: 'reversed',
             showgrid: false, zeroline: false,
             scaleanchor: 'x', scaleratio: 1 },
    margin: { l: 60, r: 40, t: 10, b: 50 },
  });
}

/* -------------------------------------------------------------------------- */
/* Per-cycle visualisations                                                    */
/* -------------------------------------------------------------------------- */

function renderCascadeTimeline(elId, seg) {
  const cascade = Array.isArray(seg.cascade) ? seg.cascade : [];
  if (cascade.length === 0) { return; }
  const depths = [...new Set(cascade.map(p => p.depth))].sort((a, b) => a - b);
  const cycles = [...new Set(cascade.map(p => p.cycle))].sort((a, b) => a - b);
  const traces = depths.map((d) => {
    const yByCycle = new Map(cascade.filter(p => p.depth === d).map(p => [p.cycle, p.firings]));
    return {
      x: cycles, y: cycles.map(c => yByCycle.get(c) || 0),
      name: `depth ${d}`,
      type: 'bar',
    };
  });
  safeReact(elId, traces, {
    height: 240, barmode: 'stack',
    xaxis: { title: 'Cycle' }, yaxis: { title: 'Firings' },
    margin: { l: 50, r: 20, t: 10, b: 40 },
    legend: { orientation: 'h' },
  });
}

function renderEnergyWaterfall(elId, seg) {
  const series = Array.isArray(seg.cycle_energy) ? seg.cycle_energy : [];
  if (series.length === 0) { return; }
  const cycles = series.map(p => p.cycle);
  const cum = { synapse: [], dendrite: [], soma: [], network: [] };
  const running = { synapse: 0, dendrite: 0, soma: 0, network: 0 };
  series.forEach(p => {
    running.synapse += p.synapse_j;
    running.dendrite += p.dendrite_j;
    running.soma += p.soma_j;
    running.network += p.network_j;
    cum.synapse.push(running.synapse);
    cum.dendrite.push(running.dendrite);
    cum.soma.push(running.soma);
    cum.network.push(running.network);
  });
  const colors = { synapse: '#5b8af5', dendrite: '#4caf50',
                   soma: '#ff9800', network: '#9c27b0' };
  const traces = ['synapse', 'dendrite', 'soma', 'network'].map(k => ({
    x: cycles, y: cum[k], name: k.charAt(0).toUpperCase() + k.slice(1),
    type: 'scatter', mode: 'lines', stackgroup: 'one',
    line: { width: 0.5, color: colors[k] }, fillcolor: colors[k],
  }));
  safeReact(elId, traces, {
    height: 260,
    xaxis: { title: 'Cycle' }, yaxis: { title: 'Cumulative Energy (J)' },
    margin: { l: 60, r: 20, t: 10, b: 40 },
    legend: { orientation: 'h' },
  });
}

function renderEfficiencyScatter(elId, seg) {
  const cores = Array.isArray(seg.per_core) ? seg.per_core : [];
  if (cores.length === 0) return;
  const xs = cores.map(c => c.spikes_fired);
  const ys = cores.map(c => c.energy_j);
  const text = cores.map(c =>
    `core ${c.core_index}<br>energy: ${fmtMj(c.energy_j)} mJ<br>spikes: ${fmtInt(c.spikes_fired)}<br>latency: ${c.core_latency}`,
  );
  safeReact(elId, [{
    x: xs, y: ys, mode: 'markers', type: 'scatter',
    marker: {
      size: 8,
      color: cores.map(c => c.core_latency),
      colorscale: 'Portland', showscale: true,
      colorbar: { title: 'Latency Layer', titleside: 'right' },
    },
    text, hoverinfo: 'text',
  }], {
    height: 280,
    xaxis: { title: 'Spikes Fired' },
    yaxis: { title: 'Energy (J)', type: 'log' },
    margin: { l: 70, r: 60, t: 10, b: 40 },
  });
}

/* Animated NoC playback — Plotly's ``animate`` matches traces by INDEX,
 * so every frame must have the same number of traces.  We use exactly
 * two: (0) the static chip-floorplan underlay coloured by per-core
 * energy so users keep spatial context while the animation runs, and
 * (1) a single scatter trace whose ``x``/``y`` arrays carry every
 * packet for the active cycle, joined by ``null`` to break segments.
 * This is what the earlier per-packet-trace version got wrong — it
 * pushed a varying number of traces per frame and Plotly silently
 * left them stale after frame 0, leaving the chart blank. */
function renderNocAnimation(elId, seg) {
  const traffic = Array.isArray(seg.noc_traffic_per_cycle) ? seg.noc_traffic_per_cycle : [];
  if (traffic.length === 0) return;
  const perTile = Array.isArray(seg.per_tile) ? seg.per_tile : [];
  const { tileW, tileH } = tileGrid(perTile);
  const geom = seg.arch_geometry || { width: 1, height: 1 };

  // Underlay: same energy-coloured floorplan grid the static view uses.
  // ``buildFloorplanGrid`` does the per-core → cell mapping for us.
  const fakeMetric = METRIC_OPTIONS.find(m => m.key === 'energy_j');
  const grid = buildFloorplanGrid(seg, fakeMetric);
  const underlay = {
    z: grid.z,
    type: 'heatmap',
    colorscale: fakeMetric.cmap,
    showscale: false,
    hoverinfo: 'text',
    text: grid.hover,
    xgap: 1, ygap: 1,
    opacity: 0.55,
  };

  function frameForCycle(cycleIdx) {
    const cycle = traffic[cycleIdx] || [];
    const xs = [], ys = [], hover = [];
    cycle.forEach(([sx, sy, dx, dy, n]) => {
      const fx = (sx + 0.5) * tileW - 0.5;
      const fy = (sy + 0.5) * tileH - 0.5;
      const tx = (dx + 0.5) * tileW - 0.5;
      const ty = (dy + 0.5) * tileH - 0.5;
      const label = `tile (${sx},${sy}) → (${dx},${dy}) · ${n} pkt`;
      xs.push(fx, tx, null);
      ys.push(fy, ty, null);
      hover.push(label, label, '');
    });
    return {
      data: [
        underlay,
        {
          x: xs, y: ys,
          mode: 'lines+markers',
          line: { color: 'rgba(91,138,245,0.85)', width: 2 },
          marker: { size: 5, color: 'rgba(91,138,245,1.0)' },
          hoverinfo: 'text', text: hover,
          showlegend: false,
        },
      ],
      name: String(cycleIdx),
    };
  }
  const frames = traffic.map((_, i) => frameForCycle(i));

  const el = document.getElementById(elId);
  if (!el || !window.Plotly) return;
  // Force the playback panel to redraw cleanly from scratch — calling
  // newPlot on a previously-animated div leaks frame state from the
  // earlier mount and re-firing the slider event then "starts" from
  // somebody else's cycle.
  try { Plotly.purge(el); } catch (_) { /* ignore */ }
  Plotly.newPlot(el, frames[0].data, {
    height: Math.max(280, grid.cellRows * 14 + 100),
    shapes: grid.shapes,
    xaxis: { title: 'Mesh X', range: [-0.5, grid.cellCols - 0.5],
             showgrid: false, zeroline: false },
    yaxis: { title: 'Mesh Y', range: [-0.5, grid.cellRows - 0.5],
             autorange: 'reversed', showgrid: false, zeroline: false,
             scaleanchor: 'x', scaleratio: 1 },
    margin: { l: 60, r: 40, t: 10, b: 70 },
    updatemenus: [{
      type: 'buttons', x: 0, y: -0.15, direction: 'left',
      buttons: [
        { method: 'animate', label: '▶ Play', args: [null, {
          frame: { duration: 500, redraw: true }, transition: { duration: 0 }, fromcurrent: true, mode: 'immediate',
        }] },
        { method: 'animate', label: '⏸ Pause', args: [[null], {
          frame: { duration: 0 }, mode: 'immediate', transition: { duration: 0 },
        }] },
      ],
    }],
    sliders: [{
      pad: { t: 30, b: 10 }, currentvalue: { prefix: 'cycle: ' },
      steps: frames.map(f => ({ method: 'animate', label: f.name,
                                args: [[f.name], { mode: 'immediate', transition: { duration: 0 },
                                                   frame: { duration: 0, redraw: true } }] })),
    }],
  });
  Plotly.addFrames(el, frames);
}

/* Per-neuron raster mini-view — fires on click of a floorplan cell. */
function attachRasterClickHandler(floorElId, rasterPanelId, seg) {
  const el = document.getElementById(floorElId);
  if (!el || !el.on) return;
  el.on('plotly_click', (data) => {
    if (!data || !data.points || !data.points.length) return;
    const pt = data.points[0];
    // Hover text format: ``core <N> (tile ...)`` — recover core_index.
    const m = pt.text && /core\s+(\d+)/.exec(String(pt.text));
    if (!m) return;
    const coreIdx = Number(m[1]);
    const core = (seg.per_core || []).find(c => c.core_index === coreIdx);
    if (!core || !core.spike_raster) return;
    const raster = core.spike_raster;        // (n_neurons × T_eff) 0/1 array
    const panel = document.getElementById(rasterPanelId);
    if (!panel) return;
    panel.style.display = 'block';
    panel.innerHTML =
      `<div style="font-weight:600;margin-bottom:6px">Spike Raster — core ${coreIdx}` +
      ` (${raster.length} neurons × ${raster[0]?.length || 0} cycles)</div>` +
      `<div id="${rasterPanelId}-chart" style="min-height:240px"></div>`;
    safeReact(`${rasterPanelId}-chart`, [{
      z: raster, type: 'heatmap',
      colorscale: [[0, '#101418'], [1, '#4caf50']],
      showscale: false, hoverinfo: 'x+y+z',
    }], {
      height: Math.max(180, raster.length * 4 + 60),
      xaxis: { title: 'Cycle' },
      yaxis: { title: 'Neuron', autorange: 'reversed' },
      margin: { l: 60, r: 20, t: 10, b: 40 },
    });
  });
}

/* -------------------------------------------------------------------------- */
/* Per-core bars                                                                */
/* -------------------------------------------------------------------------- */

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
  const si = sampleIdx, st = seg.stage_index;
  const floorId = `sanafe-floor-${si}-${st}`;
  const metricId = `sanafe-metric-${si}-${st}`;
  const nocCheckId = `sanafe-noc-${si}-${st}`;
  const nocMetricId = `sanafe-nocm-${si}-${st}`;
  const nocCongId = `sanafe-noccong-${si}-${st}`;
  const connId = `sanafe-conn-${si}-${st}`;
  const critId = `sanafe-crit-${si}-${st}`;
  const cascadeId = `sanafe-cascade-${si}-${st}`;
  const waterId = `sanafe-water-${si}-${st}`;
  const animId = `sanafe-anim-${si}-${st}`;
  const scatterId = `sanafe-scatter-${si}-${st}`;
  const rasterId = `sanafe-raster-${si}-${st}`;
  const hasNoc = Array.isArray(seg.noc_links) && seg.noc_links.length > 0;
  const hasCong = Array.isArray(seg.noc_link_load) && seg.noc_link_load.length > 0;
  const hasConn = Array.isArray(seg.connectivity) && seg.connectivity.length > 0;
  const hasCrit = Array.isArray(seg.critical_cores) && seg.critical_cores.length > 0;
  const hasCascade = Array.isArray(seg.cascade) && seg.cascade.length > 0;
  const hasWater = Array.isArray(seg.cycle_energy) && seg.cycle_energy.length > 0;
  const hasAnim = Array.isArray(seg.noc_traffic_per_cycle) && seg.noc_traffic_per_cycle.length > 0;

  return `
    <div class="card" style="margin-bottom:16px">
      <div class="card-header">Segment ${st} — <span style="font-weight:normal">${esc(seg.stage_name)}</span></div>
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
              <input type="checkbox" id="${nocCheckId}" checked> NoC routes
            </label>
            ${metricSelectorHtml(nocMetricId, NOC_METRIC_OPTIONS, 'packet_count')}` : ''}
          ${hasCong ? `
            <label style="display:flex;align-items:center;gap:4px;font-size:13px">
              <input type="checkbox" id="${nocCongId}"> NoC congestion
            </label>` : ''}
          ${hasConn ? `
            <label style="display:flex;align-items:center;gap:4px;font-size:13px">
              <input type="checkbox" id="${connId}"> Live connectivity
            </label>` : ''}
          ${hasCrit ? `
            <label style="display:flex;align-items:center;gap:4px;font-size:13px">
              <input type="checkbox" id="${critId}" checked> Critical cores
            </label>` : ''}
        </div>
        <div id="${floorId}" style="min-height:300px;margin-bottom:6px"></div>
        <div id="${rasterId}" style="display:none;margin-bottom:14px;padding:10px;background:rgba(255,255,255,0.04);border-radius:6px"></div>
        <div style="font-size:11px;color:var(--text-muted);margin-bottom:14px">Click a core cell to see its per-neuron spike raster.</div>

        <div class="grid-2" style="gap:12px;margin-bottom:14px">
          ${hasCascade ? `
          <div>
            <div style="font-weight:600;margin-bottom:6px">Latency-Cascade Timeline</div>
            <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">firings per cycle, stacked by depth</div>
            <div id="${cascadeId}" style="min-height:240px"></div>
          </div>` : '<div></div>'}
          ${hasWater ? `
          <div>
            <div style="font-weight:600;margin-bottom:6px">Energy Waterfall</div>
            <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">cumulative J per category</div>
            <div id="${waterId}" style="min-height:240px"></div>
          </div>` : '<div></div>'}
        </div>

        <div class="grid-2" style="gap:12px;margin-bottom:14px">
          ${hasAnim ? `
          <div>
            <div style="font-weight:600;margin-bottom:6px">NoC Playback</div>
            <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">per-cycle packet flow — press ▶ Play</div>
            <div id="${animId}" style="min-height:300px"></div>
          </div>` : '<div></div>'}
          <div>
            <div style="font-weight:600;margin-bottom:6px">Efficiency Scatter</div>
            <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">activity-derived energy vs spikes per core, coloured by latency layer</div>
            <div id="${scatterId}" style="min-height:300px"></div>
          </div>
        </div>

        <div class="grid-2" style="gap:12px">
          <div>
            <div style="font-weight:600;margin-bottom:6px">Per-Core Energy</div>
            <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">activity-derived, J</div>
            <div id="sanafe-core-energy-${si}-${st}" style="min-height:240px"></div>
          </div>
          <div>
            <div style="font-weight:600;margin-bottom:6px">Per-Core Spikes</div>
            <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">output firings per core</div>
            <div id="sanafe-core-spikes-${si}-${st}" style="min-height:240px"></div>
          </div>
        </div>
      </div>
    </div>`;
}

function wireSegmentControls(sampleIdx, seg) {
  const si = sampleIdx, st = seg.stage_index;
  const floorId = `sanafe-floor-${si}-${st}`;
  const ids = {
    metric: `sanafe-metric-${si}-${st}`,
    nocCheck: `sanafe-noc-${si}-${st}`,
    nocMetric: `sanafe-nocm-${si}-${st}`,
    nocCong: `sanafe-noccong-${si}-${st}`,
    conn: `sanafe-conn-${si}-${st}`,
    crit: `sanafe-crit-${si}-${st}`,
  };
  const get = id => document.getElementById(id);

  const draw = () => {
    renderFloorplan(floorId, seg, {
      metricKey: get(ids.metric)?.value || 'energy_j',
      nocOverlay: get(ids.nocCheck)?.checked || false,
      nocMetricKey: get(ids.nocMetric)?.value || 'packet_count',
      nocCongestion: get(ids.nocCong)?.checked || false,
      connectivity: get(ids.conn)?.checked || false,
      criticalCores: get(ids.crit)?.checked || false,
    });
  };
  Object.values(ids).forEach(id => {
    const el = get(id);
    if (el) el.onchange = draw;
  });
  draw();
  attachRasterClickHandler(floorId, `sanafe-raster-${si}-${st}`, seg);
  renderCascadeTimeline(`sanafe-cascade-${si}-${st}`, seg);
  renderEnergyWaterfall(`sanafe-water-${si}-${st}`, seg);
  renderNocAnimation(`sanafe-anim-${si}-${st}`, seg);
  renderEfficiencyScatter(`sanafe-scatter-${si}-${st}`, seg);
  renderPerCoreBars(si, seg);
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
