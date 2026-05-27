"""2D/3D Pareto sections and Plotly scripts for the interactive search report."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _metric_options(metric_names: List[str], selected_indices: List[int]) -> str:
    parts: List[str] = []
    for i, name in enumerate(metric_names):
        selected = " selected" if i in selected_indices else ""
        parts.append(f'<option value="{name}"{selected}>{name}</option>')
    return "".join(parts)


def render_pareto_2d_section(metric_names: List[str]) -> str:
    return f'''
    <div class="card">
        <div class="card-title">📊 2D Pareto Projections</div>
        <div class="controls">
            <div class="control-row">
                <button class="toggle-btn active" id="show-nonpareto-2d" onclick="toggle2DNonPareto()">
                    Show Non-Pareto
                </button>
            </div>
        </div>
        <div class="grid-2">
            <div class="plot-panel">
                <div class="plot-panel-header">
                    <span class="plot-panel-title">Plot A</span>
                    <div class="plot-controls">
                        <select id="x2d-a">{_metric_options(metric_names, [0])}</select>
                        <select id="y2d-a">{_metric_options(metric_names, [1])}</select>
                    </div>
                </div>
                <div id="plot2d-a"></div>
            </div>
            <div class="plot-panel">
                <div class="plot-panel-header">
                    <span class="plot-panel-title">Plot B</span>
                    <div class="plot-controls">
                        <select id="x2d-b">{_metric_options(metric_names, [2 % len(metric_names)])}</select>
                        <select id="y2d-b">{_metric_options(metric_names, [3 % len(metric_names)])}</select>
                    </div>
                </div>
                <div id="plot2d-b"></div>
            </div>
        </div>
    </div>
'''


def render_pareto_3d_section(metric_names: List[str]) -> str:
    return f'''
    <div class="card">
        <div class="card-title">🎯 3D Pareto Surface Visualization</div>
        <div class="controls">
            <div class="control-row">
                <button class="toggle-btn active" id="show-nonpareto-3d" onclick="toggle3DNonPareto()">
                    Show Non-Pareto
                </button>
                <button class="toggle-btn active" id="show-surface" onclick="toggleSurface()">
                    Show Pareto Surface
                </button>
            </div>
        </div>
        <div class="grid-2">
            <div class="plot-panel">
                <div class="plot-panel-header">
                    <span class="plot-panel-title">3D View A</span>
                    <div class="plot-controls">
                        <select id="x3d-a">{_metric_options(metric_names, [0])}</select>
                        <select id="y3d-a">{_metric_options(metric_names, [1])}</select>
                        <select id="z3d-a">{_metric_options(metric_names, [2])}</select>
                    </div>
                </div>
                <div id="plot3d-a"></div>
            </div>
            <div class="plot-panel">
                <div class="plot-panel-header">
                    <span class="plot-panel-title">3D View B</span>
                    <div class="plot-controls">
                        <select id="x3d-b">{_metric_options(metric_names, [0])}</select>
                        <select id="y3d-b">{_metric_options(metric_names, [2 % len(metric_names)])}</select>
                        <select id="z3d-b">{_metric_options(metric_names, [3 % len(metric_names)])}</select>
                    </div>
                </div>
                <div id="plot3d-b"></div>
            </div>
        </div>
    </div>
'''


def render_pareto_data_script(
    metric_names: List[str],
    candidate_data: Dict[str, Any],
    goal_by_name: Dict[str, Any],
) -> str:
    return f"""
const metricNames = {json.dumps(metric_names)};
const candidateData = {json.dumps(candidate_data)};
const objectiveGoals = {json.dumps(goal_by_name)};
let show2DNonPareto = true;
let show3DNonPareto = true;
let showSurface = true;
"""


PARETO_3D_SCRIPT = """
function get3DTraces(xMetric, yMetric, zMetric, showNonPareto, addSurface) {
    const traces = [];
    const x = candidateData[xMetric];
    const y = candidateData[yMetric];
    const z = candidateData[zMetric];
    const isPareto = candidateData.is_pareto;
    const info = candidateData.hover_info;

    if (showNonPareto) {
        const npX = [], npY = [], npZ = [], npInfo = [];
        for (let i = 0; i < x.length; i++) {
            if (!isPareto[i]) { npX.push(x[i]); npY.push(y[i]); npZ.push(z[i]); npInfo.push(info[i]); }
        }
        if (npX.length > 0) {
            traces.push({
                x: npX, y: npY, z: npZ, mode: 'markers', type: 'scatter3d', name: 'Non-Pareto',
                marker: { size: 4, color: '#475569', opacity: 0.25 },
                text: npInfo, hovertemplate: '%{text}<extra></extra>'
            });
        }
    }

    const pX = [], pY = [], pZ = [], pInfo = [];
    for (let i = 0; i < x.length; i++) {
        if (isPareto[i]) { pX.push(x[i]); pY.push(y[i]); pZ.push(z[i]); pInfo.push(info[i]); }
    }

    function getBest(arr, metric) {
        if (arr.length === 0) return null;
        const goal = objectiveGoals[metric] || 'min';
        return goal === 'max' ? Math.max(...arr) : Math.min(...arr);
    }

    const bestX = getBest(pX, xMetric);
    const bestY = getBest(pY, yMetric);
    const bestZ = getBest(pZ, zMetric);

    const bestXIdx = pX.findIndex(v => v === bestX);
    const bestYIdx = pY.findIndex(v => v === bestY);
    const bestZIdx = pZ.findIndex(v => v === bestZ);

    if (pX.length > 0 && bestX !== null) {
        const lineColors = ['#ef4444', '#22c55e', '#3b82f6'];
        const bestPoints = [
            { idx: bestXIdx, color: lineColors[0], label: 'Best ' + xMetric },
            { idx: bestYIdx, color: lineColors[1], label: 'Best ' + yMetric },
            { idx: bestZIdx, color: lineColors[2], label: 'Best ' + zMetric }
        ];

        bestPoints.forEach((bp) => {
            if (bp.idx < 0) return;
            const bx = pX[bp.idx], by = pY[bp.idx], bz = pZ[bp.idx];

            for (let i = 0; i < pX.length; i++) {
                if (i === bp.idx) continue;
                traces.push({
                    x: [bx, pX[i]], y: [by, pY[i]], z: [bz, pZ[i]],
                    mode: 'lines', type: 'scatter3d',
                    line: { color: bp.color, width: 2 },
                    opacity: 0.15,
                    hoverinfo: 'skip',
                    showlegend: false
                });
            }
        });

        const starX = [], starY = [], starZ = [], starColors = [], starLabels = [];
        bestPoints.forEach(bp => {
            if (bp.idx >= 0) {
                starX.push(pX[bp.idx]);
                starY.push(pY[bp.idx]);
                starZ.push(pZ[bp.idx]);
                starColors.push(bp.color);
                starLabels.push(bp.label);
            }
        });

        if (starX.length > 0) {
            traces.push({
                x: starX, y: starY, z: starZ,
                mode: 'markers', type: 'scatter3d', name: 'Best Points',
                marker: {
                    size: 14,
                    color: starColors,
                    symbol: 'diamond',
                    line: { color: '#fff', width: 2 }
                },
                text: starLabels,
                hovertemplate: '%{text}<extra></extra>'
            });
        }
    }

    if (addSurface && pX.length >= 3) {
        traces.push({
            x: pX, y: pY, z: pZ, type: 'mesh3d', name: 'Pareto Surface',
            opacity: 0.35,
            color: '#6366f1',
            flatshading: true,
            lighting: { ambient: 0.8, diffuse: 0.5, specular: 0.3 },
            hoverinfo: 'skip'
        });
    }

    if (pX.length > 0) {
        traces.push({
            x: pX, y: pY, z: pZ, mode: 'markers', type: 'scatter3d', name: 'Pareto',
            marker: {
                size: 7,
                color: pZ,
                colorscale: [[0, '#10b981'], [0.5, '#f59e0b'], [1, '#ef4444']],
                opacity: 1,
                line: { color: '#fff', width: 1 },
                showscale: true,
                colorbar: { thickness: 15, len: 0.6, tickfont: { color: '#94a3b8' } }
            },
            text: pInfo, hovertemplate: '%{text}<extra></extra>'
        });
    }

    return traces;
}

function update3DPlots() {
    const sceneStyle = {
        xaxis: { ...gridStyle, backgroundcolor: '#0f172a', showbackground: true },
        yaxis: { ...gridStyle, backgroundcolor: '#0f172a', showbackground: true },
        zaxis: { ...gridStyle, backgroundcolor: '#0f172a', showbackground: true },
        camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
    };

    const xA = document.getElementById('x3d-a').value;
    const yA = document.getElementById('y3d-a').value;
    const zA = document.getElementById('z3d-a').value;
    Plotly.react('plot3d-a', get3DTraces(xA, yA, zA, show3DNonPareto, showSurface), {
        ...darkLayout,
        scene: { ...sceneStyle, xaxis: { ...sceneStyle.xaxis, title: xA }, yaxis: { ...sceneStyle.yaxis, title: yA }, zaxis: { ...sceneStyle.zaxis, title: zA } },
        height: 550, showlegend: false
    });

    const xB = document.getElementById('x3d-b').value;
    const yB = document.getElementById('y3d-b').value;
    const zB = document.getElementById('z3d-b').value;
    Plotly.react('plot3d-b', get3DTraces(xB, yB, zB, show3DNonPareto, showSurface), {
        ...darkLayout,
        scene: { ...sceneStyle, xaxis: { ...sceneStyle.xaxis, title: xB }, yaxis: { ...sceneStyle.yaxis, title: yB }, zaxis: { ...sceneStyle.zaxis, title: zB } },
        height: 550, showlegend: false
    });
}

function toggle3DNonPareto() {
    show3DNonPareto = !show3DNonPareto;
    document.getElementById('show-nonpareto-3d').classList.toggle('active', show3DNonPareto);
    update3DPlots();
}

function toggleSurface() {
    showSurface = !showSurface;
    document.getElementById('show-surface').classList.toggle('active', showSurface);
    update3DPlots();
}

['x3d-a', 'y3d-a', 'z3d-a', 'x3d-b', 'y3d-b', 'z3d-b'].forEach(id => {
    document.getElementById(id).addEventListener('change', update3DPlots);
});

update3DPlots();
"""


def render_pareto_3d_script() -> str:
    return PARETO_3D_SCRIPT
