"""Evolution history section for the interactive search report."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def render_history_section(metric_names: List[str]) -> str:
    parts = [
        '''
    <div class="card">
        <div class="card-title">📈 Evolution History</div>
        <div class="grid-2" id="history-container">
''',
    ]
    for i, _name in enumerate(metric_names):
        parts.append(f'            <div class="plot-panel"><div id="history-{i}"></div></div>\n')
    parts.append('''        </div>
    </div>
''')
    return "".join(parts)


def render_history_script(
    metric_names: List[str],
    gens: List[Any],
    history_series: Dict[str, List[Any]],
) -> str:
    return f"""
const historyGens = {json.dumps(gens)};
const historySeries = {json.dumps(history_series)};

const historyColors = ['#6366f1', '#f59e0b', '#10b981', '#ef4444'];
{json.dumps(metric_names)}.forEach((name, idx) => {{
    const trace = {{
        x: historyGens,
        y: historySeries[name],
        mode: 'lines+markers',
        name: name,
        line: {{ width: 3, color: historyColors[idx % historyColors.length], shape: 'spline' }},
        marker: {{ size: 8, color: historyColors[idx % historyColors.length] }},
        fill: 'tozeroy',
        fillcolor: historyColors[idx % historyColors.length] + '15'
    }};

    Plotly.newPlot('history-' + idx, [trace], {{
        ...darkLayout,
        title: {{ text: name, font: {{ size: 14, color: '#f1f5f9' }} }},
        xaxis: {{ title: 'Generation', ...gridStyle }},
        yaxis: {{ title: 'Best Value', ...gridStyle, tickformat: name === 'accuracy' ? '.2%' : '.3s' }},
        hovermode: 'x unified',
        height: 280,
        showlegend: false
    }});
}});
"""


PARETO_2D_SCRIPT = """
function get2DTraces(xMetric, yMetric, showNonPareto) {
    const traces = [];
    const x = candidateData[xMetric];
    const y = candidateData[yMetric];
    const isPareto = candidateData.is_pareto;
    const info = candidateData.hover_info;

    if (showNonPareto) {
        const npX = [], npY = [], npInfo = [];
        for (let i = 0; i < x.length; i++) {
            if (!isPareto[i]) { npX.push(x[i]); npY.push(y[i]); npInfo.push(info[i]); }
        }
        if (npX.length > 0) {
            traces.push({
                x: npX, y: npY, mode: 'markers', type: 'scatter', name: 'Non-Pareto',
                marker: { size: 8, color: '#475569', opacity: 0.3, line: { width: 0 } },
                text: npInfo, hovertemplate: '%{text}<extra></extra>'
            });
        }
    }

    const pX = [], pY = [], pInfo = [];
    for (let i = 0; i < x.length; i++) {
        if (isPareto[i]) { pX.push(x[i]); pY.push(y[i]); pInfo.push(info[i]); }
    }
    if (pX.length > 0) {
        traces.push({
            x: pX, y: pY, mode: 'markers', type: 'scatter', name: 'Pareto',
            marker: {
                size: 14, color: '#10b981', opacity: 0.9,
                line: { color: '#059669', width: 2 },
                symbol: 'diamond'
            },
            text: pInfo, hovertemplate: '%{text}<extra></extra>'
        });
    }
    return traces;
}

function update2DPlots() {
    const xA = document.getElementById('x2d-a').value;
    const yA = document.getElementById('y2d-a').value;
    Plotly.react('plot2d-a', get2DTraces(xA, yA, show2DNonPareto), {
        ...darkLayout,
        xaxis: { title: xA, ...gridStyle },
        yaxis: { title: yA, ...gridStyle },
        hovermode: 'closest', height: 420, showlegend: false
    });

    const xB = document.getElementById('x2d-b').value;
    const yB = document.getElementById('y2d-b').value;
    Plotly.react('plot2d-b', get2DTraces(xB, yB, show2DNonPareto), {
        ...darkLayout,
        xaxis: { title: xB, ...gridStyle },
        yaxis: { title: yB, ...gridStyle },
        hovermode: 'closest', height: 420, showlegend: false
    });
}

function toggle2DNonPareto() {
    show2DNonPareto = !show2DNonPareto;
    document.getElementById('show-nonpareto-2d').classList.toggle('active', show2DNonPareto);
    update2DPlots();
}

['x2d-a', 'y2d-a', 'x2d-b', 'y2d-b'].forEach(id => {
    document.getElementById(id).addEventListener('change', update2DPlots);
});

update2DPlots();
"""


def render_pareto_2d_script() -> str:
    return PARETO_2D_SCRIPT
