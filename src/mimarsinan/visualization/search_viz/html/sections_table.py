"""Candidates table section for the interactive search report."""

from __future__ import annotations

from typing import Any, Dict, List


def _display_column_name(key: str) -> str:
    return (
        key.replace("_", " ")
        .replace("model ", "")
        .replace("obj ", "")
        .replace("hw ", "HW ")
        .title()
    )


def render_table_section(table_rows: List[Dict[str, Any]]) -> str:
    parts = [
        '''
    <div class="card">
        <div class="card-title">🔍 All Candidates Browser</div>
        <div class="table-wrapper">
            <div class="table-controls">
                <input type="text" class="search-input" id="table-search" placeholder="Search candidates..." onkeyup="filterTable()">
                <button class="toggle-btn active" id="show-nonpareto-table" onclick="toggleTableNonPareto()">
                    Show Non-Pareto
                </button>
            </div>
            <div class="table-container">
                <table id="candidates-table">
                    <thead>
                        <tr>
                            <th class="sortable" onclick="sortTable(0)">ID</th>
                            <th class="sortable" onclick="sortTable(1)">Gen</th>
                            <th class="sortable" onclick="sortTable(2)">Status</th>''',
    ]

    col_idx = 3
    if table_rows:
        sample_row = table_rows[0]
        for key in sorted(sample_row.keys()):
            if key not in ["id", "generation", "is_pareto"]:
                display_name = _display_column_name(key)
                parts.append(f'<th class="sortable" onclick="sortTable({col_idx})">{display_name}</th>')
                col_idx += 1

    parts.append('''
                        </tr>
                    </thead>
                    <tbody>''')

    for row in table_rows:
        is_pareto = row.get("is_pareto", False)
        row_class = "" if is_pareto else "non-pareto"
        parts.append(f'<tr class="{row_class}" data-pareto="{str(is_pareto).lower()}">')
        parts.append(f'<td>{row.get("id", "")}</td>')
        parts.append(f'<td><span class="badge badge-gen">G{row.get("generation", -1)}</span></td>')
        badge_class = "badge-pareto" if is_pareto else "badge-other"
        badge_text = "PARETO" if is_pareto else "OTHER"
        parts.append(f'<td><span class="badge {badge_class}">{badge_text}</span></td>')
        for key in sorted(row.keys()):
            if key not in ["id", "generation", "is_pareto"]:
                value = row.get(key, "")
                if isinstance(value, float):
                    parts.append(f'<td>{value:.4f}</td>')
                else:
                    parts.append(f"<td>{value}</td>")
        parts.append("</tr>")

    parts.append('''
                    </tbody>
                </table>
            </div>
        </div>
    </div>
''')
    return "".join(parts)


TABLE_SCRIPT = """
function toggleTableNonPareto() {
    showTableNonPareto = !showTableNonPareto;
    document.getElementById('show-nonpareto-table').classList.toggle('active', showTableNonPareto);
    const rows = document.querySelectorAll('#candidates-table tbody tr.non-pareto');
    rows.forEach(row => {
        row.style.display = showTableNonPareto ? '' : 'none';
    });
}

function filterTable() {
    const filter = document.getElementById('table-search').value.toUpperCase();
    const rows = document.querySelectorAll('#candidates-table tbody tr');
    rows.forEach(row => {
        const isNonPareto = row.classList.contains('non-pareto');
        if (!showTableNonPareto && isNonPareto) {
            row.style.display = 'none';
            return;
        }
        const text = row.textContent || row.innerText;
        row.style.display = text.toUpperCase().includes(filter) ? '' : 'none';
    });
}

let sortDirection = {};
function sortTable(columnIndex) {
    const table = document.getElementById('candidates-table');
    const tbody = table.getElementsByTagName('tbody')[0];
    const rows = Array.from(tbody.getElementsByTagName('tr'));

    const direction = sortDirection[columnIndex] === 'asc' ? 'desc' : 'asc';
    sortDirection[columnIndex] = direction;

    const headers = table.getElementsByTagName('th');
    for (let i = 0; i < headers.length; i++) headers[i].className = 'sortable';
    headers[columnIndex].className = direction === 'asc' ? 'sortable sorted-asc' : 'sortable sorted-desc';

    rows.sort((a, b) => {
        const aValue = a.getElementsByTagName('td')[columnIndex].textContent;
        const bValue = b.getElementsByTagName('td')[columnIndex].textContent;
        const aNum = parseFloat(aValue), bNum = parseFloat(bValue);
        if (!isNaN(aNum) && !isNaN(bNum)) return direction === 'asc' ? aNum - bNum : bNum - aNum;
        return direction === 'asc' ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
    });

    rows.forEach(row => tbody.appendChild(row));
}
"""


def render_table_script() -> str:
    return TABLE_SCRIPT
