# gui/snapshot/ — Pipeline Snapshot Builders

Builds JSON snapshots for the GUI from pipeline cache entries.

| File | Role |
|------|------|
| `builders.py` | Orchestrates per-step snapshot dispatch |
| `model_snapshot.py` | Model-building snapshot |
| `ir_graph_snapshot.py` | IR graph snapshot |
| `mapping_snapshot.py` | Mapping / hybrid snapshot |
| `search_snapshot.py` | Architecture search snapshot |
| `sanafe_snapshot.py` | SANA-FE results snapshot |
| `adaptation_snapshot.py` | Adaptation metrics |
| `heatmap.py` | Weight/activation heatmaps |
| `constants.py` | Snapshot key constants |
| `helpers.py` | Numeric/dict conversion helpers |
| `rebuild.py` | Snapshot rebuild on read |

## Dependents

- `gui.server`, `gui.data_collector`
