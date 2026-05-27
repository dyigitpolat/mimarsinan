# mapping/ — Model-to-Hardware Mapping

Converts PyTorch models to an intermediate representation (IR) and packs the IR into physical hardware cores.

## Subpackages

| Directory | Doc | Role |
|-----------|-----|------|
| `ir/` | [ir/ARCHITECTURE.md](ir/ARCHITECTURE.md) | `IRGraph`, `NeuralCore`, `ComputeOp` |
| `pruning/` | [pruning/ARCHITECTURE.md](pruning/ARCHITECTURE.md) | IR pruning, liveness, segmentation |
| `packing/` | [packing/ARCHITECTURE.md](packing/ARCHITECTURE.md) | Soft/hard cores, bin packing, hybrid mapping |
| `verification/` | [verification/ARCHITECTURE.md](verification/ARCHITECTURE.md) | Layout verifier, HW suggester |
| `latency/` | [latency/ARCHITECTURE.md](latency/ARCHITECTURE.md) | `IRLatency`, `ChipLatency`, upstream closure |
| `platform/` | [platform/ARCHITECTURE.md](platform/ARCHITECTURE.md) | Platform constraints, tiling structure |
| `export/` | [export/ARCHITECTURE.md](export/ARCHITECTURE.md) | Chip export and IR quantization verify |
| `layout/` | [layout/ARCHITECTURE.md](layout/ARCHITECTURE.md) | Shape-only layout SSoT |
| `mappers/` | [mappers/ARCHITECTURE.md](mappers/ARCHITECTURE.md) | Mapper hierarchy |

## Root modules (orchestration)

| File | Role |
|------|------|
| `ir_mapping.py` | `IRMapping` — materializes weights into `IRGraph` |
| `mapping_utils.py` | Mapper registration and graph utilities |
| `model_representation.py` | Dual-purpose mapper graph |

Import from subpackages directly (e.g. `from mimarsinan.mapping.ir import IRGraph`). [`__init__.py`](__init__.py) re-exports the public mapping API.

Run `python scripts/import_path_inventory.py --check-legacy-path` to audit banned legacy paths.
