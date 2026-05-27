# mapping/ -- Model-to-Hardware Mapping

Converts PyTorch models to an intermediate representation (IR) and then packs
the IR into physical hardware cores.

## Subpackages

| Directory | Role |
|-----------|------|
| `ir/` | `IRGraph`, `NeuralCore`, `ComputeOp`, legacy SoftCore conversions |
| `pruning/` | IR pruning, liveness, graph propagation, segmentation |
| `packing/` | Soft/hard cores, bin packing, hybrid mapping |
| `verification/` | Layout verifier, HW suggester, wizard layout service |
| `latency/` | `IRLatency`, `ChipLatency`, `upstream.py` |
| `platform/` | Platform constraints, coalescing, mapping structure |
| `export/` | Chip export and IR quantization |
| `layout/` | Shape-only layout SSoT |
| `mappers/` | Mapper hierarchy |

Compatibility shims at the package root (e.g. `mapping/ir.py` removed; use `from mimarsinan.mapping.ir import IRGraph` via the `ir/` package) re-export moved modules during migration.

See the module tables in the previous revision of this file for per-file contracts; paths are now `mapping/<subpackage>/<module>.py` with root-level shims where noted in `scripts/import_path_inventory.py`.
