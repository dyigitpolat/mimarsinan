""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import deploy_gate as dg
import import_gate_checkpoint as ig
import price_n2_r2 as pn
from mimarsinan.mapping.mappers.operand import operand_mappers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="N2 acceptance 5, the resident declaration (revision 2)")
    parser.add_argument("out")
    parser.add_argument("key")
    parser.add_argument("threads", nargs="?", type=int, default=1)
    pn.add_store_arguments(parser)
    args = parser.parse_args(argv)
    try:
        stores, record = pn.declaration_from_args(args)
    except pn.StoreDeclarationRefused as refused:
        print(f"REFUSED {refused}", file=sys.stderr, flush=True)
        return 2
    out, key = Path(args.out), args.key
    import torch
    torch.set_num_threads(args.threads)
    data = ig.load_gate_data()
    program = ig.build_gate_program(key, data)
    _job, _form, arm, bits = ig.JOBS[key]
    bits = bits if arm == "operand" else None
    flat = dg.pack(program.ir, dg.cores(10 ** 6, bits), scheduled=False)
    tiles = sum(dg.segment_tiles(flat))
    declared = dg.cores(tiles, bits)
    packed = dg.pack(program.ir, declared, scheduled=True)
    shape = dg.program_shape(packed)
    packing = "scheduled"
    if shape["max_passes_per_segment"] != 1:


        packing = (f"scheduling off (the scheduled packer ran {shape['multi_pass_segments']} segment(s) in "
                   f"{shape['max_passes_per_segment']} passes at {tiles} cores)")
        packed = dg.pack(program.ir, declared, scheduled=False)
        shape = dg.program_shape(packed)
        neural = [st for st in packed.stages if st.kind == "neural"]
        if all(st.schedule_segment_index is None for st in neural):

            shape.update(segments=len(neural), max_passes_per_segment=1, multi_pass_segments=0)
        if shape["max_passes_per_segment"] != 1:
            raise SystemExit(f"REFUSED: the resident declaration of {tiles} cores runs a segment in "
                             f"{shape['max_passes_per_segment']} passes")
    product_macs = sum(int(m.lowering.form.banks) * int(m.lowering.form.m) * int(m.lowering.form.k)
                       * int(m.lowering.form.n) for m in operand_mappers(program.model.get_mapper_repr()))
    rows, summary = pn.price_program(packed, declared, product_macs, stores=stores)
    result = {"key": key, "declaration": "resident: one core per tile of the program", "cores_declared": declared,
              "tiles": tiles, "widest_segment_tiles": max(dg.segment_tiles(flat)), "packing": packing,
              "packed": shape,
              "census": summary, "product_macs": int(product_macs),
              "bank_cells_per_inference": int(sum(b.cells for b in program.ir.dynamic_bank_bindings.values())),
              "rate_hz": pn.RATE_HZ, "min_lifetime_hours": pn.FLOOR_H,
              "script": {"name": "price_resident_r2.py", "revision": 2}, "stores_declared": record, "rows": rows}
    out.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({"key": key, "tiles": tiles, **summary}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
