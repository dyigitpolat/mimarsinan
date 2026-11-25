# Derivations of Printed Values

## What This Folder Holds

Some printed numbers are counts, ranges or ratios derived from experiment records by a small script.
Each script wrote one derived-value record, and each value in that record names its input files, the sha256 of each input and the field it reads.
The experiments themselves are described in their own folders under `artifact/experiments/`.
The script of the host pricing's derivation is in that experiment's folder.

## How They Were Run

Each script reads its input records, notes the sha256 of each, applies the arithmetic it states and writes its derived record. The scripts name their inputs by their places in our working tree, so a reader who runs one points those paths at the copies in the zip.

## Scripts and Their Records

| Script | Derived record in the zip |
|---|---|
| `scripts/rv3_compose/d01_derive.py` | `records/tiers/paper_locks/cut/rv3_compose/derived/rv3_values.json` |
| not published | `records/tiers/paper_locks/cut/fr_lat/derived/derived_values.json` |
| not published | `records/tiers/paper_v6/final/audit_fix/keys/derived_values.json` |
| `scripts/audit_fix2/add_keys_aud5.py` | `records/tiers/paper_v6/final/audit_fix2/keys/derived_values.json` |
| not published | `records/tiers/paper_v6/fix5/p0/keys/derived_values.json` |
| not published | `records/tiers/paper_v6/fix5/p0/keys_tab/derived_values.json` |
| not published | `records/tiers/paper_v6/fix5/p0/keys_fix/derived_values.json` |
| not published | `records/tiers/paper_v6/fix5/p0/keys_card/derived_values.json` |
| not published | `records/tiers/paper_v6/fix5/p0/keys_asm/derived_values.json` |
| `../host_cpu_pricing/scripts/add_keys_aud7.py` | `records/tiers/paper_v6/final/audit_fix3/keys/derived_values.json` |

The counts of the compiler comparison are derived the same way. Their record, `records/tiers/paper_locks/cut/cut_exhibit_place/derived/exhibit_values.json`, is published in part and holds only the four printed values, as `artifact/experiments/compiler_comparison/README.md` describes.

## How to Read the Records

Each derived record groups its values by a short group name. A value holds `value` (or, for a range, `least` and `largest`) and `inputs` (each with `file`, `sha256`, `field` and `value`), or it names its one input directly by `file`, `sha256` and `field`. Most values also state their `arithmetic`.
An input's `file` is written as a place in the zip, also when the zip does not hold the file. A path of the form `records/frozen/<name>/records/<rest>` sits in the zip at `records/frozen/<name>/<rest>`, without the inner `records/`. A file that the zip does not hold at its named place may sit in a frozen copy, at `records/frozen/<name>/` followed by the named path without its leading `records/`.
The index of printed numbers (`artifact/numbers/`) names, for every printed key, the derived record and the field it reads.

## Not Included

The scripts marked not published above and the script that derived the counts of the compiler comparison are not published, as they hold internal working notes.
Some inputs that a derived record names are not in the zip: internal check and design notes, scripts of our working tree, records that hold paths of our machines, and records of side studies that are not published. Each such input keeps its sha256 and the value read from it inside the derived record.
The authors can be asked for them.
