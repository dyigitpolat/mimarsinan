# Comparison with Two Released Compilers

## What the Paper Reports

Section V-D (Compilers) reports that we ran PIMCOMP-NN at commit `0cfbfa55` in default batch mode and CIMFlow at
commit `67f3da07` on the conversion test for their own cores, under a protocol locked beforehand and amended once
afterwards for script defects, changing no outcome class, and that both complete without an error on programs that do
not compute the network as given. Fig. 9(a) and (b) show the two compilers beside our toolchain. Table IV (Cases of
Differing Outputs and Their Detection) has two rows from these runs: PIMCOMP-NN maps the products of RegNetY to
additions and then drops them (16 and 14 products, 200 of 200 decisions changed by the additions, and its verifier
matches on 200 of 200), and CIMFlow emits no instruction for fully connected layers in 12 of 12 compiled networks.
Table V (Defect Classes in Prior Toolchains and Our Handling) marks both compilers as run and inspected by us. The
script of Fig. 9 is in `artifact/figures/`.

Printed keys (in `artifact/numbers/index.json`): `pimcompproducts`, `pimcompadditions`, `pimcompreplay` and
`cimflownofc`.

This folder holds the inspection of PIMCOMP-NN's programs behind the three PIMCOMP-NN keys, and the record that carries
all four printed values.

## Setup of the PIMCOMP-NN Inspection

Fields are named by their path inside the record, as in `artifact/numbers/index.json`.

- **Networks.** RegNetY-400MF and RegNetY-800MF with torchvision's published weights
  (`regnet_y_400mf/weights` and `regnet_y_800mf/weights` of `export_info.json`), exported to ONNX at opset 11
  (`opset`).
- **Inputs.** 200 seeded inputs per network (`inputs/n` and `inputs/seed` of `export_info.json`), in a normalized and a
  raw range (`inputs/norm` and `inputs/raw`).
- **Compiler.** PIMCOMP-NN at commit `0cfbfa55`, its front end, back end and verifier run as its README states, the back
  end in its batch pipeline mode with verification output.
- **Our toolchain.** The patched release, on the same two networks with the same published weights.

## How It Was Run

1. `scripts/pimcomp_inspection/s01_export.py` exports both networks to ONNX and draws the seeded inputs, and writes
   `export_info.json`.
2. PIMCOMP-NN's front end, back end and verifier run on the exported networks, as its README states. Our shell wrappers
   of these runs are not published, since they hold settings of our machines.
3. `scripts/pimcomp_inspection/s05b_count_instructions.py` counts, per node kind, the nodes of each batch program that
   carry an instruction.
4. `scripts/pimcomp_inspection/s08_control.py` runs ONNX Runtime on each exported network and on the same network with
   every product node set to an addition, as PIMCOMP-NN's front end represents it, and writes `control.json`.
5. `scripts/pimcomp_inspection/s09_replay.py` replays each batch program with PIMCOMP-NN's own verification code on the
   seeded inputs, and `scripts/pimcomp_inspection/s10_compare.py` compares the replay's outputs with ONNX Runtime's and
   writes `replay_compare.json`.
6. `scripts/pimcomp_inspection/s07_toolchain.py` and `scripts/pimcomp_inspection/s07b_toolchain_800.py` run our
   toolchain's conversion test on the same two networks (the toolchain's `scripts/requirement_census.py` and
   `src/mimarsinan/mapping/capability/`) and write `toolchain_census.json` and `toolchain_800_extra.json`.
7. A derivation script reads the compiled programs, `control.json`, `replay_compare.json` and the CIMFlow record and
   writes `exhibit_values.json`. It is not published, since it holds internal working notes. The record is published
   in part and holds the four printed values only.

The folder `scripts/pimcomp/` holds byte-identical copies of five of these scripts (`s01_export.py`,
`s05b_count_instructions.py`, `s08_control.py`, `s09_replay.py` and `s10_compare.py`).

## Records

| Path in the zip | What it holds | Printed keys |
|---|---|---|
| `records/tiers/paper_locks/cut/cut_exhibit_place/derived/exhibit_values.json` | the four printed values, published in part | `pimcompproducts`, `pimcompadditions`, `pimcompreplay`, `cimflownofc` |
| `records/tiers/paper_locks/cut/cut_pimcomp_measure/out/control.json` | ONNX Runtime with products and with additions, per network and input range | `pimcompadditions`, through the derivation |
| `records/tiers/paper_locks/cut/cut_pimcomp_measure/out/replay_compare.json` | PIMCOMP-NN's verifier replay against ONNX Runtime | `pimcompreplay`, through the derivation |
| `records/tiers/paper_locks/cut/cut_pimcomp_measure/out/export_info.json` | the exported networks, their weights, operator counts and the seeded inputs | none |
| `records/tiers/paper_locks/cut/cut_pimcomp_measure/out/toolchain_census.json` | our toolchain's conversion test on the two networks | none |
| `records/tiers/paper_locks/cut/cut_pimcomp_measure/out/toolchain_800_extra.json` | RegNetY-800MF's converted form in more detail | none |

## How to Read the Records

- `exhibit_values.json`: it begins with a `_note` that says it is published in part.
  `pimcomp/products/<network>/products` counts the product nodes of each network. `pimcomp/additions/both` gives `k`
  of `n` decisions changed by the additions and `pimcomp/verifier/both` `k` of `n` decisions on which the verifier
  matches, both networks together. `cimflow/no_fc_instruction` of `cimflow/compiled` gives `cimflownofc`. The inputs
  of each value, their digests and the arithmetic are in the part that is not published.
- `control.json`, `<network>/<range>/logits`: `top1_changed` of `n` decisions and the largest relative errors of the
  network with additions against the network as exported. `<network>/mul_nodes` counts the products.
- `replay_compare.json`, `<network>_<range>`: `top1_changed` of `n` decisions between the replay and ONNX Runtime,
  `ort_top1_agrees_with_saved` the agreement with the outputs PIMCOMP-NN's verifier saved, and
  `first_product_slot_nonzero_max` the largest count of nonzero values in the first product's slot of the replay's
  memory.

## Not Included

The records of both compilers' runs on the conversion test, their locked protocol with its amendment, the counts of
compiled and refused networks and the comparison's tables are not in the public copy. Neither are PIMCOMP-NN's
compiled programs, our patch of its source, the shell wrappers of its runs, the derivation script and the fields of
`exhibit_values.json` outside its published part. PIMCOMP-NN and CIMFlow are other authors' software, available from their publishers. The authors of this
paper can be asked for the withheld records and scripts.
