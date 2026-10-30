# Provenance — `hw/vendor/odin`

| Field | Value |
|---|---|
| Upstream | <https://github.com/ChFrenkel/ODIN> |
| Commit | `17819318d17b6241d4b185c13ce9b3372d453778` (short `1781931`, subject "release-ready") |
| Commit date | 2019-04-20 |
| Vendored on | 2026-08-21 |
| Subtree | `src/**` (18 Verilog files), root `LICENSE`, `doc/LICENSE` |
| Deliberately NOT vendored | `.git/`, `README.md`, `doc/figs/` (documentation images; the doc text lives upstream and is quoted where the packer tables need it) |
| Licence (HDL) | Solderpad Hardware License v2.0 — see `LICENSE`; per-file headers retained verbatim |
| Licence (documentation) | Creative Commons Attribution 4.0 International — see `doc/LICENSE` |

## Integrity

`MANIFEST.sha256` records the SHA-256 of every vendored file, in `sha256sum`
format, relative to this directory. `tests/unit/mapping/test_odin_vendor_tree.py`
re-hashes the tree against it on every default-suite run, so an accidental edit
to a file this project's exporter transcribes bit layouts from fails loud rather
than silently changing what "the stock ODIN core" means.

Verify by hand with:

    cd hw/vendor/odin && sha256sum -c MANIFEST.sha256

## The tree is READ-ONLY

Nothing in this directory is ever edited. The two upstream-mandated
substitutions (`neuron_core.v:301` / `synaptic_core.v:154` / upstream doc §5:
"the behavioral descriptions of the neuron and synapse single-port synchronous
SRAMs need to be replaced with SRAM macros ... Block RAM (BRAM) instances can be
used for FPGA implementations") live in `hw/fpga/mem/` as an overlay selected by
source-file order, never as an edit here.

## Citation

Upon usage of the documentation or source code, upstream asks for:

> C. Frenkel, M. Lefebvre, J.-D. Legat and D. Bol, "A 0.086-mm² 12.7-pJ/SOP
> 64k-Synapse 256-Neuron Online-Learning Digital Spiking Neuromorphic Processor
> in 28-nm CMOS," IEEE Transactions on Biomedical Circuits and Systems, vol. 13,
> no. 1, pp. 145-158, 2019.
