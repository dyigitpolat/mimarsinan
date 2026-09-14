# third_party/patches

Patches against installed third-party distributions. **Empty by design.**

No third-party source mimarsinan depends on needs patching today:

- **lava** is used unmodified. What mimarsinan does carry is a *runtime*
  behaviour fix, not a source change: `chip_simulation/lava_loihi/core_lava.py`
  monkeypatches `multiprocessing.set_start_method` /
  `torch.multiprocessing.set_start_method` to be idempotent before the first
  lava import, because lava's message infrastructure sets `fork` at import
  time. It applies itself through `_subtractive_lif_cls()` and stays where it
  is. lava-nc's stale `numpy<2` / `networkx<=2.8.7` / `asteval<0.10` metadata
  is handled declaratively by `[tool.uv] override-dependencies`.
- **SANA-FE** is extended through its own `plugin:` mechanism: the six
  `libmimarsinan_*.so` in `src/mimarsinan/chip_simulation/sanafe/plugins/` are
  mimarsinan source compiled against SANA-FE's headers.
- **spikingjelly** is extended by subclassing (`_LatticeIFNode` in
  `models/nn/activations/lif.py` overrides `single_step_forward` /
  `multi_step_forward`).

The hook exists so the next one is a file here, not a vendored tree.

## Layout

    third_party/patches/<distribution-name>/<NNN>-<slug>.patch

`<distribution-name>` is the name `importlib.metadata` knows (`lava-nc`, not
`lava`). The diffs are unified, `-p1`, with paths relative to the distribution
root (the site-packages directory holding the installed package).

## Applying

`scripts/apply_patches.py` (run by `make install`) applies each patch with
`patch -p1 --forward`, dry-run first, and records `patch file -> sha256` in
`<site-packages>/<dist>.mimarsinan-patches.json`, which makes re-runs no-ops.
`scripts/apply_patches.py --check` reports any patch that is not recorded as
applied and is what `tests/unit/architecture/test_third_party_contract.py`
asserts.

A patch whose content changes after it was applied cannot be re-applied to the
already-patched tree: reinstall the distribution first
(`uv sync --reinstall-package <dist>`), then re-run.
