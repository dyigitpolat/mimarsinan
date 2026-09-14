# mimarsinan
a pipeline for ***m***odeling ***i***n-***m***emory ***ar***chitectures for 
***s***piking ***n***eural ***n***etworks


## requirements

- python 3.10 (the ceiling is `lava-nc` 0.10.0's metadata)
- [uv](https://docs.astral.sh/uv/) — `pyproject.toml` + `uv.lock` are the manifest
- CUDA, for training
- clang 15 or any C++20 compiler, for the nevresim simulator compiled at run time
- the sibling `compilagent` checkout (the outer `research_stuff` repository
  provides it; a standalone clone needs
  `git clone https://github.com/dyigitpolat/compilagent.git ../compilagent`)

## setup

```bash
git submodule update --init nevresim   # the co-owned C++ simulator
make install                           # uv sync --extra dev --extra loihi, + patches
```

`make install SANAFE=1` adds the opt-in SANA-FE backend (below).
`make check-deps` runs the third-party contract test on its own;
`make check-pins` checks that every submodule pin is reachable on its remote.

To install clang:

```bash
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
sudo apt-get install libc++-17-dev libc++abi-17-dev
```

### third-party dependencies

`nevresim` is the only submodule: co-owned C++ consumed by path, with no Python
package and nothing on PyPI. Everything else is declared in `pyproject.toml`
and locked in `uv.lock`:

| inclusion | how it arrives | extra |
|---|---|---|
| `spikingjelly` | git URL at commit `fb71d12e` (the LIF numerics the golden traces are measured against) | base |
| `lava-nc` | `==0.10.0` from PyPI; `[tool.uv] override-dependencies` lifts its stale numpy/networkx/asteval pins | `loihi` |
| `sanafe` | `==2.1.1` from PyPI; its C++ headers are fetched by CMake from the `v2.1.1` tag archive, pinned by SHA256, into `build/` | `sanafe` |

`MIMARSINAN_NEVRESIM_ROOT` overrides where the simulator tree is looked for;
the default is the `nevresim/` directory beside the package's repository root,
so no entry point depends on the working directory.

`third_party/patches/<dist>/NNN-slug.patch` is the hook for patching an
installed dependency — empty by design today, applied by `make install` through
`scripts/apply_patches.py`.

## optional: SANA-FE detailed-stats backend

[SANA-FE](https://github.com/SLAM-Lab/SANA-FE) (GPL-3.0) is integrated as
an optional, opt-in pipeline step that produces per-tile and per-core
energy / latency / NoC packet stats on top of the spike-parity gate. To
enable it:

1. `make install SANAFE=1` (equivalently `bash scripts/bootstrap_sanafe.sh`):
   installs `sanafe==2.1.1` and builds the six mimarsinan-owned plugins into
   `build/mimarsinan_sanafe_plugins/`. Build deps: CMake >= 3.16, a C++17
   compiler. No SANA-FE source enters this tree; the headers are fetched into
   `build/` from the pinned tag archive.
2. Set `"enable_sanafe_simulation": true` in `deployment_parameters` (or
   toggle it in the wizard).

The step runs after `Simulation` and persists a
`SanafeStepReport` cache artifact the GUI's SANA-FE tab consumes.
mimarsinan itself stays MIT-licensed; nothing in `src/` imports
`sanafe` at module load time.

## docs
- DeepWiki: [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dyigitpolat/mimarsinan)
- [Architecture Guide](ARCHITECTURE.md)