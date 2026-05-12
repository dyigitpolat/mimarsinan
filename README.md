# mimarsinan
a pipeline for ***m***odeling ***i***n-***m***emory ***ar***chitectures for 
***s***piking ***n***eural ***n***etworks


## requirements
- python 3.10
    - nni
    - numpy
    - torch
    - torchvision
    - einops
    - matplotlib

- CUDA
- nevresim
- clang 15

## setup
- install warmup_scheduler
    `pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git`
- to clone nevresim simulator, you need to run: \
    `git submodule update --init --recursive`
- to install clang:
    - `sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"`
    - `sudo apt-get install libc++-17-dev libc++abi-17-dev`

## optional: SANA-FE detailed-stats backend

[SANA-FE](https://github.com/SLAM-Lab/SANA-FE) (GPL-3.0) is integrated as
an optional, opt-in pipeline step that produces per-tile and per-core
energy / latency / NoC packet stats on top of the spike-parity gate. To
enable it:

1. `bash scripts/bootstrap_sanafe.sh`  *(pulls the submodule and runs
   `pip install -e ./sana_fe` inside the active venv; build deps:
   CMake ≥ 3.13, GCC ≥ 8, flex, pybind11 ≥ 2.6)*
2. Set `"enable_sanafe_simulation": true` in `deployment_parameters` (or
   toggle it in the wizard).

The step runs after `Simulation` and persists a
`SanafeStepReport` cache artifact the GUI's SANA-FE tab consumes.
mimarsinan itself stays MIT-licensed; nothing in `src/` imports
`sanafe` at module load time.

## docs
- DeepWiki: [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dyigitpolat/mimarsinan)
- [Architecture Guide](ARCHITECTURE.md)