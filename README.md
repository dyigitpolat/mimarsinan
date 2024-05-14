# mimarsinan
a pipeline for ***m***odeling ***i***n-***m***emory ***ar***chitectures for 
***s***piking ***n***eural ***n***etworks


## requirements
- python 3.10
    - nni
    - torch
    - torchvision
    - einops
    - wandb
    - matplotlib

- nevresim
- clang 15

## setup
- install warmup_scheduler
    `pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git`
- to clone nevresim simulator, you need to run: \
    `git submodule update --init --recursive`
- to install clang:
    - `bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"`
    - `sudo apt-get install libc++-17-dev libc++abi-17-dev`