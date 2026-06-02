# mimarsinan custom SANA-FE plugins.
#
# Builds two shared libraries — ``libmimarsinan_dendrite.so`` and
# ``libmimarsinan_soma.so`` — that SANA-FE loads at runtime via the
# ``plugin:`` key in the arch YAML.  See:
#   * mimarsinan_dendrite.cpp — replaces the upstream ``accumulator``
#     dendrite to lift the hardcoded 1024-neurons-per-core cap.
#   * mimarsinan_soma.cpp — implements ``SubtractiveLIFReset`` semantics
#     without any Loihi-derived assumptions.
#
# Usage from the mimarsinan repo root, with the project venv active:
#   cmake -S src/mimarsinan/chip_simulation/sanafe/plugins \
#         -B build/mimarsinan_sanafe_plugins \
#         -DSANAFE_SRC=$(pwd)/sana_fe/src
#   cmake --build build/mimarsinan_sanafe_plugins --parallel
#
# ``scripts/bootstrap_sanafe.sh`` automates this.

cmake_minimum_required(VERSION 3.16)
project(MimarsinanSanafePlugins LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Path to SANA-FE's header tree.  Defaults to the in-tree submodule;
# override with ``-DSANAFE_SRC=...`` if the submodule lives elsewhere.
if(NOT DEFINED SANAFE_SRC)
    set(SANAFE_SRC "${CMAKE_SOURCE_DIR}/../../../../../sana_fe/src")
endif()

if(NOT EXISTS "${SANAFE_SRC}/pipeline.hpp")
    message(FATAL_ERROR
        "SANAFE_SRC (${SANAFE_SRC}) does not contain pipeline.hpp. "
        "Run ``git submodule update --init sana_fe`` or pass "
        "-DSANAFE_SRC=/path/to/sana_fe/src.")
endif()

add_compile_options(-Wall -Wextra -fPIC)

foreach(plugin_target IN ITEMS mimarsinan_dendrite mimarsinan_soma
        mimarsinan_ttfs_continuous_soma mimarsinan_ttfs_quantized_soma
        mimarsinan_ttfs_cycle_soma mimarsinan_ttfs_cascade_soma)
    add_library(${plugin_target} SHARED "${plugin_target}.cpp")
    target_include_directories(${plugin_target} PRIVATE "${SANAFE_SRC}")
    target_link_libraries(${plugin_target} PRIVATE ${CMAKE_DL_LIBS})
endforeach()
