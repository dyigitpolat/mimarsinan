# mimarsinan custom SANA-FE plugins.
#
# Builds the six shared libraries SANA-FE loads at runtime via the ``plugin:``
# key in the arch YAML — ``libmimarsinan_dendrite.so``, ``libmimarsinan_soma.so``
# and the four TTFS soma variants.  See:
#   * mimarsinan_dendrite.cpp — replaces the upstream ``accumulator``
#     dendrite to lift the hardcoded 1024-neurons-per-core cap.
#   * mimarsinan_soma.cpp — implements ``SubtractiveLIFReset`` semantics
#     without any Loihi-derived assumptions.
#
# The headers come from SANA-FE's v2.1.1 tag archive, fetched into the build
# directory and pinned by SHA256 -- SANA-FE is GPL-3.0 and nothing of it enters
# this MIT tree. Override with -DSANAFE_SRC=/path/to/sana_fe/src to build
# against a checkout instead.
#
# Usage from the mimarsinan repo root:
#   python scripts/build_sanafe_plugins.py
#
# or by hand:
#   cmake -S src/mimarsinan/chip_simulation/sanafe/plugins \
#         -B build/mimarsinan_sanafe_plugins
#   cmake --build build/mimarsinan_sanafe_plugins --parallel

cmake_minimum_required(VERSION 3.16)
project(MimarsinanSanafePlugins LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Path to SANA-FE's header tree. Unset means "fetch the pinned tag archive";
# ``-DSANAFE_SRC=...`` points the build at an existing checkout instead.
# The tag is the same v2.1.1 the removed submodule's gitlink pointed at
# (e88d3b5134afbc9511d33cb3b68f9a230d1dd555); PyPI publishes no sdist for
# 2.1.1, so the GitHub tag archive plus its SHA256 is the pin.
if(NOT DEFINED SANAFE_SRC)
    include(FetchContent)
    FetchContent_Declare(sanafe_src
        URL      https://github.com/SLAM-Lab/SANA-FE/archive/refs/tags/v2.1.1.tar.gz
        URL_HASH SHA256=96e202b1dd16f56176d930d4bab8b1b6b04ebac389493b309998a7cb2665f2c0
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        # SANA-FE's own CMakeLists builds the whole GPL project; naming a
        # subdirectory that carries none makes MakeAvailable populate only.
        SOURCE_SUBDIR headers-only-no-configure)
    FetchContent_MakeAvailable(sanafe_src)
    set(SANAFE_SRC "${sanafe_src_SOURCE_DIR}/src")
endif()

if(NOT EXISTS "${SANAFE_SRC}/pipeline.hpp")
    message(FATAL_ERROR
        "SANAFE_SRC (${SANAFE_SRC}) does not contain pipeline.hpp. "
        "Drop -DSANAFE_SRC to use the pinned v2.1.1 tag archive, or pass "
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
