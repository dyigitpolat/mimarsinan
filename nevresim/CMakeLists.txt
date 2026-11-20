# only for cmake --version >= 3.5.1
cmake_minimum_required(VERSION 3.5.1)

# project name
project(nevresim)

# set the C++ standard
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)

# /include as directory
include_directories(include)

# puts all .cpp files inside src to the SOURCES variable
file(GLOB SOURCES src/*.cpp)

# library folders
# add_subdirectory(lib/A_LIB_DIR)
# add_subdirectory(lib/A_LIB_DIR)

# add a switch for debug/release build
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")

# enable all warnings and treat them as errors in a portable way
if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4 /WX)
else ()
    # lots of warnings and all warnings as errors
    # add different optimization flags for debug/release builds
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -Wall -Wextra -pedantic -Werror -O0)
        message("Debug mode -- nevresim")
    elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-Wall -Wextra -pedantic -Werror -O3)
        message("Release mode -- nevresim")
    endif ()
endif ()

# compiles the files defined by SOURCES to generante the executable defined by PROJECT_NAME
add_executable(${PROJECT_NAME} ${SOURCES})

# copy "include/_tests/xor_test/xor_chip_weights.txt" to "build/include/_tests/xor_test/xor_chip_weights.txt"
file(COPY include/_tests/xor_test/xor_chip_weights.txt DESTINATION include/_tests/xor_test)
file(COPY include/_tests/xor_test/xor_inputs/ DESTINATION include/_tests/xor_test/xor_inputs)

# link libraries
# target_link_libraries(${PROJECT_NAME} A_LIB)
# target_link_libraries(${PROJECT_NAME} A_LIB)