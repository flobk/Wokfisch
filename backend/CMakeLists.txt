cmake_minimum_required(VERSION 3.4...3.18)
project(Wokfisch)

# Force CMake to use the correct Python interpreter
# Adjust path AND version as needed
# Pc
set(Python_EXECUTABLE "........./python.exe")
set(PYBIND11_PYTHON_VERSION 3.10)
# Mac
# set(Python_EXECUTABLE "/usr/local/bin/python3.12")
# set(PYBIND11_PYTHON_VERSION 3.12)
add_subdirectory(pybind11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)

# Output compiled modules directly to build directory (not Release/Debug subdirectories)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# For multi-config generators (MSVC), override per-config directories
foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR})
endforeach()

pybind11_add_module(Board Board.cpp)
pybind11_add_module(Timer Timer.cpp)
pybind11_add_module(Wokfisch Wokfisch.cpp)
## Has some clang++ specifics, need to change before its able to compile on windows
# pybind11_add_module(Board_qNNUE Board_qNNUE.cpp) 

if(MSVC)
    # Only apply optimization flags in Release mode
    target_compile_options(Board PRIVATE $<$<CONFIG:Release>:/O2 /arch:AVX2>)
    target_compile_options(Timer PRIVATE $<$<CONFIG:Release>:/O2 /arch:AVX2>)
    target_compile_options(Wokfisch PRIVATE $<$<CONFIG:Release>:/O2 /arch:AVX2>)
else()
    target_compile_options(Board PRIVATE -O3 -march=native)
    target_compile_options(Timer PRIVATE -O3 -march=native)
    target_compile_options(Wokfisch PRIVATE -O3 -march=native)
endif()
