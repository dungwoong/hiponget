# cmake -S .. -B . -DCMAKE_C_COMPILER="C:/Program Files/AMD/ROCm/6.2/bin/clang.exe" -DCMAKE_CXX_COMPILER="C:/Program Files/AMD/ROCm/6.2/bin/clang++.exe" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER_ROCM_ROOT="C:/Program Files/AMD/ROCm/6.2" -DCMAKE_HIP_ARCHITECTURES=gfx1031
# ninja -C .


cmake_minimum_required(VERSION 3.14)
project(PybindHIP LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)

find_package(HIP REQUIRED)

# 1: could not find torch/extension.h
# 2: __declspec missing
# I swear that also fixed the linking issue idek

# "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\dumpbin.exe" /dependents func.pyd

# need to os.add_dll_directory("C:/Users/kevin/AppData/Local/Programs/Python/Python311/Lib/site-packages/torch/lib") in python

include_directories("C:/Users/kevin/AppData/Roaming/Python/Python311/site-packages/pybind11/include")
include_directories("C:/Users/kevin/AppData/Local/Programs/Python/Python311/include")
include_directories("C:/Program Files/AMD/ROCm/6.2/include")
include_directories(
  "C:/Users/kevin/AppData/Roaming/Python/Python311/site-packages/torch"
  "C:/Users/kevin/AppData/Roaming/Python/Python311/site-packages/torch/include/torch/csrc/api/include"
)
link_directories("C:/Users/kevin/AppData/Local/Programs/Python/Python311/Lib/site-packages/torch/lib")
find_package(Torch REQUIRED PATHS C:/Users/kevin/AppData/Local/Programs/Python/Python311/Lib/site-packages/torch) # 1

add_library(libv1 MODULE libv1.hip)

set_target_properties(libv1 PROPERTIES PREFIX "" SUFFIX ".pyd")

target_link_libraries(libv1
    "C:/Users/kevin/AppData/Local/Programs/Python/Python311/libs/python311.lib"
    torch # 2
    torch_python # 2
)

target_include_directories(libv1 PRIVATE
    ${TORCH_INCLUDE_DIRS} # 1
)