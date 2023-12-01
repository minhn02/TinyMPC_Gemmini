# cmake_minimum_required(VERSION 3.0.0)
# 
# set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
# 
# project(TinyMPC VERSION 0.2.0 LANGUAGES CXX)
# 
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_SYSTEM_NAME Generic)
# set(CMAKE_SYSTEM_PROCESSOR riscv)
# 
# set(EIGEN_HAS_CXX11_MATH 0)
# set(STATIC_LINKING TRUE)
# set(CMAKE_EXE_LINKER_FLAGS "-static")
# 
# set(CMAKE_CXX_COMPILER riscv64-unknown-elf-g++)
# set(CMAKE_SYSROOT /home/ubuntu/Desktop/RoSE/soc/sim/firesim/.conda-env/riscv-tools/sysroot)
# 
# #set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# #set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
# #set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# 
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -lm")
# 
# include_directories(include/Eigen)
# 
# #add_subdirectory(src/tinympc)
# #add_subdirectory(examples)
# add_subdirectory(test)


cmake_minimum_required(VERSION 3.0.0)
project(TinyMPC VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(STATIC_LINKING TRUE)
set(CMAKE_EXE_LINKER_FLAGS "-static")

set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)

include_directories(include/Eigen)
include_directories(include/Gemmini)

add_subdirectory(src/tinympc)
add_subdirectory(examples)
add_subdirectory(test)
