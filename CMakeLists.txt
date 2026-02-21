cmake_minimum_required(VERSION 3.16)
project(KF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(ControlMath SHARED ${SOURCES})
target_include_directories(ControlMath PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(KF_test main.cpp)
target_link_libraries(KF_test PRIVATE ControlMath)
