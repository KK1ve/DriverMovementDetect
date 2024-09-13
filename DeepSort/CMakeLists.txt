cmake_minimum_required(VERSION 3.14)
project(DeepSort)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -MMD -Wall -Wextra -Winit-self")

# Check C++17 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    add_definitions(-DCOMPILEDWITHC17)
    message(STATUS "Using flag -std=c++17.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

set(EIGEN_VERSION 3.3)
find_package(Eigen3 ${EIGEN_VERSION} QUIET)
if(NOT EIGEN3_FOUND)
    set(BUILD_TESTING OFF CACHE INTERNAL "")
    FetchContent_Declare(eigen
            GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
            GIT_TAG         ${EIGEN_VERSION}
            GIT_SHALLOW     ON)
    FetchContent_MakeAvailable(eigen)
    unset(BUILD_TESTING CACHE)
endif()

find_package(OpenCV REQUIRED)
set(ONNXRUNTIME_DIR "/home/lsy/CPackage/onnxruntime-linux-x64-gpu-1.18.1")

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_DIR)

add_library(${PROJECT_NAME} SHARED
        ${SRC_DIR}
)
target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include "${ONNXRUNTIME_DIR}/include"
)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen ${OpenCV_LIBS} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")

