cmake_minimum_required(VERSION 3.5.1)

project("DriverMovementDetectRelease")
add_definitions("-DSEGMENTATION=0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -MMD -Wall -Wextra -Winit-self")

set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
find_package(FFMPEG REQUIRED)
include_directories(${FFMPEG_INCLUDE_DIRS})
link_directories(${FFMPEG_LIB_DIRS})
# use opencv
set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIRS})
# use libbmrt libbmlib
find_package(libsophon REQUIRED)
include_directories(${LIBSOPHON_INCLUDE_DIRS})
link_directories(${LIBSOPHON_LIB_DIRS})
# intel TBB
find_package(TBB REQUIRED)

aux_source_directory(src SRC_LIST)
aux_source_directory(include INCLUDE_LIST)

# 将源代码添加到此项目的可执行文件。
add_executable (${PROJECT_NAME} ${SRC_LIST} ${INCLUDE_LIST} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE "${CURRENT_DIR}/include")
target_link_libraries(${PROJECT_NAME} PRIVATE ByteTrack TBB::tbb ${FFMPEG_LIBS} ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so})
