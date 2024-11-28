cmake_minimum_required(VERSION 3.5.1)

project("RailwayForeignObjectDetection")
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g")
endif ()
set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
find_package(FFMPEG REQUIRED)
# use opencv
set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
# use libbmrt libbmlib
find_package(libsophon REQUIRED)
# intel TBB
find_package(TBB REQUIRED)

aux_source_directory(src SRC_LIST)
aux_source_directory(include INCLUDE_LIST)

file(GLOB SO_FILES "${FFMPEG_LIB_DIRS}/*.so" "${OpenCV_LIB_DIRS}/*.so" "${LIBSOPHON_LIB_DIRS}/*.so")

# 将源代码添加到此项目的可执行文件。
add_executable (${PROJECT_NAME} ${SRC_LIST} ${INCLUDE_LIST} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE "${CURRENT_DIR}/include" ${FFMPEG_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${LIBSOPHON_INCLUDE_DIRS})
#target_link_libraries(${PROJECT_NAME} DeepSort ${FFMPEG_LIBS} ${OpenCV_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} OpenMP::OpenMP_CXX -lpthread)
target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb ${SO_FILES})


