cmake_minimum_required(VERSION 3.5.1)

project("RailwayForeignObjectDetection")


#set(OpenCV_DIR "/opt/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_DIR "/home/lsy/CPackage/onnxruntime-linux-x64-gpu-1.18.1")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_DIR)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/include INCLUDE_DIR)

# 将源代码添加到此项目的可执行文件。
add_executable (${PROJECT_NAME} ${SRC_LIST} main.cpp)


target_include_directories(${PROJECT_NAME} PUBLIC "${ONNXRUNTIME_DIR}/include" ${INCLUDE_LIST})
target_link_libraries(${PROJECT_NAME} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" ${OpenCV_LIBS} OpenMP::OpenMP_CXX pthread)

