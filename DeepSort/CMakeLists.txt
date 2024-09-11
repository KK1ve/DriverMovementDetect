cmake_minimum_required(VERSION 2.6)
project(deepsort)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
set(ONNXRUNTIME_DIR "/home/lsy/CPackage/onnxruntime-linux-x64-gpu-1.18.1")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/include INCLUDE_DIR)
add_executable (${PROJECT_NAME} ${SRC_DIR} ${INCLUDE_DIR})
# ===== deepsort =====
target_include_directories(${PROJECT_NAME} PUBLIC "${ONNXRUNTIME_DIR}/include")
target_link_libraries(${PROJECT_NAME} ByteTrack "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" ${OpenCV_LIBS} )
