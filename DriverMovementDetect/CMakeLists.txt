cmake_minimum_required(VERSION 3.5.1)

project("DriverMovementDetect")


#set(OpenCV_DIR "/opt/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_DIR "/home/lsy/CPackage/onnxruntime-linux-x64-gpu-1.18.1")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


aux_source_directory(src SRC_LIST)
aux_source_directory(include INCLUDE_LIST)

# 将源代码添加到此项目的可执行文件。
add_executable (${PROJECT_NAME} ${SRC_LIST} ${INCLUDE_LIST} main.cpp)


target_include_directories(${PROJECT_NAME} PUBLIC "${ONNXRUNTIME_DIR}/include" ${CMAKE_SOURCE_DIR}/DeepSort/include)
target_link_libraries(${PROJECT_NAME} DeepSort "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" ${OpenCV_LIBS} OpenMP::OpenMP_CXX pthread)

