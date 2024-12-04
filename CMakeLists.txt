# CMakeList.txt: DriverActionDetect 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)


project ("6A")

# 获取当前目录路径
set(CURRENT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# 将当前目录路径定义为预处理器宏
add_definitions("-DCURRENT_DIR=\"${CURRENT_DIR}\"")
add_definitions(-DFEATURE_VECTOR_DIM=512)
add_subdirectory("DeepSort")
add_subdirectory("ByteTrack")
add_subdirectory("Test")
add_subdirectory("DriverMovementDetect")
add_subdirectory("RailwayForeignObjectDetection" )
add_subdirectory("RailwayForeignObjectDetectionRelease" )
add_subdirectory("DriverMovementDetectRelease" )


