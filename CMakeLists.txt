# CMakeList.txt: DriverActionDetect 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)


project ("DriverActionDetect")

set(ncnn_DIR D:/CPackage/ncnn-20240102-windows-vs2022/x64/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

set(OpenCV_DIR D:/CPackage/opencv/build/x64/vc16/lib)
find_package(OpenCV REQUIRED)

find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/model_zoo)

# 将源代码添加到此项目的可执行文件。
add_executable (DriverActionDetect "DriverActionDetect.cpp" "DriverActionDetect.h" "YoloPose.cpp" "YoloPose.h" "yolov5.cpp" "yolov5.h" "yolov8.cpp" "yolov8.h"  "pose-classification.cpp" "pose-classification.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET DriverActionDetect PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。

# 获取当前目录路径
set(CURRENT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# 将当前目录路径定义为预处理器宏
add_definitions("-DCURRENT_DIR=\"${CURRENT_DIR}\"")

target_link_libraries(DriverActionDetect ncnn ${OpenCV_LIBS} )

if(OpenMP_CXX_FOUND)
    target_link_libraries(DriverActionDetect OpenMP::OpenMP_CXX)
endif()
