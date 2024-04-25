project(YOLO)

cmake_minimum_required(VERSION 3.10.2)

set(ncnn_DIR D:/CPackage/ncnn-20240102-windows-vs2022/x64/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

set(OpenCV_DIR D:/CPackage/opencv/build/x64/vc16/lib)
find_package(OpenCV REQUIRED)


add_executable(main main.cpp "yolov5.cpp" "yolov5.h" "yoloface.h" "yoloface.cpp")

# 获取当前目录路径
set(CURRENT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# 将当前目录路径定义为预处理器宏
add_definitions("-DCURRENT_DIR=\"${CURRENT_DIR}\"")

target_link_libraries(main ncnn ${OpenCV_LIBS})
