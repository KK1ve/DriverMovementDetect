cmake_minimum_required(VERSION 3.5.1)

project("Test")

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
find_package(TBB REQUIRED)

# 将源代码添加到此项目的可执行文件。
add_executable (${PROJECT_NAME} test.cpp test.h)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/DeepSort/include)
target_link_libraries(${PROJECT_NAME} TBB::tbb OpenMP::OpenMP_CXX pthread)