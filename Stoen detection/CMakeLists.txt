cmake_minimum_required(VERSION 2.8)

project(Skip)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} /usr/include)

add_executable(Skip skip.cpp)

target_link_libraries(Skip ${OpenCV_LIBS})

