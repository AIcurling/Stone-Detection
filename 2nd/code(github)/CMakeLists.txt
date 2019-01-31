cmake_minimum_required(VERSION 2.8)

project(algorithm2)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} /usr/include/)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

add_executable(algorithm2 robot.cpp /usr/include/ueye.h /usr/include/ueye_deprecated.h)

target_link_libraries(algorithm2 ${OpenCV_LIBS} /usr/lib/libueye_api.so)

