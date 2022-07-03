cmake_minimum_required(VERSION 3.0.2)
project(depth_segmenter)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  sensor_msgs
  std_msgs
)

find_package(OpenCV REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

message(STATUS "OpenCV = ${OpenCV}")

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}  
)

catkin_package()

add_executable(depth_segmenter
  src/main.cpp
)

target_link_libraries(depth_segmenter ${catkin_LIBRARIES} ${OpenCV_LIBS})