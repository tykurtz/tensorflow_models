cmake_minimum_required(VERSION 2.8.3)
project(tensorflow_models)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_geometry
  image_transport
  message_filters
  nodelet
  roscpp
  roslib
  sensor_msgs
)

## System dependencies are found with CMake's conventions

# Taken from https://github.com/PatWie/tensorflow-cmake/blob/master/inference/cc/CMakeLists.txt
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
find_package(TensorFlow REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )

TensorFlow_REQUIRE_C_LIBRARY()
TensorFlow_REQUIRE_SOURCE()

find_package(OpenCV REQUIRED)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES ${PROJECT_NAME}
 CATKIN_DEPENDS cv_bridge image_geometry image_transport message_filters nodelet sensor_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  TensorFlow_DEP
)

add_library(path_estimation src/deeplab.cpp)
add_library(object_detection src/object_detection.cpp)
add_library(image_io src/image_io.cpp)

target_link_libraries(path_estimation PUBLIC TensorFlow_DEP)
target_link_libraries(object_detection PUBLIC ${OpenCV_LIBRARIES} TensorFlow_DEP)
target_link_libraries(image_io PUBLIC ${OpenCV_LIBRARIES} TensorFlow_DEP)

add_library(${PROJECT_NAME}
  src/deeplab_ros_wrapper.cpp
  src/object_detection_ros_wrapper.cpp
)
target_link_libraries(${PROJECT_NAME} PUBLIC ${catkin_LIBRARIES} object_detection image_io path_estimation)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(estimate_path src/nodes/path_estimation.cpp)
target_link_libraries(estimate_path PUBLIC path_estimation image_io)

add_executable(detect_objects src/nodes/detect_objects.cpp)
target_link_libraries(detect_objects PUBLIC object_detection image_io)

add_executable(ros_inference_node src/nodes/ros_inference_node.cpp)
target_link_libraries(ros_inference_node PUBLIC ${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} PRIVATE TensorFlow_DEP)

add_executable(ros_object_detection_node src/nodes/ros_object_detection_node.cpp)
target_link_libraries(ros_object_detection_node PUBLIC ${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} PRIVATE TensorFlow_DEP)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )
