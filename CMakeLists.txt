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
find_package(OpenCV REQUIRED)

# Setup tensorflow lite paths
find_package(Threads REQUIRED)
set(TF_LITE_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/source_builds/tensorflow_src
  ${CMAKE_CURRENT_SOURCE_DIR}/source_builds/minimal_build/flatbuffers/include
)
set(TF_LITE_LIBS
  ${CMAKE_CURRENT_SOURCE_DIR}/source_builds/minimal_build/tensorflow-lite/libtensorflow-lite.a
  ${CMAKE_CURRENT_SOURCE_DIR}/source_builds/minimal_build/_deps/flatbuffers-build/libflatbuffers.a
  ${CMAKE_CURRENT_SOURCE_DIR}/source_builds/minimal_build/_deps/ruy-build/libruy.a
  ${CMAKE_CURRENT_SOURCE_DIR}/source_builds/minimal_build/_deps/xnnpack-build/libXNNPACK.a
  ${CMAKE_CURRENT_SOURCE_DIR}/source_builds/minimal_build/pthreadpool/libpthreadpool.a
  ${CMAKE_CURRENT_SOURCE_DIR}/source_builds/minimal_build/cpuinfo/libcpuinfo.a
  ${CMAKE_CURRENT_SOURCE_DIR}/source_builds/minimal_build/clog/libclog.a
  ${CMAKE_CURRENT_SOURCE_DIR}/source_builds/minimal_build/_deps/fft2d-build/libfft2d_fftsg.a
  ${CMAKE_CURRENT_SOURCE_DIR}/source_builds/minimal_build/_deps/fft2d-build/libfft2d_fftsg2d.a
  ${CMAKE_CURRENT_SOURCE_DIR}/source_builds/minimal_build/_deps/farmhash-build/libfarmhash.a
  ${CMAKE_DL_LIBS}
  Threads::Threads
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES ${PROJECT_NAME}
 CATKIN_DEPENDS cv_bridge image_geometry image_transport message_filters nodelet sensor_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${TF_LITE_INCLUDE_DIRS}
  # ${catkin_INCLUDE_DIRS}
  # ${OpenCV_INCLUDE_DIRS}
)
add_executable(tf_lite_node src/nodes/tf_lite_node.cpp)
target_link_libraries(tf_lite_node ${TF_LITE_LIBS})


## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/deeplab.cpp
#   src/deeplab_ros_wrapper.cpp
#   src/object_detection.cpp
#   src/object_detection_ros_wrapper.cpp
#   src/image_io.cpp
# )
# target_link_libraries(${PROJECT_NAME} PUBLIC ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} PRIVATE TensorFlow_DEP)

# ## Declare a C++ executable
# ## With catkin_make all packages are built within a single CMake context
# ## The recommended prefix ensures that target names across packages don't collide
# add_executable(inference_node src/nodes/inference_node.cpp)
# target_link_libraries(inference_node PUBLIC ${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} PRIVATE TensorFlow_DEP)

# add_executable(object_detection_node src/nodes/object_detection_node.cpp)
# target_link_libraries(object_detection_node PUBLIC ${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} PRIVATE TensorFlow_DEP)

# add_executable(ros_inference_node src/nodes/ros_inference_node.cpp)
# target_link_libraries(ros_inference_node PUBLIC ${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} PRIVATE TensorFlow_DEP)

# add_executable(ros_object_detection_node src/nodes/ros_object_detection_node.cpp)
# target_link_libraries(ros_object_detection_node PUBLIC ${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} PRIVATE TensorFlow_DEP)

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
