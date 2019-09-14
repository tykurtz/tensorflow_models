#include <ros/ros.h>
#include "tensorflow_models/object_detection_ros_wrapper.h"

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "object_detection_node");

  tensorflow_models::ObjectDetectionRosWrapper object_detector;

  ros::spin();
  return 0;
}
