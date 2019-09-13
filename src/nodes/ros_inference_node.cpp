#include <ros/ros.h>
#include "tensorflow_models/deeplab_ros_wrapper.h"

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "semantic_segmentation_node");

  deeplab::DeepLabv3RosWrapper deep_lab_ros;

  ros::spin();
  return 0;
}
