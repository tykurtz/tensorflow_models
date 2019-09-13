#include <ros/ros.h>
#include "tensorflow_models/object_detection.h"

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "object_detection_node");

  object_detection::ObjectDetection object_detector;

  // Load an image from disk
  tensorflow::Tensor image_tensor;
  TF_CHECK_OK(object_detector.readImageFromDisk(ros::package::getPath("tensorflow_models") + "/models/walmart.jpg", image_tensor));

  std::vector<tensorflow::Tensor> network_output;
  TF_CHECK_OK(object_detector.run_object_detection(image_tensor, network_output));

  // Print the results
  for (const auto& output : network_output) {
    std::cout << output.DebugString() << std::endl;
  }

  // Save tensor to disk
  // TF_CHECK_OK(object_detector.saveTensorToDisk(ros::package::getPath("tensorflow_models") + "object_detection_out.jpg", network_output));

  return 0;
}
