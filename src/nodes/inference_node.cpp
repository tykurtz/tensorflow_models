#include "tensorflow_models/deeplab.h"
#include <ros/ros.h>

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "semantic_segmentation_node");

  deeplab::DeepLabv3 deep_lab;

  // Load an image from disk
  tensorflow::Tensor image_tensor;
  TF_CHECK_OK(deep_lab.readImageFromDisk(ros::package::getPath("tensorflow_models") + "/models/walmart.jpg", image_tensor));

  tensorflow::Tensor output;
  TF_CHECK_OK(deep_lab.run_semantic_segmentation(image_tensor, output));

  // Print the results
  std::cout << output.DebugString() << std::endl;  // Tensor<type: int64 shape: [1,288,513] values: [[6 6 6...]]...>

  TF_CHECK_OK(deep_lab.run_softmax_single_class(image_tensor, output));

  std::cout << output.DebugString() << std::endl;

  // Save tensor to disk
  TF_CHECK_OK(deep_lab.saveTensorToDisk(ros::package::getPath("tensorflow_models") + "/output.jpg", output));

  return 0;
}
