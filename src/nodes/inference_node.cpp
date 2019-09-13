#include <ros/package.h>
#include <ros/ros.h>
#include "tensorflow_models/deeplab.h"
#include "tensorflow_models/image_io.h"

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "semantic_segmentation_node");

  std::string model_path = ros::package::getPath("tensorflow_models") +
                           "/models/deeplabv3_mnv2_ade20k_train_2018_12_03/"
                           "frozen_inference_graph.pb";

  deeplab::DeepLabv3 deep_lab(model_path);

  // Load an image from disk
  tensorflow::Tensor image_tensor, processed_image_tensor;
  TF_CHECK_OK(tensorflow_models::readImageFromDisk(ros::package::getPath("tensorflow_models") + "/test/walmart.jpg", image_tensor));

  // Prep it for the network
  TF_CHECK_OK(deep_lab.pre_process_image(image_tensor, processed_image_tensor));

  tensorflow::Tensor output;
  TF_CHECK_OK(deep_lab.run_semantic_segmentation(processed_image_tensor, output));

  // Print the results
  std::cout << output.DebugString() << std::endl;  // Tensor<type: int64 shape: [1,288,513] values: [[6 6 6...]]...>

  TF_CHECK_OK(deep_lab.run_softmax_single_class(processed_image_tensor, output));

  std::cout << output.DebugString() << std::endl;

  // Save tensor to disk
  TF_CHECK_OK(tensorflow_models::saveTensorToDisk(ros::package::getPath("tensorflow_models") + "/test/output.jpg", output));

  return 0;
}
