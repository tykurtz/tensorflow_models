#include <ros/package.h>
#include <ros/ros.h>
#include "tensorflow_models/object_detection.h"
#include "tensorflow_models/image_io.h"

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "object_detection_node");

  std::string model_path = ros::package::getPath("tensorflow_models") +
                           "/models/ssd_mobilenet_v1_fpn_coco"
                           "frozen_inference_graph.pb";
  object_detection::ObjectDetection object_detector(model_path);

  // Load an image from disk
  tensorflow::Tensor image_tensor;
  TF_CHECK_OK(tensorflow_models::readImageFromDisk(ros::package::getPath("tensorflow_models") + "/test/walmart.jpg", image_tensor));

  std::vector<tensorflow::Tensor> network_output;
  TF_CHECK_OK(object_detector.run_object_detection(image_tensor, network_output));

  // Print the results
  for (const auto& output : network_output) {
    std::cout << output.DebugString() << std::endl;
  }

  // Save tensor to disk
  // TF_CHECK_OK(tensorflow_models::saveTensorToDisk(ros::package::getPath("tensorflow_models") + "/test/object_detection_out.jpg", network_output));

  return 0;
}
