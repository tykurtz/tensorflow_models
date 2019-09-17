#include <ros/package.h>
#include <ros/ros.h>
#include <opencv2/imgcodecs.hpp>
#include "tensorflow_models/image_io.h"
#include "tensorflow_models/object_detection.h"

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "object_detection_node");

  std::string model_path = ros::package::getPath("tensorflow_models") +
                           "/models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/" +
                           "frozen_inference_graph.pb";
  tensorflow_models::ObjectDetection object_detector(model_path);

  // Load an image from disk
  tensorflow::Tensor image_tensor;
  TF_CHECK_OK(tensorflow_models::readImageFromDisk(ros::package::getPath("tensorflow_models") + "/test/walmart with more people.jpeg", image_tensor));

  std::vector<tensorflow::Tensor> network_output;
  TF_CHECK_OK(object_detector.run_object_detection(image_tensor, network_output));

  // Print the results
  for (const auto& output : network_output) {
    std::cout << output.DebugString(20) << std::endl;
  }

  cv::Mat draw_image;
  object_detector.draw_detection_boxes(network_output, image_tensor, draw_image);

  cv::cvtColor(draw_image, draw_image, cv::COLOR_RGB2BGR);
  cv::imwrite("/home/pv20bot/output.jpg", draw_image);

  return 0;
}
