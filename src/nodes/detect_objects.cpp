#include <opencv2/imgcodecs.hpp>
#include "tensorflow_models/image_io.h"
#include "tensorflow_models/object_detection.h"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0]
              << " <model path> <input image path> <output image path>"
              << std::endl;
    return EXIT_FAILURE;
  }

  auto model_path = std::string(argv[1]);
  auto input_image_path = std::string(argv[2]);
  auto output_image_path = std::string(argv[3]);

  tensorflow_models::ObjectDetection object_detector(model_path);

  // Load an image from disk
  tensorflow::Tensor image_tensor;
  TF_CHECK_OK(tensorflow_models::ReadImageFromDisk(input_image_path, image_tensor));

  std::vector<tensorflow::Tensor> network_output;
  TF_CHECK_OK(object_detector.RunObjectDetection(image_tensor, network_output));

  // Print the results
  for (const auto& output : network_output) {
    std::cout << output.DebugString(20) << std::endl;
  }

  cv::Mat draw_image;
  object_detector.DrawDetectionBoxes(network_output, image_tensor, draw_image);

  cv::cvtColor(draw_image, draw_image, cv::COLOR_RGB2BGR);
  cv::imwrite(output_image_path, draw_image);

  return 0;
}
