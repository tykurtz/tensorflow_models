#include "tensorflow_models/deeplab.h"
#include "tensorflow_models/image_io.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <model path> <input image path> <output image path>" << std::endl;
    return EXIT_FAILURE;
  }

  auto model_path = std::string(argv[1]);
  auto input_image_path = std::string(argv[2]);
  auto output_image_path = std::string(argv[3]);

  tensorflow_models::DeepLabv3 deep_lab(model_path);

  // Load an image from disk
  tensorflow::Tensor image_tensor, processed_image_tensor;
  TF_CHECK_OK(tensorflow_models::ReadImageFromDisk(input_image_path, image_tensor));

  // Prep it for the network
  TF_CHECK_OK(deep_lab.PreprocessImage(image_tensor, processed_image_tensor));

  tensorflow::Tensor output;
  TF_CHECK_OK(deep_lab.RunSemanticSegmentation(processed_image_tensor, output));

  // Print the results
  std::cout << output.DebugString() << std::endl;  // Tensor<type: int64 shape: [1,288,513] values: [[6 6 6...]]...>

  TF_CHECK_OK(deep_lab.RunSoftmaxSingleClass(processed_image_tensor, output));

  std::cout << output.DebugString() << std::endl;

  // Save tensor to disk
  TF_CHECK_OK(tensorflow_models::SaveTensorToDisk(output_image_path, output));

  return 0;
}
