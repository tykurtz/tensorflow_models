#include <opencv/cv.h>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow_models {

/**
 * Enables running semantic segmentation and softmax on pretrained networks from https://github.com/tensorflow/models/tree/master/research/deeplab
 */
class DeepLabv3 {
 public:
  /**
   * Initializes a tensorflow session with the necessary graphs to do semantic segmentation, softmax, and read/save images.
   */
  DeepLabv3(const std::string& model_path, bool verbose = true, tensorflow::Session* session = nullptr);
  ~DeepLabv3() = default;

  tensorflow::Status RunSemanticSegmentation(const tensorflow::Tensor& image_tensor, tensorflow::Tensor& output_tensor);

  /**
   * Run the image through the network, apply the softmax operation, and return the tensor from the specified class.
   *
   * For the ADE20K class list, please see https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv
   */
  tensorflow::Status RunSoftmaxSingleClass(const tensorflow::Tensor& image_tensor, tensorflow::Tensor& output_tensor, int class_label = 4);

  /**
   * Converts cv Mat -> Tensor, feeds through network, and converts tensor output -> cv::Mat
   */
  bool RunSoftmaxSingleClass(const cv::Mat& image, cv::Mat& output_image, int class_label = 4);

  /**
   * Processes a cv::Mat object into the right tensorflow input
   */
  tensorflow::Status PreprocessImage(const cv::Mat& input_image, tensorflow::Tensor& output_image_tensor);

  /**
   * Runs through the prep network which scales the image to meet the constraint max(height, width) <= 513
   */
  tensorflow::Status PreprocessImage(const tensorflow::Tensor& input_image_tensor, tensorflow::Tensor& output_image_tensor);

 protected:
  void FindCorrectHeightWidth(const int image_height, const int image_width, int& resized_height, int& resized_width);

  tensorflow::Status InitializePreprocessNetwork();
  tensorflow::Status InitializeSoftmaxNetwork();

  tensorflow::Session* session_;

  std::string SOFTMAX_INPUT_NAME = "softmax_input";
  std::string SOFTMAX_OUTPUT_NAME = "softmax_output";
  std::string PREP_INPUT_NAME = "prep_input";
  std::string PREP_OUTPUT_NAME = "prep_output";
  std::string CLASS_INPUT_NAME = "class_input";

  bool verbose_;
};

}  // namespace tensorflow_models
