#include <opencv/cv.h>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"

#include <fstream>

namespace deeplab {

class DeepLabv3 {
 public:
  /**
   * Initializes a tensorflow session with the necessary graphs to do semantic segmentation, softmax, and read/save images.
   * TODO Add optional session* as argument
   */
  DeepLabv3(const std::string& model_path, bool verbose = true);
  ~DeepLabv3() = default;

  tensorflow::Status run_semantic_segmentation(const tensorflow::Tensor& image_tensor, tensorflow::Tensor& output_tensor);

  /**
   * Run the image through the network, apply the softmax operation, and return the tensor from the specified class.
   * TODO Add a class argument
   * TODO Add a method that returns vector<tensor>
   * For the ADE20K class list, please see https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv
   */
  tensorflow::Status run_softmax_single_class(const tensorflow::Tensor& image_tensor, tensorflow::Tensor& output_tensor, int class_label=4);

  /**
   * Converts cv Mat -> Tensor, feeds through network, and converts tensor output -> cv::Mat
   *
   */
  bool run_softmax_single_class(const cv::Mat& image, cv::Mat& output_image, int class_label=4);

  /**
   * Processes a cv::Mat object into the right tensorflow input
   */
  tensorflow::Status pre_process_image(const cv::Mat& input_image, tensorflow::Tensor& output_image_tensor);

  /**
   * Runs through the prep network which scales the image to meet the constraint max(height, width) <= 513
   */
  tensorflow::Status pre_process_image(const tensorflow::Tensor& input_image_tensor, tensorflow::Tensor& output_image_tensor);

 protected:

  tensorflow::Status initialize_preprocess_network();
  tensorflow::Status initialize_softmax_network();

  tensorflow::Session* session_;

  std::string SOFTMAX_INPUT_NAME = "softmax_input";
  std::string SOFTMAX_OUTPUT_NAME = "softmax_output";
  std::string PREP_INPUT_NAME = "prep_input";
  std::string PREP_OUTPUT_NAME = "prep_output";
  std::string CLASS_INPUT_NAME = "class_input";

  bool verbose_;
};

}  // namespace deeplab
