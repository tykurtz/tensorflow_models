#include <opencv/cv.h>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"

namespace object_detection {

class ObjectDetection {
 public:
  ObjectDetection(const std::string& model_path, bool verbose = true);
  ~ObjectDetection() = default;

  tensorflow::Status run_object_detection(const tensorflow::Tensor& image_tensor, std::vector<tensorflow::Tensor>& network_outputs);

  bool run_object_detection(const cv::Mat& image, cv::Mat& output_image);

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

  tensorflow::Session* session_;
  bool verbose_;
};

}  // namespace object_detection
