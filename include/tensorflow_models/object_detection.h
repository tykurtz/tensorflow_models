#include <opencv/cv.h>
#include <opencv2/imgproc.hpp>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow_models {

class ObjectDetection {
 public:
  ObjectDetection(const std::string& model_path, bool verbose = true, tensorflow::Session* session = nullptr);
  ~ObjectDetection() = default;

  /**
   * Feeds the image to the network, and returns 4 tensors.
   * Label list https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt
   *
   * Example output:
   * num_detections    = network_outputs[0]    Tensor<type: float shape: [1] values: 3>
   * detection_classes = network_outputs[1]    Tensor<type: float shape: [1,100] values: [82 82 47 1 1 1 1 ...]...>
   * detection_boxes   = network_outputs[2]    Tensor<type: float shape: [1,100,4] values: [[0.01099509 0.501422465 0.691698432 0.85768348][0.0397521853 0.548089147 0.912597954 0.958036065][0.927556217 0.776757598 0.999857843 0.830113888][0 0 0 0][0 0 0 0]]...>
   * detection_scores  = network_outputs[3]    Tensor<type: float shape: [1,100] values: [0.411729336 0.405611962 0.308594882 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0...]...>
   */
  tensorflow::Status RunObjectDetection(const tensorflow::Tensor& image_tensor, std::vector<tensorflow::Tensor>& network_outputs);

  bool RunObjectDetection(const cv::Mat& image, cv::Mat& output_image);

  /**
   * Converts a cv::Mat to a tensor with dimensions {1, height, width, 3}
   */
  void ConvertCvImageToTensor(const cv::Mat& input_image, tensorflow::Tensor& output_image_tensor);

  /**
   * Does a zero-copy conversion from tensorflow::Tensor to cv::Mat.
   * WARNING: This ties cv_image to the lifetime of image_tensor. Undefined behavior is probably going to occur if cv_image is used after image_tensor goes out of scope. A member variable input_tensor_buffer_ is used to make this persistent
   */
  void ConvertTensorToCvImage(tensorflow::Tensor& image_tensor, cv::Mat& cv_image);

  /**
   * Given the network outputs and an image tensor, draw the bounding box detections on a cv Mat
   * WARNING: This method directly modifies image_tensor. As the memory is modified in place, draw_image is tied to the lifetime of image_tensor
   */
  void DrawDetectionBoxes(const std::vector<tensorflow::Tensor>& network_outputs, tensorflow::Tensor& image_tensor, cv::Mat& draw_image);

  /**
   * Given a populated cv Mat, draw the bounding boxes on them
   */
  void DrawDetectionBoxes(const std::vector<tensorflow::Tensor>& network_outputs, cv::Mat& draw_image);

  void SetDetectionThreshold(float threshold) {
    detection_threshold_ = threshold;
  }

 protected:
  tensorflow::Session* session_;
  bool verbose_;

  float detection_threshold_ = .45;

  tensorflow::Tensor input_tensor_buffer_;
};

}  // namespace tensorflow_models
