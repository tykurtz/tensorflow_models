#include <opencv/cv.h>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"

// Note : Adding ROS as a dependency here is pure laziness on my part. Ideally this class should be ROS-independent
// TODO Remove ros::packagae::getPath and change ROS_DEBUG to passing back debug strings
#include <ros/package.h>
#include <ros/ros.h>

#include <fstream>

namespace object_detection {

class ObjectDetection {
 public:
  ObjectDetection();
  ~ObjectDetection() = default;

  /**
   * TODO Move to utils
   */
  tensorflow::Status readImageFromDisk(const tensorflow::string& file_name, tensorflow::Tensor& processed_image_tensor);

  /**
   * TODO Move to utils
   */
  tensorflow::Status saveTensorToDisk(const tensorflow::string& file_name, const tensorflow::Tensor& tensor);

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
};

}  // namespace object_detection
