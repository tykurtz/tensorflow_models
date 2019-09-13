#include <opencv/cv.h>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"

// Note : Adding ROS as a dependency here is pure laziness on my part. Ideally this class should be ROS-independent
// TODO Remove ros::packagae::getPath and change ROS_DEBUG to passing back debug strings
#include <ros/package.h>
#include <ros/ros.h>

#include <fstream>

namespace deeplab {

class DeepLabv3 {
 public:
  /**
   * Initializes a tensorflow session with the necessary graphs to do semantic segmentation, softmax, and read/save images.
   * TODO Add optional session* as argument
   */
  DeepLabv3();
  ~DeepLabv3() = default;

  /**
   * Reads image from disk and processes it to the right format to feed into the network
   */
  tensorflow::Status readImageFromDisk(const tensorflow::string& file_name, tensorflow::Tensor& processed_image_tensor);

  /**
   * TODO Method clean up. Currently it creates a new session to save.
   */
  tensorflow::Status saveTensorToDisk(const tensorflow::string& file_name, const tensorflow::Tensor& tensor);

  tensorflow::Status run_semantic_segmentation(const tensorflow::Tensor& image_tensor, tensorflow::Tensor& output_tensor);

  /**
   * Run the image through the network, apply the softmax operation, and return the tensor from the specified class.
   * TODO Link to labels csv for ADE20K
   * TODO Add a class argument
   * TODO Add a method that returns vector<tensor>
   */
  tensorflow::Status run_softmax_single_class(const tensorflow::Tensor& image_tensor, tensorflow::Tensor& output_tensor);

  /**
   * Converts cv Mat -> Tensor, feeds through network, and converts tensor output -> cv::Mat
   */
  bool run_softmax_single_class(const cv::Mat& image, cv::Mat& output_image);

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
};

}  // namespace deeplab
