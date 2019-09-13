#include "tensorflow_models/object_detection.h"

namespace object_detection {

ObjectDetection::ObjectDetection(const std::string& model_path, bool verbose) : verbose_(verbose) {
  tensorflow::SessionOptions options;

  // Initialize a tensorflow session
  options.config.mutable_gpu_options()->set_allow_growth(true);
  TF_CHECK_OK(tensorflow::NewSession(options, &session_));

  // Read in the deeplab graph from disk
  tensorflow::GraphDef graph_def;
  TF_CHECK_OK(ReadBinaryProto(tensorflow::Env::Default(), model_path, &graph_def));

  // Add the graph to the session
  TF_CHECK_OK(session_->Create(graph_def));

  TF_CHECK_OK(initialize_preprocess_network());
}

tensorflow::Status ObjectDetection::run_object_detection(const tensorflow::Tensor& image_tensor, std::vector<tensorflow::Tensor>& network_outputs) {
  // Setup inputs and outputs:
  std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {
      {"image_tensor:0", image_tensor}};

  std::vector<tensorflow::string> output_tensor_names = {"num_detections", "detection_classes", "detection_boxes", "detection_scores"};

  std::vector<tensorflow::Tensor>
      outputs;
  TF_CHECK_OK(session_->Run(inputs, output_tensor_names, {}, &network_outputs));
}

bool ObjectDetection::run_object_detection(const cv::Mat& image, cv::Mat& output_image) {
  // TODO
}

tensorflow::Status ObjectDetection::pre_process_image(const cv::Mat& input_image, tensorflow::Tensor& output_image_tensor) {
  // TODO
}

tensorflow::Status ObjectDetection::pre_process_image(const tensorflow::Tensor& input_image_tensor, tensorflow::Tensor& output_image_tensor) {
  // TODO
}

tensorflow::Status ObjectDetection::initialize_preprocess_network() {
  // TODO
}
}  // namespace object_detection
