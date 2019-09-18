#include "tensorflow_models/object_detection.h"

namespace tensorflow_models {

ObjectDetection::ObjectDetection(const std::string& model_path, bool verbose, tensorflow::Session* session) : verbose_(verbose), session_(session) {
  // Read in the deeplab graph from disk
  tensorflow::GraphDef graph_def;
  TF_CHECK_OK(ReadBinaryProto(tensorflow::Env::Default(), model_path, &graph_def));

  if (session_ == nullptr) {
    tensorflow::SessionOptions options;

    // Initialize a tensorflow session
    options.config.mutable_gpu_options()->set_allow_growth(true);
    TF_CHECK_OK(tensorflow::NewSession(options, &session_));

    TF_CHECK_OK(session_->Create(graph_def));
  } else {
    TF_CHECK_OK(session_->Extend(graph_def));
  }
}

tensorflow::Status ObjectDetection::RunObjectDetection(const tensorflow::Tensor& image_tensor, std::vector<tensorflow::Tensor>& network_outputs) {
  // Setup inputs and outputs:
  std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {
      {"image_tensor:0", image_tensor}};

  std::vector<tensorflow::string> output_tensor_names = {"num_detections:0", "detection_classes:0", "detection_boxes:0", "detection_scores:0"};

  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session_->Run(inputs, output_tensor_names, {}, &network_outputs));

  return tensorflow::Status::OK();
}

bool ObjectDetection::RunObjectDetection(const cv::Mat& image, cv::Mat& output_image) {
  // Convert input image to tensor
  ConvertCvImageToTensor(image, input_tensor_buffer_);

  std::vector<tensorflow::Tensor> output_tensors;
  TF_CHECK_OK(RunObjectDetection(input_tensor_buffer_, output_tensors));
  if (verbose_) {
    for (const auto& output_tensor : output_tensors) {
      std::cout << output_tensor.DebugString(20) << std::endl;
    }
  }

  cv::Mat draw_image;
  DrawDetectionBoxes(output_tensors, input_tensor_buffer_, output_image);
  if (verbose_) {
    std::cout << output_image.cols << " " << output_image.rows << " " << output_image.channels() << std::endl;
  }

  return true;
}

void ObjectDetection::ConvertCvImageToTensor(const cv::Mat& input_image, tensorflow::Tensor& output_image_tensor) {
  int height, width;
  height = input_image.rows;
  width = input_image.cols;

  // If the tensor is already in the right format, don't reallocate memory
  if (output_image_tensor.shape() != tensorflow::TensorShape({1, height, width, 3}) || output_image_tensor.dtype() != tensorflow::DT_UINT8) {
    output_image_tensor = tensorflow::Tensor(tensorflow::DT_UINT8, tensorflow::TensorShape({1, height, width, 3}));
  }
  uint8_t* p = output_image_tensor.flat<uint8_t>().data();

  cv::Mat target_buffer(height, width, CV_8UC3, p);
  input_image.convertTo(target_buffer, CV_8UC3);
}

void ObjectDetection::ConvertTensorToCvImage(tensorflow::Tensor& image_tensor, cv::Mat& cv_image) {
  if (image_tensor.dims() > 3) {
    // If the input image tensor is in batch format, just take the first image of the batch
    cv_image = cv::Mat(image_tensor.dim_size(1), image_tensor.dim_size(2), CV_8UC3, image_tensor.flat<uint8_t>().data());
  } else {
    cv_image = cv::Mat(image_tensor.dim_size(0), image_tensor.dim_size(1), CV_8UC3, image_tensor.flat<uint8_t>().data());
  }
}

void ObjectDetection::DrawDetectionBoxes(const std::vector<tensorflow::Tensor>& network_outputs, tensorflow::Tensor& image_tensor, cv::Mat& draw_image) {
  ConvertTensorToCvImage(image_tensor, draw_image);
  DrawDetectionBoxes(network_outputs, draw_image);
}

void ObjectDetection::DrawDetectionBoxes(const std::vector<tensorflow::Tensor>& network_outputs, cv::Mat& draw_image) {
  int number_of_detections = int(network_outputs.at(0).flat<float>()(0));
  auto detection_classes = network_outputs.at(1).flat<float>();
  auto detection_boxes = network_outputs.at(2).tensor<float, 3>();
  auto detection_scores = network_outputs.at(3).flat<float>();

  for (int i = 0; i < number_of_detections; i++) {
    int detection_class = int(detection_classes(i));
    float y_min, x_min, y_max, x_max;
    y_min = detection_boxes(0, i, 0);
    x_min = detection_boxes(0, i, 1);
    y_max = detection_boxes(0, i, 2);
    x_max = detection_boxes(0, i, 3);

    float detection_score = detection_scores(i);

    if (detection_score < detection_threshold_)
      continue;

    cv::Point upper_left_point(x_min * draw_image.cols, y_min * draw_image.rows);
    cv::Point lower_right_point(x_max * draw_image.cols, y_max * draw_image.rows);

    // TODO Make a better map from class -> color
    cv::rectangle(draw_image, upper_left_point, lower_right_point, cv::Scalar(detection_class, detection_class, detection_class), 8);
  }
}
}  // namespace tensorflow_models
