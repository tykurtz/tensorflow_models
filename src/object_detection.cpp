#include "tensorflow_models/object_detection.h"

namespace object_detection {

ObjectDetection::ObjectDetection() {
  tensorflow::SessionOptions options;

  // Initialize a tensorflow session
  options.config.mutable_gpu_options()->set_allow_growth(true);
  TF_CHECK_OK(tensorflow::NewSession(options, &session_));

  // Read in the deeplab graph from disk
  tensorflow::GraphDef graph_def;
  TF_CHECK_OK(ReadBinaryProto(tensorflow::Env::Default(),
                              ros::package::getPath("tensorflow_models") +
                                  "/models/ssd_mobilenet_v1_fpn_coco"
                                  "frozen_inference_graph.pb",
                              &graph_def));

  // Add the graph to the session
  TF_CHECK_OK(session_->Create(graph_def));

  TF_CHECK_OK(initialize_preprocess_network());
}

tensorflow::Status ObjectDetection::readImageFromDisk(const tensorflow::string& file_name, tensorflow::Tensor& processed_image_tensor) {
  using namespace ::tensorflow::ops;

  ROS_DEBUG("Reading file_name ");
  auto root = tensorflow::Scope::NewRootScope();

  tensorflow::string input_name = "file_reader";
  tensorflow::string output_name = "resized_image";
  auto file_reader = ReadFile(root.WithOpName(input_name), file_name);

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  std::string file_name_sub = file_name.substr(file_name.find_last_of('.') + 1);

  if (file_name_sub == "png") {
    image_reader = DecodePng(root.WithOpName(output_name), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (file_name_sub == "gif") {
    image_reader = DecodeGif(root.WithOpName(output_name), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(
        root.WithOpName(output_name), file_reader,
        DecodeJpeg::Channels(wanted_channels));
  }

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::vector<tensorflow::Tensor> out_tensors;
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, &out_tensors));

  tensorflow::Tensor image_tensor = out_tensors.at(0);

  ROS_DEBUG_STREAM("Successfully loaded an image : " << image_tensor.DebugString());
  TF_CHECK_OK(pre_process_image(image_tensor, processed_image_tensor));

  return tensorflow::Status::OK();
}

tensorflow::Status ObjectDetection::saveTensorToDisk(const tensorflow::string& file_name, const tensorflow::Tensor& tensor) {
  using namespace ::tensorflow::ops;
  auto root = tensorflow::Scope::NewRootScope();
  auto encode_jpg = EncodeJpeg(root.WithOpName("encode"), tensor);

  tensorflow::GraphDef graph;
  TF_CHECK_OK(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

  TF_CHECK_OK(session->Create(graph));

  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session->Run({}, {"encode"}, {}, &outputs));
  std::ofstream(file_name, std::ios::binary) << outputs[0].scalar<std::string>()();

  return tensorflow::Status::OK();
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
