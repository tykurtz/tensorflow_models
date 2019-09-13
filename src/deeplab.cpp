#include "tensorflow_models/deeplab.h"

namespace deeplab {

DeepLabv3::DeepLabv3() {
  tensorflow::SessionOptions options;

  // Initialize a tensorflow session
  options.config.mutable_gpu_options()->set_allow_growth(true);
  TF_CHECK_OK(tensorflow::NewSession(options, &session_));

  // Read in the deeplab graph from disk
  tensorflow::GraphDef graph_def;
  TF_CHECK_OK(ReadBinaryProto(tensorflow::Env::Default(),
                              ros::package::getPath("tensorflow_models") +
                                  "/models/deeplabv3_mnv2_ade20k_train_2018_12_03/"
                                  "frozen_inference_graph.pb",
                              &graph_def));

  // Add the graph to the session
  TF_CHECK_OK(session_->Create(graph_def));

  TF_CHECK_OK(initialize_preprocess_network());
  TF_CHECK_OK(initialize_softmax_network());
}

tensorflow::Status DeepLabv3::readImageFromDisk(const tensorflow::string& file_name,
                                                tensorflow::Tensor& processed_image_tensor) {
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

tensorflow::Status DeepLabv3::saveTensorToDisk(const tensorflow::string& file_name, const tensorflow::Tensor& tensor) {
  using namespace ::tensorflow::ops;
  auto root = tensorflow::Scope::NewRootScope();
  // auto transpose = Transpose(root, cast, {1, 2, 0});
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

tensorflow::Status DeepLabv3::run_semantic_segmentation(const tensorflow::Tensor& image_tensor, tensorflow::Tensor& output_tensor) {
  // Setup inputs and outputs:
  std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {
      {"ImageTensor:0", image_tensor}};

  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session_->Run(inputs, {"SemanticPredictions:0"}, {}, &outputs));

  output_tensor = outputs.at(0);
  ROS_DEBUG_STREAM("Semantic segmentation output " << output_tensor.DebugString());
}

tensorflow::Status DeepLabv3::run_softmax_single_class(const tensorflow::Tensor& image_tensor, tensorflow::Tensor& output_tensor) {
  // Setup inputs and outputs:
  std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {
      {"ImageTensor:0", image_tensor}};

  // ResizeBilinear_2:0 is the last layer before applying the arg_max operation
  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session_->Run(inputs, {"ResizeBilinear_2:0"}, {}, &outputs));

  inputs = {{SOFTMAX_INPUT_NAME, outputs.at(0)}};
  TF_CHECK_OK(session_->Run({inputs}, {SOFTMAX_OUTPUT_NAME}, {}, &outputs));

  output_tensor = outputs.at(0);

  return tensorflow::Status::OK();
}

bool DeepLabv3::run_softmax_single_class(const cv::Mat& image, cv::Mat& output_image) {
  // Convert input image to tensor
  tensorflow::Tensor input_tensor, output_tensor;
  TF_CHECK_OK(pre_process_image(image, input_tensor));

  TF_CHECK_OK(run_softmax_single_class(input_tensor, output_tensor));

  // Convert output tensor to image
  output_image = cv::Mat(output_tensor.dim_size(0), output_tensor.dim_size(1), CV_8UC1, output_tensor.flat<uint8_t>().data());

  return true;
}

tensorflow::Status DeepLabv3::pre_process_image(const cv::Mat& input_image, tensorflow::Tensor& output_image_tensor) {
  int height, width;
  height = input_image.rows;
  width = input_image.cols;

  tensorflow::Tensor unscaled_tensor(tensorflow::DT_UINT8, tensorflow::TensorShape({height, width, 3}));
  uint8_t* p = unscaled_tensor.flat<uint8_t>().data();

  cv::Mat target_buffer(height, width, CV_8UC3, p);
  input_image.convertTo(target_buffer, CV_8UC3);

  TF_CHECK_OK(pre_process_image(unscaled_tensor, output_image_tensor));
  ROS_DEBUG_STREAM("Processed an image : " << output_image_tensor.DebugString());

  return tensorflow::Status();
}

tensorflow::Status DeepLabv3::pre_process_image(const tensorflow::Tensor& input_image_tensor, tensorflow::Tensor& output_image_tensor) {
  std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {
      {PREP_INPUT_NAME, input_image_tensor}};

  std::vector<tensorflow::Tensor> out_tensors;
  TF_CHECK_OK(session_->Run(inputs, {PREP_OUTPUT_NAME}, {}, &out_tensors));

  output_image_tensor = out_tensors.at(0);

  ROS_DEBUG_STREAM("Successfully processed : " << output_image_tensor.DebugString());

  return tensorflow::Status::OK();
}

tensorflow::Status DeepLabv3::initialize_preprocess_network() {
  // Add preprocess graph
  using namespace ::tensorflow::ops;
  auto root = tensorflow::Scope::NewRootScope();
  auto prep_input_tensor = Placeholder(root.WithOpName(PREP_INPUT_NAME), tensorflow::DT_UINT8);
  auto dims_expander = ExpandDims(root.WithOpName("prep_dims_expand"), prep_input_tensor, 0);

  // TODO Add in logic for getting proper height and width
  auto resized = ResizeBilinear(root.WithOpName("prep_resize"), dims_expander,
                                Const(root.WithOpName("size"), {288, 513}));

  auto casted = Cast(root.WithOpName(PREP_OUTPUT_NAME), resized, tensorflow::DT_UINT8);

  tensorflow::GraphDef preprocess_graph;
  TF_CHECK_OK(root.ToGraphDef(&preprocess_graph));

  TF_CHECK_OK(session_->Extend(preprocess_graph));
  return tensorflow::Status::OK();
}

tensorflow::Status DeepLabv3::initialize_softmax_network() {
  // Add softmax graph
  using namespace ::tensorflow::ops;
  auto root = tensorflow::Scope::NewRootScope();
  auto input_tensor = Placeholder(root.WithOpName(SOFTMAX_INPUT_NAME), tensorflow::DT_FLOAT);
  auto softmax_tensor = Softmax(root, input_tensor); // Tensor<type: float shape: [1,513,513,151]
  auto squeeze_tensor = Squeeze(root.WithOpName("likelihood_squeeze"), softmax_tensor); // Tensor<type: float shape: [513,513,151]
  auto slice_tensor = Slice(root.WithOpName("likelihood_slice"), squeeze_tensor, Const(root, {0, 0, 4}), Const(root, {288, 513, 1})); // Tensor<type: float shape: [288,513,1]
  auto multiply_tensor = Multiply(root.WithOpName("likelihood_multiply"), slice_tensor, Cast(root.WithOpName("likelihood_multiply_cast"), Const(root, 255.0), tensorflow::DT_FLOAT)); // Scale this to 255 for saving an image
  auto cast = Cast(root.WithOpName(SOFTMAX_OUTPUT_NAME), multiply_tensor, tensorflow::DT_UINT8);

  tensorflow::GraphDef softmax_graph;
  TF_CHECK_OK(root.ToGraphDef(&softmax_graph));

  TF_CHECK_OK(session_->Extend(softmax_graph));
  return tensorflow::Status::OK();
}
}  // namespace deeplab
