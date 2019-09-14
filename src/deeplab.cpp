#include "tensorflow_models/deeplab.h"

namespace tensorflow_models {

DeepLabv3::DeepLabv3(const std::string& model_path, bool verbose, tensorflow::Session* session) : verbose_(verbose), session_(session) {
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
    // Add the graph to the session
    TF_CHECK_OK(session_->Extend(graph_def));
  }

  TF_CHECK_OK(initialize_preprocess_network());
  TF_CHECK_OK(initialize_softmax_network());
}

tensorflow::Status DeepLabv3::run_semantic_segmentation(const tensorflow::Tensor& image_tensor, tensorflow::Tensor& output_tensor) {
  // Setup inputs and outputs:
  std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {
      {"ImageTensor:0", image_tensor}};

  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session_->Run(inputs, {"SemanticPredictions:0"}, {}, &outputs));

  output_tensor = outputs.at(0);
  if (verbose_)
    std::cout << "Semantic segmentation output " << output_tensor.DebugString() << std::endl;
}

tensorflow::Status DeepLabv3::run_softmax_single_class(const tensorflow::Tensor& image_tensor, tensorflow::Tensor& output_tensor, int class_label) {
  // Setup inputs and outputs:
  std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {
      {"ImageTensor:0", image_tensor}};

  // ResizeBilinear_2:0 is the last layer before applying the arg_max operation
  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session_->Run(inputs, {"ResizeBilinear_2:0"}, {}, &outputs));

  tensorflow::Tensor class_label_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({1}));
  class_label_tensor.vec<int32_t>()(0) = class_label;  // This looks awful. Surely there's a better way...?

  inputs = {
      {SOFTMAX_INPUT_NAME, outputs.at(0)},
      {CLASS_INPUT_NAME, class_label_tensor}};
  TF_CHECK_OK(session_->Run({inputs}, {SOFTMAX_OUTPUT_NAME}, {}, &outputs));

  output_tensor = outputs.at(0);

  return tensorflow::Status::OK();
}

bool DeepLabv3::run_softmax_single_class(const cv::Mat& image, cv::Mat& output_image, int class_label) {
  // Convert input image to tensor
  tensorflow::Tensor input_tensor, output_tensor;
  TF_CHECK_OK(pre_process_image(image, input_tensor));

  TF_CHECK_OK(run_softmax_single_class(input_tensor, output_tensor, class_label));

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
  if (verbose_)
    std::cout << "Processed an image : " << output_image_tensor.DebugString() << std::endl;

  return tensorflow::Status::OK();
}

tensorflow::Status DeepLabv3::pre_process_image(const tensorflow::Tensor& input_image_tensor, tensorflow::Tensor& output_image_tensor) {
  std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {
      {PREP_INPUT_NAME, input_image_tensor}};

  std::vector<tensorflow::Tensor> out_tensors;
  TF_CHECK_OK(session_->Run(inputs, {PREP_OUTPUT_NAME}, {}, &out_tensors));

  output_image_tensor = out_tensors.at(0);

  if (verbose_)
    std::cout << "Successfully processed : " << output_image_tensor.DebugString() << std::endl;

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
  auto class_input_tensor = Placeholder(root.WithOpName(CLASS_INPUT_NAME), tensorflow::DT_INT32);
  auto softmax_tensor = Softmax(root, input_tensor);                                     // Tensor<type: float shape: [1,513,513,151]
  auto squeeze_tensor = Squeeze(root.WithOpName("likelihood_squeeze"), softmax_tensor);  // Tensor<type: float shape: [513,513,151]

  auto const_tensor = Const(root, {0, 0});
  auto concat_tensor = Concat(root.WithOpName("likelihood_concat"), std::initializer_list<tensorflow::Input>({const_tensor, class_input_tensor}), 0);
  auto slice_tensor = Slice(root.WithOpName("likelihood_slice"), squeeze_tensor, concat_tensor, Const(root, {288, 513, 1}));                                                           // Tensor<type: float shape: [288,513,1]
  auto multiply_tensor = Multiply(root.WithOpName("likelihood_multiply"), slice_tensor, Cast(root.WithOpName("likelihood_multiply_cast"), Const(root, 255.0), tensorflow::DT_FLOAT));  // Scale this to 255 for saving an image
  auto cast = Cast(root.WithOpName(SOFTMAX_OUTPUT_NAME), multiply_tensor, tensorflow::DT_UINT8);

  tensorflow::GraphDef softmax_graph;
  TF_CHECK_OK(root.ToGraphDef(&softmax_graph));

  TF_CHECK_OK(session_->Extend(softmax_graph));
  return tensorflow::Status::OK();
}
}  // namespace tensorflow_models
