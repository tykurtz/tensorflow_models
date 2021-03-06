#include "tensorflow_models/image_io.h"

namespace tensorflow_models {

// Modified from https://github.com/jhjin/tensorflow-cpp/blob/master/app.cc
tensorflow::Status ReadImageFromDisk(const tensorflow::string& file_name,
                                     tensorflow::Tensor& processed_image_tensor) {
  using namespace ::tensorflow::ops;

  std::cout << "Reading file_name " << std::endl;
  auto root = tensorflow::Scope::NewRootScope();

  tensorflow::string input_name = "file_reader";
  tensorflow::string output_name = "resized_image";
  auto file_reader = ReadFile(root.WithOpName(input_name), file_name);

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  std::string file_name_sub = file_name.substr(file_name.find_last_of('.') + 1);

  if (file_name_sub == "png") {
    image_reader = DecodePng(root.WithOpName("image_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (file_name_sub == "gif") {
    image_reader = DecodeGif(root.WithOpName("image_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(
        root.WithOpName("image_reader"), file_reader,
        DecodeJpeg::Channels(wanted_channels));
  }
  auto dims_expander = ExpandDims(root.WithOpName(output_name), image_reader, 0);

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::vector<tensorflow::Tensor> out_tensors;
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, &out_tensors));

  processed_image_tensor = out_tensors.at(0);
  TF_CHECK_OK(session->Close());

  std::cout << "Successfully loaded an image : " << processed_image_tensor.DebugString() << std::endl;

  return tensorflow::Status::OK();
}

tensorflow::Status SaveTensorToDisk(const tensorflow::string& file_name, const tensorflow::Tensor& tensor) {
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
  std::ofstream(file_name, std::ios::binary) << outputs[0].scalar<tensorflow::tstring>()();

  TF_CHECK_OK(session->Close());

  return tensorflow::Status::OK();
}
}  // namespace tensorflow_models
