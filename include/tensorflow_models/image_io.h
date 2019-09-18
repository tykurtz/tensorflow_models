#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"

#include <fstream>

namespace tensorflow_models {

tensorflow::Status ReadImageFromDisk(const tensorflow::string& file_name, tensorflow::Tensor& processed_image_tensor);

tensorflow::Status SaveTensorToDisk(const tensorflow::string& file_name, const tensorflow::Tensor& tensor);

}  // namespace tensorflow_models
