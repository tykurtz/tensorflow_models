#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <memory>

#include "tensorflow_models/deeplab.h"

namespace tensorflow_models {

class DeepLabv3RosWrapper {
 public:
  DeepLabv3RosWrapper();
  ~DeepLabv3RosWrapper() = default;

 protected:
  std::unique_ptr<DeepLabv3> deeplab_;

  ros::NodeHandle node_handle_, private_node_handle_;
  std::shared_ptr<image_transport::ImageTransport> image_transport_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

  void ImageCallback(const sensor_msgs::ImageConstPtr& rgb_image);
};
}  // namespace tensorflow_models
