#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <memory>

#include "tensorflow_models/object_detection.h"

namespace tensorflow_models {

class ObjectDetectionRosWrapper {
 public:
  ObjectDetectionRosWrapper();
  ~ObjectDetectionRosWrapper() = default;

 protected:
  std::unique_ptr<ObjectDetection> object_detector_;

  ros::NodeHandle node_handle_, private_node_handle_;
  std::shared_ptr<image_transport::ImageTransport> image_transport_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

  void image_cb(const sensor_msgs::ImageConstPtr& rgb_image);
};
}  // namespace tensorflow_models
