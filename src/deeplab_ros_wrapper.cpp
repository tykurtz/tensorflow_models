#include "tensorflow_models/deeplab_ros_wrapper.h"

namespace deeplab {

DeepLabv3RosWrapper::DeepLabv3RosWrapper() : image_transport_(new image_transport::ImageTransport(node_handle_)), private_node_handle_("~") {
  std::string model_path;
  if (!private_node_handle_.getParam("model_path", model_path)) {
    model_path = ros::package::getPath("tensorflow_models") +
                 "/models/deeplabv3_mnv2_ade20k_train_2018_12_03/"
                 "frozen_inference_graph.pb";
    ROS_WARN_STREAM("Could not find model_path on parameter server. Using default : " << model_path);
  }

  deeplab_ = std::make_unique<DeepLabv3>(model_path);

  image_transport::TransportHints hints("raw", ros::TransportHints(), private_node_handle_);
  image_sub_ = image_transport_->subscribe("/camera/color/image_raw", 1, std::bind(&DeepLabv3RosWrapper::image_cb, this, std::placeholders::_1));
  image_pub_ = image_transport_->advertise("/perception/floor_likelihood", 1);
}

void DeepLabv3RosWrapper::image_cb(const sensor_msgs::ImageConstPtr& rgb_image) {
  cv_bridge::CvImageConstPtr input_image = cv_bridge::toCvShare(rgb_image, "rgb8");

  cv_bridge::CvImage out_msg;
  deeplab_->run_softmax_single_class(input_image->image, out_msg.image);

  out_msg.header = rgb_image->header;
  out_msg.encoding = sensor_msgs::image_encodings::MONO8;

  image_pub_.publish(out_msg.toImageMsg());
}

}  // namespace deeplab
