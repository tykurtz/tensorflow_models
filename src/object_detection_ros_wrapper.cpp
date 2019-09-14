#include "tensorflow_models/object_detection_ros_wrapper.h"

namespace tensorflow_models {

ObjectDetectionRosWrapper::ObjectDetectionRosWrapper() : image_transport_(new image_transport::ImageTransport(node_handle_)), private_node_handle_("~") {
  std::string model_path;
  if (!private_node_handle_.getParam("model_path", model_path)) {
    model_path = ros::package::getPath("tensorflow_models") +
                 "/models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/" +
                 "frozen_inference_graph.pb";
    ROS_WARN_STREAM("Could not find model_path on parameter server. Using default : " << model_path);
  }

  object_detector_ = std::make_unique<ObjectDetection>(model_path);

  image_transport::TransportHints hints("raw", ros::TransportHints(), private_node_handle_);
  image_sub_ = image_transport_->subscribe("/camera/color/image_raw", 1, std::bind(&ObjectDetectionRosWrapper::image_cb, this, std::placeholders::_1));
  image_pub_ = image_transport_->advertise("/perception/detected_objects_draw", 1);

  // TODO Add detected object msgs and publish those.
}

void ObjectDetectionRosWrapper::image_cb(const sensor_msgs::ImageConstPtr& rgb_image) {
  cv_bridge::CvImageConstPtr input_image = cv_bridge::toCvShare(rgb_image, "rgb8");

  cv_bridge::CvImage out_msg;
  object_detector_->run_object_detection(input_image->image, out_msg.image);

  out_msg.header = rgb_image->header;
  out_msg.encoding = sensor_msgs::image_encodings::RGB8;

  image_pub_.publish(out_msg.toImageMsg());
}

}  // namespace tensorflow_models
