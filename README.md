# Tensorflow Models in ROS
This repository is a C++ ROS wrapper around several different networks pulled from [tensorflow/models](https://github.com/tensorflow/models)

There are currently four target functionalities from this repository.

1. 2D bounding box object detectors from [tensorflow/models/research/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)
2. Semantic segmentation using [deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)
3. 2D "free space" estimation using a modified deeplab network.
4. Planned - Free space estimation using mean-variance estimators as model ensembles (see https://arxiv.org/pdf/1612.01474.pdf)

## 2D Bounding box object detectors
Models taken from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

## Semantic segmentation using deeplab
Models taken from https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md

## Free-space estimation in 2D for indoor robots
ADE20K is a dataset that contains many examples of indoor images with segmentation labels. The approach here was to modify DeepLabv3 trained on ADE20K by taking the linear outputs before the ArgMax layer, and applying a softmax operation. The intention is that by selecting the floor class layer from the softmax output, this gives a probability estimate of drivable terrain.

NOTE: It's important to use this as a starting point and to fine-tune the model for your target environment. While ADE20K has the advantage of containing training examples of indoor scenes, the labeling policy isn't appropriate for this task in particular (see https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv for list of classes). For example, labeling policies that I would find to be more robust are cityscapes including dynamic and static object classes, and wilddash including a 'void' class denoting invalid sensor input. ADE20K does not have a 'catch all' type label for generic objects nor a void label for sensor failures. Going through the dataset, one can see many examples of objects on the floor being included in the floor class.

## Free-space estimation using mean-variance estimators as model ensembles
TODO

# Roadmap
* Add example images to documentation
* Remove build dependency on tensorflow python wheel
* Add Dockerfile, docker image, and run commands
* Nodelet implementation
* Add separate launch file for semantic segmentation
* Add script to pull models instead of saving on github (Squash after doing this)
* Add MVE model ensemble
* Better colormap for object detection draw
* Colormap output for semantic segmentation
* Semantic segmentation output

# Getting started

## Building
Follow https://www.tensorflow.org/install/source to setup CUDA, install bazel, and build tensorflow from source. Be sure to verify the correct kernel, tensorflow, bazel, CUDA, and cudnn versions. You can reference the dockerfile for more details on building from source.

```sh
bazel build -c opt tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip
bazel build -c opt -j 8 //tensorflow:libtensorflow_cc.so
bazel build -c opt -j 8 //tensorflow:libtensorflow.so

pip install /tmp/pip/tensorflow-2.4.0-cp36-cp36m-linux_x86_64.whl

catkin build --this --cmake-args -DTensorFlow_SOURCE_DIR:PATH=$(pwd)/source_builds/tensorflow -DTensorFlow_BUILD_DIR:PATH=$(pwd)/source_builds/tensorflow/bazel-bin/tensorflow
```

## Testing images

```sh
# Test estimation of driveable terrain on a single image
rosrun tensorflow_models estimate_path \
    $(rospack find tensorflow_models)/models/deeplabv3_mnv2_ade20k_train_2018_12_03/frozen_inference_graph.pb \
    $(rospack find tensorflow_models)/test/walmart.jpg \
    $(rospack find tensorflow_models)/test/output.jpg

# Test object detection on a single image
rosrun tensorflow_models detect_objects \
    $(rospack find tensorflow_models)/models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb \
    $(rospack find tensorflow_models)/test/walmart_with_people.jpg \
    $(rospack find tensorflow_models)/test/output_object_detection.jpg
```

## Docker
Using docker with GPU support will be the recommended approach, but is currently WIP.

# Motivation
The primary goal is efficiency with the target language being C++. This gives access to image_transport and nodelets, which cuts down on unnecessary serializing/deserializing of images. Another goal was to target tensorflow/models,

https://github.com/tensorflow/models/tree/master/research/object_detection for bounding box detection.
https://github.com/tensorflow/models/tree/master/research/deeplab for segmentation, and for 2D 'free space' perception.

This comes with the caveat that you need to build tensorflow from source, see https://github.com/tradr-project/tensorflow_ros_cpp#c-abi-difference-problems

Care was taken to minimize dependencies into deeplab and object_detection so ROS agnostic projects might benefit from these classes.

## Related work
https://github.com/leggedrobotics/darknet_ros
- C++ and actively maintained. Very good starting point for bounding box detection. Only supports YOLO v3, no nodelet support

https://github.com/UbiquityRobotics/dnn_detect
- C++, uses opencv DNN interface. Another good starting point, example uses SSD with a mobilenet backbone. Utilizes image_transport, no nodelet support

https://github.com/osrf/tensorflow_object_detector
- Python only. Followed a similar approach by targetting a (now outdated) fork of tensorflow/models.

### Tensorflow C++ integration with ROS
https://github.com/tradr-project/tensorflow_ros_cpp
https://github.com/tradr-project/tensorflow_ros_test
- Catkin-friendly C++ bindings for tensorflow.

https://github.com/PatWie/tensorflow-cmake
- Source of the FindTensorflow.cmake file in this project
