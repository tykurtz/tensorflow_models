# Tensorflow Models in ROS
This repository is a C++ ROS wrapper around several different networks pulled from [tensorflow/models](https://github.com/tensorflow/models)

There are currently four target functionalities from this repository.

1. 2D bounding box object detectors from [tensorflow/models/research/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)
2. Semantic segmentation using [deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)
3. 2D "free space" estimation using a modified deeplab network.
4. Semantic segmentation using mean-variance estimators as model ensembles (see https://arxiv.org/pdf/1612.01474.pdf)

TODO Add bounding box object detector

TODO Nodelet implementation

TODO Add separate launch file for semantic segmentation

TODO Add script to pull models instead of saving on github

TODO Add MVE model ensemble

## 2D Bounding box object detectors

## Semantic segmentation using deeplab

## Free-space estimation in 2D for indoor robots
ADE20K is a dataset that contains many examples of indoor images with segmentation labels. The approach here was to modify DeepLabv3 trained on ADE20K by taking the linear outputs before the ArgMax layer, and applying a softmax operation. By selecting the floor class layer from the softmax output, this gives a probability estimate of drivable terrain.

NOTE: It's important to use this as a starting point and to fine-tune the model for your target environment. While ADE20K has the advantage of containing many indoor scenes, the labeling policy isn't appropriate for this task in particular (see https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv for list of classes). For example, labeling policies that I would find to be more robust are cityscapes including dynamic and static object classes, and wilddash including a 'void' class denoting invalid sensor input. ADE20K does not have a 'catch all' type label for generic objects nor a void label for sensor failures. Going through the dataset, one can see many examples of objects on the floor being included in the floor class.

## Free-space estimation using mean-variance estimators as model ensembles

# Getting started
## Docker

Using docker with GPU support is the recommended approach, due to possible complications with a source build of tensorflow.
TODO Add Dockerfile, docker image, and run commands


## From source
### Dependencies
Requires a source build of tensorflow.

# Example installation of tensorflow from source
```sh
# TODO Install Bazel 0.24.1

git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r1.14

# TODO Investigate some config options.

bazel build -c opt --copt=-mavx --config=cuda \
    --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
    tensorflow/tools/pip_package:build_pip_package && \
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip

bazel build -c opt --copt=-mavx --config=cuda \
    --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
    //tensorflow:libtensorflow_cc.so

bazel build -c opt --copt=-mavx --config=cuda \
    --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
    //tensorflow:libtensorflow.so


# TODO Pip install from /tmp/pip
```

### Environment setup
Some environmental variables need to be setup to help FindTensorflow.cmake

An example is given below
```sh
export TENSORFLOW_BUILD_DIR=/home/pv20bot/coding/source_builds/tensorflow_build
export TENSORFLOW_SOURCE_DIR=/home/pv20bot/coding/source_builds/tensorflow
```

# Motivation
The primary goal is efficiency with the target language being C++. This gives us access to image_transport and nodelets, which cuts down on unnecessary serializing/deserializing of images. Another goal is flexibility and compatability by targeting tensorflow/models,

https://github.com/tensorflow/models/tree/master/research/object_detection for bounding box detection.
https://github.com/tensorflow/models/tree/master/research/deeplab for segmentation, and for 2D 'free space' perception.

This comes with the caveat that you need to build tensorflow from source, see https://github.com/tradr-project/tensorflow_ros_cpp#c-abi-difference-problems

## Related work
https://github.com/leggedrobotics/darknet_ros
- C++ and actively maintained. Very good starting point. Only supports YOLO v3, no nodelet support

https://github.com/UbiquityRobotics/dnn_detect
- C++, uses opencv DNN interface. Another good starting point, example uses SSD with a mobilenet backbone. Utilizes image_transport, no nodelet support

https://github.com/osrf/tensorflow_object_detector
- Python only. Followed a similar approach by targetting an old fork of tensorflow/models.

### Tensorflow C++ integration with ROS
https://github.com/tradr-project/tensorflow_ros_cpp
https://github.com/tradr-project/tensorflow_ros_test
- Catkin-friendly C++ bindings for tensorflow.

https://github.com/PatWie/tensorflow-cmake
- Source of the FindTensorflow.cmake file in this project
