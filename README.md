# Tensorflow Models in ROS
This repository is a C++ ROS wrapper around several different networks pulled from [tensorflow/models](https://github.com/tensorflow/models)

There are currently four target functionalities from this repository.

1. 2D bounding box object detectors from [tensorflow/models/research/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)
2. Semantic segmentation using [deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)
3. 2D "free space" estimation using a modified deeplab network.

## 2D Bounding box object detectors
Models taken from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

## Semantic segmentation using deeplab
Models taken from https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md

## Free-space estimation in 2D for indoor robots
ADE20K is a dataset that contains many examples of indoor images with segmentation labels. The approach here was to modify DeepLabv3 trained on ADE20K by taking the linear outputs before the ArgMax layer, and applying a softmax operation. The intention is that by selecting the floor class layer from the softmax output, this gives a probability estimate of drivable terrain.

NOTE: It's important to use this as a starting point and to fine-tune the model for your target environment. While ADE20K has the advantage of containing training examples of indoor scenes, the labeling policy isn't appropriate for this task in particular (see https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv for list of classes). For example, labeling policies that I would find to be more robust are cityscapes including dynamic and static object classes, and wilddash including a 'void' class denoting invalid sensor input. ADE20K does not have a 'catch all' type label for generic objects nor a void label for sensor failures. Going through the dataset, one can see many examples of objects on the floor being included in the floor class.

# Getting started
## Source build
Requires a source build of tensorflow lite.
Follow https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/minimal
or

```sh
# Install cmake/tensorflow from source to get to C++ bindings
mkdir source_builds; cd source_builds

# System version of cmake on 18.04 isn't recent enough. Installing something more recent
wget https://github.com/Kitware/CMake/releases/download/v3.19.2/cmake-3.19.2.tar.gz
tar -xvf cmake-3.19.2.tar.gz
cd cmake-3.19.2/
cmake .
make -j16
sudo make install
cd ..

# Open a new shell due to conflicts with system cmake
# Install tensorflow
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
mkdir minimal_build
cd minimal_build
cmake ../tensorflow_src/tensorflow/lite/examples/minimal
cmake --build . -j

# Install ROS dependencies for this package
rosdep install --from-paths . --ignore-src -r -y --os=ubuntu:bionic
catkin build --this
```

## Testing

```sh
rosrun tensorflow_models tf_lite_node $(pwd)/models/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite
# INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
# === Pre-invoke Interpreter State ===
# Interpreter has 184 tensors and 64 nodes
# Inputs: 175
# Outputs: 167 168 169 170

# Tensor   0 BoxPredictor_0/BoxEncodingPredictor/BiasAdd kTfLiteUInt8  kTfLiteArenaRw       4332 bytes ( 0.0 MB)  1 19 19 12
# Tensor   1 BoxPredictor_0/BoxEncodingPredictor/Conv2D_bias kTfLiteInt32   kTfLiteMmapRo         48 bytes ( 0.0 MB)  12

rosrun --prefix 'gdb --args' tensorflow_models tf_lite_node $(pwd)/models/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite
```


## Docker

WIP.

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

# Roadmap
* Add example images to documentation
* Nodelet implementation
* Add separate launch file for semantic segmentation
* Add script to pull models instead of saving on github (Squash after doing this)
* Better colormap for object detection draw
* Colormap output for semantic segmentation
* Semantic segmentation output
* Add Dockerfile, docker image, and run commands
* Add MVE model ensemble https://arxiv.org/pdf/1612.01474.pdf
