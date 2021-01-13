# Used https://github.com/PatWie/tensorflow-cmake/tree/master/dockerfiles/ as reference
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt update                          && \
    apt install -y --no-install-recommends \
      cmake                                \
      g++                                  \
      git                                  \
      pkg-config                           \
      python-dev                           \
      python-pip                           \
      python-setuptools                    \
      python3-dev                          \
      python3-pip                          \
      python3-setuptools                   \
      unzip                                \
      wget                                 \
      zip                                  \
      zlib1g-dev

RUN pip install pip numpy wheel --no-cache-dir && \
    pip install -U --user keras_preprocessing --no-deps --no-cache-dir


# RUN mkdir /source_builds                                 && \
#     cd /source_builds                                    && \
#     wget https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh && \
#     chmod +x bazel-0.26.1-installer-linux-x86_64.sh && \
#     ./bazel-0.26.1-installer-linux-x86_64.sh

RUN sudo apt install curl gnupg && \
  curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg && \
  sudo mv bazel.gpg /etc/apt/trusted.gpg.d/ && \
  echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list && \
  sudo apt update && sudo apt install bazel-3.1.0 && \
  sudo ln -s /usr/bin/bazel-3.1.0 /usr/local/bin/bazel

RUN cd /source_builds                                    && \
  git clone https://github.com/tensorflow/tensorflow.git && \
  cd tensorflow                                          && \
  git checkout r1.14

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.0,3.5,5.2,6.0,6.1
ENV TF_CUDA_VERSION=10.1
ENV TF_NCCL_VERSION=2.4.8
ENV TF_CUDNN_VERSION=7

WORKDIR /source_builds/tensorflow/
RUN bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --config=cuda \
      --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
      tensorflow/tools/pip_package:build_pip_package
RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip
RUN bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --config=cuda \
      --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
      //tensorflow:libtensorflow_cc.so
RUN bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --config=cuda \
      --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
      //tensorflow:libtensorflow.so


# RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
# RUN echo "startup --batch" >>/etc/bazel.bazelrc

# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
# RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >>/etc/bazel.bazelrc
# RUN cd tensorflow_models/models && \
#   wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz && \
#   tar -xvf ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz

# TODO Pip install from /tmp/pip


# ## From source
# ### Getting the models
# ```sh
# cd tensorflow_models/models
# wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
# tar -xvf ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
# ```

# ### Environment setup
# Some environmental variables need to be setup to help FindTensorflow.cmake

# An example is given below
# ```sh
# export TENSORFLOW_BUILD_DIR=/home/pv20bot/coding/source_builds/tensorflow_build
# export TENSORFLOW_SOURCE_DIR=/home/pv20bot/coding/source_builds/tensorflow
# ```
