# TensorFlow Lite C++ Image Recognition for Arm CCA 
In this repository we provide the guide and files to build a binary which is able to generate inference result given a tensorflow lite model and an a .png image. We used the binary in another project to simulate machine learning inference on a realm virtual machine (look at [GuaranTEE](https://github.com/comet-cc/GuaranTEE)) but, it is executable in any linux environment on Arm64 architecture with c++ shared library support. 

First, you need to install bazel [Installing Bazel](https://bazel.build/install) and tensorflow [Install TensorFlow 2](https://www.tensorflow.org/install).

### Clone TensorFlow repository
```
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```
### Place our repository into TensorFlow repository
```
sudo rm -r tensorflow_src/tensorflow/lite/examples/label_image/ 
git clone https://github.com/comet-cc/TFlite-CCA.git ./tensorflow_src/tensorflow/lite/examples/label_image
```
### Build source code

```
cd ./tensorflow_src
bazel build -c opt --config=elinux_aarch64 \
  //tensorflow/lite/examples/label_image:realm_inference
```
After succesfully building, you can find `realm_inference` binary at: `bazel-bin/tensorflow/lite/examples/label_image/`
