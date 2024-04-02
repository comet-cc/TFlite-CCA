

1 Install Bazel using [Installing Bazel](https://bazel.build/install) 

2 Install TensorFlow using [Install TensorFl](https://www.tensorflow.org/install)

3 Clone TensorFlow repository
'''
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
'''
4 Place our repository into TensorFlow repository
'''
sudo rm -r tensorflow_src/tensorflow/lite/examples/label_image/ 
git clone https://github.com/SinaAb7/TFlite_CCA.git ./tensorflow_src/tensorflow/lite/examples/
'''

5 Cross compile the source code

'''
bazel build -c opt --config=elinux_aarch64 \
  //tensorflow/lite/examples/label_image:realm_inference
'''
