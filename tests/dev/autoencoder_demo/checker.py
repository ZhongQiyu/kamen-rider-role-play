# checker.py

import tensorflow as tf

# 打印 CUDA 和 cuDNN 版本
build_info = tf.sysconfig.get_build_info()
print(f"CUDA version: {build_info['cuda_version']}")
print(f"cuDNN version: {build_info['cudnn_version']}")
