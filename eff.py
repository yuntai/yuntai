import time
import sys
sys.path = ['/home/yuntai/github/k/keras-applications'] + sys.path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from pathlib import Path

import tensorflow as tf
from dataset import build_dataset
import hparams
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

from tensorflow.keras.layers import (
  Activation,
  BatchNormalization,
  Concatenate,
  Conv2D,
  Dense,
  DepthwiseConv2D,
  Dropout,
  Flatten,
  GlobalMaxPool2D,
  Input,
  Lambda,
  MaxPooling2D,
  ReLU,
  Reshape,
  Softmax,
  UpSampling2D,
)

from tensorflow.keras import (
  Model,
  Sequential,
)

import tensorflow.keras.backend as K
def swish_activation(x):
  return (K.sigmoid(x) * x)
tf.keras.utils.get_custom_objects().update(
  {'swish': Activation(swish_activation)})

import keras_applications
from tensorflow.python.keras.applications import keras_modules_injection 

import numpy as np

phi = 1
image_dims = 128,128
backbone_gen = keras_modules_injection(
  keras_applications.efficientnet.__dict__[f'EfficientNetB{phi}'])
backbone = backbone_gen(include_top=False, 
                        input_shape=[image_dims[0], image_dims[1], 3])
backbone.trainable = False
print(backbone.layers[0])
