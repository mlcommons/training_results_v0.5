# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ResNet50 model for Keras.

Adapted from tf.keras.applications.resnet50.ResNet50().

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import utils


L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def identity_block(input_tensor, kernel_size, filters, stage, block):
  """The identity block is the block that has no conv layer at shortcut.

  # Arguments
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of
          middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names

  # Returns
      Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(filters1, (1, 1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2a')(input_tensor)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2a')(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(filters2, kernel_size,
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2b')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2b')(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(filters3, (1, 1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2c')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2c')(x)

  x = layers.add([x, input_tensor])
  x = layers.Activation('relu')(x)
  return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
  """A block that has a conv layer at shortcut.

  # Arguments
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of
          middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the second conv layer in the block.

  # Returns
      Output tensor for the block.

  Note that from stage 3,
  the second conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well
  """
  filters1, filters2, filters3 = filters
  if backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(filters1, (1, 1), kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2a')(input_tensor)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2a')(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(filters2, kernel_size, strides=strides, padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2b')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2b')(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(filters3, (1, 1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2c')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name=bn_name_base + '2c')(x)

  shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                           kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                           bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                           name=conv_name_base + '1')(input_tensor)
  shortcut = layers.BatchNormalization(axis=bn_axis,
                                       momentum=BATCH_NORM_DECAY,
                                       epsilon=BATCH_NORM_EPSILON,
                                       name=bn_name_base + '1')(shortcut)

  x = layers.add([x, shortcut])
  x = layers.Activation('relu')(x)
  return x


def ResNet50(num_classes):
  """Instantiates the ResNet50 architecture.

  Args:
    num_classes: `int` number of classes for image classification.

  Returns:
      A Keras model instance.
  """
  # Determine proper input shape
  if backend.image_data_format() == 'channels_first':
    input_shape = (3, 224, 224)
    bn_axis = 1
  else:
    input_shape = (224, 224, 3)
    bn_axis = 3

  img_input = layers.Input(shape=input_shape)
  x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
  x = layers.Conv2D(64, (7, 7),
                    strides=(2, 2),
                    padding='valid',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name='conv1')(x)
  x = layers.BatchNormalization(axis=bn_axis,
                                momentum=BATCH_NORM_DECAY,
                                epsilon=BATCH_NORM_EPSILON,
                                name='bn_conv1')(x)
  x = layers.Activation('relu')(x)
  x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
  x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

  x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

  x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

  x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

  x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = layers.Dense(
      num_classes, activation='softmax',
      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
      bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
      name='fc1000')(x)

  # Create model.
  return models.Model(img_input, x, name='resnet50')
