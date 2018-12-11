# Copyright 2018 Google. All Rights Reserved.
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
"""SSD (via ResNet50) model definition.

Defines the SSD model and loss functions from this paper:

https://arxiv.org/pdf/1708.02002

Uses the ResNet model as a basis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages
from mlperf_compliance import mlperf_log
import ssd_constants


def cross_replica_average(inputs, num_shards=None, num_shards_per_group=None):
  """Customized cross replica sum op."""
  # if num_shards_per_group is defined, apply distributed batch norm.
  group_assignment = None
  if num_shards_per_group > 0:
    if num_shards % num_shards_per_group != 0:
      raise ValueError(
          'num_shards: %d mod num_shards_per_group: %d, should be 0' %
          (num_shards, num_shards_per_group))
    num_groups = num_shards // num_shards_per_group
    group_assignment = [[
        x for x in range(num_shards) if x // num_shards_per_group == y
    ] for y in range(num_groups)]

  return tpu_ops.cross_replica_sum(inputs, group_assignment) / math_ops.cast(
      num_shards_per_group, inputs.dtype)


def distributed_batch_norm(inputs,
                           decay=ssd_constants.BATCH_NORM_DECAY,
                           epsilon=ssd_constants.BATCH_NORM_EPSILON,
                           is_training=True,
                           gamma_initializer=None,
                           num_shards=None,
                           distributed_group_size=4,
                           scope=None):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

  Note: When is_training is True the moving_mean and moving_variance need to be
  updated, by default the update_ops are placed in `tf.GraphKeys.UPDATE_OPS` so
  they need to be added as a dependency to the `train_op`, example:

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
      updates = tf.group(*update_ops)
      total_loss = control_flow_ops.with_dependencies([updates], total_loss)

  One can set updates_collections=None to force the updates in place, but that
  can have speed penalty, especially in distributed settings.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
    decay: Decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance.
    epsilon: Small float added to variance to avoid dividing by zero.
    is_training: Whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    gamma_initializer:  Initializers for gamma.
    num_shards: Number of shards that participate in the global reduction.
      Default is set to None, that will skip the cross replica sum in and
      normalize across local examples only.
    distributed_group_size: Number of replicas to normalize across in the
      distributed batch normalization.
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.
     Raises:

  Raises:
    ValueError: If input shape is not fully defined.
  """

  with tf.variable_scope(scope, 'batch_normalization', [inputs], reuse=None):
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined `C` dimension %s.' %
                       (inputs.name, params_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta = tf.get_variable(
        'beta',
        shape=params_shape,
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=True)
    gamma = tf.get_variable(
        'gamma',
        dtype=tf.float32,
        shape=params_shape,
        initializer=gamma_initializer,
        trainable=True)
    # Disable partition setting for moving_mean and moving_variance
    # as assign_moving_average op below doesn't support partitioned variable.
    scope = tf.get_variable_scope()
    partitioner = scope.partitioner
    scope.set_partitioner(None)
    moving_mean = tf.get_variable(
        'moving_mean',
        shape=params_shape,
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=False)
    moving_variance = tf.get_variable(
        'moving_variance',
        shape=params_shape,
        initializer=tf.ones_initializer(),
        trainable=False)
    # Restore scope's partitioner setting.
    scope.set_partitioner(partitioner)

    # Add cross replica sum to do subset mean and variance calculation
    # First compute mean and variance
    if is_training:
      # Execute a distributed batch normalization
      axis = 3
      inputs_dtype = inputs.dtype
      inputs = tf.cast(inputs, tf.float32)
      ndims = len(inputs_shape)
      reduction_axes = [i for i in range(ndims) if i != axis]
      counts, mean_ss, variance_ss, _ = tf.nn.sufficient_statistics(
          inputs, reduction_axes, keep_dims=False)
      mean_variance_ss = tf.concat([mean_ss, variance_ss], 0)
      mean_variance_ss = cross_replica_average(mean_variance_ss, num_shards,
                                               distributed_group_size)
      num_elements = tf.reduce_prod(mean_ss.get_shape())
      mean_ss = tf.slice(mean_variance_ss, [0], [num_elements])
      variance_ss = tf.slice(mean_variance_ss, [num_elements], [num_elements])

      mean, variance = tf.nn.normalize_moments(
          counts, mean_ss, variance_ss, shift=None)
      outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma,
                                          epsilon)
      outputs = tf.cast(outputs, inputs_dtype)
    else:
      outputs, mean, variance = tf.nn.fused_batch_norm(
          inputs,
          gamma,
          beta,
          mean=moving_mean,
          variance=moving_variance,
          epsilon=epsilon,
          is_training=False,
          data_format='NHWC')

    if is_training:
      update_moving_mean = moving_averages.assign_moving_average(
          moving_mean,
          tf.cast(mean, moving_mean.dtype),
          decay,
          zero_debias=False)
      update_moving_variance = moving_averages.assign_moving_average(
          moving_variance,
          tf.cast(variance, moving_variance.dtype),
          decay,
          zero_debias=False)
      tf.add_to_collection('update_ops', update_moving_mean)
      tf.add_to_collection('update_ops', update_moving_variance)

    outputs.set_shape(inputs_shape)
    return outputs


def batch_norm_relu(inputs,
                    is_training_bn,
                    params,
                    relu=True,
                    init_zero=False,
                    data_format='channels_last',
                    name=None):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training_bn: `bool` for whether the model is training.
    params: params of the model, a dict including `distributed_group_size`
        and `num_shards`.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    name: the name of the batch normalization layer

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  if params['distributed_group_size'] > 1:
    inputs = distributed_batch_norm(
        inputs=inputs,
        decay=ssd_constants.BATCH_NORM_DECAY,
        epsilon=ssd_constants.BATCH_NORM_EPSILON,
        is_training=is_training_bn,
        gamma_initializer=gamma_initializer,
        num_shards=params['num_shards'],
        distributed_group_size=params['distributed_group_size'])
  else:
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=ssd_constants.BATCH_NORM_DECAY,
        epsilon=ssd_constants.BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=is_training_bn,
        fused=True,
        gamma_initializer=gamma_initializer,
        name=name)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(
        inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format='channels_last'):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def residual_block(inputs,
                   filters,
                   is_training_bn,
                   strides,
                   params,
                   use_projection=False,
                   data_format='channels_last'):
  """Standard building block for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training_bn: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    params: params of the model, a dict.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs
  if use_projection:
    # Projection shortcut in first layer to match filters and strides
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(
        shortcut, is_training_bn, params, relu=False, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs, is_training_bn, params, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      params,
      relu=False,
      init_zero=True,
      data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def block_group(inputs,
                filters,
                block_fn,
                blocks,
                strides,
                is_training_bn,
                name,
                params,
                data_format='channels_last',
                use_projection=True):
  """Creates one group of blocks for the ResNet model.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
    is_training_bn: `bool` for whether the model is training.
    name: `str`name for the Tensor output of the block layer.
    params: params of the model, a dict.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.

  Returns:
    The output `Tensor` of the block layer.
  """
  # Only the first block per block_group uses projection shortcut and strides.
  inputs = block_fn(
      inputs,
      filters,
      is_training_bn,
      strides,
      params,
      use_projection=use_projection,
      data_format=data_format)

  for _ in range(1, blocks):
    inputs = block_fn(
        inputs, filters, is_training_bn, 1, params, data_format=data_format)

  return tf.identity(inputs, name)


def resnet_v1_generator(block_fn, layers, params, data_format='channels_last'):
  """Generator of ResNet v1 model with classification layers removed.

    Our actual ResNet network.  We return the output of c2, c3,c4,c5
    N.B. batch norm is always run with trained parameters, as we use very small
    batches when training the object layers.

  Args:
    block_fn: `function` for the block to use within the model. Either
        `residual_block` or `bottleneck_block`.
    layers: list of 4 `int`s denoting the number of blocks to include in each
      of the 4 block groups. Each group consists of blocks that take inputs of
      the same resolution.
    params: params of the model, a dict.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns the
    output `Tensor` of the ResNet model.
  """
  def model(inputs, is_training_bn=False):
    """Creation of the model graph."""
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=64,
        kernel_size=7,
        strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = batch_norm_relu(
        inputs, is_training_bn, params, data_format=data_format)

    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=3,
        strides=2,
        padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    c2 = block_group(
        inputs=inputs,
        filters=64,
        blocks=layers[0],
        strides=1,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        params=params,
        name='block_group1',
        data_format=data_format,
        use_projection=False)
    c3 = block_group(
        inputs=c2,
        filters=128,
        blocks=layers[1],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        params=params,
        name='block_group2',
        data_format=data_format)
    c4 = block_group(
        inputs=c3,
        filters=256,
        blocks=layers[2],
        strides=1,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        params=params,
        name='block_group3',
        data_format=data_format)
    c5 = block_group(
        inputs=c4,
        filters=512,
        blocks=layers[3],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        params=params,
        name='block_group4',
        data_format=data_format)
    return c2, c3, c4, c5

  return model


def resnet_v1(resnet_depth, params, data_format='channels_last'):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      34: {'block': residual_block, 'layers': [3, 4, 6, 3]}
  }

  if resnet_depth not in model_params:
    raise ValueError('Not a valid resnet_depth:', resnet_depth)

  resnet_params = model_params[resnet_depth]
  mlperf_log.ssd_print(
      key=mlperf_log.BACKBONE, value='resnet{}'.format(resnet_depth))
  return resnet_v1_generator(resnet_params['block'], resnet_params['layers'],
                             params, data_format)


def class_net(images, level, num_classes):
  """Class prediction network for SSD."""
  return tf.layers.conv2d(
      images,
      num_classes * ssd_constants.NUM_DEFAULTS_BY_LEVEL[level],
      kernel_size=(3, 3),
      padding='same',
      activation=None,
      name='class-%d' % (level),
  )


def box_net(images, level):
  """Box regression network for SSD."""
  return tf.layers.conv2d(
      images,
      4 * ssd_constants.NUM_DEFAULTS_BY_LEVEL[level],
      kernel_size=(3, 3),
      padding='same',
      activation=None,
      name='box-%d' % (level),
  )


def ssd(features, params, is_training_bn=False):
  """SSD classification and regression model."""
  # upward layers
  with tf.variable_scope('resnet%s' % ssd_constants.RESNET_DEPTH):
    resnet_fn = resnet_v1(ssd_constants.RESNET_DEPTH, params)
    _, _, u4, _ = resnet_fn(features, is_training_bn)

  with tf.variable_scope('ssd'):
    feats = {}
    # output channels for mlperf logging.
    out_channels = [256]
    feats[3] = u4
    feats[4] = tf.layers.conv2d(
        feats[3],
        filters=256,
        kernel_size=(1, 1),
        padding='same',
        activation=tf.nn.relu,
        name='block7-conv1x1')
    feats[4] = tf.layers.conv2d(
        feats[4],
        filters=512,
        strides=(2, 2),
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu,
        name='block7-conv3x3')
    out_channels.append(512)
    feats[5] = tf.layers.conv2d(
        feats[4],
        filters=256,
        kernel_size=(1, 1),
        padding='same',
        activation=tf.nn.relu,
        name='block8-conv1x1')
    feats[5] = tf.layers.conv2d(
        feats[5],
        filters=512,
        strides=(2, 2),
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu,
        name='block8-conv3x3')
    out_channels.append(512)
    feats[6] = tf.layers.conv2d(
        feats[5],
        filters=128,
        kernel_size=(1, 1),
        padding='same',
        activation=tf.nn.relu,
        name='block9-conv1x1')
    feats[6] = tf.layers.conv2d(
        feats[6],
        filters=256,
        strides=(2, 2),
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu,
        name='block9-conv3x3')
    out_channels.append(256)
    feats[7] = tf.layers.conv2d(
        feats[6],
        filters=128,
        kernel_size=(1, 1),
        padding='same',
        activation=tf.nn.relu,
        name='block10-conv1x1')
    feats[7] = tf.layers.conv2d(
        feats[7],
        filters=256,
        kernel_size=(3, 3),
        padding='valid',
        activation=tf.nn.relu,
        name='block10-conv3x3')
    out_channels.append(256)
    feats[8] = tf.layers.conv2d(
        feats[7],
        filters=128,
        kernel_size=(1, 1),
        padding='same',
        activation=tf.nn.relu,
        name='block11-conv1x1')
    feats[8] = tf.layers.conv2d(
        feats[8],
        filters=256,
        kernel_size=(3, 3),
        padding='valid',
        activation=tf.nn.relu,
        name='block11-conv3x3')
    out_channels.append(256)
    mlperf_log.ssd_print(
        key=mlperf_log.LOC_CONF_OUT_CHANNELS, value=out_channels)

    class_outputs = {}
    box_outputs = {}
    min_level = ssd_constants.MIN_LEVEL
    max_level = ssd_constants.MAX_LEVEL
    num_classes = ssd_constants.NUM_CLASSES
    mlperf_log.ssd_print(
        key=mlperf_log.NUM_DEFAULTS_PER_CELL, value=ssd_constants.NUM_DEFAULTS)

    with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
      for level in range(min_level, max_level + 1):
        class_outputs[level] = class_net(
            feats[level], level, num_classes)

    with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
      for level in range(min_level, max_level + 1):
        box_outputs[level] = box_net(
            feats[level], level)

  return class_outputs, box_outputs
