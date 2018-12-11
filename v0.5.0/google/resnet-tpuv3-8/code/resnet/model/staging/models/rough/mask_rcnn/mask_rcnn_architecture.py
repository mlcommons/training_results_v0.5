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
"""Mask-RCNN (via ResNet) model definition.

Uses the ResNet model as a basis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import anchors
from object_detection import balanced_positive_negative_sampler
from mlperf_compliance import mlperf_log


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-4
_RESNET_MAX_LEVEL = 5
_EPSILON = 1e-8
_NMS_TILE_SIZE = 512


def batch_norm_relu(inputs,
                    is_training_bn,
                    relu=True,
                    init_zero=False,
                    data_format='channels_last',
                    name=None):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training_bn: `bool` for whether the model is training.
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

  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
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
        shortcut, is_training_bn, relu=False, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      relu=False,
      init_zero=True,
      data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def bottleneck_block(inputs,
                     filters,
                     is_training_bn,
                     strides,
                     use_projection=False,
                     data_format='channels_last'):
  """Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training_bn: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
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
    # Projection shortcut only in first block within a group. Bottleneck blocks
    # end with 4 times the number of filters.
    filters_out = 4 * filters
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(
        shortcut, is_training_bn, relu=False, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
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
                data_format='channels_last'):
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
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block layer.
  """
  # Only the first block per block_group uses projection shortcut and strides.
  inputs = block_fn(
      inputs,
      filters,
      is_training_bn,
      strides,
      use_projection=True,
      data_format=data_format)

  for _ in range(1, blocks):
    inputs = block_fn(
        inputs, filters, is_training_bn, 1, data_format=data_format)

  return tf.identity(inputs, name)


def resnet_v1_generator(block_fn, layers, data_format='channels_last'):
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
    inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

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
        name='block_group1',
        data_format=data_format)
    c3 = block_group(
        inputs=c2,
        filters=128,
        blocks=layers[1],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group2',
        data_format=data_format)
    c4 = block_group(
        inputs=c3,
        filters=256,
        blocks=layers[2],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group3',
        data_format=data_format)
    c5 = block_group(
        inputs=c4,
        filters=512,
        blocks=layers[3],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group4',
        data_format=data_format)
    return c2, c3, c4, c5

  return model


def resnet_v1(resnet_depth, data_format='channels_last'):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': residual_block, 'layers': [2, 2, 2, 2]},
      34: {'block': residual_block, 'layers': [3, 4, 6, 3]},
      50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
  }

  if resnet_depth not in model_params:
    raise ValueError('Not a valid resnet_depth:', resnet_depth)

  mlperf_log.maskrcnn_print(key=mlperf_log.BACKBONE,
                            value='resnet{}'.format(resnet_depth))
  params = model_params[resnet_depth]
  return resnet_v1_generator(
      params['block'], params['layers'], data_format)


def nearest_upsampling(data, scale):
  """Nearest neighbor upsampling implementation.

  Args:
    data: A tensor with a shape of [batch, height_in, width_in, channels].
    scale: An integer multiple to scale resolution of input data.
  Returns:
    data_up: A tensor with a shape of
      [batch, height_in*scale, width_in*scale, channels]. Same dtype as input
      data.
  """
  with tf.name_scope('nearest_upsampling'):
    bs, h, w, c = data.get_shape().as_list()
    bs = -1 if bs is None else bs
    # Use reshape to quickly upsample the input.  The nearest pixel is selected
    # implicitly via broadcasting.
    data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
        [1, 1, scale, 1, scale, 1], dtype=data.dtype)
    return tf.reshape(data, [bs, h * scale, w * scale, c])


def _bbox_overlap(boxes, gt_boxes):
  """Calculates the overlap between proposal and ground truth boxes.

  Some `gt_boxes` may have been padded.  The returned `iou` tensor for these
  boxes will be -1.

  Args:
    boxes: a tensor with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates in [ymin, xmin, ymax, xmax] form.
    gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
      tensor might have paddings with a negative value.
  Returns:
    iou: a tensor with as a shape of [batch_size, N, MAX_NUM_INSTANCES].
  """
  with tf.name_scope('bbox_overlap'):
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=2)
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(
        value=gt_boxes, num_or_size_splits=4, axis=2)

    # Calculates the intersection area.
    i_xmin = tf.maximum(bb_x_min, tf.transpose(gt_x_min, [0, 2, 1]))
    i_xmax = tf.minimum(bb_x_max, tf.transpose(gt_x_max, [0, 2, 1]))
    i_ymin = tf.maximum(bb_y_min, tf.transpose(gt_y_min, [0, 2, 1]))
    i_ymax = tf.minimum(bb_y_max, tf.transpose(gt_y_max, [0, 2, 1]))
    i_area = tf.maximum((i_xmax - i_xmin), 0) * tf.maximum((i_ymax - i_ymin), 0)

    # Calculates the union area.
    bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
    gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
    # Adds a small epsilon to avoid divide-by-zero.
    u_area = bb_area + tf.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

    # Calculates IoU.
    iou = i_area / u_area

    # Fills -1 for padded ground truth boxes.
    padding_mask = tf.less(i_xmin, tf.zeros_like(i_xmin))
    iou = tf.where(padding_mask, -tf.ones_like(iou), iou)

    return iou


def _add_class_assignments(iou, scaled_gt_boxes, gt_labels):
  """Computes object category assignment for each box.

  Args:
    iou: a tensor for the iou matrix with a shape of
      [batch_size, K, MAX_NUM_INSTANCES]. K is the number of post-nms RoIs
      (i.e., rpn_post_nms_topn).
    scaled_gt_boxes: a tensor with a shape of
      [batch_size, MAX_NUM_INSTANCES, 4]. This tensor might have paddings with
      negative values. The coordinates of gt_boxes are in the pixel coordinates
      of the scaled image scale.
    gt_labels: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
      tensor might have paddings with a value of -1.
  Returns:
    max_boxes: a tensor with a shape of [batch_size, K, 4], representing
      the ground truth coordinates of each roi.
    max_classes: a int32 tensor with a shape of [batch_size, K], representing
      the ground truth class of each roi.
    max_overlap: a tensor with a shape of [batch_size, K], representing
      the maximum overlap of each roi.
    argmax_iou: a tensor with a shape of [batch_size, K], representing the iou
      argmax.
  """
  with tf.name_scope('add_class_assignments'):
    batch_size, _, _ = iou.get_shape().as_list()
    argmax_iou = tf.argmax(iou, axis=2, output_type=tf.int32)
    indices = tf.reshape(
        argmax_iou + tf.expand_dims(
            tf.range(batch_size) * tf.shape(gt_labels)[1], 1), [-1])
    max_classes = tf.reshape(
        tf.gather(tf.reshape(gt_labels, [-1, 1]), indices), [batch_size, -1])
    max_overlap = tf.reduce_max(iou, axis=2)
    bg_mask = tf.equal(max_overlap, tf.zeros_like(max_overlap))
    max_classes = tf.where(bg_mask, tf.zeros_like(max_classes), max_classes)

    max_boxes = tf.reshape(
        tf.gather(tf.reshape(scaled_gt_boxes, [-1, 4]), indices),
        [batch_size, -1, 4])
    max_boxes = tf.where(
        tf.tile(tf.expand_dims(bg_mask, axis=2), [1, 1, 4]),
        tf.zeros_like(max_boxes), max_boxes)
  return max_boxes, max_classes, max_overlap, argmax_iou


def encode_box_targets(boxes, gt_boxes, gt_labels, bbox_reg_weights):
  """Encodes predicted boxes with respect to ground truth boxes."""
  with tf.name_scope('encode_box_targets'):
    box_targets = anchors.batch_encode_box_targets_op(
        boxes, gt_boxes, bbox_reg_weights)
    # If a target is background, the encoded box target should be zeros.
    mask = tf.tile(
        tf.expand_dims(tf.equal(gt_labels, tf.zeros_like(gt_labels)), axis=2),
        [1, 1, 4])
    box_targets = tf.where(mask, tf.zeros_like(box_targets), box_targets)
  return box_targets


def proposal_label_op(boxes, gt_boxes, gt_labels, image_info,
                      batch_size_per_im=512, fg_fraction=0.25, fg_thresh=0.5,
                      bg_thresh_hi=0.5, bg_thresh_lo=0., is_training=True):
  """Assigns the proposals with ground truth labels and performs subsmpling.

  Given proposal `boxes`, `gt_boxes`, and `gt_labels`, the function uses the
  following algorithm to generate the final `batch_size_per_im` RoIs.
  1. Calculates the IoU between each proposal box and each gt_boxes.
  2. Assigns each proposal box with a ground truth class and box label by
     choosing the largest overlap.
  3. Samples `batch_size_per_im` boxes from all proposal boxes, and returns
     box_targets, class_targets, and RoIs.
  The reference implementations of #1 and #2 are here: https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/json_dataset.py  # pylint: disable=line-too-long
  The reference implementation of #3 is here: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py.  # pylint: disable=line-too-long

  Args:
    boxes: a tensor with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates of scaled images in
      [ymin, xmin, ymax, xmax] form.
    gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
      tensor might have paddings with a value of -1. The coordinates of gt_boxes
      are in the pixel coordinates of the original image scale.
    gt_labels: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
      tensor might have paddings with a value of -1.
    image_info: a tensor of shape [batch_size, 5] where the three columns
      encode the input image's [height, width, scale,
      original_height, original_width]. Height and width are for
      the input to the network, not the original image; scale is the scale
      factor used to scale the network input size to the original image size.
      See dataloader.DetectionInputProcessor for details. The last two are
      original height and width.
    batch_size_per_im: a integer represents RoI minibatch size per image.
    fg_fraction: a float represents the target fraction of RoI minibatch that
      is labeled foreground (i.e., class > 0).
    fg_thresh: a float represents the overlap threshold for an RoI to be
      considered foreground (if >= fg_thresh).
    bg_thresh_hi: a float represents the overlap threshold for an RoI to be
      considered background (class = 0 if overlap in [LO, HI)).
    bg_thresh_lo: a float represents the overlap threshold for an RoI to be
      considered background (class = 0 if overlap in [LO, HI)).
    is_training: a boolean that indicates the training mode, which performs
      subsampling; otherwise, no subsampling.
  Returns:
    box_targets: a tensor with a shape of [batch_size, K, 4]. The tensor
      contains the ground truth pixel coordinates of the scaled images for each
      roi. K is the number of sample RoIs (e.g., batch_size_per_im).
    class_targets: a integer tensor with a shape of [batch_size, K]. The tensor
      contains the ground truth class for each roi.
    rois: a tensor with a shape of [batch_size, K, 4], representing the
      coordinates of the selected RoI.
    proposal_to_label_map: a tensor with a shape of [batch_size, K]. This tensor
      keeps the mapping between proposal to labels. proposal_to_label_map[i]
      means the index of the ground truth instance for the i-th proposal.
  """
  with tf.name_scope('proposal_label'):
    batch_size = boxes.shape[0]
    # Scales ground truth boxes to the scaled image coordinates.
    image_scale = 1 / image_info[:, 2]
    scaled_gt_boxes = gt_boxes * tf.reshape(image_scale, [batch_size, 1, 1])

    # The reference implementation intentionally includes ground truth boxes in
    # the proposals. see https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/json_dataset.py#L359.  # pylint: disable=line-too-long
    if is_training:
      boxes = tf.concat([boxes, scaled_gt_boxes], axis=1)
    iou = _bbox_overlap(boxes, scaled_gt_boxes)

    (pre_sample_box_targets, pre_sample_class_targets, max_overlap,
     proposal_to_label_map) = _add_class_assignments(
         iou, scaled_gt_boxes, gt_labels)

    # Generates a random sample of RoIs comprising foreground and background
    # examples. reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py#L132  # pylint: disable=line-too-long
    positives = tf.greater(max_overlap,
                           fg_thresh * tf.ones_like(max_overlap))
    negatives = tf.logical_and(
        tf.greater_equal(max_overlap,
                         bg_thresh_lo * tf.ones_like(max_overlap)),
        tf.less(max_overlap,
                bg_thresh_hi * tf.ones_like(max_overlap)))
    pre_sample_class_targets = tf.where(
        negatives, tf.zeros_like(pre_sample_class_targets),
        pre_sample_class_targets)
    proposal_to_label_map = tf.where(
        negatives, tf.zeros_like(proposal_to_label_map),
        proposal_to_label_map)

    # Returns box/class targets and rois before sampling for evaluation.
    if not is_training:
      return (pre_sample_box_targets, pre_sample_class_targets,
              boxes, proposal_to_label_map)

    # Handles ground truth paddings.
    ignore_mask = tf.less(
        tf.reduce_min(iou, axis=2), tf.zeros_like(max_overlap))
    # indicator includes both positive and negative labels.
    # labels includes only positives labels.
    # positives = indicator & labels.
    # negatives = indicator & !labels.
    # ignore = !indicator.
    labels = positives
    pos_or_neg = tf.logical_or(positives, negatives)
    indicator = tf.logical_and(pos_or_neg, tf.logical_not(ignore_mask))

    all_samples = []
    sampler = (
        balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
            positive_fraction=fg_fraction, is_static=True))
    # Batch-unroll the sub-sampling process.
    for i in range(batch_size):
      samples = sampler.subsample(
          indicator[i], batch_size_per_im, labels[i])
      all_samples.append(samples)
    all_samples = tf.stack([all_samples], axis=0)[0]
    # A workaround to get the indices from the boolean tensors.
    _, samples_indices = tf.nn.top_k(tf.to_int32(all_samples),
                                     k=batch_size_per_im, sorted=True)
    # Contructs indices for gather.
    samples_indices = tf.reshape(
        samples_indices + tf.expand_dims(
            tf.range(batch_size) * tf.shape(boxes)[1], 1), [-1])
    rois = tf.reshape(
        tf.gather(tf.reshape(boxes, [-1, 4]), samples_indices),
        [batch_size, -1, 4])
    class_targets = tf.reshape(
        tf.gather(
            tf.reshape(pre_sample_class_targets, [-1, 1]), samples_indices),
        [batch_size, -1])
    sample_box_targets = tf.reshape(
        tf.gather(tf.reshape(pre_sample_box_targets, [-1, 4]), samples_indices),
        [batch_size, -1, 4])
    sample_proposal_to_label_map = tf.reshape(
        tf.gather(tf.reshape(proposal_to_label_map, [-1, 1]), samples_indices),
        [batch_size, -1])
  return sample_box_targets, class_targets, rois, sample_proposal_to_label_map


def _top_k(scores, k, topk_sorted):
  """A wrapper that returns top-k scores and indices with batch dimension.

  Args:
    scores: a tensor with a shape of [batch_size, N]. N is the number of scores.
    k: an integer for selecting the top-k elements.
    topk_sorted: a boolean to sort the top-k elements.
  Returns:
    top_k_scores: the selected top-k scores with a shape of [batch_size, k].
    gather_indices: the indices to gather the elements. It has a shape of
      [batch_size, k].
  """
  with tf.name_scope('top_k_wrapper'):
    batch_size = scores.shape[0]
    top_k_scores, top_k_indices = tf.nn.top_k(
        scores, k=k, sorted=topk_sorted)
    # Contructs indices for gather.
    batch_indices = tf.tile(
        tf.expand_dims(tf.range(batch_size), axis=1), [1, k])
    gather_indices = tf.stack([batch_indices, top_k_indices], axis=2)
    return top_k_scores, gather_indices


def _filter_boxes(scores, boxes, rpn_min_size, image_info):
  """Filters boxes whose height or width is smaller than rpn_min_size.

  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/ops/generate_proposals.py  # pylint: disable=line-too-long

  Args:
    scores: a tensor with a shape of [batch_size, N].
    boxes: a tensor with a shape of [batch_size, N, 4]. The proposals
      are in pixel coordinates.
    rpn_min_size: a integer that represents the smallest length of the image
      height or width.
    image_info: a tensor of shape [batch_size, 5] where the three columns
      encode the input image's [height, width, scale,
      original_height, original_width]. `scale` is the scale
      factor used to scale the network input size to the original image size.
      See dataloader.DetectionInputProcessor for details.
  Returns:
    scores: a tensor with a shape of [batch_size, anchors]. Same shape and dtype
      as input scores.
    proposals: a tensor with a shape of [batch_size, anchors, 4]. Same shape and
      dtype as input boxes.
  """
  with tf.name_scope('filter_boxes'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=2)
    image_info = tf.cast(tf.expand_dims(image_info, axis=2), dtype=boxes.dtype)
    # The following tensors have a shape of [batch_size, 1, 1].
    image_height = image_info[:, 0:1, :]
    image_width = image_info[:, 1:2, :]
    image_scale = image_info[:, 2:3, :]
    min_size = tf.cast(tf.maximum(rpn_min_size, 1), dtype=boxes.dtype)

    # Proposal center is computed relative to the scaled input image.
    hs = y_max - y_min + 1
    ws = x_max - x_min + 1
    y_ctr = y_min + hs / 2
    x_ctr = x_min + ws / 2
    height_mask = tf.greater_equal(hs, min_size * image_scale)
    width_mask = tf.greater_equal(ws, min_size * image_scale)
    center_mask = tf.logical_and(
        tf.less(y_ctr, image_height), tf.less(x_ctr, image_width))
    mask = tf.logical_and(tf.logical_and(height_mask, width_mask),
                          center_mask)[:, :, 0]
    scores = tf.where(mask, scores, tf.zeros_like(scores))
    boxes = tf.cast(tf.expand_dims(mask, 2), boxes.dtype) * boxes

  return scores, boxes


def _self_suppression(iou, _, iou_sum):
  batch_size = tf.shape(iou)[0]
  can_suppress_others = tf.cast(
      tf.reshape(tf.reduce_max(iou, 1) <= 0.5, [batch_size, -1, 1]), iou.dtype)
  iou_suppressed = tf.reshape(
      tf.cast(tf.reduce_max(can_suppress_others * iou, 1) <= 0.5, iou.dtype),
      [batch_size, -1, 1]) * iou
  iou_sum_new = tf.reduce_sum(iou_suppressed, [1, 2])
  return [
      iou_suppressed,
      tf.reduce_any(iou_sum - iou_sum_new > 0.5), iou_sum_new
  ]


def _cross_suppression(boxes, box_slice, iou_threshold, inner_idx):
  batch_size = tf.shape(boxes)[0]
  new_slice = tf.slice(boxes, [0, inner_idx * _NMS_TILE_SIZE, 0],
                       [batch_size, _NMS_TILE_SIZE, 4])
  iou = _bbox_overlap(new_slice, box_slice)
  ret_slice = tf.expand_dims(
      tf.cast(tf.reduce_all(iou < iou_threshold, [1]), box_slice.dtype),
      2) * box_slice
  return boxes, ret_slice, iou_threshold, inner_idx + 1


def _suppression_loop_body(boxes, iou_threshold, output_size, idx):
  """Process boxes in the range [idx*_NMS_TILE_SIZE, (idx+1)*_NMS_TILE_SIZE).

  Args:
    boxes: a tensor with a shape of [batch_size, anchors, 4].
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    output_size: an int32 tensor of size [batch_size]. Representing the number
      of selected boxes for each batch.
    idx: an integer scalar representing induction variable.

  Returns:
    boxes: updated boxes.
    iou_threshold: pass down iou_threshold to the next iteration.
    output_size: the updated output_size.
    idx: the updated induction variable.
  """
  num_tiles = tf.shape(boxes)[1] // _NMS_TILE_SIZE
  batch_size = tf.shape(boxes)[0]

  # Iterates over tiles that can possibly suppress the current tile.
  box_slice = tf.slice(boxes, [0, idx * _NMS_TILE_SIZE, 0],
                       [batch_size, _NMS_TILE_SIZE, 4])
  _, box_slice, _, _ = tf.while_loop(
      lambda _boxes, _box_slice, _threshold, inner_idx: inner_idx < idx,
      _cross_suppression, [boxes, box_slice, iou_threshold,
                           tf.constant(0)])

  # Iterates over the current tile to compute self-suppression.
  iou = _bbox_overlap(box_slice, box_slice)
  mask = tf.expand_dims(
      tf.reshape(tf.range(_NMS_TILE_SIZE), [1, -1]) > tf.reshape(
          tf.range(_NMS_TILE_SIZE), [-1, 1]), 0)
  iou *= tf.cast(tf.logical_and(mask, iou >= iou_threshold), iou.dtype)
  suppressed_iou, _, _ = tf.while_loop(
      lambda _iou, loop_condition, _iou_sum: loop_condition, _self_suppression,
      [iou, tf.constant(True),
       tf.reduce_sum(iou, [1, 2])])
  suppressed_box = tf.reduce_sum(suppressed_iou, 1) > 0
  box_slice *= tf.expand_dims(1.0 - tf.cast(suppressed_box, box_slice.dtype), 2)

  # Uses box_slice to update the input boxes.
  mask = tf.reshape(
      tf.cast(tf.equal(tf.range(num_tiles), idx), boxes.dtype), [1, -1, 1, 1])
  boxes = tf.tile(tf.expand_dims(
      box_slice, [1]), [1, num_tiles, 1, 1]) * mask + tf.reshape(
          boxes, [batch_size, num_tiles, _NMS_TILE_SIZE, 4]) * (1 - mask)
  boxes = tf.reshape(boxes, [batch_size, -1, 4])

  # Updates output_size.
  output_size += tf.reduce_sum(
      tf.cast(tf.reduce_any(box_slice > 0, [2]), tf.int32), [1])
  return boxes, iou_threshold, output_size, idx + 1


def _non_max_suppression_padded(
    scores, boxes, max_output_size, iou_threshold, level):
  """A wrapper that handles non-maximum suppression.

  Assumption:
    * The boxes are sorted by scores unless the box is a dot (all coordinates
      are zero).
    * Boxes with higher scores can be used to suppress boxes with lower scores.

  The overal design of the algorithm is to handle boxes tile-by-tile:

  boxes = boxes.pad_to_multiply_of(tile_size)
  num_tiles = len(boxes) // tile_size
  output_boxes = []
  for i in range(num_tiles):
    box_tile = boxes[i*tile_size : (i+1)*tile_size]
    for j in range(i - 1):
      suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
      iou = _bbox_overlap(box_tile, suppressing_tile)
      # if the box is suppressed in iou, clear it to a dot
      box_tile *= _update_boxes(iou)
    # Iteratively handle the diagnal tile.
    iou = _box_overlap(box_tile, box_tile)
    iou_changed = True
    while iou_changed:
      # boxes that are not suppressed by anything else
      suppressing_boxes = _get_suppressing_boxes(iou)
      # boxes that are suppressed by suppressing_boxes
      suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
      # clear iou to 0 for boxes that are suppressed, as they cannot be used
      # to suppress other boxes any more
      new_iou = _clear_iou(iou, suppressed_boxes)
      iou_changed = (new_iou != iou)
      iou = new_iou
    # remaining boxes that can still suppress others, are selected boxes.
    output_boxes.append(_get_suppressing_boxes(iou))
    if len(output_boxes) >= max_output_size:
      break

  Args:
    scores: a tensor with a shape of [batch_size, anchors].
    boxes: a tensor with a shape of [batch_size, anchors, 4].
    max_output_size: a scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    level: a integer for the level that the function operates on.
  Returns:
    nms_scores: a tensor with a shape of [batch_size, anchors]. It has same
      dtype as input scores.
    nms_proposals: a tensor with a shape of [batch_size, anchors, 4]. It has
      same dtype as input boxes.
  """
  with tf.name_scope('nms_l%d' % level):
    batch_size = tf.shape(boxes)[0]
    num_boxes = tf.shape(boxes)[1]
    pad = tf.cast(
        tf.ceil(tf.cast(num_boxes, tf.float32) / _NMS_TILE_SIZE),
        tf.int32) * _NMS_TILE_SIZE - num_boxes
    boxes = tf.pad(tf.cast(boxes, tf.float32), [[0, 0], [0, pad], [0, 0]])
    scores = tf.pad(tf.cast(scores, tf.float32), [[0, 0], [0, pad]])
    num_boxes += pad

    def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
      return tf.logical_and(
          tf.reduce_min(output_size) < max_output_size,
          idx < num_boxes // _NMS_TILE_SIZE)

    selected_boxes, _, output_size, _ = tf.while_loop(
        _loop_cond, _suppression_loop_body, [
            boxes, iou_threshold,
            tf.zeros([batch_size], tf.int32),
            tf.constant(0)
        ])
    idx = num_boxes - tf.cast(
        tf.nn.top_k(
            tf.cast(tf.reduce_any(selected_boxes > 0, [2]), tf.int32) *
            tf.expand_dims(tf.range(num_boxes, 0, -1), 0), max_output_size)[0],
        tf.int32)
    idx = tf.minimum(idx, num_boxes - 1)
    idx = tf.reshape(
        idx + tf.reshape(tf.range(batch_size) * num_boxes, [-1, 1]), [-1])
    boxes = tf.reshape(
        tf.gather(tf.reshape(boxes, [-1, 4]), idx),
        [batch_size, max_output_size, 4])
    boxes = boxes * tf.cast(
        tf.reshape(tf.range(max_output_size), [1, -1, 1]) < tf.reshape(
            output_size, [-1, 1, 1]), boxes.dtype)
    scores = tf.reshape(
        tf.gather(tf.reshape(scores, [-1, 1]), idx),
        [batch_size, max_output_size])
    scores = scores * tf.cast(
        tf.reshape(tf.range(max_output_size), [1, -1]) < tf.reshape(
            output_size, [-1, 1]), scores.dtype)
    return scores, boxes


def _proposal_op_per_level(scores, boxes, anchor_boxes, image_info,
                           rpn_pre_nms_topn, rpn_post_nms_topn,
                           rpn_nms_threshold, rpn_min_size, level):
  """Proposes RoIs for the second stage nets.

  This proposal op performs the following operations.
    1. for each location i in a (H, W) grid:
         generate A anchor boxes centered on cell i
         apply predicted bbox deltas to each of the A anchors at cell i
    2. clip predicted boxes to image
    3. remove predicted boxes with either height or width < threshold
    4. sort all (proposal, score) pairs by score from highest to lowest
    5. take the top rpn_pre_nms_topn proposals before NMS
    6. apply NMS with a loose threshold (0.7) to the remaining proposals
    7. take after_nms_topN proposals after NMS
    8. return the top proposals
  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/ops/generate_proposals.py  # pylint: disable=line-too-long

  Args:
    scores: a tensor with a shape of
      [batch_size, height, width, num_anchors].
    boxes: a tensor with a shape of
      [batch_size, height, width, num_anchors * 4], in the encoded form.
    anchor_boxes: an Anchors object that contains the anchors with a shape of
      [batch_size, height, width, num_anchors * 4].
    image_info: a tensor of shape [batch_size, 5] where the three columns
      encode the input image's [height, width, scale,
      original_height, original_width]. Height and width are for
      the input to the network, not the original image; scale is the scale
      factor used to scale the network input size to the original image size.
      See dataloader.DetectionInputProcessor for details. The last two are
      original height and width. See dataloader.DetectionInputProcessor for
      details.
    rpn_pre_nms_topn: a integer number of top scoring RPN proposals to keep
      before applying NMS. This is *per FPN level* (not total).
    rpn_post_nms_topn: a integer number of top scoring RPN proposals to keep
      after applying NMS. This is the total number of RPN proposals produced.
    rpn_nms_threshold: a float number between 0 and 1 as the NMS threshold
      used on RPN proposals.
    rpn_min_size: a integer number as the minimum proposal height and width as
      both need to be greater than this number. Note that this number is at
      origingal image scale; not scale used during training or inference).
    level: a integer number for the level that the function operates on.
  Returns:
    scores: a tensor with a shape of [batch_size, rpn_post_nms_topn, 1]
      representing the scores of the proposals. It has same dtype as input
      scores.
    boxes: a tensor with a shape of [batch_size, rpn_post_nms_topn, 4]
      represneting the boxes of the proposals. The boxes are in normalized
      coordinates with a form of [ymin, xmin, ymax, xmax]. It has same dtype as
      input boxes.

  """
  with tf.name_scope('proposal-l%d' % level):
    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take the top rpn_pre_nms_topn proposals before NMS
    batch_size, h, w, num_anchors = scores.get_shape().as_list()
    scores = tf.reshape(scores, [batch_size, -1])
    boxes = tf.reshape(boxes, [batch_size, -1, 4])
    # Map scores to [0, 1] for convenince of setting min score.
    scores = tf.sigmoid(scores)

    topk_limit = (h * w * num_anchors if h * w * num_anchors < rpn_pre_nms_topn
                  else rpn_pre_nms_topn)
    anchor_boxes = tf.reshape(anchor_boxes, [batch_size, -1, 4])
    scores, top_k_indices = tf.nn.top_k(scores, k=topk_limit)
    boxes_indices = tf.reshape(
        top_k_indices + tf.expand_dims(
            tf.range(batch_size) * tf.shape(boxes)[1], 1), [-1])
    boxes = tf.reshape(
        tf.gather(tf.reshape(boxes, [-1, 4]), boxes_indices),
        [batch_size, -1, 4])
    anchor_indices = tf.reshape(
        top_k_indices + tf.expand_dims(
            tf.range(batch_size) * tf.shape(anchor_boxes)[1], 1), [-1])
    anchor_boxes = tf.reshape(
        tf.gather(tf.reshape(anchor_boxes, [-1, 4]), anchor_indices),
        [batch_size, -1, 4])

    # Transforms anchors into proposals via bbox transformations.
    boxes = anchors.batch_decode_box_outputs_op(anchor_boxes, boxes)

    # 2. clip proposals to image (may result in proposals with zero area
    # that will be removed in the next step)
    boxes = anchors.clip_boxes(boxes, image_info[:, :2])

    # 3. remove predicted boxes with either height or width < min_size
    scores, boxes = _filter_boxes(scores, boxes, rpn_min_size, image_info)

    # 6. apply loose nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    post_nms_topk_limit = (topk_limit if topk_limit < rpn_post_nms_topn else
                           rpn_post_nms_topn)
    if rpn_nms_threshold > 0:
      scores, boxes = _non_max_suppression_padded(
          scores, boxes, max_output_size=post_nms_topk_limit,
          iou_threshold=rpn_nms_threshold, level=level)

    scores, top_k_indices = _top_k(scores, k=post_nms_topk_limit,
                                   topk_sorted=True)
    boxes = tf.gather_nd(boxes, top_k_indices)

    return scores, boxes


def proposal_op(scores_outputs, box_outputs, all_anchors, image_info,
                rpn_pre_nms_topn, rpn_post_nms_topn, rpn_nms_threshold,
                rpn_min_size):
  """Proposes RoIs for the second stage nets.

  This proposal op performs the following operations.
    1. propose rois at each level.
    2. collect all proposals.
    3. keep rpn_post_nms_topn proposals by their sorted scores from the highest
       to the lowest.
  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/ops/collect_and_distribute_fpn_rpn_proposals.py  # pylint: disable=line-too-long

  Args:
    scores_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4]
    all_anchors: an Anchors object that contains the all anchors.
    image_info: a tensor of shape [batch_size, 5] where the three columns
      encode the input image's [height, width, scale,
      original_height, original_width]. Height and width are for
      the input to the network, not the original image; scale is the scale
      factor used to scale the network input size to the original image size.
      See dataloader.DetectionInputProcessor for details. The last two are
      original height and width. See dataloader.DetectionInputProcessor for
      details.
    rpn_pre_nms_topn: a integer number of top scoring RPN proposals to keep
      before applying NMS. This is *per FPN level* (not total).
    rpn_post_nms_topn: a integer number of top scoring RPN proposals to keep
      after applying NMS. This is the total number of RPN proposals produced.
    rpn_nms_threshold: a float number between 0 and 1 as the NMS threshold
      used on RPN proposals.
    rpn_min_size: a integer number as the minimum proposal height and width as
      both need to be greater than this number. Note that this number is at
      origingal image scale; not scale used during training or inference).
  Returns:
    scores: a tensor with a shape of [batch_size, rpn_post_nms_topn, 1]
      representing the scores of the proposals.
    rois: a tensor with a shape of [batch_size, rpn_post_nms_topn, 4]
      representing the boxes of the proposals. The boxes are in normalized
      coordinates with a form of [ymin, xmin, ymax, xmax].
  """
  with tf.name_scope('proposal'):
    levels = scores_outputs.keys()
    scores = []
    rois = []
    anchor_boxes = all_anchors.get_unpacked_boxes()
    for level in levels:
      # Expands the batch dimension for anchors as anchors do not have batch
      # dimension. Note that batch_size is invariant across levels.
      batch_size = scores_outputs[level].shape[0]
      anchor_boxes_batch = tf.cast(
          tf.tile(tf.expand_dims(anchor_boxes[level], axis=0),
                  [batch_size, 1, 1, 1]),
          dtype=scores_outputs[level].dtype)
      scores_per_level, boxes_per_level = _proposal_op_per_level(
          scores_outputs[level], box_outputs[level], anchor_boxes_batch,
          image_info, rpn_pre_nms_topn, rpn_post_nms_topn, rpn_nms_threshold,
          rpn_min_size, level)
      scores.append(scores_per_level)
      rois.append(boxes_per_level)
    scores = tf.concat(scores, axis=1)
    rois = tf.concat(rois, axis=1)

    with tf.name_scope('post_nms_topk'):
      # Selects the top-k rois, k being rpn_post_nms_topn or the number of total
      # anchors after non-max suppression.
      post_nms_num_anchors = scores.shape[1]
      post_nms_topk_limit = (
          post_nms_num_anchors if post_nms_num_anchors < rpn_post_nms_topn
          else rpn_post_nms_topn)

      top_k_scores, top_k_indices = _top_k(scores, k=post_nms_topk_limit,
                                           topk_sorted=True)
      top_k_rois = tf.gather_nd(rois, top_k_indices)
    top_k_scores = tf.stop_gradient(top_k_scores)
    top_k_rois = tf.stop_gradient(top_k_rois)
    return top_k_scores, top_k_rois


def rpn_net(features, min_level=2, max_level=6, num_anchors=3):
  """Region Proposal Network (RPN) for Mask-RCNN."""
  scores_outputs = {}
  box_outputs = {}
  with tf.variable_scope('rpn_net', reuse=tf.AUTO_REUSE):

    def shared_rpn_heads(features, num_anchors):
      """Shared RPN heads."""
      # TODO(chiachenc): check the channel depth of the first convoultion.
      features = tf.layers.conv2d(
          features,
          256,
          kernel_size=(3, 3),
          strides=(1, 1),
          activation=tf.nn.relu,
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          padding='same',
          name='rpn')
      # Proposal classification scores
      scores = tf.layers.conv2d(
          features,
          num_anchors,
          kernel_size=(1, 1),
          strides=(1, 1),
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          padding='valid',
          name='rpn-class')
      # Proposal bbox regression deltas
      bboxes = tf.layers.conv2d(
          features,
          4 * num_anchors,
          kernel_size=(1, 1),
          strides=(1, 1),
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          padding='valid',
          name='rpn-box')

      return scores, bboxes

    for level in range(min_level, max_level + 1):
      scores_output, box_output = shared_rpn_heads(features[level], num_anchors)
      scores_outputs[level] = scores_output
      box_outputs[level] = box_output

  return scores_outputs, box_outputs


def faster_rcnn_heads(features, boxes, num_classes=91, mlp_head_dim=1024):
  """Box and class branches for the Mask-RCNN model.

  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py  # pylint: disable=line-too-long

  Args:
    features: A dictionary with key as pyramid level and value as features.
      The features are in shape of [batch_size, height_l, width_l, num_filters].
    boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    num_classes: a integer for the number of classes.
    mlp_head_dim: a integer that is the hidden dimension in the fully-connected
      layers.
  Returns:
    class_outputs: a tensor with a shape of
      [batch_size, num_rois, num_classes], representing the class predictions.
    box_outputs: a tensor with a shape of
      [batch_size, num_rois, num_classes * 4], representing the box predictions.
  """
  with tf.variable_scope('faster_rcnn_heads'):

    # Performs multi-level RoIAlign.
    roi_features = multilevel_crop_and_resize(features, boxes, output_size=7)

    # reshape inputs beofre FC.
    batch_size, num_rois, _, _, _ = roi_features.get_shape().as_list()
    roi_features = tf.reshape(roi_features, [batch_size, num_rois, -1])
    net = tf.layers.dense(roi_features, units=mlp_head_dim,
                          activation=tf.nn.relu, name='fc6')
    net = tf.layers.dense(net, units=mlp_head_dim,
                          activation=tf.nn.relu, name='fc7')

    class_outputs = tf.layers.dense(
        net, num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        name='class-predict')
    box_outputs = tf.layers.dense(
        net, num_classes * 4,
        kernel_initializer=tf.random_normal_initializer(stddev=0.001),
        bias_initializer=tf.zeros_initializer(),
        name='box-predict')
    return class_outputs, box_outputs


def mask_rcnn_heads(features, fg_box_rois, num_classes=91, mrcnn_resolution=28):
  """Mask branch for the Mask-RCNN model.

  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/mask_rcnn_heads.py  # pylint: disable=line-too-long

  Args:
    features: A dictionary with key as pyramid level and value as features.
      The features are in shape of [batch_size, height_l, width_l, num_filters].
    fg_box_rois: A 3-D Tensor of shape [batch_size, num_masks, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    num_classes: a integer for the number of classes.
    mrcnn_resolution: a integer that is the resolution of masks.
  Returns:
    mask_outputs: a tensor with a shape of
      [batch_size, num_masks, mask_height, mask_width, num_classes],
      representing the mask predictions.
    fg_gather_indices: a tensor with a shape of [batch_size, num_masks, 2],
      representing the fg mask targets.
  Raises:
    ValueError: If boxes is not a rank-3 tensor or the last dimension of
      boxes is not 4.
  """

  def _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out):
    """Returns the stddev of random normal initialization as MSRAFill."""
    # Reference: https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.h#L445-L463  # pylint: disable=line-too-long
    # For example, kernel size is (3, 3) and fan out is 256, stddev is 0.029.
    # stddev = (2/(3*3*256))^0.5 = 0.029
    return (2 / (kernel_size[0] * kernel_size[1] * fan_out)) ** 0.5

  if fg_box_rois.shape.ndims != 3:
    raise ValueError('fg_box_rois must be of rank 3.')
  if fg_box_rois.shape[2] != 4:
    raise ValueError(
        'fg_box_rois.shape[1] is {:d}, but must be divisible by 4.'.format(
            fg_box_rois.shape[1])
    )
  with tf.variable_scope('mask_rcnn_heads'):
    batch_size, num_masks, _ = fg_box_rois.get_shape().as_list()
    # Performs multi-level RoIAlign.
    features = multilevel_crop_and_resize(features, fg_box_rois, output_size=14)
    net = tf.reshape(
        features,
        [batch_size * num_masks, 14, 14, -1])

    # TODO(chiachenc): check what is MSRAFill initialization in the reference.
    for i in range(4):
      kernel_size = (3, 3)
      fan_out = 256
      init_stddev = _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)
      net = tf.layers.conv2d(
          net,
          fan_out,
          kernel_size=kernel_size,
          strides=(1, 1),
          padding='same',
          dilation_rate=(1, 1),
          activation=tf.nn.relu,
          kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
          bias_initializer=tf.zeros_initializer(),
          name='mask-conv-l%d' % i)

    kernel_size = (2, 2)
    fan_out = 256
    init_stddev = _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)
    net = tf.layers.conv2d_transpose(
        net,
        fan_out,
        kernel_size=kernel_size,
        strides=(2, 2),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
        bias_initializer=tf.zeros_initializer(),
        name='conv5-mask')

    kernel_size = (1, 1)
    fan_out = num_classes
    init_stddev = _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)
    mask_outputs = tf.layers.conv2d(
        net,
        fan_out,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding='valid',
        kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
        bias_initializer=tf.zeros_initializer(),
        name='mask_fcn_logits')
    mask_outputs = tf.reshape(
        mask_outputs,
        [batch_size, num_masks, mrcnn_resolution, mrcnn_resolution, -1])

    return mask_outputs


def select_fg_for_masks(class_targets, box_targets, boxes,
                        proposal_to_label_map, max_num_fg=128):
  """Selects the fore ground objects for mask branch during training.

  Args:
    class_targets: a tensor of shape [batch_size, num_boxes]  representing the
      class label for each box.
    box_targets: a tensor with a shape of [batch_size, num_boxes, 4]. The tensor
      contains the ground truth pixel coordinates of the scaled images for each
      roi.
    boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    proposal_to_label_map: a tensor with a shape of [batch_size, num_boxes].
      This tensor keeps the mapping between proposal to labels.
      proposal_to_label_map[i] means the index of the ground truth instance for
      the i-th proposal.
    max_num_fg: a integer represents the number of masks per image.
  Returns:
    class_targets, boxes, proposal_to_label_map, box_targets that have
    foreground objects.
  """
  with tf.name_scope('select_fg_for_masks'):
    # Masks are for positive (fg) objects only. Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py  # pylint: disable=line-too-long
    batch_size = boxes.shape[0]
    _, fg_indices = tf.nn.top_k(
        tf.to_float(tf.greater(class_targets, 0)), k=max_num_fg)
    # Contructs indices for gather.
    indices = tf.reshape(
        fg_indices + tf.expand_dims(
            tf.range(batch_size) * tf.shape(class_targets)[1], 1), [-1])

    fg_class_targets = tf.reshape(
        tf.gather(tf.reshape(class_targets, [-1, 1]), indices),
        [batch_size, -1])
    fg_box_targets = tf.reshape(
        tf.gather(tf.reshape(box_targets, [-1, 4]), indices),
        [batch_size, -1, 4])
    fg_box_rois = tf.reshape(
        tf.gather(tf.reshape(boxes, [-1, 4]), indices), [batch_size, -1, 4])
    fg_proposal_to_label_map = tf.reshape(
        tf.gather(tf.reshape(proposal_to_label_map, [-1, 1]), indices),
        [batch_size, -1])

  return (fg_class_targets, fg_box_targets, fg_box_rois,
          fg_proposal_to_label_map)


def resnet_fpn(features,
               min_level=3,
               max_level=7,  # pylint: disable=unused-argument
               resnet_depth=50,
               is_training_bn=False):
  """ResNet feature pyramid networks."""
  # upward layers
  with tf.variable_scope('resnet%s' % resnet_depth):
    resnet_fn = resnet_v1(resnet_depth)
    u2, u3, u4, u5 = resnet_fn(features, is_training_bn)

  feats_bottom_up = {
      2: u2,
      3: u3,
      4: u4,
      5: u5,
  }

  with tf.variable_scope('resnet_fpn'):
    # lateral connections
    feats_lateral = {}
    for level in range(min_level, _RESNET_MAX_LEVEL + 1):
      feats_lateral[level] = tf.layers.conv2d(
          feats_bottom_up[level],
          filters=256,
          kernel_size=(1, 1),
          padding='same',
          name='l%d' % level)

    # add top-down path
    feats = {_RESNET_MAX_LEVEL: feats_lateral[_RESNET_MAX_LEVEL]}
    for level in range(_RESNET_MAX_LEVEL - 1, min_level - 1, -1):
      feats[level] = nearest_upsampling(
          feats[level + 1], 2) + feats_lateral[level]

    # add post-hoc 3x3 convolution kernel
    for level in range(min_level, _RESNET_MAX_LEVEL + 1):
      feats[level] = tf.layers.conv2d(
          feats[level],
          filters=256,
          strides=(1, 1),
          kernel_size=(3, 3),
          padding='same',
          name='post_hoc_d%d' % level)

    # Use original FPN P6 level implementation from CVPR'17 FPN paper instead of
    # coarse FPN levels introduced for RetinaNet.
    # Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/modeling/FPN.py  # pylint: disable=line-too-long
    feats[6] = tf.layers.max_pooling2d(
        inputs=feats[5],
        pool_size=1,
        strides=2,
        padding='valid',
        name='p6')

  return feats


def mask_rcnn(features, labels, all_anchors, mode, params):
  """Mask-RCNN classification and regression model."""
  min_level = params['min_level']
  max_level = params['max_level']
  # create feature pyramid networks
  fpn_feats = resnet_fpn(features, min_level, max_level, params['resnet_depth'],
                         params['is_training_bn'])

  def rpn_fn(feats):
    rpn_score_outputs, rpn_box_outputs = rpn_net(
        feats, min_level, max_level,
        len(params['aspect_ratios'] * params['num_scales']))
    return rpn_score_outputs, rpn_box_outputs

  # Box and class part (Fast-RCNN).
  def faster_rcnn_fn(feats, rpn_score_outputs, rpn_box_outputs):
    """Generates box and class outputs."""
    # Uses different NMS top-k parameters in different modes.
    mlperf_log.maskrcnn_print(key=mlperf_log.RPN_PRE_NMS_TOP_N_TRAIN,
                              value=params['rpn_pre_nms_topn'])
    mlperf_log.maskrcnn_print(key=mlperf_log.RPN_PRE_NMS_TOP_N_TEST,
                              value=params['test_rpn_pre_nms_topn'])
    mlperf_log.maskrcnn_print(key=mlperf_log.RPN_POST_NMS_TOP_N_TRAIN,
                              value=params['rpn_post_nms_topn'])
    mlperf_log.maskrcnn_print(key=mlperf_log.RPN_POST_NMS_TOP_N_TEST,
                              value=params['test_rpn_post_nms_topn'])
    rpn_pre_nms_topn = (
        params['rpn_pre_nms_topn'] if mode == tf.estimator.ModeKeys.TRAIN else
        params['test_rpn_pre_nms_topn'])
    rpn_post_nms_topn = (
        params['rpn_post_nms_topn'] if mode == tf.estimator.ModeKeys.TRAIN else
        params['test_rpn_post_nms_topn'])
    _, box_rois = proposal_op(rpn_score_outputs, rpn_box_outputs, all_anchors,
                              labels['image_info'], rpn_pre_nms_topn,
                              rpn_post_nms_topn, params['rpn_nms_threshold'],
                              params['rpn_min_size'])
    box_rois = tf.to_float(box_rois)

    mlperf_log.maskrcnn_print(key=mlperf_log.FG_IOU_THRESHOLD,
                              value=params['fg_thresh'])
    mlperf_log.maskrcnn_print(key=mlperf_log.BG_IOU_THRESHOLD,
                              value=params['bg_thresh_hi'])

    (box_targets, class_targets, box_rois,
     proposal_to_label_map) = proposal_label_op(
         box_rois, labels['groundtruth_data'][:, :, :4],
         labels['groundtruth_data'][:, :, 6], labels['image_info'],
         batch_size_per_im=params['batch_size_per_im'],
         fg_fraction=params['fg_fraction'], fg_thresh=params['fg_thresh'],
         bg_thresh_hi=params['bg_thresh_hi'],
         bg_thresh_lo=params['bg_thresh_lo'],
         is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    class_outputs, box_outputs = faster_rcnn_heads(
        feats, box_rois, num_classes=params['num_classes'],
        mlp_head_dim=params['fast_rcnn_mlp_head_dim'])
    return (class_outputs, box_outputs, class_targets, box_targets, box_rois,
            proposal_to_label_map)

  # Mask part (Mask-RCNN).
  def mask_rcnn_fn(feats, class_targets=None, box_targets=None, box_rois=None,
                   proposal_to_label_map=None, detections=None):
    """Generates mask outputs (and mask targets during training)."""

    if mode == tf.estimator.ModeKeys.TRAIN:
      (class_targets, box_targets, box_rois,
       proposal_to_label_map) = select_fg_for_masks(
           class_targets, box_targets, box_rois, proposal_to_label_map,
           max_num_fg=int(params['batch_size_per_im'] * params['fg_fraction']))
      mask_targets = get_mask_targets(
          box_rois, proposal_to_label_map, box_targets,
          labels['cropped_gt_masks'], params['mrcnn_resolution'])
      mask_outputs = mask_rcnn_heads(
          feats, box_rois, num_classes=params['num_classes'],
          mrcnn_resolution=params['mrcnn_resolution'])
      return (mask_outputs, class_targets, box_targets, box_rois,
              proposal_to_label_map, mask_targets)
    else:
      box_rois = detections[:, :, 1:5]
      mask_outputs = mask_rcnn_heads(
          feats, box_rois, num_classes=params['num_classes'],
          mrcnn_resolution=params['mrcnn_resolution'])
      return mask_outputs

  return fpn_feats, rpn_fn, faster_rcnn_fn, mask_rcnn_fn


def remove_variables(variables, resnet_depth=50):
  """Removes low-level variables from the input.

  Removing low-level parameters (e.g., initial convolution layer) from training
  usually leads to higher training speed and slightly better testing accuracy.
  The intuition is that the low-level architecture (e.g., ResNet-50) is able to
  capture low-level features such as edges; therefore, it does not need to be
  fine-tuned for the detection task.

  Args:
    variables: all the variables in training
    resnet_depth: the depth of ResNet model

  Returns:
    var_list: a list containing variables for training

  """
  # Freeze at conv2 based on reference model.
  # Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/modeling/ResNet.py  # pylint: disable=line-too-long
  remove_list = []
  prefix = 'resnet{}/'.format(resnet_depth)
  remove_list.append(prefix + 'conv2d/')
  remove_list.append(prefix + 'batch_normalization/')
  for i in range(1, 11):
    remove_list.append(prefix + 'conv2d_{}/'.format(i))
    remove_list.append(prefix + 'batch_normalization_{}/'.format(i))

  def _is_kept(variable):
    for rm_str in remove_list:
      if rm_str in variable.name:
        return False
    return True

  var_list = [v for v in variables if _is_kept(v)]
  return var_list


def get_mask_targets(fg_boxes, fg_proposal_to_label_map, fg_box_targets,
                     mask_gt_labels, output_size=28):
  """Crop and resize on multilevel feature pyramid.

  Args:
    fg_boxes: A 3-D tensor of shape [batch_size, num_masks, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    fg_proposal_to_label_map: A tensor of shape [batch_size, num_masks].
    fg_box_targets: a float tensor representing the box label for each box
      with a shape of [batch_size, num_masks, 4].
    mask_gt_labels: A tensor with a shape of [batch_size, M, H+4, W+4]. M is
      NUM_MAX_INSTANCES (i.e., 100 in this implementation) in each image, while
      H and W are ground truth mask size. The `+4` comes from padding of two
      zeros in both directions of height and width dimension.
    output_size: A scalar to indicate the output crop size.

  Returns:
    A 4-D tensor representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size].
  """
  # TODO(chiachenc): this function is embarassingly similar to
  #   `multilevel_crop_and_resize`. Two functions shall be refactored.
  with tf.name_scope('get_mask_targets'):
    (batch_size, num_instances, max_feature_height,
     max_feature_width) = mask_gt_labels.get_shape().as_list()
    _, num_masks = fg_proposal_to_label_map.get_shape().as_list()
    features_all = mask_gt_labels
    height_dim_size = max_feature_width
    level_dim_size = max_feature_height * height_dim_size
    batch_dim_size = num_instances * level_dim_size

    # proposal_to_label_map might have a -1 paddings.
    levels = tf.maximum(fg_proposal_to_label_map, 0)

    # Projects box location and sizes to corresponding cropped ground truth
    # mask coordinates.
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(
        value=fg_boxes, num_or_size_splits=4, axis=2)
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(
        value=fg_box_targets, num_or_size_splits=4, axis=2)
    valid_feature_width = max_feature_width - 4
    valid_feature_height = max_feature_height - 4
    y_transform = (bb_y_min - gt_y_min) * valid_feature_height / (
        gt_y_max - gt_y_min + _EPSILON) + 2
    x_transform = (bb_x_min - gt_x_min) * valid_feature_width / (
        gt_x_max - gt_x_min + _EPSILON) + 2
    h_transform = (bb_y_max - bb_y_min) * valid_feature_height / (
        gt_y_max - gt_y_min + _EPSILON)
    w_transform = (bb_x_max - bb_x_min) * valid_feature_width / (
        gt_x_max - gt_x_min + _EPSILON)

    # Compute y and x coordinate indices.
    box_grid_x = []
    box_grid_y = []
    for i in range(output_size):
      box_grid_x.append(x_transform + (0.5 + i) * w_transform / output_size)
      box_grid_y.append(y_transform + (0.5 + i) * h_transform / output_size)
    box_grid_x = tf.stack(box_grid_x, axis=2)
    box_grid_y = tf.stack(box_grid_y, axis=2)

    box_grid_y0 = tf.floor(box_grid_y)
    box_grid_x0 = tf.floor(box_grid_x)

    # Compute indices for gather operation.
    box_gridx0x1 = tf.stack([box_grid_x0, box_grid_x0 + 1], axis=3)
    box_gridy0y1 = tf.stack([box_grid_y0, box_grid_y0 + 1], axis=3)

    # Check boundary.
    box_gridx0x1 = tf.minimum(
        tf.to_float(max_feature_width-1), tf.maximum(0., box_gridx0x1))
    box_gridy0y1 = tf.minimum(
        tf.to_float(max_feature_height-1), tf.maximum(0., box_gridy0y1))

    x_indices = tf.cast(
        tf.reshape(box_gridx0x1,
                   [batch_size, num_masks, output_size * 2]), dtype=tf.int32)
    y_indices = tf.cast(
        tf.reshape(box_gridy0y1,
                   [batch_size, num_masks, output_size * 2]), dtype=tf.int32)

    indices = tf.reshape(
        tf.tile(tf.reshape(tf.range(batch_size) * batch_dim_size,
                           [batch_size, 1, 1, 1]),
                [1, num_masks, output_size * 2, output_size * 2]) +
        tf.tile(tf.reshape(levels * level_dim_size,
                           [batch_size, num_masks, 1, 1]),
                [1, 1, output_size * 2, output_size * 2]) +
        tf.tile(tf.reshape(y_indices * height_dim_size,
                           [batch_size, num_masks, output_size * 2, 1]),
                [1, 1, 1, output_size * 2]) +
        tf.tile(tf.reshape(x_indices,
                           [batch_size, num_masks, 1, output_size * 2]),
                [1, 1, output_size * 2, 1]), [-1])

    features_r2 = tf.reshape(features_all, [-1, 1])
    features_per_box = tf.reshape(
        tf.gather(features_r2, indices),
        [batch_size, num_masks, output_size * 2, output_size * 2])

    # The RoIAlign feature f can be computed by bilinear interpolation of four
    # neighboring feature points f0, f1, f2, and f3.
    # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
    #                       [f10, f11]]
    # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
    # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
    ly = box_grid_y - box_grid_y0
    lx = box_grid_x - box_grid_x0
    hy = 1.0 - ly
    hx = 1.0 - lx
    kernel_x = tf.reshape(tf.stack([hx, lx], axis=3),
                          [batch_size, num_masks, 1, output_size*2])
    kernel_y = tf.reshape(tf.stack([hy, ly], axis=3),
                          [batch_size, num_masks, output_size*2, 1])
    # Use implicit broadcast to generate the interpolation kernel. The
    # multiplier `4` is for avg pooling.
    interpolation_kernel = kernel_y * kernel_x * 4

    # Interpolate the gathered features with computed interpolation kernels.
    features_per_box *= tf.cast(interpolation_kernel,
                                dtype=features_per_box.dtype)
    features_per_box = tf.transpose(features_per_box, perm=[0, 2, 3, 1])
    features_per_box = tf.nn.avg_pool(
        features_per_box, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    features_per_box = tf.transpose(features_per_box, perm=[0, 3, 1, 2])

    # Masks are binary outputs.
    features_per_box = tf.where(
        tf.greater_equal(features_per_box, 0.5), tf.ones_like(features_per_box),
        tf.zeros_like(features_per_box))

    # mask_targets depend on box RoIs, which have gradients. This stop_gradient
    # prevents the flow of gradient to box RoIs.
    features_per_box = tf.stop_gradient(features_per_box)
  return features_per_box


def multilevel_crop_and_resize(features, boxes, output_size=7):
  """Crop and resize on multilevel feature pyramid.

    Following the ROIAlign technique (see https://arxiv.org/pdf/1703.06870.pdf,
    figure 3 for reference), we want to sample pixel level feature information
    from our feature map at the box boundaries.  For each feature map, we select
    an (output_size, output_size) set of pixels corresponding to our box
    location, and then use bilinear interpolation to select the feature value
    for each pixel.

    For performance, we perform the gather and interpolation on all layers as a
    single operation. This is op the multi-level features are first stacked and
    gathered into [2*output_size, 2*output_size] feature points. Then bilinear
    interpolation is performed on the gathered feature points to generate
    [output_size, output_size] RoIAlign feature map.

    Here is the step-by-step algorithm:
    1. Pad all multi-level features to a fixed spatial dimension and stack them
       into a Tensor of shape [batch_size, level, height, width, num_filters].
    2. The multi-level features are then gathered into a
       [batch_size, num_boxes, output_size*2, output_size*2, num_filters]
       Tensor. The Tensor contains four neighboring feature points for each
       vertice in the output grid.
       Instead of performing gather operation in one-step, a two-step gather
       algorithm is performed. First, the Tensor containining multi-level
       features is gathered into
       [batch_size, num_boxes, output_size*2, width, num_filters].
       Then the tensor is transposed to
       [batch_size, num_boxes, width, output_size*2, num_filters]
       then gathered to
       [batch_size, num_boxes, output_size*2, output_size*2, num_filters].
       The 2-step gather algorithm makes sure each gather operation performs on
       large contiguous memory.
    3. Compute the interpolation kernel of shape
       [batch_size, num_boxes, output_size*2, output_size*2]. The last 2 axis
       can be seen as stacking 2x2 interpolation kernels for all vertices in the
       output grid.
    4. Element-wise multiply the gathered features and interpolation kernel.
       Then apply 2x2 average pooling to reduce spatial dimension to
       output_size.

  Args:
    features: A dictionary with key as pyramid level and value as features.
      The features are in shape of [batch_size, height_l, width_l, num_filters].
    boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    output_size: A scalar to indicate the output crop size.

  Returns:
    A 5-D tensor representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size, num_filters].
  """
  with tf.name_scope('multilevel_crop_and_resize'):
    levels = features.keys()
    min_level = min(levels)
    max_level = max(levels)
    (batch_size, max_feature_height,
     max_feature_width, num_filters) = features[min_level].get_shape().as_list()
    _, num_boxes, _ = boxes.get_shape().as_list()
    # Stack feature pyramid into a features_all of shape
    # [batch_size, levels, height, width, num_filters].
    features_all = []
    for level in range(min_level, max_level + 1):
      features_all.append(tf.image.pad_to_bounding_box(
          features[level], 0, 0, max_feature_height, max_feature_width))
    features_all = tf.stack(features_all, axis=1)
    height_dim_size = max_feature_width
    level_dim_size = max_feature_height * height_dim_size
    batch_dim_size = len(levels) * level_dim_size

    # Assign boxes to the right level.
    box_width = boxes[:, :, 3] - boxes[:, :, 1]
    box_height = boxes[:, :, 2] - boxes[:, :, 0]
    areas_sqrt = tf.sqrt(box_height * box_width)
    levels = tf.cast(tf.floordiv(tf.log(tf.div(areas_sqrt, 224.0)),
                                 tf.log(2.0)) + 4.0, dtype=tf.int32)
    # Map levels between [min_level, max_level].
    levels = tf.minimum(max_level, tf.maximum(levels, min_level))

    # Project box location and sizes to corresponding feature levels.
    scale_to_level = tf.cast(
        tf.pow(tf.constant(2.0), tf.cast(levels, tf.float32)),
        dtype=boxes.dtype)
    boxes /= tf.expand_dims(scale_to_level, axis=2)
    box_width /= scale_to_level
    box_height /= scale_to_level

    # Map levels to [0, max_level-min_level].
    levels -= min_level

    # Compute y and x coordinate indices.
    box_grid_x = []
    box_grid_y = []
    for i in range(output_size):
      box_grid_x.append(boxes[:, :, 1] + (i + 0.5) * box_width / output_size)
      box_grid_y.append(boxes[:, :, 0] + (i + 0.5) * box_height / output_size)
    box_grid_x = tf.stack(box_grid_x, axis=2)
    box_grid_y = tf.stack(box_grid_y, axis=2)

    box_grid_y0 = tf.floor(box_grid_y)
    box_grid_x0 = tf.floor(box_grid_x)

    # Compute indices for gather operation.
    box_grid_x0 = tf.maximum(0., box_grid_x0)
    box_grid_y0 = tf.maximum(0., box_grid_y0)
    boundary = tf.cast(
        tf.expand_dims([[tf.cast(max_feature_width, tf.float32)]] / tf.pow(
            [[2.0]], tf.cast(levels, tf.float32)) - 1, 2), box_grid_x0.dtype)
    box_gridx0x1 = tf.stack([
        tf.minimum(box_grid_x0, boundary),
        tf.minimum(box_grid_x0 + 1, boundary)
    ],
                            axis=3)
    box_gridy0y1 = tf.stack([
        tf.minimum(box_grid_y0, boundary),
        tf.minimum(box_grid_y0 + 1, boundary)
    ],
                            axis=3)

    x_indices = tf.cast(
        tf.reshape(box_gridx0x1,
                   [batch_size, num_boxes, output_size * 2]), dtype=tf.int32)
    y_indices = tf.cast(
        tf.reshape(box_gridy0y1,
                   [batch_size, num_boxes, output_size * 2]), dtype=tf.int32)

    indices = tf.reshape(
        tf.tile(tf.reshape(tf.range(batch_size) * batch_dim_size,
                           [batch_size, 1, 1, 1]),
                [1, num_boxes, output_size * 2, output_size * 2]) +
        tf.tile(tf.reshape(levels * level_dim_size,
                           [batch_size, num_boxes, 1, 1]),
                [1, 1, output_size * 2, output_size * 2]) +
        tf.tile(tf.reshape(y_indices * height_dim_size,
                           [batch_size, num_boxes, output_size * 2, 1]),
                [1, 1, 1, output_size * 2]) +
        tf.tile(tf.reshape(x_indices,
                           [batch_size, num_boxes, 1, output_size * 2]),
                [1, 1, output_size * 2, 1]), [-1])

    features_r2 = tf.reshape(features_all, [-1, num_filters])
    features_per_box = tf.reshape(
        tf.gather(features_r2, indices),
        [batch_size, num_boxes, output_size * 2, output_size * 2, num_filters])

    # The RoIAlign feature f can be computed by bilinear interpolation of four
    # neighboring feature points f0, f1, f2, and f3.
    # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
    #                       [f10, f11]]
    # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
    # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
    ly = box_grid_y - box_grid_y0
    lx = box_grid_x - box_grid_x0
    hy = 1.0 - ly
    hx = 1.0 - lx
    kernel_x = tf.reshape(tf.stack([hx, lx], axis=3),
                          [batch_size, num_boxes, 1, output_size*2])
    kernel_y = tf.reshape(tf.stack([hy, ly], axis=3),
                          [batch_size, num_boxes, output_size*2, 1])
    # Use implicit broadcast to generate the interpolation kernel. The
    # multiplier `4` is for avg pooling.
    interpolation_kernel = kernel_y * kernel_x * 4

    # Interpolate the gathered features with computed interpolation kernels.
    features_per_box *= tf.cast(
        tf.expand_dims(interpolation_kernel, axis=4),
        dtype=features_per_box.dtype)
    features_per_box = tf.reshape(
        features_per_box,
        [batch_size * num_boxes, output_size*2, output_size*2, num_filters])
    features_per_box = tf.nn.avg_pool(
        features_per_box, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    features_per_box = tf.reshape(
        features_per_box,
        [batch_size, num_boxes, output_size, output_size, num_filters])

  return features_per_box
