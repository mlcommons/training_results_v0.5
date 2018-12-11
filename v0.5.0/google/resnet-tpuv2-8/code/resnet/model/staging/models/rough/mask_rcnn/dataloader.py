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
"""Data loader and processing.

Defines input_fn of Mask-RCNN for TF Estimator. The input_fn includes training
data for category classification, bounding box regression, and number of
positive examples to normalize the loss during training.

"""

import tensorflow as tf

import anchors
from object_detection import preprocessor
from object_detection import tf_example_decoder
from mlperf_compliance import mlperf_log

MAX_NUM_INSTANCES = 100


class InputProcessor(object):
  """Base class of Input processor."""

  def __init__(self, image, output_size, short_side_image_size,
               long_side_max_image_size):
    """Initializes a new `InputProcessor`.

    This InputProcessor is tailored for MLPerf. The reference implementation
    resizes images as the following:
      1. Resize the short side to 800 pixels while keeping the aspect ratio.
      2. Clip the long side at a maximum of 1333 pixels.
    In order to comply with MLPerf reference models, this TPU implementation
    follows the same guideline. However, due to the fact that TPU uses a fixed
    size image, this implementation pads the resized image with zero such that
    the image size is 1408x1408.

    Args:
      image: The input image before processing.
      output_size: The output image size after calling resize_and_crop_image
        function.
      short_side_image_size: The image size for the short side. This is analogy
        to cfg.TRAIN.scales in the MLPerf reference model.
      long_side_max_image_size: The maximum image size for the long side. This
        is analogy to cfg.TRAIN.max_size in the MLPerf reference model.
    """
    self._image = image
    self._output_size = output_size
    self._short_side_image_size = short_side_image_size
    self._long_side_max_image_size = long_side_max_image_size
    # Parameters to control rescaling and shifting during preprocessing.
    # Image scale defines scale from original image to scaled image.
    self._image_scale = tf.constant(1.0)
    # The integer height and width of scaled image.
    self._scaled_height = tf.shape(image)[0]
    self._scaled_width = tf.shape(image)[1]
    self._ori_height = tf.shape(image)[0]
    self._ori_width = tf.shape(image)[1]
    # The x and y translation offset to crop scaled image to the output size.
    self._crop_offset_y = tf.constant(0)
    self._crop_offset_x = tf.constant(0)

  def normalize_image(self):
    """Normalize the image to zero mean and unit variance."""
    # The image normalization is identical to Cloud TPU ResNet.
    self._image = tf.image.convert_image_dtype(self._image, dtype=tf.float32)
    offset = tf.constant([0.485, 0.456, 0.406])
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    self._image -= offset

    # This is simlar to `PIXEL_MEANS` in the reference. Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/core/config.py#L909  # pylint: disable=line-too-long
    mlperf_log.maskrcnn_print(key=mlperf_log.INPUT_NORMALIZATION_STD,
                              value=[0.229, 0.224, 0.225])
    scale = tf.constant([0.229, 0.224, 0.225])
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    self._image /= scale

  def set_training_random_scale_factors(self, scale_min, scale_max):
    """Set the parameters for multiscale training."""
    # Select a random scale factor.
    random_scale_factor = tf.random_uniform([], scale_min, scale_max)
    scaled_size = tf.to_int32(random_scale_factor *
                              tf.to_float(self._short_side_image_size))

    # Recompute the accurate scale_factor using rounded scaled image size.
    height = tf.shape(self._image)[0]
    width = tf.shape(self._image)[1]
    min_image_size = tf.to_float(tf.minimum(height, width))
    image_scale = tf.to_float(scaled_size) / min_image_size

    # Select non-zero random offset (x, y) if scaled image is larger than
    # self._output_size.
    scaled_height = tf.to_int32(tf.to_float(height) * image_scale)
    scaled_width = tf.to_int32(tf.to_float(width) * image_scale)

    self._image_scale = image_scale
    self._scaled_height = scaled_height
    self._scaled_width = scaled_width
    self._crop_offset_x, self._crop_offset_y = self._get_offset(scale_min,
                                                                scale_max)
    return image_scale

  def set_scale_factors_to_mlperf_reference_size(self):
    """Set the parameters to resize the image according to MLPerf reference."""
    # Compute the scale_factor using rounded scaled image size.
    height = tf.shape(self._image)[0]
    width = tf.shape(self._image)[1]
    min_image_size = tf.to_float(tf.minimum(height, width))
    image_scale = tf.to_float(self._short_side_image_size) / min_image_size
    scaled_height = tf.to_int32(tf.to_float(height) * image_scale)
    scaled_width = tf.to_int32(tf.to_float(width) * image_scale)
    self._image_scale = image_scale
    self._scaled_height = scaled_height
    self._scaled_width = scaled_width
    return image_scale

  def resize_and_crop_image(self, method=tf.image.ResizeMethod.BILINEAR):
    """Resize input image and crop it to the self._output dimension."""
    scaled_image = tf.image.resize_images(
        self._image, [self._scaled_height, self._scaled_width], method=method)
    scaled_image = scaled_image[
        self._crop_offset_y:self._crop_offset_y + self.get_height_length(),
        self._crop_offset_x:self._crop_offset_x + self.get_width_length(), :]

    is_height_short_side = tf.less(self._scaled_height, self._scaled_width)
    output_image = tf.cond(
        is_height_short_side,
        lambda: tf.image.pad_to_bounding_box(scaled_image, 0, 0, self._output_size[0], self._output_size[1]),  # pylint: disable=line-too-long
        lambda: tf.image.pad_to_bounding_box(scaled_image, 0, 0, self._output_size[1], self._output_size[0])  # pylint: disable=line-too-long
    )

    return output_image

  @property
  def offset_x(self):
    return self._crop_offset_x

  @property
  def offset_y(self):
    return self._crop_offset_y

  def _get_offset(self, scale_min, scale_max):
    """Calculates the random offset in (x, y) in an image."""
    is_height_short_side = tf.less(self._scaled_height, self._scaled_width)
    offset_y = tf.where(
        is_height_short_side,
        tf.to_float(self._scaled_height - self._short_side_image_size),
        tf.to_float(self._scaled_height - self._long_side_max_image_size))
    offset_x = tf.where(
        is_height_short_side,
        tf.to_float(self._scaled_width - self._long_side_max_image_size),
        tf.to_float(self._scaled_width - self._short_side_image_size),
        )
    offset_y = tf.maximum(0.0, offset_y) * tf.random_uniform([], 0, 1)
    offset_x = tf.maximum(0.0, offset_x) * tf.random_uniform([], 0, 1)
    offset_y = tf.to_int32(offset_y)
    offset_x = tf.to_int32(offset_x)
    # If scale_min and scale_max are both 1.0, do not set offsets.
    offset_y = tf.where(
        tf.logical_and(
            tf.equal(scale_min, scale_max), tf.equal(scale_min, 1.0)),
        0, offset_y)
    offset_x = tf.where(
        tf.logical_and(
            tf.equal(scale_min, scale_max), tf.equal(scale_min, 1.0)),
        0, offset_x)
    return offset_x, offset_y

  def get_height_length(self):
    is_height_short_side = tf.less(self._scaled_height, self._scaled_width)
    return tf.where(is_height_short_side,
                    self._short_side_image_size,
                    tf.minimum(self._scaled_height - self.offset_y,
                               self._long_side_max_image_size))

  def get_width_length(self):
    is_height_short_side = tf.less(self._scaled_height, self._scaled_width)
    return tf.where(is_height_short_side,
                    tf.minimum(self._scaled_width - self.offset_x,
                               self._long_side_max_image_size),
                    self._short_side_image_size)

  @property
  def get_original_height(self):
    # Return original image height.
    return self._ori_height

  @property
  def get_original_width(self):
    # Return original image width.
    return self._ori_width


class InstanceSegmentationInputProcessor(InputProcessor):
  """Input processor for object detection."""

  def __init__(self, image, output_size, short_side_image_size,
               long_side_max_image_size, boxes=None, classes=None, masks=None):
    InputProcessor.__init__(self, image, output_size, short_side_image_size,
                            long_side_max_image_size)
    mlperf_log.maskrcnn_print(key=mlperf_log.MIN_IMAGE_SIZE,
                              value=short_side_image_size)
    mlperf_log.maskrcnn_print(key=mlperf_log.MAX_IMAGE_SIZE,
                              value=long_side_max_image_size)
    self._boxes = boxes
    self._classes = classes
    self._masks = masks

  def random_horizontal_flip(self):
    """Randomly flip input image and bounding boxes."""
    self._image, self._boxes, self._masks = preprocessor.random_horizontal_flip(
        self._image, boxes=self._boxes, masks=self._masks)

  def clip_boxes(self, boxes):
    """Clip boxes to fit in an image."""
    boxes = tf.where(tf.less(boxes, 0), tf.zeros_like(boxes), boxes)
    is_height_short_side = tf.less(self._scaled_height, self._scaled_width)
    bound = tf.where(
        is_height_short_side,
        tf.convert_to_tensor(
            [self._output_size[0] - 1, self._output_size[1] - 1] * 2,
            dtype=tf.float32),
        tf.convert_to_tensor(
            [self._output_size[1] - 1, self._output_size[0] - 1] * 2,
            dtype=tf.float32))
    boxes = tf.where(
        tf.greater(boxes, bound), bound * tf.ones_like(boxes), boxes)
    return boxes

  def resize_and_crop_boxes(self):
    """Resize boxes and crop it to the self._output dimension."""
    boxlist = preprocessor.box_list.BoxList(self._boxes)
    boxes = preprocessor.box_list_scale(
        boxlist, self._scaled_height, self._scaled_width).get()
    # Adjust box coordinates based on the offset.
    box_offset = tf.stack([self._crop_offset_y, self._crop_offset_x,
                           self._crop_offset_y, self._crop_offset_x,])
    boxes -= tf.to_float(tf.reshape(box_offset, [1, 4]))
    # Clip the boxes.
    boxes = self.clip_boxes(boxes)
    # Filter out ground truth boxes that are all zeros and corresponding classes
    # and masks.
    indices = tf.where(tf.not_equal(tf.reduce_sum(boxes, axis=1), 0))
    boxes = tf.gather_nd(boxes, indices)
    classes = tf.gather_nd(self._classes, indices)
    self._masks = tf.gather_nd(self._masks, indices)
    return boxes, classes

  def resize_and_crop_masks(self,
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    """Resize masks and crop it to the self._output dimension."""
    # Resize the 3D tensor self._masks to a 4D scaled_masks.
    scaled_masks = tf.image.resize_images(
        tf.expand_dims(self._masks, -1),
        [self._scaled_height, self._scaled_width], method=method)

    scaled_masks = scaled_masks[
        :, self._crop_offset_y:self._crop_offset_y + self.get_height_length(),
        self._crop_offset_x:self._crop_offset_x + self.get_width_length(), :]

    is_height_short_side = tf.less(self._scaled_height, self._scaled_width)

    def padded_bounding_box_fn():
      return tf.cond(
          is_height_short_side,
          lambda: tf.image.pad_to_bounding_box(scaled_masks, 0, 0, self._output_size[0], self._output_size[1]),  # pylint: disable=line-too-long
          lambda: tf.image.pad_to_bounding_box(scaled_masks, 0, 0, self._output_size[1], self._output_size[0])  # pylint: disable=line-too-long
      )

    def zeroed_box_fn():
      return tf.cond(
          is_height_short_side,
          lambda: tf.zeros([0, self._output_size[0], self._output_size[1], 1]),
          lambda: tf.zeros([0, self._output_size[1], self._output_size[0], 1]))

    num_masks = tf.shape(scaled_masks)[0]
    # Check if there is any instance in this image or not.
    # pylint: disable=g-long-lambda
    scaled_masks = tf.cond(num_masks > 0, padded_bounding_box_fn, zeroed_box_fn)
    scaled_masks = scaled_masks[:, :, :, 0]
    # pylint: enable=g-long-lambda

    return scaled_masks

  def crop_gt_masks(self, instance_masks, boxes, gt_mask_size, image_size):
    """Crops the ground truth binary masks and resize to fixed-size masks."""
    num_boxes = tf.shape(boxes)[0]
    num_masks = tf.shape(instance_masks)[0]
    assert_length = tf.Assert(
        tf.equal(num_boxes, num_masks), [num_masks])
    is_height_short_side = tf.less(self._scaled_height, self._scaled_width)
    scale_sizes = tf.where(
        is_height_short_side,
        tf.convert_to_tensor(
            [image_size[0], image_size[1]] * 2, dtype=tf.float32),
        tf.convert_to_tensor(
            [image_size[1], image_size[0]] * 2, dtype=tf.float32))
    boxes = boxes / scale_sizes
    with tf.control_dependencies([assert_length]):
      cropped_gt_masks = tf.image.crop_and_resize(
          image=tf.expand_dims(instance_masks, axis=-1), boxes=boxes,
          box_ind=tf.range(num_masks, dtype=tf.int32),
          crop_size=[gt_mask_size, gt_mask_size],
          method='bilinear')[:, :, :, 0]
    cropped_gt_masks = tf.pad(
        cropped_gt_masks, paddings=tf.constant([[0, 0,], [2, 2,], [2, 2]]),
        mode='CONSTANT', constant_values=0.)
    return cropped_gt_masks

  @property
  def image_scale(self):
    # Return image scale from original image to scaled image.
    return self._image_scale

  @property
  def image_scale_to_original(self):
    # Return image scale from scaled image to original image.
    return 1.0 / self._image_scale


def pad_to_fixed_size(data, pad_value, output_shape):
  """Pad data to a fixed length at the first dimension.

  Args:
    data: Tensor to be padded to output_shape.
    pad_value: A constant value assigned to the paddings.
    output_shape: The output shape of a 2D tensor.

  Returns:
    The Padded tensor with output_shape [max_num_instances, dimension].
  """
  max_num_instances = output_shape[0]
  dimension = output_shape[1]
  data = tf.reshape(data, [-1, dimension])
  num_instances = tf.shape(data)[0]
  assert_length = tf.Assert(
      tf.less_equal(num_instances, max_num_instances), [num_instances])
  with tf.control_dependencies([assert_length]):
    pad_length = max_num_instances - num_instances
  paddings = pad_value * tf.ones([pad_length, dimension])
  padded_data = tf.concat([data, paddings], axis=0)
  padded_data = tf.reshape(padded_data, output_shape)
  return padded_data


class InputReader(object):
  """Input reader for dataset."""

  def __init__(self, file_pattern, mode=tf.estimator.ModeKeys.TRAIN):
    self._file_pattern = file_pattern
    self._max_num_instances = MAX_NUM_INSTANCES
    self._mode = mode

  def __call__(self, params):
    image_size = params['dynamic_image_size'] if params[
        'dynamic_input_shapes'] else (params['image_size'],
                                      params['image_size'])
    input_anchors = anchors.Anchors(
        params['min_level'], params['max_level'], params['num_scales'],
        params['aspect_ratios'], params['anchor_scale'], image_size)
    anchor_labeler = anchors.AnchorLabeler(
        input_anchors, params['num_classes'], params['rpn_positive_overlap'],
        params['rpn_negative_overlap'], params['rpn_batch_size_per_im'],
        params['rpn_fg_fraction'])

    if params['dynamic_input_shapes']:
      height_long_side_image_size = image_size[::-1]
      height_long_side_input_anchors = anchors.Anchors(
          params['min_level'], params['max_level'], params['num_scales'],
          params['aspect_ratios'], params['anchor_scale'],
          height_long_side_image_size)
      height_long_side_anchor_labeler = anchors.AnchorLabeler(
          height_long_side_input_anchors, params['num_classes'],
          params['rpn_positive_overlap'], params['rpn_negative_overlap'],
          params['rpn_batch_size_per_im'], params['rpn_fg_fraction'])

    example_decoder = tf_example_decoder.TfExampleDecoder(
        use_instance_mask=True)

    def _dataset_parser(value):
      """Parse data to a fixed dimension input image and learning targets.

      Args:
        value: A dictionary contains an image and groundtruth annotations.

      Returns:
        image: Image tensor that is preproessed to have normalized value and
          fixed dimension [image_size, image_size, 3]
        cls_targets_dict: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, num_anchors]. The height_l and width_l
          represent the dimension of class logits at l-th level.
        box_targets_dict: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, num_anchors * 4]. The height_l and
          width_l represent the dimension of bounding box regression output at
          l-th level.
        num_positives: Number of positive anchors in the image.
        source_id: Source image id. Default value -1 if the source id is empty
          in the groundtruth annotation.
        image_scale: Scale of the proccessed image to the original image.
        boxes: Groundtruth bounding box annotations. The box is represented in
          [y1, x1, y2, x2] format. The tennsor is padded with -1 to the fixed
          dimension [self._max_num_instances, 4].
        is_crowds: Groundtruth annotations to indicate if an annotation
          represents a group of instances by value {0, 1}. The tennsor is
          padded with 0 to the fixed dimension [self._max_num_instances].
        areas: Groundtruth areas annotations. The tennsor is padded with -1
          to the fixed dimension [self._max_num_instances].
        classes: Groundtruth classes annotations. The tennsor is padded with -1
          to the fixed dimension [self._max_num_instances].
      """
      with tf.name_scope('parser'):
        data = example_decoder.decode(value)
        source_id = data['source_id']
        image = data['image']
        instance_masks = data['groundtruth_instance_masks']
        boxes = data['groundtruth_boxes']
        classes = data['groundtruth_classes']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        areas = data['groundtruth_area']
        is_crowds = data['groundtruth_is_crowd']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        if not params['use_category']:
          classes = tf.cast(tf.greater(classes, 0), dtype=tf.float32)

        if (params['skip_crowd_during_training'] and
            self._mode == tf.estimator.ModeKeys.TRAIN):
          indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
          classes = tf.gather_nd(classes, indices)
          boxes = tf.gather_nd(boxes, indices)
          instance_masks = tf.gather_nd(instance_masks, indices)

        input_processor = InstanceSegmentationInputProcessor(
            image, image_size, params['short_side_image_size'],
            params['long_side_max_image_size'], boxes, classes, instance_masks)
        input_processor.normalize_image()
        if (self._mode == tf.estimator.ModeKeys.TRAIN and
            params['input_rand_hflip']):
          input_processor.random_horizontal_flip()
        if self._mode == tf.estimator.ModeKeys.TRAIN:
          input_processor.set_training_random_scale_factors(
              params['train_scale_min'], params['train_scale_max'])
        else:
          input_processor.set_scale_factors_to_mlperf_reference_size()
        image = input_processor.resize_and_crop_image()
        boxes, classes = input_processor.resize_and_crop_boxes()
        instance_masks = input_processor.resize_and_crop_masks()
        cropped_gt_masks = input_processor.crop_gt_masks(
            instance_masks, boxes, params['gt_mask_size'], image_size)

        # Assign anchors.
        if params['dynamic_input_shapes']:
          is_height_short_side = tf.less(
              input_processor._scaled_height,  # pylint: disable=protected-access
              input_processor._scaled_width)  # pylint: disable=protected-access
          score_targets, box_targets = tf.cond(
              is_height_short_side,
              lambda: anchor_labeler.label_anchors(boxes, classes),
              lambda: height_long_side_anchor_labeler.label_anchors(boxes, classes))  # pylint: disable=line-too-long
        else:
          score_targets, box_targets = anchor_labeler.label_anchors(
              boxes, classes)

        source_id = tf.where(tf.equal(source_id, tf.constant('')), '-1',
                             source_id)
        source_id = tf.string_to_number(source_id)

        image_scale = input_processor.image_scale_to_original
        scaled_height = input_processor.get_height_length()
        scaled_width = input_processor.get_width_length()
        image_info = tf.stack(
            [tf.to_float(scaled_height),
             tf.to_float(scaled_width),
             image_scale,
             tf.to_float(input_processor.get_original_height),
             tf.to_float(input_processor.get_original_width),
            ])
        # Pad groundtruth data for evaluation.
        boxes *= image_scale
        is_crowds = tf.cast(is_crowds, dtype=tf.float32)
        boxes = pad_to_fixed_size(boxes, -1, [self._max_num_instances, 4])
        is_crowds = pad_to_fixed_size(is_crowds, 0,
                                      [self._max_num_instances, 1])
        areas = pad_to_fixed_size(areas, -1, [self._max_num_instances, 1])
        classes = pad_to_fixed_size(classes, -1, [self._max_num_instances, 1])
        # Pads cropped_gt_masks.
        cropped_gt_masks = tf.reshape(
            cropped_gt_masks, [self._max_num_instances, -1])
        cropped_gt_masks = pad_to_fixed_size(
            cropped_gt_masks, -1,
            [self._max_num_instances, (params['gt_mask_size'] + 4) ** 2])
        cropped_gt_masks = tf.reshape(
            cropped_gt_masks,
            [self._max_num_instances, params['gt_mask_size'] + 4,
             params['gt_mask_size'] + 4])

        if params['use_bfloat16']:
          image = tf.cast(image, dtype=tf.bfloat16)
        return (image, score_targets, box_targets, source_id, image_info,
                boxes, is_crowds, areas, classes, cropped_gt_masks)

    # batch_size = params['batch_size']
    batch_size = params['batch_size'] if 'batch_size' in params else 1
    dataset = tf.data.Dataset.list_files(
        self._file_pattern, shuffle=(self._mode == tf.estimator.ModeKeys.TRAIN))
    if self._mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.repeat()

    # Prefetch data from files.
    def _prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename).prefetch(1)
      return dataset

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            _prefetch_dataset, cycle_length=32,
            sloppy=(self._mode == tf.estimator.ModeKeys.TRAIN)))
    if self._mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(64)

    # Parse the fetched records to input tensors for model function.
    dataset = dataset.map(_dataset_parser, num_parallel_calls=64)

    if params['dynamic_input_shapes']:

      def key_func(image, *args):
        del args
        return tf.cast(tf.shape(image)[0], dtype=tf.int64)

      def reduce_func(unused_key, dataset):
        return dataset.batch(batch_size, drop_remainder=True)

      dataset = dataset.apply(
          tf.contrib.data.group_by_window(
              key_func=key_func,
              reduce_func=reduce_func,
              window_size=params['global_batch_size']))
    else:
      dataset = dataset.prefetch(batch_size)
      dataset = dataset.batch(batch_size, drop_remainder=True)

    def _process_example(images, score_targets, box_targets, source_ids,
                         image_info, boxes, is_crowds, areas, classes,
                         cropped_gt_masks):
      """Processes one batch of data."""
      # Transposes images from (N, H, W, C)->(H, W, N, C). As batch size is
      # less than 8, the batch goes to the second minor dimension.
      if (params['transpose_input'] and
          self._mode == tf.estimator.ModeKeys.TRAIN):
        images = tf.transpose(images, [1, 2, 0, 3])

      labels = {}
      for level in range(params['min_level'], params['max_level'] + 1):
        labels['score_targets_%d' % level] = score_targets[level]
        labels['box_targets_%d' % level] = box_targets[level]
      # Concatenate groundtruth annotations to a tensor.
      groundtruth_data = tf.concat([boxes, is_crowds, areas, classes], axis=2)
      labels['source_ids'] = source_ids
      labels['groundtruth_data'] = groundtruth_data
      labels['image_info'] = image_info
      labels['cropped_gt_masks'] = cropped_gt_masks
      if self._mode == tf.estimator.ModeKeys.PREDICT:
        features = dict(
            images=images,
            image_info=image_info,
            groundtruth_data=groundtruth_data,
            source_ids=source_ids)
        return features
      elif params['dynamic_input_shapes']:
        # For dynamic input shapes, we have 2 TPU programs. A tf.cond op is run
        # on the host side to decide which TPU program to launch. As we have
        # data prefetch in device side, the data for evaluating the shape needs
        # to sent back from device to host. Thus we retun `images` shape here
        # explictly to avoid copy the entire `images` back.
        return tf.shape(images), images, labels
      else:
        return images, labels

    dataset = dataset.map(_process_example)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset
