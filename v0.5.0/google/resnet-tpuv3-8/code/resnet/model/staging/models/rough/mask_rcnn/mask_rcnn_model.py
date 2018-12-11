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
"""Model defination for the Mask-RCNN Model.

Defines model_fn of Mask-RCNN for TF Estimator. The model_fn includes Mask-RCNN
model architecture, loss function, learning rate schedule, and evaluation
procedure.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

import anchors
import coco_metric
import mask_rcnn_architecture
from mlperf_compliance import mlperf_log

_DEFAULT_BATCH_SIZE = 16
_WEIGHT_DECAY = 1e-4


def update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule.

  This function adjusts the learning schedule based on the given batch size and
  other LR-schedule-related parameters. The default values specified in the
  default_hparams() are for training with a batch size of 64 and COCO dataset.

  For other batch sizes that train with the same schedule w.r.t. the number of
  epochs, this function handles the learning rate schedule.

    For batch size=16, the default values are listed below:
      learning_rate=0.02,
      lr_warmup_epoch=1.0,
      first_lr_drop_epoch=8.0,
      second_lr_drop_epoch=11.0;
    The values are converted to a LR schedule listed below:
      adjusted_learning_rate=0.02,
      lr_warmup_step=7500,
      first_lr_drop_step=60000,
      second_lr_drop_step=82500;
    For batch size=8, the default values will have the following LR shedule:
      adjusted_learning_rate=0.01,
      lr_warmup_step=15000,
      first_lr_drop_step=120000,
      second_lr_drop_step=165000;

  For training with different schedules, such as extended schedule with double
  number of epochs, adjust the values in default_hparams(). Note that the
  values are w.r.t. a batch size of 16.

    For batch size=16, 1x schedule (default values),
      learning_rate=0.02,
      lr_warmup_step=7500,
      first_lr_drop_step=60000,
      second_lr_drop_step=82500;
    For batch size=16, 2x schedule, *lr_drop_epoch are doubled.
      first_lr_drop_epoch=16.0,
      second_lr_drop_epoch=22.0;
    The values are converted to a LR schedule listed below:
      adjusted_learning_rate=0.02,
      lr_warmup_step=7500,
      first_lr_drop_step=120000,
      second_lr_drop_step=165000.

  Args:
    params: a parameter dictionary that includes learning_rate,
      lr_warmup_epoch, first_lr_drop_epoch, and second_lr_drop_epoch.
  """
  # params['batch_size'] is per-shard within model_fn if use_tpu=true.
  batch_size = (params['batch_size'] * params['num_shards'] if params['use_tpu']
                else params['batch_size'])
  scaling_factor = batch_size / _DEFAULT_BATCH_SIZE
  # Learning rate is proportional to the batch size
  params['adjusted_learning_rate'] = params['learning_rate'] * scaling_factor
  steps_per_epoch = params['num_examples_per_epoch'] / batch_size
  params['lr_warmup_step'] = int(
      params['lr_warmup_epoch'] * steps_per_epoch * scaling_factor)
  params['first_lr_drop_step'] = int(
      params['first_lr_drop_epoch'] * steps_per_epoch +
      params['lr_warmup_step'] * (1.0 - 1.0/scaling_factor))
  params['second_lr_drop_step'] = int(
      params['second_lr_drop_epoch'] * steps_per_epoch +
      params['lr_warmup_step'] * (1.0 - 1.0/scaling_factor))


def learning_rate_schedule(adjusted_learning_rate, lr_warmup_init,
                           lr_warmup_step, first_lr_drop_step,
                           second_lr_drop_step, global_step):
  """Handles linear scaling rule, gradual warmup, and LR decay."""
  # lr_warmup_init is the starting learning rate; the learning rate is linearly
  # scaled up to the full learning rate after `lr_warmup_steps` before decaying.
  linear_warmup = (lr_warmup_init +
                   (tf.cast(global_step, dtype=tf.float32) / lr_warmup_step *
                    (adjusted_learning_rate - lr_warmup_init)))
  learning_rate = tf.where(global_step < lr_warmup_step,
                           linear_warmup, adjusted_learning_rate)
  lr_schedule = [[1.0, lr_warmup_step],
                 [0.1, first_lr_drop_step],
                 [0.01, second_lr_drop_step]]
  for mult, start_global_step in lr_schedule:
    learning_rate = tf.where(global_step < start_global_step, learning_rate,
                             adjusted_learning_rate * mult)
  return learning_rate


def _rpn_score_loss(score_outputs, score_targets, normalizer=1.0):
  """Computes score loss."""
  # score_targets has three values: (1) score_targets[i]=1, the anchor is a
  # positive sample. (2) score_targets[i]=0, negative. (3) score_targets[i]=-1,
  # the anchor is don't care (ignore).
  with tf.name_scope('rpn_score_loss'):
    mask = tf.logical_or(tf.equal(score_targets, 1), tf.equal(score_targets, 0))
    score_targets = tf.maximum(score_targets, tf.zeros_like(score_targets))
    # RPN score loss is sum over all except ignored samples.
    score_loss = tf.losses.sigmoid_cross_entropy(
        score_targets, score_outputs, weights=mask,
        reduction=tf.losses.Reduction.SUM)
    score_loss /= normalizer
    return score_loss


def _rpn_box_loss(box_outputs, box_targets, normalizer=1.0, delta=1./9):
  """Computes box regression loss."""
  # delta is typically around the mean value of regression target.
  # for instances, the regression targets of 512x512 input with 6 anchors on
  # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
  with tf.name_scope('rpn_box_loss'):
    mask = tf.not_equal(box_targets, 0.0)
    # The loss is normalized by the sum of non-zero weights before additional
    # normalizer provided by the function caller.
    box_loss = tf.losses.huber_loss(
        box_targets,
        box_outputs,
        weights=mask,
        delta=delta,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    box_loss /= normalizer
    return box_loss


def rpn_loss(score_outputs, box_outputs, labels, params):
  """Computes total RPN detection loss.

  Computes total RPN detection loss including box and score from all levels.
  Args:
    score_outputs: an OrderDict with keys representing levels and values
      representing scores in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
    labels: the dictionary that returned from dataloader that includes
      groundturth targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
  Returns:
    total_rpn_loss: a float tensor representing total loss reduced from
      score and box losses from all levels.
    rpn_score_loss: a float tensor representing total score loss.
    rpn_box_loss: a float tensor representing total box regression loss.
  """
  with tf.name_scope('rpn_loss'):
    levels = score_outputs.keys()

    score_losses = []
    box_losses = []
    for level in levels:
      score_targets_at_level = labels['score_targets_%d' % level]
      box_targets_at_level = labels['box_targets_%d' % level]
      score_losses.append(
          _rpn_score_loss(
              score_outputs[level],
              score_targets_at_level,
              normalizer=tf.to_float(
                  params['batch_size'] * params['rpn_batch_size_per_im'])))
      box_losses.append(
          _rpn_box_loss(box_outputs[level], box_targets_at_level))

    # Sum per level losses to total loss.
    rpn_score_loss = tf.add_n(score_losses)
    rpn_box_loss = params['rpn_box_loss_weight'] * tf.add_n(box_losses)
    total_rpn_loss = rpn_score_loss + rpn_box_loss
    return total_rpn_loss, rpn_score_loss, rpn_box_loss


def _fast_rcnn_class_loss(class_outputs, class_targets_one_hot, normalizer=1.0):
  """Computes classification loss."""
  with tf.name_scope('fast_rcnn_class_loss'):
    # The loss is normalized by the sum of non-zero weights before additional
    # normalizer provided by the function caller.
    class_loss = tf.losses.softmax_cross_entropy(
        class_targets_one_hot, class_outputs,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    class_loss /= normalizer
    return class_loss


def _fast_rcnn_box_loss(box_outputs, box_targets, class_targets, normalizer=1.0,
                        delta=1.):
  """Computes box regression loss."""
  # delta is typically around the mean value of regression target.
  # for instances, the regression targets of 512x512 input with 6 anchors on
  # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].
  with tf.name_scope('fast_rcnn_box_loss'):
    mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2),
                   [1, 1, 4])
    # The loss is normalized by the sum of non-zero weights before additional
    # normalizer provided by the function caller.
    box_loss = tf.losses.huber_loss(
        box_targets,
        box_outputs,
        weights=mask,
        delta=delta,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    box_loss /= normalizer
    return box_loss


def fast_rcnn_loss(class_outputs, box_outputs, class_targets, box_targets,
                   params):
  """Computes the box and class loss (Fast-RCNN branch) of Mask-RCNN.

  This function implements the classification and box regression loss of the
  Fast-RCNN branch in Mask-RCNN. As the `box_outputs` produces `num_classes`
  boxes for each RoI, the reference model expands `box_targets` to match the
  shape of `box_outputs` and selects only the target that the RoI has a maximum
  overlap. (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py)  # pylint: disable=line-too-long
  Instead, this function selects the `box_outputs` by the `class_targets` so
  that it doesn't expand `box_targets`.

  The loss computation has two parts: (1) classification loss is softmax on all
  RoIs. (2) box loss is smooth L1-loss on only positive samples of RoIs.
  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py  # pylint: disable=line-too-long


  Args:
    class_outputs: a float tensor representing the class prediction for each box
      with a shape of [batch_size, num_boxes, num_classes].
    box_outputs: a float tensor representing the box prediction for each box
      with a shape of [batch_size, num_boxes, num_classes * 4].
    class_targets: a float tensor representing the class label for each box
      with a shape of [batch_size, num_boxes].
    box_targets: a float tensor representing the box label for each box
      with a shape of [batch_size, num_boxes, 4].
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
  Returns:
    total_loss: a float tensor representing total loss reducing from
      class and box losses from all levels.
    cls_loss: a float tensor representing total class loss.
    box_loss: a float tensor representing total box regression loss.
  """
  with tf.name_scope('fast_rcnn_loss'):
    class_targets = tf.to_int32(class_targets)
    class_targets_one_hot = tf.one_hot(class_targets, params['num_classes'])
    class_loss = _fast_rcnn_class_loss(
        class_outputs, class_targets_one_hot)

    # Selects the box from `box_outputs` based on `class_targets`, with which
    # the box has the maximum overlap.
    batch_size, num_rois, _ = box_outputs.get_shape().as_list()
    box_outputs = tf.reshape(box_outputs,
                             [batch_size, num_rois, params['num_classes'], 4])

    box_indices = tf.reshape(
        class_targets + tf.tile(
            tf.expand_dims(
                tf.range(batch_size) * num_rois * params['num_classes'], 1),
            [1, num_rois]) + tf.tile(
                tf.expand_dims(tf.range(num_rois) * params['num_classes'], 0),
                [batch_size, 1]), [-1])

    box_outputs = tf.matmul(
        tf.one_hot(
            box_indices,
            batch_size * num_rois * params['num_classes'],
            dtype=box_outputs.dtype), tf.reshape(box_outputs, [-1, 4]))
    box_outputs = tf.reshape(box_outputs, [batch_size, -1, 4])

    box_loss = (params['fast_rcnn_box_loss_weight'] *
                _fast_rcnn_box_loss(box_outputs, box_targets, class_targets))
    total_loss = class_loss + box_loss
    return total_loss, class_loss, box_loss


def mask_rcnn_loss(mask_outputs, mask_targets, select_class_targets, params):
  """Computes the mask loss of Mask-RCNN.

  This function implements the mask loss of Mask-RCNN. As the `mask_outputs`
  produces `num_classes` masks for each RoI, the reference model expands
  `mask_targets` to match the shape of `mask_outputs` and selects only the
  target that the RoI has a maximum overlap. (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py)  # pylint: disable=line-too-long
  Instead, this function selects the `mask_outputs` by the `class_targets` so
  that it doesn't expand `mask_targets`.

  Args:
    mask_outputs: a float tensor representing the class prediction for each mask
      with a shape of
      [batch_size, num_masks, mask_height, mask_width, num_classes].
    mask_targets: a float tensor representing the binary mask of ground truth
      labels for each mask with a shape of
      [batch_size, num_masks, mask_height, mask_width].
    select_class_targets: a tensor with a shape of [batch_size, num_masks],
      representing the foreground mask targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
  Returns:
    mask_loss: a float tensor representing total mask loss.
  """
  with tf.name_scope('mask_loss'):
    # Selects the mask from `mask_outputs` based on `class_targets`, with which
    # the mask has the maximum overlap.
    (batch_size, num_masks, mask_height, mask_width,
     _) = mask_outputs.get_shape().as_list()
    mask_outputs = tf.transpose(mask_outputs, [0, 1, 4, 2, 3])
    # Contructs indices for gather.
    batch_indices = tf.tile(
        tf.expand_dims(tf.range(batch_size), axis=1), [1, num_masks])
    mask_indices = tf.tile(
        tf.expand_dims(tf.range(num_masks), axis=0), [batch_size, 1])
    gather_indices = tf.stack(
        [batch_indices, mask_indices, tf.to_int32(select_class_targets)],
        axis=2)
    mask_outputs = tf.gather_nd(mask_outputs, gather_indices)

    weights = tf.tile(
        tf.reshape(tf.greater(select_class_targets, 0),
                   [batch_size, num_masks, 1, 1]),
        [1, 1, mask_height, mask_width])
    loss = tf.losses.sigmoid_cross_entropy(
        mask_targets, mask_outputs, weights=weights,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    return params['mrcnn_weight_loss_mask'] * loss


def _model_fn(features, labels, mode, params, model, variable_filter_fn=None):
  """Model defination for the Mask-RCNN model based on ResNet.

  Args:
    features: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include score targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    model: the Mask-RCNN model outputs class logits and box regression outputs.
    variable_filter_fn: the filter function that takes trainable_variables and
      returns the variable list after applying the filter rule.

  Returns:
    tpu_spec: the TPUEstimatorSpec to run training, evaluation, or prediction.
  """
  if mode == tf.estimator.ModeKeys.PREDICT:
    labels = features
    features = labels.pop('images')

  if params['transpose_input'] and mode == tf.estimator.ModeKeys.TRAIN:
    features = tf.transpose(features, [2, 0, 1, 3])

  image_size = params['dynamic_image_size'] if params[
      'dynamic_input_shapes'] else (params['image_size'], params['image_size'])
  all_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                params['num_scales'], params['aspect_ratios'],
                                params['anchor_scale'], image_size)

  def _model_outputs():
    """Generates outputs from the model."""
    fpn_feats, rpn_fn, faster_rcnn_fn, mask_rcnn_fn = model(
        features, labels, all_anchors, mode, params)
    rpn_score_outputs, rpn_box_outputs = rpn_fn(fpn_feats)
    (class_outputs, box_outputs, class_targets, box_targets, box_rois,
     proposal_to_label_map) = faster_rcnn_fn(fpn_feats, rpn_score_outputs,
                                             rpn_box_outputs)
    encoded_box_targets = mask_rcnn_architecture.encode_box_targets(
        box_rois, box_targets, class_targets, params['bbox_reg_weights'])

    if mode != tf.estimator.ModeKeys.TRAIN:
      # Use TEST.NMS in the reference for this value. Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/core/config.py#L227  # pylint: disable=line-too-long
      mlperf_log.maskrcnn_print(key=mlperf_log.NMS_THRESHOLD,
                                value=params['test_nms'])

      # The mask branch takes inputs from different places in training vs in
      # eval/predict. In training, the mask branch uses proposals combined with
      # labels to produce both mask outputs and targets. At test time, it uses
      # the post-processed predictions to generate masks.
      # Generate detections one image at a time.
      batch_size, _, _ = class_outputs.get_shape().as_list()
      detections = []
      softmax_class_outputs = tf.nn.softmax(class_outputs)
      for i in range(batch_size):
        detections.append(
            anchors.generate_detections_per_image_op(
                softmax_class_outputs[i], box_outputs[i], box_rois[i],
                labels['source_ids'][i], labels['image_info'][i],
                params['test_detections_per_image'],
                params['test_rpn_post_nms_topn'], params['test_nms'],
                params['bbox_reg_weights'])
            )
      detections = tf.stack(detections, axis=0)
      mask_outputs = mask_rcnn_fn(fpn_feats, detections=detections)
    else:
      (mask_outputs, select_class_targets, select_box_targets, select_box_rois,
       select_proposal_to_label_map, mask_targets) = mask_rcnn_fn(
           fpn_feats, class_targets, box_targets, box_rois,
           proposal_to_label_map)
    # Performs post-processing for eval/predict.
    if mode != tf.estimator.ModeKeys.TRAIN:
      batch_size, num_instances, _, _, _ = mask_outputs.get_shape().as_list()
      mask_outputs = tf.transpose(mask_outputs, [0, 1, 4, 2, 3])
      # Compute indices for batch, num_detections, and class.
      batch_indices = tf.tile(
          tf.reshape(tf.range(batch_size), [batch_size, 1]),
          [1, num_instances])
      instance_indices = tf.tile(
          tf.reshape(tf.range(num_instances), [1, num_instances]),
          [batch_size, 1])
      class_indices = tf.to_int32(detections[:, :, 6])
      gather_indices = tf.stack(
          [batch_indices, instance_indices, class_indices], axis=2)
      mask_outputs = tf.gather_nd(mask_outputs, gather_indices)
    model_outputs = {
        'rpn_score_outputs': rpn_score_outputs,
        'rpn_box_outputs': rpn_box_outputs,
        'class_outputs': class_outputs,
        'box_outputs': box_outputs,
        'class_targets': class_targets,
        'box_targets': encoded_box_targets,
        'box_rois': box_rois,
        'mask_outputs': mask_outputs,
    }
    if mode == tf.estimator.ModeKeys.TRAIN:
      model_outputs.update({
          'select_class_targets': select_class_targets,
          'select_box_targets': select_box_targets,
          'select_box_rois': select_box_rois,
          'select_proposal_to_label_map': select_proposal_to_label_map,
          'mask_targets': mask_targets,})
    else:
      model_outputs.update({'detections': detections})
    return model_outputs

  if params['use_bfloat16']:
    with tf.contrib.tpu.bfloat16_scope():
      model_outputs = _model_outputs()
      def cast_outputs_to_float(d):
        for k, v in six.iteritems(d):
          if isinstance(v, dict):
            cast_outputs_to_float(v)
          else:
            if k != 'select_proposal_to_label_map':
              d[k] = tf.cast(v, tf.float32)
      cast_outputs_to_float(model_outputs)
  else:
    model_outputs = _model_outputs()

  # First check if it is in PREDICT mode.
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {}
    predictions['detections'] = model_outputs['detections']
    predictions['mask_outputs'] = tf.nn.sigmoid(model_outputs['mask_outputs'])
    predictions['image_info'] = labels['image_info']

    if params['use_tpu']:
      return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Load pretrained model from checkpoint.
  if params['resnet_checkpoint'] and mode == tf.estimator.ModeKeys.TRAIN:

    def scaffold_fn():
      """Loads pretrained model through scaffold function."""
      tf.train.init_from_checkpoint(params['resnet_checkpoint'], {
          '/': 'resnet%s/' % params['resnet_depth'],
      })
      return tf.train.Scaffold()
  else:
    scaffold_fn = None

  # Set up training loss and learning rate.
  update_learning_rate_schedule_parameters(params)
  global_step = tf.train.get_or_create_global_step()
  learning_rate = learning_rate_schedule(
      params['adjusted_learning_rate'], params['lr_warmup_init'],
      params['lr_warmup_step'], params['first_lr_drop_step'],
      params['second_lr_drop_step'], global_step)
  # score_loss and box_loss are for logging. only total_loss is optimized.
  total_rpn_loss, rpn_score_loss, rpn_box_loss = rpn_loss(
      model_outputs['rpn_score_outputs'], model_outputs['rpn_box_outputs'],
      labels, params)

  (total_fast_rcnn_loss, fast_rcnn_class_loss,
   fast_rcnn_box_loss) = fast_rcnn_loss(
       model_outputs['class_outputs'], model_outputs['box_outputs'],
       model_outputs['class_targets'], model_outputs['box_targets'], params)
  # Only training has the mask loss. Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/model_builder.py  # pylint: disable=line-too-long
  if mode == tf.estimator.ModeKeys.TRAIN:
    mask_loss = mask_rcnn_loss(
        model_outputs['mask_outputs'], model_outputs['mask_targets'],
        model_outputs['select_class_targets'], params)
  else:
    mask_loss = 0.
  var_list = variable_filter_fn(
      tf.trainable_variables(),
      params['resnet_depth']) if variable_filter_fn else None
  total_loss = (total_rpn_loss + total_fast_rcnn_loss + mask_loss +
                _WEIGHT_DECAY * tf.add_n(
                    [tf.nn.l2_loss(v) for v in var_list
                     if 'batch_normalization' not in v.name and 'bias' not in v.name]))

  host_call = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    mlperf_log.maskrcnn_print(
        key=mlperf_log.OPT_NAME, value='tf.train.MomentumOptimizer')
    mlperf_log.maskrcnn_print(
        key=mlperf_log.OPT_MOMENTUM, value=params['momentum'])
    mlperf_log.maskrcnn_print(
        key=mlperf_log.OPT_WEIGHT_DECAY, value=_WEIGHT_DECAY)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=params['momentum'])
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    grads_and_vars = optimizer.compute_gradients(total_loss, var_list)
    gradients, variables = zip(*grads_and_vars)
    grads_and_vars = []
    # Special treatment for biases (beta is named as bias in reference model)
    # Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/modeling/optimizer.py#L109  # pylint: disable=line-too-long
    for grad, var in zip(gradients, variables):
      if 'beta' in var.name or 'bias' in var.name:
        grad = 2.0 * grad
      grads_and_vars.append((grad, var))
    minimize_op = optimizer.apply_gradients(grads_and_vars,
                                            global_step=global_step)

    with tf.control_dependencies(update_ops):
      train_op = minimize_op

    if params['use_host_call']:
      def host_call_fn(global_step, total_loss, total_rpn_loss, rpn_score_loss,
                       rpn_box_loss, total_fast_rcnn_loss, fast_rcnn_class_loss,
                       fast_rcnn_box_loss, mask_loss, learning_rate):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          global_step: `Tensor with shape `[batch, ]` for the global_step.
          total_loss: `Tensor` with shape `[batch, ]` for the training loss.
          total_rpn_loss: `Tensor` with shape `[batch, ]` for the training RPN
            loss.
          rpn_score_loss: `Tensor` with shape `[batch, ]` for the training RPN
            score loss.
          rpn_box_loss: `Tensor` with shape `[batch, ]` for the training RPN
            box loss.
          total_fast_rcnn_loss: `Tensor` with shape `[batch, ]` for the
            training Mask-RCNN loss.
          fast_rcnn_class_loss: `Tensor` with shape `[batch, ]` for the
            training Mask-RCNN class loss.
          fast_rcnn_box_loss: `Tensor` with shape `[batch, ]` for the
            training Mask-RCNN box loss.
          mask_loss: `Tensor` with shape `[batch, ]` for the training Mask-RCNN
            mask loss.
          learning_rate: `Tensor` with shape `[batch, ]` for the learning_rate.

        Returns:
          List of summary ops to run on the CPU host.
        """
        # Outfeed supports int32 but global_step is expected to be int64.
        global_step = tf.reduce_mean(global_step)
        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        with (tf.contrib.summary.create_file_writer(
            params['model_dir'],
            max_queue=params['iterations_per_loop']).as_default()):
          with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(
                'total_loss', tf.reduce_mean(total_loss), step=global_step)
            tf.contrib.summary.scalar(
                'total_rpn_loss', tf.reduce_mean(total_rpn_loss),
                step=global_step)
            tf.contrib.summary.scalar(
                'rpn_score_loss', tf.reduce_mean(rpn_score_loss),
                step=global_step)
            tf.contrib.summary.scalar(
                'rpn_box_loss', tf.reduce_mean(rpn_box_loss), step=global_step)
            tf.contrib.summary.scalar(
                'total_fast_rcnn_loss', tf.reduce_mean(total_fast_rcnn_loss),
                step=global_step)
            tf.contrib.summary.scalar(
                'fast_rcnn_class_loss', tf.reduce_mean(fast_rcnn_class_loss),
                step=global_step)
            tf.contrib.summary.scalar(
                'fast_rcnn_box_loss', tf.reduce_mean(fast_rcnn_box_loss),
                step=global_step)
            tf.contrib.summary.scalar(
                'mask_loss', tf.reduce_mean(mask_loss), step=global_step)
            tf.contrib.summary.scalar(
                'learning_rate', tf.reduce_mean(learning_rate),
                step=global_step)

            return tf.contrib.summary.all_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      global_step_t = tf.reshape(global_step, [1])
      total_loss_t = tf.reshape(total_loss, [1])
      total_rpn_loss_t = tf.reshape(total_rpn_loss, [1])
      rpn_score_loss_t = tf.reshape(rpn_score_loss, [1])
      rpn_box_loss_t = tf.reshape(rpn_box_loss, [1])
      total_fast_rcnn_loss_t = tf.reshape(total_fast_rcnn_loss, [1])
      fast_rcnn_class_loss_t = tf.reshape(fast_rcnn_class_loss, [1])
      fast_rcnn_box_loss_t = tf.reshape(fast_rcnn_box_loss, [1])
      mask_loss_t = tf.reshape(mask_loss, [1])
      learning_rate_t = tf.reshape(learning_rate, [1])
      host_call = (host_call_fn,
                   [global_step_t, total_loss_t, total_rpn_loss_t,
                    rpn_score_loss_t, rpn_box_loss_t, total_fast_rcnn_loss_t,
                    fast_rcnn_class_loss_t, fast_rcnn_box_loss_t,
                    mask_loss_t, learning_rate_t])
  else:
    train_op = None

  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op,
      host_call=host_call,
      scaffold_fn=scaffold_fn)


def mask_rcnn_model_fn(features, labels, mode, params):
  """Mask-RCNN model."""
  with tf.variable_scope('', reuse=tf.AUTO_REUSE):
    return _model_fn(
        features,
        labels,
        mode,
        params,
        model=mask_rcnn_architecture.mask_rcnn,
        variable_filter_fn=mask_rcnn_architecture.remove_variables)


def default_hparams():
  return tf.contrib.training.HParams(
      # input preprocessing parameters
      image_size=1408,
      short_side_image_size=800,
      long_side_max_image_size=1333,
      input_rand_hflip=True,
      train_scale_min=1.0,
      train_scale_max=1.0,
      gt_mask_size=224,
      # dataset specific parameters
      num_classes=91,
      skip_crowd_during_training=False,
      use_category=True,
      # Region Proposal Network
      rpn_positive_overlap=0.7,
      rpn_negative_overlap=0.3,
      rpn_batch_size_per_im=256,
      rpn_fg_fraction=0.5,
      rpn_pre_nms_topn=2000,
      rpn_post_nms_topn=1000,
      rpn_nms_threshold=0.7,
      rpn_min_size=0.,
      # Proposal layer.
      batch_size_per_im=512,
      fg_fraction=0.25,
      fg_thresh=0.5,
      bg_thresh_hi=0.5,
      bg_thresh_lo=0.,
      # Faster-RCNN heads.
      fast_rcnn_mlp_head_dim=1024,
      bbox_reg_weights=(10., 10., 5., 5.),
      # Mask-RCNN heads.
      mrcnn_resolution=28,
      # evaluation
      test_detections_per_image=100,
      test_nms=0.5,
      test_rpn_pre_nms_topn=1000,
      test_rpn_post_nms_topn=1000,
      test_rpn_nms_thresh=0.7,
      # model architecture
      min_level=2,
      max_level=6,
      num_scales=1,
      aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
      anchor_scale=8.0,
      resnet_depth=50,
      # is batchnorm training mode
      is_training_bn=False,
      # optimization
      momentum=0.9,
      learning_rate=0.02,
      lr_warmup_init=0.0067,
      lr_warmup_epoch=0.067,
      first_lr_drop_epoch=8.0,
      second_lr_drop_epoch=10.67,
      # localization loss
      delta=0.1,
      rpn_box_loss_weight=1.0,
      fast_rcnn_box_loss_weight=1.0,
      mrcnn_weight_loss_mask=1.0,
      # enable bfloat
      use_bfloat16=True,
      # enable host_call
      use_host_call=False,
      # enable dynamic input shapes
      dynamic_image_size=(896, 1408),
  )
