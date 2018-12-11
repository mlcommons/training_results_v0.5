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
"""Model defination for the SSD Model.

Defines model_fn of SSD for TF Estimator. The model_fn includes SSD
model architecture, loss function, learning rate schedule, and evaluation
procedure.

T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from object_detection import box_coder
from object_detection import box_list
from object_detection import faster_rcnn_box_coder

from tensorflow.contrib.tpu.python.tpu import bfloat16
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.python.estimator import model_fn as model_fn_lib

from mlperf_compliance import mlperf_log
import dataloader
import ssd_architecture
import ssd_constants


def select_top_k_scores(scores_in, pre_nms_num_detections=5000):
  """Select top_k scores and indices for each class.

  Args:
    scores_in: a Tensor with shape [batch_size, N, num_classes], which stacks
      class logit outputs on all feature levels. The N is the number of total
      anchors on all levels. The num_classes is the number of classes predicted
      by the model.
    pre_nms_num_detections: Number of candidates before NMS.

  Returns:
    scores and indices: Tensors with shape [batch_size, pre_nms_num_detections,
      num_classes].
  """
  scores_trans = tf.transpose(scores_in, perm=[0, 2, 1])

  top_k_scores, top_k_indices = tf.nn.top_k(
      scores_trans, k=pre_nms_num_detections, sorted=False)

  return tf.transpose(top_k_scores, [0, 2, 1]), tf.transpose(
      top_k_indices, [0, 2, 1])


def concat_outputs(cls_outputs, box_outputs):
  """Concatenate predictions into a single tensor.

  This function takes the dicts of class and box prediction tensors and
  concatenates them into a single tensor for comparison with the ground truth
  boxes and class labels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width,
      num_anchors * num_classses].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
  Returns:
    concatenanted cls_outputs and box_outputs.
  """
  assert set(cls_outputs.keys()) == set(box_outputs.keys())

  # This sort matters. The labels assume a certain order based on
  # ssd_constants.FEATURE_SIZES, and this sort matches that convention.
  keys = sorted(cls_outputs.keys())
  batch_size = int(cls_outputs[keys[0]].shape[0])

  flat_cls = []
  flat_box = []

  mlperf_log.ssd_print(
      key=mlperf_log.FEATURE_SIZES, value=ssd_constants.FEATURE_SIZES)

  for i, k in enumerate(keys):
    # TODO(taylorrobie): confirm that this reshape, transpose,
    # reshape is correct.
    scale = ssd_constants.FEATURE_SIZES[i]
    split_shape = (ssd_constants.NUM_DEFAULTS[i], ssd_constants.NUM_CLASSES)
    assert cls_outputs[k].shape[3] == split_shape[0] * split_shape[1]
    intermediate_shape = (batch_size, scale, scale) + split_shape
    final_shape = (batch_size, scale ** 2 * split_shape[0], split_shape[1])
    flat_cls.append(tf.reshape(tf.transpose(tf.reshape(
        cls_outputs[k], intermediate_shape), (0, 3, 1, 2, 4)), final_shape))

    split_shape = (ssd_constants.NUM_DEFAULTS[i], 4)
    assert box_outputs[k].shape[3] == split_shape[0] * split_shape[1]
    intermediate_shape = (batch_size, scale, scale) + split_shape
    final_shape = (batch_size, scale ** 2 * split_shape[0], split_shape[1])
    flat_box.append(tf.reshape(tf.transpose(tf.reshape(
        box_outputs[k], intermediate_shape), (0, 3, 1, 2, 4)), final_shape))

  return tf.concat(flat_cls, axis=1), tf.concat(flat_box, axis=1)


def _localization_loss(pred_loc, gt_loc, gt_label, num_matched_boxes):
  """Computes the localization loss.

  Computes the localization loss using smooth l1 loss.
  Args:
    pred_loc: a flatten tensor that includes all predicted locations. The shape
      is [batch_size, num_anchors, 4].
    gt_loc: a tensor representing box regression targets in
      [batch_size, num_anchors, 4].
    gt_label: a tensor that represents the classification groundtruth targets.
      The shape is [batch_size, num_anchors, 1].
    num_matched_boxes: the number of anchors that are matched to a groundtruth
      targets, used as the loss normalizater. The shape is [batch_size].
  Returns:
    box_loss: a float32 representing total box regression loss.
  """
  mask = tf.greater(tf.squeeze(gt_label), 0)
  float_mask = tf.cast(mask, tf.float32)

  smooth_l1 = tf.reduce_sum(tf.losses.huber_loss(
      gt_loc, pred_loc,
      reduction=tf.losses.Reduction.NONE
  ), axis=2)
  smooth_l1 = tf.multiply(smooth_l1, float_mask)
  box_loss = tf.reduce_sum(smooth_l1, axis=1)

  # TODO(taylorrobie): Confirm that normalizing by the number of boxes matches
  # reference
  return tf.reduce_mean(box_loss / num_matched_boxes)


def _classification_loss(pred_label, gt_label, num_matched_boxes):
  """Computes the classification loss.

  Computes the classification loss with hard negative mining.
  Args:
    pred_label: a flatten tensor that includes all predicted class. The shape
      is [batch_size, num_anchors, num_classes].
    gt_label: a tensor that represents the classification groundtruth targets.
      The shape is [batch_size, num_anchors, 1].
    num_matched_boxes: the number of anchors that are matched to a groundtruth
      targets. This is used as the loss normalizater.
  Returns:
    box_loss: a float32 representing total box regression loss.
  """
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      gt_label, pred_label, reduction=tf.losses.Reduction.NONE)

  mask = tf.greater(tf.squeeze(gt_label), 0)
  float_mask = tf.cast(mask, tf.float32)

  # Hard example mining
  neg_masked_cross_entropy = cross_entropy * (1 - float_mask)
  relative_position = tf.contrib.framework.argsort(tf.contrib.framework.argsort(
      neg_masked_cross_entropy, direction='DESCENDING'))
  num_neg_boxes = tf.minimum(
      tf.to_int32(num_matched_boxes) * ssd_constants.NEGS_PER_POSITIVE,
      ssd_constants.NUM_SSD_BOXES)
  top_k_neg_mask = tf.cast(tf.less(
      relative_position,
      tf.tile(num_neg_boxes[:, tf.newaxis], (1, ssd_constants.NUM_SSD_BOXES))
  ), tf.float32)

  class_loss = tf.reduce_sum(
      tf.multiply(cross_entropy, float_mask + top_k_neg_mask), axis=1)

  # TODO(taylorrobie): Confirm that normalizing by the number of boxes matches
  # reference
  return tf.reduce_mean(class_loss / num_matched_boxes)


def detection_loss(cls_outputs, box_outputs, labels):
  """Computes total detection loss.

  Computes total detection loss including box and class loss from all levels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
    labels: the dictionary that returned from dataloader that includes
      groundturth targets.
  Returns:
    total_loss: a float32 representing total loss reducing from class and box
      losses from all levels.
    cls_loss: a float32 representing total class loss.
    box_loss: a float32 representing total box regression loss.
  """

  flattened_cls, flattened_box = concat_outputs(cls_outputs, box_outputs)
  box_loss = _localization_loss(
      flattened_box, labels[ssd_constants.BOXES],
      labels[ssd_constants.CLASSES], labels[ssd_constants.NUM_MATCHED_BOXES])
  class_loss = _classification_loss(
      flattened_cls, labels[ssd_constants.CLASSES],
      labels[ssd_constants.NUM_MATCHED_BOXES])

  return class_loss + box_loss, class_loss, box_loss


def update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule.

  Args:
    params: a parameter dictionary that includes learning_rate, lr_warmup_epoch,
      first_lr_drop_epoch, and second_lr_drop_epoch.
  """
  # params['batch_size'] is per-shard within model_fn if use_tpu=true.
  batch_size = (
      params['batch_size'] * params['num_shards']
      if params['use_tpu'] else params['batch_size'])
  # Learning rate is proportional to the batch size
  steps_per_epoch = params['num_examples_per_epoch'] / batch_size
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['first_lr_drop_step'] = int(
      params['first_lr_drop_epoch'] * steps_per_epoch)
  params['second_lr_drop_step'] = int(
      params['second_lr_drop_epoch'] * steps_per_epoch)


def learning_rate_schedule(params, global_step):
  """Handles learning rate scaling, linear warmup, and learning rate decay.

  Args:
    params: A dictionary that defines hyperparameters of model.
    global_step: A tensor representing current global step.

  Returns:
    A tensor representing current learning rate.
  """
  base_learning_rate = params['base_learning_rate']
  lr_warmup_step = params['lr_warmup_step']
  first_lr_drop_step = params['first_lr_drop_step']
  second_lr_drop_step = params['second_lr_drop_step']
  batch_size = (params['batch_size'] * params['num_shards'] if params['use_tpu']
                else params['batch_size'])
  scaling_factor = batch_size / ssd_constants.DEFAULT_BATCH_SIZE
  adjusted_learning_rate = base_learning_rate * scaling_factor
  learning_rate = (tf.cast(global_step, dtype=tf.float32) /
                   lr_warmup_step) * adjusted_learning_rate
  lr_schedule = [[1.0, lr_warmup_step], [0.1, first_lr_drop_step],
                 [0.01, second_lr_drop_step]]
  for mult, start_global_step in lr_schedule:
    learning_rate = tf.where(global_step < start_global_step, learning_rate,
                             adjusted_learning_rate * mult)
  return learning_rate


def _model_fn(features, labels, mode, params, model):
  """Model defination for the SSD model based on ResNet-50.

  Args:
    features: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include class targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    model: the SSD model outputs class logits and box regression outputs.

  Returns:
    spec: the EstimatorSpec or TPUEstimatorSpec to run training, evaluation,
      or prediction.
  """
  if mode == tf.estimator.ModeKeys.PREDICT:
    labels = features
    features = labels.pop('image')

  # Manually apply the double transpose trick for training data.
  if params['transpose_input'] and mode != tf.estimator.ModeKeys.PREDICT:
    features = tf.transpose(features, [3, 0, 1, 2])
    labels[ssd_constants.BOXES] = tf.transpose(
        labels[ssd_constants.BOXES], [2, 0, 1])
    labels[ssd_constants.CLASSES] = tf.transpose(
        labels[ssd_constants.CLASSES], [2, 0, 1])

  # Normalize the image to zero mean and unit variance.
  mlperf_log.ssd_print(key=mlperf_log.DATA_NORMALIZATION_MEAN,
                       value=ssd_constants.NORMALIZATION_MEAN)
  mlperf_log.ssd_print(key=mlperf_log.DATA_NORMALIZATION_STD,
                       value=ssd_constants.NORMALIZATION_STD)

  features -= tf.constant(
      ssd_constants.NORMALIZATION_MEAN, shape=[1, 1, 3], dtype=features.dtype)

  features /= tf.constant(
      ssd_constants.NORMALIZATION_STD, shape=[1, 1, 3], dtype=features.dtype)

  def _model_outputs():
    return model(
        features, params, is_training_bn=(mode == tf.estimator.ModeKeys.TRAIN))

  if params['use_bfloat16']:
    with bfloat16.bfloat16_scope():
      cls_outputs, box_outputs = _model_outputs()
      levels = cls_outputs.keys()
      for level in levels:
        cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
        box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
  else:
    cls_outputs, box_outputs = _model_outputs()
    levels = cls_outputs.keys()

  # First check if it is in PREDICT mode.
  if mode == tf.estimator.ModeKeys.PREDICT:
    flattened_cls, flattened_box = concat_outputs(cls_outputs, box_outputs)
    mlperf_log.ssd_print(
        key=mlperf_log.SCALES, value=ssd_constants.BOX_CODER_SCALES)
    ssd_box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=ssd_constants.BOX_CODER_SCALES)

    anchors = box_list.BoxList(
        tf.convert_to_tensor(dataloader.DefaultBoxes()('ltrb')))

    decoded_boxes = box_coder.batch_decode(
        encoded_boxes=flattened_box, box_coder=ssd_box_coder, anchors=anchors)

    pred_scores = tf.nn.softmax(flattened_cls, axis=2)

    pred_scores, indices = select_top_k_scores(pred_scores,
                                               ssd_constants.MAX_NUM_EVAL_BOXES)

    predictions = dict(
        labels,
        indices=indices,
        pred_scores=pred_scores,
        pred_box=decoded_boxes,
    )

    if params['visualize_dataloader']:
      # this is for inference visualization.
      predictions['image'] = features

    if params['use_tpu']:
      return tpu_estimator.TPUEstimatorSpec(mode=mode, predictions=predictions)

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Load pretrained model from checkpoint.
  if params['resnet_checkpoint'] and mode == tf.estimator.ModeKeys.TRAIN:

    def scaffold_fn():
      """Loads pretrained model through scaffold function."""
      tf.train.init_from_checkpoint(params['resnet_checkpoint'], {
          '/': 'resnet%s/' % ssd_constants.RESNET_DEPTH,
      })
      return tf.train.Scaffold()
  else:
    scaffold_fn = None

  # Set up training loss and learning rate.
  update_learning_rate_schedule_parameters(params)
  global_step = tf.train.get_or_create_global_step()
  learning_rate = learning_rate_schedule(params, global_step)
  mlperf_log.ssd_print(key=mlperf_log.OPT_LR, deferred=True)
  # cls_loss and box_loss are for logging. only total_loss is optimized.
  total_loss, cls_loss, box_loss = detection_loss(
      cls_outputs, box_outputs, labels)

  total_loss += params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  host_call = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=ssd_constants.MOMENTUM)
    if params['use_tpu']:
      optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    mlperf_log.ssd_print(
        key=mlperf_log.OPT_NAME, value='tf.train.MomentumOptimizer')
    # TODO(wangtao): figure out how to log learning rate.
    # mlperf_log.ssd_print(key=mlperf_log.OPT_LR, value=learning_rate)
    mlperf_log.ssd_print(
        key=mlperf_log.OPT_MOMENTUM, value=ssd_constants.MOMENTUM)
    mlperf_log.ssd_print(
        key=mlperf_log.OPT_WEIGHT_DECAY, value=params['weight_decay'])

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if params['device'] == 'gpu':
      # GPU uses tf.group to avoid dependency overhead on update_ops; also,
      # multi-GPU requires a different EstimatorSpec class object
      train_op = tf.group(optimizer.minimize(total_loss, global_step),
                          update_ops)
      return model_fn_lib.EstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op, scaffold=scaffold_fn())
    else:
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step)

    if params['use_host_call']:
      def host_call_fn(global_step, total_loss, cls_loss, box_loss,
                       learning_rate):
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
          cls_loss: `Tensor` with shape `[batch, ]` for the training cls loss.
          box_loss: `Tensor` with shape `[batch, ]` for the training box loss.
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
                'cls_loss', tf.reduce_mean(cls_loss), step=global_step)
            tf.contrib.summary.scalar(
                'box_loss', tf.reduce_mean(box_loss), step=global_step)
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
      cls_loss_t = tf.reshape(cls_loss, [1])
      box_loss_t = tf.reshape(box_loss, [1])
      learning_rate_t = tf.reshape(learning_rate, [1])
      host_call = (host_call_fn,
                   [global_step_t, total_loss_t, cls_loss_t, box_loss_t,
                    learning_rate_t])
  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    raise NotImplementedError

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics,
      scaffold_fn=scaffold_fn)


def ssd_model_fn(features, labels, mode, params):
  """SSD model."""
  return _model_fn(features, labels, mode, params, model=ssd_architecture.ssd)


def default_hparams():
  # TODO(taylorrobie): replace params useages with global constants.
  return tf.contrib.training.HParams(

      # TODO(taylorrobie): I don't respect this everywhere.
      # enable bfloat
      use_bfloat16=True,
      use_host_call=True,
      num_examples_per_epoch=120000,
      lr_warmup_epoch=1.0,
      first_lr_drop_epoch=42.6,
      second_lr_drop_epoch=53.3,
      weight_decay=ssd_constants.WEIGHT_DECAY,
      base_learning_rate=ssd_constants.BASE_LEARNING_RATE,
      visualize_dataloader=False,
      distributed_group_size=1,
      eval_every_checkpoint=False,
      transpose_input=True,
      train_with_low_level_api=True,
      eval_with_low_level_api=True,
  )
