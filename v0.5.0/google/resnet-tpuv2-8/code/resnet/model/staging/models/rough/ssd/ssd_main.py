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
"""Training script for SSD.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading
import numpy as np
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.training.python.training import evaluation

from mlperf_compliance import mlperf_log
import async_checkpoint
import coco_metric
import dataloader
import ssd_constants
import ssd_model

# Cloud TPU Cluster Resolvers
tf.flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
tf.flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
tf.flags.DEFINE_string(
    'tpu_name', default=None,
    help='Name of the Cloud TPU for Cluster Resolvers. You must specify either '
    'this flag or --master.')

tf.flags.DEFINE_string(
    'eval_master', default='',
    help='GRPC URL of the eval master. Set to an appropiate value when running '
    'on CPU/GPU')
tf.flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than CPUs')
tf.flags.DEFINE_string('model_dir', None, 'Location of model_dir')
tf.flags.DEFINE_string('resnet_checkpoint', '',
                       'Location of the ResNet checkpoint to use for model '
                       'initialization.')
tf.flags.DEFINE_string('hparams', '',
                       'Comma separated k=v pairs of hyperparameters.')
tf.flags.DEFINE_integer(
    'num_shards', default=8, help='Number of shards (TPU cores) for '
    'training.')
tf.flags.DEFINE_integer(
    'eval_num_shards', default=8, help='Number of shards (TPU cores) for eval')
tf.flags.DEFINE_integer('train_batch_size', 64, 'training batch size')
tf.flags.DEFINE_integer('eval_batch_size', 1, 'evaluation batch size')
tf.flags.DEFINE_integer('eval_samples', 5000, 'The number of samples for '
                        'evaluation.')
tf.flags.DEFINE_integer(
    'iterations_per_loop', 1000, 'Number of iterations per TPU training loop')
tf.flags.DEFINE_string(
    'training_file_pattern', None,
    'Glob for training data files (e.g., COCO train - minival set)')
tf.flags.DEFINE_string(
    'validation_file_pattern', None,
    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
tf.flags.DEFINE_string(
    'val_json_file',
    None,
    'COCO validation JSON containing golden bounding boxes.')
tf.flags.DEFINE_integer('num_examples_per_epoch', 120000,
                        'Number of examples in one epoch')
tf.flags.DEFINE_integer('num_epochs', 64, 'Number of epochs for training')
tf.flags.DEFINE_string('mode', 'train',
                       'Mode to run: train or eval (default: train)')
tf.flags.DEFINE_bool('eval_after_training', False, 'Run one eval after the '
                     'training finishes.')
tf.flags.DEFINE_integer(
    'keep_checkpoint_max', 32,
    'Maximum number of checkpoints to keep.')

# For Eval mode
tf.flags.DEFINE_integer('min_eval_interval', 180,
                        'Minimum seconds between evaluations.')
tf.flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')
tf.flags.DEFINE_string('device', 'tpu', 'device to train (default: tpu)')
tf.flags.DEFINE_bool('use_async_checkpoint', True, 'Use async checkpoint')
tf.flags.DEFINE_integer('eval_epoch', 0, 'Epoch to eval.')

FLAGS = tf.flags.FLAGS
SUCCESS = False


def next_checkpoint(model_dir, timeout_mins=240):
  """Yields successive checkpoints from model_dir."""
  last_ckpt = None
  last_step = 0
  while True:
    # Get the latest checkpoint.
    last_ckpt = tf.contrib.training.wait_for_new_checkpoint(
        model_dir, last_ckpt, seconds_to_sleep=0, timeout=60 * timeout_mins)
    # Get all the checkpoint from the model dir.
    ckpt_path = tf.train.get_checkpoint_state(model_dir)
    all_model_checkpoint_paths = ckpt_path.all_model_checkpoint_paths

    ckpt_step = np.inf
    next_ckpt = None
    # Find the next checkpoint to eval based on last_step.
    for ckpt in all_model_checkpoint_paths:
      step = int(os.path.basename(ckpt).split('-')[1])
      if step > last_step and step < ckpt_step:
        ckpt_step = step
        next_ckpt = ckpt

    # If all the checkpoints have been evaluated.
    if last_ckpt is None and next_ckpt is None:
      tf.logging.info(
          'Eval timeout: no new checkpoints within %dm' % timeout_mins)
      break

    if next_ckpt is not None:
      last_step = ckpt_step
      last_ckpt = next_ckpt

    yield last_ckpt


def construct_run_config(iterations_per_loop):
  """Construct the run config."""
  tpu_cluster_resolver = None
  run_local = FLAGS.device == 'gpu'  # cpu will be run on the TPU host
  # controller CPU, not the local CPU
  if FLAGS.use_tpu and run_local:
    raise RuntimeError('--use_tpu should be set to False if --device is set '
                       'to "gpu"')

  if FLAGS.tpu_name:
    tpu_cluster_resolver = (
        tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name,
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project))
    tpu_grpc_url = tpu_cluster_resolver.get_master()
    tf.Session.reset(tpu_grpc_url)
  elif not run_local:
    raise RuntimeError('Either --tpu_name must be set or --device must be set '
                       'to "gpu"')

  # Check device
  if FLAGS.use_tpu and FLAGS.device != 'tpu':
    raise RuntimeError("--device must be 'tpu' when --use_tpu=True.")
  if not FLAGS.use_tpu and FLAGS.device == 'tpu':
    raise RuntimeError("--device must be either 'gpu' or 'cpu' "
                       'when --use_tpu=False.')

  # Parse hparams
  hparams = ssd_model.default_hparams()
  hparams.parse(FLAGS.hparams)

  params = dict(
      hparams.values(),
      num_shards=FLAGS.num_shards,
      num_examples_per_epoch=FLAGS.num_examples_per_epoch,
      use_tpu=FLAGS.use_tpu,
      resnet_checkpoint=FLAGS.resnet_checkpoint,
      val_json_file=FLAGS.val_json_file,
      mode=FLAGS.mode,
      device=FLAGS.device,
      model_dir=FLAGS.model_dir,
      iterations_per_loop=iterations_per_loop,
      steps_per_epoch=FLAGS.num_examples_per_epoch // FLAGS.train_batch_size,
  )

  if run_local:
    # if FLAGS.mode != 'train':
    #   # TODO(taylorrobie): Add other modes
    #   raise NotImplementedError('set --mode=train on gpu (other modes will be '
    #                             'added later)')

    if FLAGS.eval_after_training:
      raise NotImplementedError('eval_after_training is not yet supported for '
                                'gpu')

    session_config = tf.ConfigProto(allow_soft_placement=True)

    # TODO(taylorrobie): Multi GPU
    train_distribute = tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")

    return tf.estimator.RunConfig(
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        save_checkpoints_steps=ssd_constants.CHECKPOINT_FREQUENCY,
        train_distribute=train_distribute,
        session_config=session_config), params

  return tpu_config.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      save_checkpoints_steps=None
      if FLAGS.use_async_checkpoint else ssd_constants.CHECKPOINT_FREQUENCY,
      log_step_count_steps=iterations_per_loop,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False),
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop,
          FLAGS.num_shards,
          per_host_input_for_training=tpu_config.InputPipelineConfig
          .PER_HOST_V2)), params


def coco_eval(predictions, current_epoch, current_step, summary_writer):
  """Call the coco library to get the eval metrics."""
  global SUCCESS
  eval_results = coco_metric.compute_map(predictions, FLAGS.val_json_file)
  mlperf_log.ssd_print(key=mlperf_log.EVAL_STOP, value=current_epoch)
  mlperf_log.ssd_print(key=mlperf_log.EVAL_SIZE, value=FLAGS.eval_samples)
  mlperf_log.ssd_print(
      key=mlperf_log.EVAL_ACCURACY,
      value={
          'epoch': current_epoch,
          'value': eval_results['COCO/AP']
      })
  mlperf_log.ssd_print(
      key=mlperf_log.EVAL_TARGET, value=ssd_constants.EVAL_TARGET)
  mlperf_log.ssd_print(
      key=mlperf_log.EVAL_ITERATION_ACCURACY,
      value={
          'iteration': current_step,
          'value': eval_results['COCO/AP']
      })
  if eval_results['COCO/AP'] >= ssd_constants.EVAL_TARGET and not SUCCESS:
    mlperf_log.ssd_print(key=mlperf_log.RUN_STOP, value={'success': 'true'})
    mlperf_log.ssd_print(key=mlperf_log.RUN_FINAL)
    SUCCESS = True
  tf.logging.info('Eval results: %s' % eval_results)
  # Write out eval results for the checkpoint.
  with tf.Graph().as_default():
    summaries = []
    for metric in eval_results:
      summaries.append(
          tf.Summary.Value(tag=metric, simple_value=eval_results[metric]))
    tf_summary = tf.Summary(value=list(summaries))
    summary_writer.add_summary(tf_summary, current_step)


def main(argv):
  del argv  # Unused.
  global SUCCESS

  # Check data path
  if FLAGS.mode in ('train',
                    'train_and_eval') and FLAGS.training_file_pattern is None:
    raise RuntimeError('You must specify --training_file_pattern for training.')
  if FLAGS.mode in ('eval', 'train_and_eval', 'eval_once'):
    if FLAGS.validation_file_pattern is None:
      raise RuntimeError('You must specify --validation_file_pattern '
                         'for evaluation.')

  run_config, params = construct_run_config(FLAGS.iterations_per_loop)

  if FLAGS.mode != 'eval':
    mlperf_log.ssd_print(key=mlperf_log.RUN_START)
  # TPU Estimator
  if FLAGS.mode == 'train':
    if FLAGS.device == 'gpu':
      train_params = dict(params)
      train_params['batch_size'] = FLAGS.train_batch_size
      train_estimator = tf.estimator.Estimator(
          model_fn=ssd_model.ssd_model_fn, model_dir=FLAGS.model_dir,
          config=run_config, params=train_params)
    else:
      train_estimator = tpu_estimator.TPUEstimator(
          model_fn=ssd_model.ssd_model_fn,
          use_tpu=FLAGS.use_tpu,
          train_batch_size=FLAGS.train_batch_size,
          config=run_config,
          params=params)

    tf.logging.info(params)
    mlperf_log.ssd_print(key=mlperf_log.TRAIN_LOOP)
    mlperf_log.ssd_print(key=mlperf_log.TRAIN_EPOCH, value=0)
    hooks = []
    if FLAGS.use_async_checkpoint:
      hooks.append(
          async_checkpoint.AsyncCheckpointSaverHook(
              checkpoint_dir=FLAGS.model_dir,
              save_steps=max(100, FLAGS.iterations_per_loop)))
    train_estimator.train(
        input_fn=dataloader.SSDInputReader(
            FLAGS.training_file_pattern,
            params['transpose_input'],
            is_training=True),
        steps=int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                  FLAGS.train_batch_size),
        hooks=hooks)

    if FLAGS.eval_after_training:
      eval_estimator = tpu_estimator.TPUEstimator(
          model_fn=ssd_model.ssd_model_fn,
          use_tpu=FLAGS.use_tpu,
          train_batch_size=FLAGS.train_batch_size,
          predict_batch_size=FLAGS.eval_batch_size,
          config=run_config,
          params=params)

      predictions = list(eval_estimator.predict(
          input_fn=dataloader.SSDInputReader(
              FLAGS.validation_file_pattern, is_training=False)))

      eval_results = coco_metric.compute_map(predictions, FLAGS.val_json_file)

      tf.logging.info('Eval results: %s' % eval_results)

  elif FLAGS.mode == 'train_and_eval':
    output_dir = os.path.join(FLAGS.model_dir, 'eval')
    tf.gfile.MakeDirs(output_dir)
    # Summary writer writes out eval metrics.
    summary_writer = tf.summary.FileWriter(output_dir)

    current_step = 0
    mlperf_log.ssd_print(key=mlperf_log.TRAIN_LOOP)
    threads = []
    for eval_step in ssd_constants.EVAL_STEPS:
      # Compute the actual eval steps based on the actural train_batch_size
      steps = int(eval_step * ssd_constants.DEFAULT_BATCH_SIZE /
                  FLAGS.train_batch_size)
      run_config, params = construct_run_config(steps)
      current_epoch = current_step // params['steps_per_epoch']
      # TODO(wangtao): figure out how to log for each epoch.
      mlperf_log.ssd_print(key=mlperf_log.TRAIN_EPOCH, value=current_epoch)

      tf.logging.info('Starting training cycle for %d steps.' % steps)
      if FLAGS.device == 'gpu':
        train_params = dict(params)
        train_params['batch_size'] = FLAGS.train_batch_size
        train_estimator = tf.estimator.Estimator(
            model_fn=ssd_model.ssd_model_fn, model_dir=FLAGS.model_dir,
            config=run_config, params=train_params)
      else:
        train_estimator = tpu_estimator.TPUEstimator(
            model_fn=ssd_model.ssd_model_fn,
            use_tpu=FLAGS.use_tpu,
            train_batch_size=FLAGS.train_batch_size,
            config=run_config,
            params=params)

      tf.logging.info(params)
      train_estimator.train(
          input_fn=dataloader.SSDInputReader(FLAGS.training_file_pattern,
                                             params['transpose_input'],
                                             is_training=True),
          steps=steps)

      if SUCCESS:
        break

      current_step = current_step + steps
      current_epoch = current_step // params['steps_per_epoch']
      tf.logging.info('Starting evaluation cycle at step %d.' % current_step)
      mlperf_log.ssd_print(key=mlperf_log.EVAL_START, value=current_epoch)
      # Run evaluation at the given step.
      if FLAGS.device == 'gpu':
        eval_params = dict(params)
        eval_params['batch_size'] = FLAGS.eval_batch_size
        eval_estimator = tf.estimator.Estimator(
            model_fn=ssd_model.ssd_model_fn, model_dir=FLAGS.model_dir,
            config=run_config, params=eval_params)
      else:
        eval_estimator = tpu_estimator.TPUEstimator(
            model_fn=ssd_model.ssd_model_fn,
            use_tpu=FLAGS.use_tpu,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=FLAGS.eval_batch_size,
            config=run_config,
            params=params)

      predictions = list(eval_estimator.predict(
          input_fn=dataloader.SSDInputReader(
              FLAGS.validation_file_pattern, is_training=False)))

      t = threading.Thread(
          target=coco_eval,
          args=(predictions, current_epoch, current_step, summary_writer))
      threads.append(t)
      t.start()

    for t in threads:
      t.join()

    # success is a string right now as boolean is not JSON serializable.
    if not SUCCESS:
      mlperf_log.ssd_print(key=mlperf_log.RUN_STOP, value={'success': 'false'})
      mlperf_log.ssd_print(key=mlperf_log.RUN_FINAL)
    summary_writer.close()

  elif FLAGS.mode == 'eval':
    if FLAGS.device == 'gpu':
      eval_params = dict(params)
      eval_params['batch_size'] = FLAGS.eval_batch_size
      eval_estimator = tf.estimator.Estimator(
          model_fn=ssd_model.ssd_model_fn, model_dir=FLAGS.model_dir,
          config=run_config, params=eval_params)
    else:
      eval_estimator = tpu_estimator.TPUEstimator(
          model_fn=ssd_model.ssd_model_fn,
          use_tpu=FLAGS.use_tpu,
          train_batch_size=FLAGS.train_batch_size,
          predict_batch_size=FLAGS.eval_batch_size,
          config=run_config,
          params=params)

    output_dir = os.path.join(eval_estimator.model_dir, 'eval')
    tf.gfile.MakeDirs(output_dir)
    # Summary writer writes out eval metrics.
    summary_writer = tf.summary.FileWriter(output_dir)

    def terminate_eval():
      tf.logging.info('Terminating eval after %d seconds of no checkpoints' %
                      FLAGS.eval_timeout)
      return True

    eval_steps = np.cumsum(ssd_constants.EVAL_STEPS).tolist()
    eval_epochs = [
        steps * ssd_constants.DEFAULT_BATCH_SIZE / FLAGS.train_batch_size //
        params['steps_per_epoch'] for steps in eval_steps
    ]

    # For 8x8 slices and above.
    if FLAGS.train_batch_size >= 4096:
      eval_epochs = [i * 2 for i in eval_epochs]

    tf.logging.info('Eval epochs: %s' % eval_epochs)
    # Run evaluation when there's a new checkpoint
    threads = []
    for ckpt in next_checkpoint(FLAGS.model_dir):
      if SUCCESS:
        break
      current_step = int(os.path.basename(ckpt).split('-')[1])
      current_epoch = current_step // params['steps_per_epoch']
      tf.logging.info('current epoch: %s' % current_epoch)
      if not params[
          'eval_every_checkpoint'] and current_epoch not in eval_epochs:
        continue

      tf.logging.info('Starting to evaluate.')
      try:
        mlperf_log.ssd_print(key=mlperf_log.EVAL_START, value=current_epoch)

        predictions = list(
            eval_estimator.predict(
                checkpoint_path=ckpt,
                input_fn=dataloader.SSDInputReader(
                    FLAGS.validation_file_pattern, is_training=False)))

        t = threading.Thread(
            target=coco_eval,
            args=(predictions, current_epoch, current_step, summary_writer))
        threads.append(t)
        t.start()

        # Terminate eval job when final checkpoint is reached
        total_step = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                         FLAGS.train_batch_size)
        if current_step >= total_step:
          tf.logging.info(
              'Evaluation finished after training step %d' % current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long
        # after the CPU job tells it to start evaluating. In this case,
        # the checkpoint file could have been deleted already.
        tf.logging.info(
            'Checkpoint %s no longer exists, skipping checkpoint' % ckpt)

    for t in threads:
      t.join()

    if not SUCCESS:
      mlperf_log.ssd_print(key=mlperf_log.RUN_STOP, value={'success': 'false'})
      mlperf_log.ssd_print(key=mlperf_log.RUN_FINAL)
    summary_writer.close()
  elif FLAGS.mode == 'eval_once':
    eval_estimator = tpu_estimator.TPUEstimator(
        model_fn=ssd_model.ssd_model_fn,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.eval_batch_size,
        config=run_config,
        params=params)

    output_dir = os.path.join(eval_estimator.model_dir, 'eval')
    tf.gfile.MakeDirs(output_dir)
    # Summary writer writes out eval metrics.
    summary_writer = tf.summary.FileWriter(output_dir)

    # Run evaluation when there's a new checkpoint
    for ckpt in next_checkpoint(FLAGS.model_dir):
      current_step = int(os.path.basename(ckpt).split('-')[1])
      current_epoch = current_step // params['steps_per_epoch']
      tf.logging.info('current epoch: %s' % current_epoch)
      if FLAGS_eval_epoch < current_epoch:
        continue
      if FLAGS_eval_epoch > current_epoch:
        break

      tf.logging.info('Starting to evaluate.')
      try:
        mlperf_log.ssd_print(key=mlperf_log.EVAL_START, value=current_epoch)

        predictions = list(
            eval_estimator.predict(
                checkpoint_path=ckpt,
                input_fn=dataloader.SSDInputReader(
                    FLAGS.validation_file_pattern, is_training=False)))

        coco_eval(predictions, current_epoch, current_step, summary_write)

        # Terminate eval job when final checkpoint is reached
        total_step = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                         FLAGS.train_batch_size)
        if current_step >= total_step:
          if not SUCCESS:
            mlperf_log.ssd_print(
                key=mlperf_log.RUN_STOP, value={'success': 'false'})
            mlperf_log.ssd_print(key=mlperf_log.RUN_FINAL)
          tf.logging.info(
              'Evaluation finished after training step %d' % current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long
        # after the CPU job tells it to start evaluating. In this case,
        # the checkpoint file could have been deleted already.
        tf.logging.info(
            'Checkpoint %s no longer exists, skipping checkpoint' % ckpt)

    summary_writer.close()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
