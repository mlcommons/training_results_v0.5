# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Estimator functions supporting running on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import subprocess
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import lookup_ops
from mlperf_compliance import mlperf_log
import async_checkpoint
import gnmt_model
from utils import evaluation_utils
from utils import iterator_utils
from utils import misc_utils
from utils import nmt_utils
from utils import vocab_utils


def make_model_fn(hparams):
  """Construct a GNMT model function for training."""

  def _model_fn(features, labels, mode, params):
    """Model function."""
    del labels, params
    # Create a GNMT model for training.
    # assert (hparams.encoder_type == "gnmt" or
    #        hparams.attention_architecture in ["gnmt", "gnmt_v2"])
    model = gnmt_model.GNMTModel(hparams, mode=mode, features=features)
    if mode == tf.contrib.learn.ModeKeys.INFER:
      predicted_ids = model.predicted_ids
      # make sure outputs is of shape [batch_size, time] or [beam_width,
      # batch_size, time] when using beam search.
      if hparams.time_major:
        predicted_ids = tf.transpose(predicted_ids, [2, 1, 0])
      elif predicted_ids.shape.ndims == 3:
        # beam search output in [batch_size, time, beam_width] shape.
        predicted_ids = tf.transpose(predicted_ids, [2, 0, 1])
      # Get the top predictions from beam search.
      predicted_ids = tf.gather_nd(predicted_ids, [0])
      predictions = {"predictions": predicted_ids}
      return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)

    elif mode == tf.contrib.learn.ModeKeys.TRAIN:
      loss = tf.zeros([], dtype=tf.float32)
      train_op = model.update

    else:
      raise ValueError("Unknown mode in model_fn: %s" % mode)

    if hparams.use_tpu:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op)
    else:
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  return _model_fn


def make_input_fn(hparams, mode):
  """Construct a input function for training."""

  def _input_fn(params):
    """Input function."""
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
      tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
    else:
      src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
      tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    src_dataset = tf.data.TextLineDataset(src_file)
    tgt_dataset = tf.data.TextLineDataset(tgt_file)

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      if "context" in params:
        batch_size = params["batch_size"]
        num_hosts = params["context"].num_hosts
        # TODO(dehao): update to use current_host once available in API.
        current_host = params["context"].current_input_fn_deployment()[1]
      else:
        num_hosts = 1
        current_host = 0
        batch_size = hparams.batch_size
      mlperf_log.gnmt_print(key=mlperf_log.INPUT_BATCH_SIZE, value=batch_size)
      mlperf_log.gnmt_print(
          key=mlperf_log.TRAIN_HP_MAX_SEQ_LEN, value=hparams.src_max_len)
      return iterator_utils.get_iterator(
          src_dataset,
          tgt_dataset,
          src_vocab_table,
          tgt_vocab_table,
          batch_size=batch_size,
          sos=hparams.sos,
          eos=hparams.eos,
          random_seed=hparams.random_seed,
          num_buckets=hparams.num_buckets,
          src_max_len=hparams.src_max_len,
          tgt_max_len=hparams.tgt_max_len,
          output_buffer_size=None,
          skip_count=None,
          num_shards=num_hosts,
          shard_index=current_host,
          reshuffle_each_iteration=True,
          use_char_encode=hparams.use_char_encode,
          filter_oversized_sequences=True)
    else:
      return iterator_utils.get_infer_iterator(
          src_dataset,
          src_vocab_table,
          batch_size=hparams.infer_batch_size,
          eos=hparams.eos,
          src_max_len=hparams.src_max_len,
          use_char_encode=hparams.use_char_encode)

  def _synthetic_input_fn(params):
    """Fake inputs for debugging and benchmarking."""
    del params
    batch_size = hparams.batch_size
    src_max_len = hparams.src_max_len
    tgt_max_len = hparams.tgt_max_len
    features = {
        "source":
            tf.random_uniform(
                dtype=tf.int32,
                minval=1,
                maxval=10,
                seed=1,
                shape=(batch_size, src_max_len)),
        "target_input":
            tf.random_uniform(
                dtype=tf.int32,
                minval=1,
                maxval=10,
                seed=2,
                shape=(batch_size, tgt_max_len)),
        "target_output":
            tf.random_uniform(
                dtype=tf.int32,
                minval=1,
                maxval=10,
                seed=3,
                shape=(batch_size, tgt_max_len)),
        "source_sequence_length":
            tf.constant([src_max_len] * batch_size),
        "target_sequence_length":
            tf.constant([tgt_max_len] * batch_size)
    }
    return features

  if hparams.use_synthetic_data:
    return _synthetic_input_fn
  else:
    return _input_fn


def get_distribution_strategy(num_gpus):
  if num_gpus == 0:
    return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
  else:
    return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)


def get_sacrebleu(trans_file, detokenizer_file):
  """Detokenize the trans_file and get the sacrebleu score."""
  assert tf.gfile.Exists(detokenizer_file)
  local_detokenizer_file = "/tmp/detokenizer.perl"
  if not tf.gfile.Exists(local_detokenizer_file):
    tf.gfile.Copy(detokenizer_file, local_detokenizer_file)

  assert tf.gfile.Exists(trans_file)
  local_trans_file = "/tmp/newstest2014_out.tok.de"
  if tf.gfile.Exists(local_trans_file):
    tf.gfile.Remove(local_trans_file)
  tf.gfile.Copy(trans_file, local_trans_file)

  detok_trans_path = "/tmp/newstest2014_out.detok.de"
  if tf.gfile.Exists(detok_trans_path):
    tf.gfile.Remove(detok_trans_path)

  # Detokenize the trans_file.
  cmd = "cat %s | perl %s -l de | cat > %s" % (
      local_trans_file, local_detokenizer_file, detok_trans_path)
  subprocess.run(cmd, shell=True)
  assert tf.gfile.Exists(detok_trans_path)

  # run sacrebleu
  cmd = ("cat %s | sacrebleu -t wmt14/full -l en-de --score-only -lc --tokenize"
         " intl") % (detok_trans_path)
  sacrebleu = subprocess.run([cmd], stdout=subprocess.PIPE, shell=True)
  return float(sacrebleu.stdout.strip())


def _convert_ids_to_strings(tgt_vocab_file, ids):
  """Convert prediction ids to words."""
  with tf.Session() as sess:
    reverse_target_vocab_table = lookup_ops.index_to_string_table_from_file(
        tgt_vocab_file, default_value=vocab_utils.UNK)
    sess.run(tf.tables_initializer())
    translations = sess.run(
        reverse_target_vocab_table.lookup(
            tf.to_int64(tf.convert_to_tensor(np.asarray(ids)))))
  return translations


def get_metric(hparams, estimator):
  """Run inference and compute metric."""
  predictions = estimator.predict(
      make_input_fn(hparams, tf.contrib.learn.ModeKeys.INFER))
  predicted_ids = []
  for prediction in predictions:
    predicted_ids.append(prediction["predictions"])

  mlperf_log.gnmt_print(
      key=mlperf_log.EVAL_SIZE, value=hparams.examples_to_infer)
  if hparams.examples_to_infer < len(predicted_ids):
    predicted_ids = predicted_ids[0:hparams.examples_to_infer]
  translations = _convert_ids_to_strings(hparams.tgt_vocab_file, predicted_ids)

  trans_file = os.path.join(
      hparams.out_dir, "newstest2014_out_{}.tok.de".format(
          estimator.get_variable_value(tf.GraphKeys.GLOBAL_STEP)))
  trans_dir = os.path.dirname(trans_file)
  if not tf.gfile.Exists(trans_dir):
    tf.gfile.MakeDirs(trans_dir)
  tf.logging.info("Writing to file %s" % trans_file)
  with codecs.getwriter("utf-8")(tf.gfile.GFile(trans_file,
                                                mode="wb")) as trans_f:
    trans_f.write("")  # Write empty string to ensure file is created.
    for translation in translations:
      sentence = nmt_utils.get_translation(
          translation,
          tgt_eos=hparams.eos,
          subword_option=hparams.subword_option)
      trans_f.write((sentence + b"\n").decode("utf-8"))

  # Evaluation
  output_dir = os.path.join(estimator.model_dir, "eval")
  tf.gfile.MakeDirs(output_dir)
  summary_writer = tf.summary.FileWriter(output_dir)

  ref_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)

  metric = "bleu"
  if hparams.use_borg:
    score = evaluation_utils.evaluate(ref_file, trans_file, metric,
                                      hparams.subword_option)
  else:
    score = get_sacrebleu(trans_file, hparams.detokenizer_file)
  with tf.Graph().as_default():
    summaries = []
    summaries.append(tf.Summary.Value(tag=metric, simple_value=score))
  tf_summary = tf.Summary(value=list(summaries))
  summary_writer.add_summary(
      tf_summary, estimator.get_variable_value(tf.GraphKeys.GLOBAL_STEP))

  misc_utils.print_out("  %s: %.1f" % (metric, score))

  summary_writer.close()
  return score


def _get_tpu_run_config(hparams, standalone_training=False):
  master = None
  cluster = None
  if hparams.tpu_name is None:
    master = hparams.master
  else:
    cluster = tf.contrib.cluster_resolver.TPUClusterResolver(hparams.tpu_name)
  steps_per_epoch = int(hparams.num_examples_per_epoch/hparams.batch_size)
  # Save one checkpoint for each epoch.
  return tf.contrib.tpu.RunConfig(
      master=master,
      cluster=cluster,
      model_dir=hparams.out_dir,
      save_checkpoints_steps=None if (hparams.use_async_checkpoint and
                                      standalone_training) else steps_per_epoch,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=steps_per_epoch,
          per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
          .PER_HOST_V1))


def _get_tgt_sos_eos_id(hparams):
  with tf.Session() as sess:
    _, tgt_vocab_table = vocab_utils.create_vocab_tables(
        hparams.src_vocab_file, hparams.tgt_vocab_file, hparams.share_vocab)
    tgt_sos_id = tf.cast(
        tgt_vocab_table.lookup(tf.constant(hparams.sos)), tf.int32)
    tgt_eos_id = tf.cast(
        tgt_vocab_table.lookup(tf.constant(hparams.eos)), tf.int32)
    sess.run(tf.tables_initializer())
    tgt_sos_id = sess.run(tgt_sos_id, {})
    tgt_eos_id = sess.run(tgt_eos_id, {})
    return tgt_sos_id, tgt_eos_id


def train_fn(hparams):
  """Train function."""
  hparams.tgt_sos_id, hparams.tgt_eos_id = _get_tgt_sos_eos_id(hparams)
  model_fn = make_model_fn(hparams)
  input_fn = make_input_fn(hparams, tf.contrib.learn.ModeKeys.TRAIN)

  mlperf_log.gnmt_print(key=mlperf_log.TRAIN_LOOP)
  mlperf_log.gnmt_print(key=mlperf_log.TRAIN_EPOCH, value=0)
  mlperf_log.gnmt_print(key=mlperf_log.INPUT_SIZE,
                        value=hparams.num_examples_per_epoch)

  if hparams.use_tpu:
    run_config = _get_tpu_run_config(hparams, True)
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        config=run_config,
        use_tpu=hparams.use_tpu,
        train_batch_size=hparams.batch_size,
        eval_batch_size=hparams.batch_size,
        predict_batch_size=hparams.infer_batch_size)
  else:
    distribution_strategy = get_distribution_strategy(hparams.num_gpus)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=hparams.out_dir,
        config=tf.estimator.RunConfig(train_distribute=distribution_strategy))

  hooks = []
  if hparams.use_async_checkpoint:
    hooks.append(
        async_checkpoint.AsyncCheckpointSaverHook(
            checkpoint_dir=hparams.out_dir,
            save_steps=int(
                hparams.num_examples_per_epoch / hparams.batch_size)))

  estimator.train(
      input_fn=input_fn, max_steps=hparams.num_train_steps, hooks=hooks)
  # Return value is not used
  return 0.0


def eval_fn(hparams):
  """Inference function."""
  hparams.tgt_sos_id, hparams.tgt_eos_id = _get_tgt_sos_eos_id(hparams)
  model_fn = make_model_fn(hparams)
  run_config = _get_tpu_run_config(hparams, False)

  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      config=run_config,
      use_tpu=hparams.use_tpu,
      train_batch_size=hparams.batch_size,
      eval_batch_size=hparams.batch_size,
      predict_batch_size=hparams.infer_batch_size)
  return get_metric(hparams, estimator)


def train_and_eval_fn(hparams):
  """Train and evaluation function."""
  hparams.tgt_sos_id, hparams.tgt_eos_id = 1, 2
  model_fn = make_model_fn(hparams)
  input_fn = make_input_fn(hparams, tf.contrib.learn.ModeKeys.TRAIN)
  run_config = _get_tpu_run_config(hparams, False)
  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      config=run_config,
      use_tpu=hparams.use_tpu,
      train_batch_size=hparams.batch_size,
      eval_batch_size=hparams.batch_size,
      predict_batch_size=hparams.infer_batch_size)

  score = 0.0
  mlperf_log.gnmt_print(key=mlperf_log.TRAIN_LOOP)
  mlperf_log.gnmt_print(key=mlperf_log.EVAL_TARGET, value=hparams.target_bleu)
  for i in range(hparams.max_train_epochs):
    mlperf_log.gnmt_print(key=mlperf_log.TRAIN_EPOCH, value=i)
    tf.logging.info("Start training epoch %d", i)
    mlperf_log.gnmt_print(key=mlperf_log.INPUT_SIZE,
                          value=hparams.num_examples_per_epoch)
    steps_per_epoch = int(hparams.num_examples_per_epoch/hparams.batch_size)
    max_steps = steps_per_epoch * (i + 1)
    estimator.train(input_fn=input_fn, max_steps=max_steps)
    mlperf_log.gnmt_print(
        key=mlperf_log.TRAIN_CHECKPOINT, value=("Under " + hparams.out_dir))
    tf.logging.info("End training epoch %d", i)
    mlperf_log.gnmt_print(key=mlperf_log.EVAL_START)
    score = get_metric(hparams, estimator)
    tf.logging.info("Score after epoch %d: %f", i, score)
    mlperf_log.gnmt_print(key=mlperf_log.EVAL_ACCURACY,
                          value={"value": score, "epoch": i})
    mlperf_log.gnmt_print(key=mlperf_log.EVAL_STOP, value=i)
    if score >= hparams.target_bleu:
      mlperf_log.gnmt_print(mlperf_log.RUN_STOP, {"success": True})
      return score

  mlperf_log.gnmt_print(mlperf_log.RUN_STOP, {"success": False})
  return score
