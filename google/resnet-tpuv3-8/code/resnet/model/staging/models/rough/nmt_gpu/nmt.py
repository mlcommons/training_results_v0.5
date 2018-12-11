# Copyright 2017 Google Inc. All Rights Reserved.
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

"""TensorFlow NMT model implementation."""
from __future__ import print_function

import argparse
import os
import random
import sys

# import matplotlib.image as mpimg
import numpy as np
import time
import tensorflow as tf

from mlperf_compliance import mlperf_log
import estimator
from utils import evaluation_utils
from utils import iterator_utils
from utils import misc_utils as utils
from utils import vocab_utils
from variable_mgr import constants

utils.check_tensorflow_version()

FLAGS = None


def add_arguments(parser):
  """Build ArgumentParser."""
  parser.register("type", "bool", lambda v: v.lower() == "true")

  # network
  parser.add_argument(
      "--num_units", type=int, default=1024, help="Network size.")
  parser.add_argument(
      "--num_layers", type=int, default=4, help="Network depth.")
  parser.add_argument("--num_encoder_layers", type=int, default=None,
                      help="Encoder depth, equal to num_layers if None.")
  parser.add_argument("--num_decoder_layers", type=int, default=None,
                      help="Decoder depth, equal to num_layers if None.")
  parser.add_argument(
      "--encoder_type",
      type=str,
      default="gnmt",
      help="""\
      uni | bi | gnmt.
      For bi, we build num_encoder_layers/2 bi-directional layers.
      For gnmt, we build 1 bi-directional layer, and (num_encoder_layers - 1)
        uni-directional layers.\
      """)
  parser.add_argument(
      "--residual",
      type="bool",
      nargs="?",
      const=True,
      default=True,
      help="Whether to add residual connections.")
  parser.add_argument("--time_major", type="bool", nargs="?", const=True,
                      default=True,
                      help="Whether to use time-major mode for dynamic RNN.")
  parser.add_argument("--num_embeddings_partitions", type=int, default=0,
                      help="Number of partitions for embedding vars.")

  # attention mechanisms
  parser.add_argument(
      "--attention",
      type=str,
      default="normed_bahdanau",
      help="""\
      luong | scaled_luong | bahdanau | normed_bahdanau or set to "" for no
      attention\
      """)
  parser.add_argument(
      "--attention_architecture",
      type=str,
      default="gnmt_v2",
      help="""\
      standard | gnmt | gnmt_v2.
      standard: use top layer to compute attention.
      gnmt: GNMT style of computing attention, use previous bottom layer to
          compute attention.
      gnmt_v2: similar to gnmt, but use current bottom layer to compute
          attention.\
      """)
  parser.add_argument(
      "--output_attention", type="bool", nargs="?", const=True,
      default=True,
      help="""\
      Only used in standard attention_architecture. Whether use attention as
      the cell output at each timestep.
      .\
      """)
  parser.add_argument(
      "--pass_hidden_state", type="bool", nargs="?", const=True,
      default=True,
      help="""\
      Whether to pass encoder's hidden state to decoder when using an attention
      based model.\
      """)

  # optimizer
  parser.add_argument(
      "--optimizer", type=str, default="adam", help="sgd | adam")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=5e-4,
      help="Learning rate. Adam: 0.001 | 0.0001")
  parser.add_argument("--warmup_steps", type=int, default=0,
                      help="How many steps we inverse-decay learning.")
  parser.add_argument("--warmup_scheme", type=str, default="t2t", help="""\
      How to warmup learning rates. Options include:
        t2t: Tensor2Tensor's way, start with lr 100 times smaller, then
             exponentiate until the specified lr.\
      """)
  parser.add_argument(
      "--decay_scheme", type=str, default="", help="""\
      How we decay learning rate. Options include:
        luong234: after 2/3 num train steps, we start halving the learning rate
          for 4 times before finishing.
        luong5: after 1/2 num train steps, we start halving the learning rate
          for 5 times before finishing.\
        luong10: after 1/2 num train steps, we start halving the learning rate
          for 10 times before finishing.\
      """)

  parser.add_argument(
      "--num_train_steps", type=int, default=100000, help="Num steps to train.")
  parser.add_argument(
      "--max_train_epochs", type=int, default=8, help="Max number of epochs.")
  parser.add_argument("--num_examples_per_epoch", type=int, default=4068191,
                      help="Number of examples in one epoch")
  parser.add_argument(
      "--target_bleu", type=float, default=22.0, help="Target bleu.")
  parser.add_argument("--colocate_gradients_with_ops", type="bool", nargs="?",
                      const=True,
                      default=True,
                      help=("Whether try colocating gradients with "
                            "corresponding op"))
  parser.add_argument("--label_smoothing", type=float, default=0.1,
                      help=("If nonzero, smooth the labels towards "
                            "1/num_classes."))

  # initializer
  parser.add_argument("--init_op", type=str, default="uniform",
                      help="uniform | glorot_normal | glorot_uniform")
  parser.add_argument("--init_weight", type=float, default=0.1,
                      help=("for uniform init_op, initialize weights "
                            "between [-this, this]."))

  # data
  parser.add_argument(
      "--src", type=str, default="en", help="Source suffix, e.g., en.")
  parser.add_argument(
      "--tgt", type=str, default="de", help="Target suffix, e.g., de.")
  parser.add_argument(
      "--data_dir", type=str, default="",
      help="Training/eval data directory.")

  parser.add_argument(
      "--train_prefix",
      type=str,
      default="train.tok.clean.bpe.32000",
      help="Train prefix, expect files with src/tgt suffixes.")
  parser.add_argument(
      "--dev_prefix",
      type=str,
      default="newstest2014.tok.bpe.32000",
      help="Dev prefix, expect files with src/tgt suffixes.")
  parser.add_argument(
      "--test_prefix",
      type=str,
      default="newstest2014.tok.bpe.32000",
      help="Test prefix, expect files with src/tgt suffixes.")

  parser.add_argument(
      "--output_dir", type=str, default="",
      help="Store log/model files.")

  # Vocab
  parser.add_argument(
      "--vocab_prefix",
      type=str,
      default="vocab.bpe.32000",
      help="""\
      Vocab prefix, expect files with src/tgt suffixes.\
      """)

  parser.add_argument(
      "--embed_prefix",
      type=str,
      default=None,
      help="""\
      Pretrained embedding prefix, expect files with src/tgt suffixes.
      The embedding files should be Glove formatted txt files.\
      """)
  parser.add_argument("--sos", type=str, default="<s>",
                      help="Start-of-sentence symbol.")
  parser.add_argument("--eos", type=str, default="</s>",
                      help="End-of-sentence symbol.")
  parser.add_argument(
      "--share_vocab",
      type="bool",
      nargs="?",
      const=True,
      default=True,
      help="""\
      Whether to use the source vocab and embeddings for both source and
      target.\
      """)
  parser.add_argument("--check_special_token", type="bool", default=True,
                      help="""\
                      Whether check special sos, eos, unk tokens exist in the
                      vocab files.\
                      """)

  # Sequence lengths
  parser.add_argument(
      "--src_max_len",
      type=int,
      default=50,
      help="Max length of src sequences during training.")
  parser.add_argument(
      "--tgt_max_len",
      type=int,
      default=50,
      help="Max length of tgt sequences during training.")
  parser.add_argument("--src_max_len_infer", type=int, default=None,
                      help="Max length of src sequences during inference.")
  parser.add_argument("--tgt_max_len_infer", type=int, default=80,
                      help="""\
      Max length of tgt sequences during inference.  Also use to restrict the
      maximum decoding length.\
      """)

  # Default settings works well (rarely need to change)
  parser.add_argument("--unit_type", type=str, default="lstm",
                      help="lstm | gru | layer_norm_lstm | nas")
  parser.add_argument("--forget_bias", type=float, default=1.0,
                      help="Forget bias for BasicLSTMCell.")
  parser.add_argument("--dropout", type=float, default=0.2,
                      help="Dropout rate (not keep_prob)")
  parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                      help="Clip gradients to this norm.")
  parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")

  parser.add_argument("--steps_per_stats", type=int, default=5,
                      help=("How many training steps to do per stats logging."
                            "Save checkpoint every 10x steps_per_stats"))
  parser.add_argument("--max_train", type=int, default=0,
                      help="Limit on the size of training data (0: no limit).")
  parser.add_argument(
      "--num_buckets",
      type=int,
      default=1,
      help="Put data into similar-length buckets.")

  # SPM
  parser.add_argument("--subword_option", type=str, default="bpe",
                      choices=["", "bpe", "spm"],
                      help="""\
                      Set to bpe or spm to activate subword desegmentation.\
                      """)

  # Experimental encoding feature.
  parser.add_argument("--use_char_encode", type="bool", default=False,
                      help="""\
                      Whether to split each word or bpe into character, and then
                      generate the word-level representation from the character
                      reprentation.
                      """)

  # Misc
  parser.add_argument(
      "--save_checkpoints_steps", type=int, default=1000,
      help="save_checkpoints_steps")
  parser.add_argument(
      "--num_gpus", type=int, default=1, help="Number of gpus in each worker.")
  parser.add_argument(
      "--log_device_placement",
      type="bool",
      nargs="?",
      const=True,
      default=True,
      help="Debug GPU allocation.")
  parser.add_argument("--steps_per_external_eval", type=int, default=None,
                      help="""\
      How many training steps to do per external evaluation.  Automatically set
      based on data if None.\
      """)
  parser.add_argument("--hparams_path", type=str, default=None,
      help=("Path to standard hparams json file that overrides"
            "hparams values from FLAGS."))
  parser.add_argument(
      "--random_seed",
      type=int,
      default=1,
      help="Random seed (>0, set a specific seed).")
  parser.add_argument("--override_loaded_hparams", type="bool", nargs="?",
                      const=True, default=False,
                      help="Override loaded hparams with values specified")
  parser.add_argument("--num_keep_ckpts", type=int, default=5,
                      help="Max number of checkpoints to keep.")
  parser.add_argument("--avg_ckpts", type="bool", nargs="?",
                      const=True, default=False, help=("""\
                      Average the last N checkpoints for external evaluation.
                      N can be controlled by setting --num_keep_ckpts.\
                      """))
  parser.add_argument("--language_model", type="bool", nargs="?",
                      const=True, default=False,
                      help="True to train a language model, ignoring encoder")

  # Inference
  parser.add_argument("--ckpt", type=str, default="",
                      help="Checkpoint file to load a model for inference.")
  parser.add_argument("--inference_input_file", type=str, default=None,
                      help="Set to the text to decode.")
  parser.add_argument("--inference_list", type=str, default=None,
                      help=("A comma-separated list of sentence indices "
                            "(0-based) to decode."))
  parser.add_argument(
      "--infer_batch_size",
      type=int,
      default=64,
      help="Batch size for inference mode.")
  parser.add_argument("--detokenizer_file", type=str,
                      default="",
                      help=("""Detokenizer script file."""))
  parser.add_argument("--use_borg", type="bool", default=False)

  # Advanced inference arguments
  parser.add_argument("--infer_mode", type=str, default="beam_search",
                      choices=["greedy", "sample", "beam_search"],
                      help="Which type of decoder to use during inference.")
  parser.add_argument("--beam_width", type=int, default=5,
                      help=("""\
      beam width when using beam search decoder. If 0 (default), use standard
      decoder with greedy helper.\
      """))
  parser.add_argument(
      "--length_penalty_weight",
      type=float,
      default=0.6,
      help="Length penalty for beam search.")
  parser.add_argument(
      "--coverage_penalty_weight",
      type=float,
      default=0.1,
      help="Coverage penalty for beam search.")
  parser.add_argument("--sampling_temperature", type=float,
                      default=0.0,
                      help=("""\
      Softmax sampling temperature for inference decoding, 0.0 means greedy
      decoding. This option is ignored when using beam search.\
      """))
  parser.add_argument("--num_translations_per_input", type=int, default=1,
                      help=("""\
      Number of translations generated for each sentence. This is only used for
      inference.\
      """))

  # Job info
  parser.add_argument("--jobid", type=int, default=0,
                      help="Task id of the worker.")
  parser.add_argument("--num_workers", type=int, default=1,
                      help="Number of workers (inference only).")
  parser.add_argument("--num_inter_threads", type=int, default=0,
                      help="number of inter_op_parallelism_threads")
  parser.add_argument("--num_intra_threads", type=int, default=0,
                      help="number of intra_op_parallelism_threads")

  # Fp16
  parser.add_argument("--use_fp16", type="bool", default=True,
                      help="use_fp16 for training and inference")
  parser.add_argument(
      "--fp16_loss_scale",
      type=float,
      default=128,
      help="If fp16 is enabled, the loss is multiplied by this amount "
      "right before gradients are computed, then each gradient "
      "is divided by this amount. Mathematically, this has no "
      "effect, but it helps avoid fp16 underflow. Set to 1 to "
      "effectively disable.")
  parser.add_argument(
      "--enable_auto_loss_scale",
      type="bool",
      default=False,
      help="If True and use_fp16 is True, automatically adjust the "
      "loss scale during training.")
  parser.add_argument(
      "--fp16_inc_loss_scale_every_n",
      type=int,
      default=1000,
      help="If fp16 is enabled and enable_auto_loss_scale is "
      "True, increase the loss scale every n steps.")
  parser.add_argument(
      "--check_tower_loss_numerics",
      type="bool",
      default=False,  # Set to false for xla.compile()
      help="whether to check tower loss numerics")
  parser.add_argument(
      "--use_fp32_batch_matmul",
      type="bool",
      default=True,
      help="Whether to use fp32 batch matmul")

  # Performance
  # XLA
  parser.add_argument(
      "--force_inputs_padding",
      type="bool",
      default=False,
      help="Force padding input batch to src_max_len and tgt_max_len")
  parser.add_argument(
      "--use_xla",
      type="bool",
      default=False,
      help="Use xla to compile a few selected locations, mostly Defuns.")
  parser.add_argument(
      "--use_xla_compile",
      type="bool",
      default=False,
      help="Use xla.compile() for each tower's fwd and bak pass.")
  parser.add_argument(
      "--use_autojit_xla",
      type="bool",
      default=False,
      help="Use auto jit xla.")
  # GPU knobs
  parser.add_argument(
      "--use_pintohost_optimizer",
      type="bool",
      default=False,
      help="whether to use PinToHost optimizer")
  parser.add_argument(
      "--use_cudnn_lstm",
      type="bool",
      default=False,
      help="whether to use cudnn_lstm for encoder, non residual layers")
  parser.add_argument(
      "--use_loose_bidi_cudnn_lstm",
      type="bool",
      default=False,
      help="whether to use loose bidi cudnn_lstm")
  parser.add_argument(
      "--use_fused_lstm",
      type="bool",
      default=False,
      help="whether to use fused lstm and variant. If enabled, training will "
      "use LSTMBlockFusedCell, infer will use LSTMBlockCell when appropriate.")
  parser.add_argument(
      "--use_fused_lstm_dec",
      type="bool",
      default=False,
      help="whether to use fused lstm for decoder (training only).")
  parser.add_argument(
      "--gpu_indices",
      type=str,
      default="",
      help="Indices of worker GPUs in ring order")
  parser.add_argument(
      "--gpu_thread_mode",
      type=str,
      default="global",
      help="Methods to assign GPU host work to threads. "
      "global: all GPUs and CPUs share the same global threads; "
      "gpu_private: a private threadpool for each GPU; "
      "gpu_shared: all GPUs share the same threadpool.")
  parser.add_argument(
      "--per_gpu_thread_count",
      type=int,
      default=0,
      help="The number of threads to use for GPU. Only valid when "
      "gpu_thread_mode is not global.")
  parser.add_argument(
      "--sync_on_finish",
      type="bool",
      default=False,
      help="Enable/disable whether the devices are synced after each "
      "step.")
  parser.add_argument(
      "--force_gpu_compatible",
      type="bool",
      default=False,
      help="whether to enable force_gpu_compatible in GPU_Options")
  # Graph knobs
  parser.add_argument("--parallel_iterations", type=int, default=10,
                      help="number of parallel iterations in dynamic_rnn")
  parser.add_argument("--use_dist_strategy", type="bool", default=False,
                      help="whether to use distribution strategy")
  parser.add_argument(
      "--hierarchical_copy",
      type="bool",
      default=False,
      help="Use hierarchical copies. Currently only optimized for "
      "use on a DGX-1 with 8 GPUs and may perform poorly on "
      "other hardware. Requires --num_gpus > 1, and only "
      "recommended when --num_gpus=8")
  parser.add_argument(
      "--network_topology",
      type=constants.NetworkTopology,
      default=constants.NetworkTopology.DGX1,
      choices=list(constants.NetworkTopology))
  parser.add_argument(
      "--enable_layout_optimizer",
      type="bool",
      default=False,
      help="whether to enable layout optimizer")
  parser.add_argument(
      "--use_block_lstm",
      type="bool",
      default=False,
      help="whether to use block lstm")
  parser.add_argument(
      "--use_defun",
      type="bool",
      default=False,
      help="whether to use Defun")

  # Gradient tricks
  parser.add_argument(
      "--gradient_repacking",
      type=int,
      default=0,
      help="Use gradient repacking. It"
      "currently only works with replicated mode. At the end of"
      "of each step, it repacks the gradients for more efficient"
      "cross-device transportation. A non-zero value specifies"
      "the number of split packs that will be formed.")
  parser.add_argument(
      "--compact_gradient_transfer",
      type="bool",
      default=True,
      help="Compact gradient as much as possible for cross-device transfer and "
      "aggregation.")
  parser.add_argument(
      "--all_reduce_spec",
      type=str,
      default="nccl",
      help="A specification of the all_reduce algorithm to be used "
      "for reducing gradients.  For more details, see "
      "parse_all_reduce_spec in variable_mgr.py.  An "
      "all_reduce_spec has BNF form:\n"
      "int ::= positive whole number\n"
      "g_int ::= int[KkMGT]?\n"
      "alg_spec ::= alg | alg#int\n"
      "range_spec ::= alg_spec | alg_spec/alg_spec\n"
      "spec ::= range_spec | range_spec:g_int:range_spec\n"
      "NOTE: not all syntactically correct constructs are "
      "supported.\n\n"
      "Examples:\n "
      "\"xring\" == use one global ring reduction for all "
      "tensors\n"
      "\"pscpu\" == use CPU at worker 0 to reduce all tensors\n"
      "\"nccl\" == use NCCL to locally reduce all tensors.  "
      "Limited to 1 worker.\n"
      "\"nccl/xring\" == locally (to one worker) reduce values "
      "using NCCL then ring reduce across workers.\n"
      "\"pscpu:32k:xring\" == use pscpu algorithm for tensors of "
      "size up to 32kB, then xring for larger tensors.")
  parser.add_argument(
      "--agg_small_grads_max_bytes",
      type=int,
      default=0,
      help="If > 0, try to aggregate tensors of less than this "
      "number of bytes prior to all-reduce.")
  parser.add_argument(
      "--agg_small_grads_max_group",
      type=int,
      default=10,
      help="When aggregating small tensors for all-reduce do not "
      "aggregate more than this many into one new tensor.")
  parser.add_argument(
      "--allreduce_merge_scope",
      type=int,
      default=1,
      help="Establish a name scope around this many "
      "gradients prior to creating the all-reduce operations. "
      "It may affect the ability of the backend to merge "
      "parallel ops.")
  # Other knobs
  parser.add_argument(
      "--local_parameter_device",
      type=str,
      default="gpu",
      help="Device to use as parameter server: cpu or gpu. For "
      "distributed training, it can affect where caching of "
      "variables happens.")
  parser.add_argument(
      "--autotune_threshold",
      type=int,
      default=0,
      help="The autotune threshold for the models")
  parser.add_argument(
      "--datasets_num_private_threads",
      type=int,
      default=None,
      help="Number of threads for a private threadpool created for "
      "all datasets computation. By default, we pick an "
      "appropriate number. If set to 0, we use the default "
      "tf-Compute threads for dataset operations.")
  parser.add_argument(
      "--winograd_nonfused",
      type="bool",
      default=True,
      help="Enable/disable using the Winograd non-fused algorithms.")
  parser.add_argument(
      "--batchnorm_persistent",
      type="bool",
      default=True,
      help="Enable/disable using the CUDNN_BATCHNORM_SPATIAL_PERSISTENT "
      "mode for batchnorm.")
  parser.add_argument(
      "--device",
      type=str,
      default="gpu",
      help="Device to use for computation: cpu or gpu")
  parser.add_argument(
      "--allow_growth",
      type="bool",
      default=False,
      help="whether to enable allow_growth in GPU_Options")
  parser.add_argument(
      "--use_resource_vars",
      type="bool",
      default=False,
      help="Use resource variables instead of normal variables. "
      "Resource variables are slower, but this option is useful "
      "for debugging their performance.")
  # Performance tuning specific to MKL.
  parser.add_argument(
      "--mkl",
      type="bool",
      default=False,
      help="If true, set MKL environment variables.")
  parser.add_argument(
      "--kmp_blocktime",
      type=int,
      default=30,
      help="The time, in milliseconds, that a thread should wait, "
      "after completing the execution of a parallel region, "
      "before sleeping")
  parser.add_argument(
      "--kmp_affinity",
      type=str,
      default="granularity=fine,verbose,compact,1,0",
      help="Restricts execution of certain threads (virtual execution "
      "units) to a subset of the physical processing units in a "
      "multiprocessor computer.")
  parser.add_argument(
      "--kmp_settings", type=int, default=1,
      help="If set to 1, MKL settings will be printed.")

  # Debug
  parser.add_argument("--debug", type="bool", default=False,
                      help="Debug train and eval")
  parser.add_argument("--show_metrics", type="bool", default=True,
                      help="whether to show detailed metrics")
  parser.add_argument("--build_graph_only", type="bool", default=False,
                      help="whehter or not just building the graph")
  parser.add_argument("--clip_grads", type="bool", default=True,
                      help="whether to clip gradients")
  parser.add_argument("--profile", type="bool", default=False,
                      help="If generate profile")
  parser.add_argument("--profile_save_steps", type=int, default=10,
                      help="Save timeline every N steps.")

  # TPU
  parser.add_argument("--use_dynamic_rnn", type="bool", default=True)
  parser.add_argument("--master", type=str, default="")
  parser.add_argument("--use_synthetic_data", type="bool", default=False)
  parser.add_argument(
      "--iterations_per_loop",
      type=int,
      default=100,
      help="the number of iterations to run on TPU before returning to host")
  parser.add_argument(
      "--mode", type=str, default="train_and_eval",
      choices=["train", "train_and_eval", "infer"])
  parser.add_argument(
      "--run_name",
      type=str,
      default="",
      help=
      "if set, load ckpt from /gs://ij-d/home/mlperf-nmt/'run_name'"
  )


def create_hparams(flags):
  """Create training hparams."""
  return tf.contrib.training.HParams(
      # Data
      src=flags.src,
      tgt=flags.tgt,
      train_prefix=os.path.join(flags.data_dir, flags.train_prefix),
      dev_prefix=os.path.join(flags.data_dir, flags.dev_prefix),
      test_prefix=os.path.join(flags.data_dir, flags.test_prefix),
      vocab_prefix=os.path.join(flags.data_dir, flags.vocab_prefix),
      embed_prefix=flags.embed_prefix,
      output_dir=flags.output_dir,

      # Networks
      num_units=flags.num_units,
      num_encoder_layers=(flags.num_encoder_layers or flags.num_layers),
      num_decoder_layers=(flags.num_decoder_layers or flags.num_layers),
      dropout=flags.dropout,
      unit_type=flags.unit_type,
      encoder_type=flags.encoder_type,
      residual=flags.residual,
      time_major=flags.time_major,
      num_embeddings_partitions=flags.num_embeddings_partitions,

      # Attention mechanisms
      attention=flags.attention,
      attention_architecture=flags.attention_architecture,
      output_attention=flags.output_attention,
      pass_hidden_state=flags.pass_hidden_state,

      # Train
      optimizer=flags.optimizer,
      num_train_steps=flags.num_train_steps,
      max_train_epochs=flags.max_train_epochs,
      num_examples_per_epoch=flags.num_examples_per_epoch,
      target_bleu=flags.target_bleu,
      label_smoothing=flags.label_smoothing,
      batch_size=flags.batch_size,
      init_op=flags.init_op,
      init_weight=flags.init_weight,
      max_gradient_norm=flags.max_gradient_norm,
      learning_rate=flags.learning_rate,
      warmup_steps=flags.warmup_steps,
      warmup_scheme=flags.warmup_scheme,
      decay_scheme=flags.decay_scheme,
      colocate_gradients_with_ops=flags.colocate_gradients_with_ops,

      # Data constraints
      num_buckets=flags.num_buckets,
      max_train=flags.max_train,
      src_max_len=flags.src_max_len,
      tgt_max_len=flags.tgt_max_len,

      # Inference
      src_max_len_infer=flags.src_max_len_infer,
      tgt_max_len_infer=flags.tgt_max_len_infer,
      infer_batch_size=flags.infer_batch_size,
      detokenizer_file=flags.detokenizer_file,
      use_borg=flags.use_borg,

      # Advanced inference arguments
      infer_mode=flags.infer_mode,
      beam_width=flags.beam_width,
      length_penalty_weight=flags.length_penalty_weight,
      coverage_penalty_weight=flags.coverage_penalty_weight,
      sampling_temperature=flags.sampling_temperature,
      num_translations_per_input=flags.num_translations_per_input,

      # Vocab
      sos=flags.sos if flags.sos else vocab_utils.SOS,
      eos=flags.eos if flags.eos else vocab_utils.EOS,
      subword_option=flags.subword_option,
      check_special_token=flags.check_special_token,
      use_char_encode=flags.use_char_encode,

      # Misc
      forget_bias=flags.forget_bias,
      num_gpus=flags.num_gpus,
      save_checkpoints_steps=flags.save_checkpoints_steps,
      epoch_step=0,  # record where we were within an epoch.
      steps_per_stats=flags.steps_per_stats,
      steps_per_external_eval=flags.steps_per_external_eval,
      share_vocab=flags.share_vocab,
      log_device_placement=flags.log_device_placement,
      random_seed=flags.random_seed,
      override_loaded_hparams=flags.override_loaded_hparams,
      num_keep_ckpts=flags.num_keep_ckpts,
      avg_ckpts=flags.avg_ckpts,
      language_model=flags.language_model,
      num_intra_threads=flags.num_intra_threads,
      num_inter_threads=flags.num_inter_threads,

      # Fp16
      use_fp16=flags.use_fp16,
      fp16_loss_scale=flags.fp16_loss_scale,
      enable_auto_loss_scale=flags.enable_auto_loss_scale,
      fp16_inc_loss_scale_every_n=flags.fp16_inc_loss_scale_every_n,
      check_tower_loss_numerics=flags.check_tower_loss_numerics,
      use_fp32_batch_matmul=flags.use_fp32_batch_matmul,

      # Performance
      # GPU knbs
      force_inputs_padding=flags.force_inputs_padding,
      use_xla=flags.use_xla,
      use_xla_compile=flags.use_xla_compile,
      use_autojit_xla=flags.use_autojit_xla,
      use_pintohost_optimizer=flags.use_pintohost_optimizer,
      use_cudnn_lstm=flags.use_cudnn_lstm,
      use_loose_bidi_cudnn_lstm=flags.use_loose_bidi_cudnn_lstm,
      use_fused_lstm=flags.use_fused_lstm,
      use_fused_lstm_dec=flags.use_fused_lstm_dec,
      gpu_indices=flags.gpu_indices,
      gpu_thread_mode=flags.gpu_thread_mode,
      per_gpu_thread_count=flags.per_gpu_thread_count,
      sync_on_finish=flags.sync_on_finish,
      force_gpu_compatible=flags.force_gpu_compatible,
      # Graph knobs
      parallel_iterations=flags.parallel_iterations,
      use_dynamic_rnn=flags.use_dynamic_rnn,
      use_dist_strategy=flags.use_dist_strategy,
      hierarchical_copy=flags.hierarchical_copy,
      network_topology=flags.network_topology,
      enable_layout_optimizer=flags.enable_layout_optimizer,
      use_block_lstm=flags.use_block_lstm,
      # Grad tricks
      gradient_repacking=flags.gradient_repacking,
      compact_gradient_transfer=flags.compact_gradient_transfer,
      all_reduce_spec=flags.all_reduce_spec,
      agg_small_grads_max_bytes=flags.agg_small_grads_max_bytes,
      agg_small_grads_max_group=flags.agg_small_grads_max_group,
      allreduce_merge_scope=flags.allreduce_merge_scope,
      # Other knobs
      local_parameter_device=("cpu" if flags.num_gpus ==0
                              else flags.local_parameter_device),
      autotune_threshold=flags.autotune_threshold,
      datasets_num_private_threads=flags.datasets_num_private_threads,
      winograd_nonfused=flags.winograd_nonfused,
      batchnorm_persistent=flags.batchnorm_persistent,
      device=flags.device,
      allow_growth=flags.allow_growth,
      use_resource_vars=flags.use_resource_vars,
      mkl=flags.mkl,
      kmp_blocktime=flags.kmp_blocktime,
      kmp_affinity=flags.kmp_affinity,
      kmp_settings=flags.kmp_settings,

      # Debug
      debug=flags.debug,
      build_graph_only=flags.build_graph_only,
      clip_grads=flags.clip_grads,
      profile=flags.profile,
      profile_save_steps=flags.profile_save_steps,
      show_metrics=flags.show_metrics,

      # TPU
      master=flags.master,
      use_synthetic_data=flags.use_synthetic_data,
      iterations_per_loop=flags.iterations_per_loop,
      mode=flags.mode,
      run_name=flags.run_name)


def _add_argument(hparams, key, value, update=True):
  """Add an argument to hparams; if exists, change the value if update==True."""
  if hasattr(hparams, key):
    if update:
      setattr(hparams, key, value)
  else:
    hparams.add_hparam(key, value)


def extend_hparams(hparams):
  """Add new arguments to hparams."""
  # Sanity checks
  if hparams.encoder_type == "bi" and hparams.num_encoder_layers % 2 != 0:
    raise ValueError("For bi, num_encoder_layers %d should be even" %
                     hparams.num_encoder_layers)
  if (hparams.attention_architecture in ["gnmt"] and
      hparams.num_encoder_layers < 2):
    raise ValueError("For gnmt attention architecture, "
                     "num_encoder_layers %d should be >= 2" %
                     hparams.num_encoder_layers)
  if hparams.subword_option and hparams.subword_option not in ["spm", "bpe"]:
    raise ValueError("subword option must be either spm, or bpe")
  if hparams.infer_mode == "beam_search" and hparams.beam_width <= 0:
    raise ValueError("beam_width must greater than 0 when using beam_search"
                     "decoder.")
  if hparams.infer_mode == "sample" and hparams.sampling_temperature <= 0.0:
    raise ValueError("sampling_temperature must greater than 0.0 when using"
                     "sample decoder.")

  # Different number of encoder / decoder layers
  assert hparams.num_encoder_layers and hparams.num_decoder_layers
  if hparams.num_encoder_layers != hparams.num_decoder_layers:
    hparams.pass_hidden_state = False
    utils.print_out("Num encoder layer %d is different from num decoder layer"
                    " %d, so set pass_hidden_state to False" % (
                        hparams.num_encoder_layers,
                        hparams.num_decoder_layers))

  # Set residual layers
  num_encoder_residual_layers = 0
  num_decoder_residual_layers = 0
  if hparams.residual:
    if hparams.num_encoder_layers > 1:
      num_encoder_residual_layers = hparams.num_encoder_layers - 1
    if hparams.num_decoder_layers > 1:
      num_decoder_residual_layers = hparams.num_decoder_layers - 1

    if hparams.encoder_type == "gnmt":
      # The first unidirectional layer (after the bi-directional layer) in
      # the GNMT encoder can't have residual connection due to the input is
      # the concatenation of fw_cell and bw_cell's outputs.
      num_encoder_residual_layers = hparams.num_encoder_layers - 2

      # Compatible for GNMT models
      if hparams.num_encoder_layers == hparams.num_decoder_layers:
        num_decoder_residual_layers = num_encoder_residual_layers
  _add_argument(hparams, "num_encoder_residual_layers",
                num_encoder_residual_layers)
  _add_argument(hparams, "num_decoder_residual_layers",
                num_decoder_residual_layers)

  # Language modeling
  if hparams.language_model:
    hparams.attention = ""
    hparams.attention_architecture = ""
    hparams.pass_hidden_state = False
    hparams.share_vocab = True
    hparams.src = hparams.tgt
    utils.print_out("For language modeling, we turn off attention and "
                    "pass_hidden_state; turn on share_vocab; set src to tgt.")

  ## Vocab
  # Get vocab file names first
  if hparams.vocab_prefix:
    src_vocab_file = hparams.vocab_prefix + "." + hparams.src
    tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt
  else:
    raise ValueError("hparams.vocab_prefix must be provided.")

  # Source vocab
  src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
      src_vocab_file,
      hparams.output_dir,
      check_special_token=hparams.check_special_token,
      sos=hparams.sos,
      eos=hparams.eos,
      unk=vocab_utils.UNK)

  # Target vocab
  if hparams.share_vocab:
    utils.print_out("  using source vocab for target")
    tgt_vocab_file = src_vocab_file
    tgt_vocab_size = src_vocab_size
  else:
    tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(
        tgt_vocab_file,
        hparams.output_dir,
        check_special_token=hparams.check_special_token,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_utils.UNK)
  mlperf_log.gnmt_print(key=mlperf_log.PREPROC_VOCAB_SIZE,
                        value={"src": src_vocab_size, "tgt": tgt_vocab_size})
  _add_argument(hparams, "src_vocab_size", src_vocab_size)
  _add_argument(hparams, "tgt_vocab_size", tgt_vocab_size)
  _add_argument(hparams, "src_vocab_file", src_vocab_file)
  _add_argument(hparams, "tgt_vocab_file", tgt_vocab_file)

  # Num embedding partitions
  _add_argument(
      hparams, "num_enc_emb_partitions", hparams.num_embeddings_partitions)
  _add_argument(
      hparams, "num_dec_emb_partitions", hparams.num_embeddings_partitions)

  # Pretrained Embeddings
  _add_argument(hparams, "src_embed_file", "")
  _add_argument(hparams, "tgt_embed_file", "")
  if hparams.embed_prefix:
    src_embed_file = hparams.embed_prefix + "." + hparams.src
    tgt_embed_file = hparams.embed_prefix + "." + hparams.tgt

    if tf.gfile.Exists(src_embed_file):
      utils.print_out("  src_embed_file %s exist" % src_embed_file)
      hparams.src_embed_file = src_embed_file

      utils.print_out(
          "For pretrained embeddings, set num_enc_emb_partitions to 1")
      hparams.num_enc_emb_partitions = 1
    else:
      utils.print_out("  src_embed_file %s doesn't exist" % src_embed_file)

    if tf.gfile.Exists(tgt_embed_file):
      utils.print_out("  tgt_embed_file %s exist" % tgt_embed_file)
      hparams.tgt_embed_file = tgt_embed_file

      utils.print_out(
          "For pretrained embeddings, set num_dec_emb_partitions to 1")
      hparams.num_dec_emb_partitions = 1
    else:
      utils.print_out("  tgt_embed_file %s doesn't exist" % tgt_embed_file)

  # Evaluation
  metric = "bleu"
  best_metric_dir = os.path.join(hparams.output_dir, "best_" + metric)
  tf.gfile.MakeDirs(best_metric_dir)
  _add_argument(hparams, "best_" + metric, 0, update=False)
  _add_argument(hparams, "best_" + metric + "_dir", best_metric_dir)

  if hparams.avg_ckpts:
    best_metric_dir = os.path.join(hparams.output_dir, "avg_best_" + metric)
    tf.gfile.MakeDirs(best_metric_dir)
    _add_argument(hparams, "avg_best_" + metric, 0, update=False)
    _add_argument(hparams, "avg_best_" + metric + "_dir", best_metric_dir)

  return hparams


def create_or_load_hparams(default_hparams, hparams_path):
  """Create hparams or load hparams from output_dir."""
  hparams = utils.maybe_parse_standard_hparams(default_hparams, hparams_path)
  hparams = extend_hparams(hparams)
  # Print HParams
  utils.print_hparams(hparams)
  return hparams


def run_main(flags, default_hparams, estimator_fn):
  """Run main."""
  # Job
  jobid = flags.jobid
  utils.print_out("# Job id %d" % jobid)

  # Random
  random_seed = flags.random_seed
  if random_seed is not None and random_seed > 0:
    utils.print_out("# Set random seed to %d" % random_seed)
    random.seed(random_seed + jobid)
    np.random.seed(random_seed + jobid)
    tf.set_random_seed(random_seed)

  # Model output directory
  output_dir = flags.output_dir
  if output_dir and not tf.gfile.Exists(output_dir):
    utils.print_out("# Creating output directory %s ..." % output_dir)
    tf.gfile.MakeDirs(output_dir)

  # Load hparams.
  hparams = create_or_load_hparams(default_hparams, flags.hparams_path)

  # Train or Evaluation
  estimator_fn(hparams)
  return hparams


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.use_fp16 and FLAGS.use_dist_strategy:
    raise ValueError("use_fp16 and use_dist_strategy aren't compatible")

  # Set up hacky envvars.
  # Hack that affects Defun in attention_wrapper.py
  active_xla_option_nums = np.sum([FLAGS.use_xla, FLAGS.use_autojit_xla,
                                   FLAGS.use_xla_compile])
  if active_xla_option_nums > 1:
    raise ValueError(
        "Only one of use_xla, use_xla_compile, use_autojit_xla can be set")

  os.environ["use_xla"] = str(FLAGS.use_xla).lower()
  if FLAGS.use_xla:
    os.environ["use_defun"] = str(True).lower()
  else:
    os.environ["use_defun"] = str(FLAGS.use_defun).lower()
  utils.print_out("use_defun is %s for attention" % os.environ["use_defun"])

  # TODO(jamesqin): retire this config after Cuda9.1
  os.environ["use_fp32_batch_matmul"] = ("true" if FLAGS.use_fp32_batch_matmul
                                         else "false")
  os.environ["use_xla_compile"] = "true" if FLAGS.use_xla_compile else "false"
  os.environ["force_inputs_padding"] = (
      "true" if FLAGS.force_inputs_padding else "false")

  if FLAGS.mode == "train":
    utils.print_out("Running training mode.")
    FLAGS.num_buckets = 5
    default_hparams = create_hparams(FLAGS)
    run_main(FLAGS, default_hparams, estimator.train_fn)
  elif FLAGS.mode == "infer":
    utils.print_out("Running inference mode.")
    # Random
    random_seed = FLAGS.random_seed
    if random_seed is not None and random_seed > 0:
      utils.print_out("# Set random seed to %d" % random_seed)
      random.seed(random_seed)
      np.random.seed(random_seed)
      tf.set_random_seed(random_seed)

    # Model output directory
    output_dir = FLAGS.output_dir
    if output_dir and not tf.gfile.Exists(output_dir):
      utils.print_out("# Creating output directory %s ..." % output_dir)
      tf.gfile.MakeDirs(output_dir)

    # Load hparams.
    default_hparams = create_hparams(FLAGS)
    default_hparams.num_buckets = 1
    # The estimator model_fn is written in a way allowing train hparams to be
    # passed in infer mode.
    hparams = create_or_load_hparams(default_hparams, FLAGS.hparams_path)
    utils.print_out("infer_hparams:")
    utils.print_hparams(hparams)

    # Run evaluation when there's a new checkpoint
    for i, ckpt in enumerate(
        evaluation_utils.get_all_checkpoints(FLAGS.output_dir)):
      tf.logging.info("Starting to evaluate...")
      eval_start = time.time()
      bleu_score = estimator.eval_fn(hparams, ckpt)
      eval_end = time.time()
      utils.print_out("eval time for %d th ckpt: %.2f mins" %
                      (i, (eval_end - eval_start) / 60.), f=sys.stderr)
  else:
    assert FLAGS.mode == "train_and_eval"
    utils.print_out("Running train and eval mode.")

    # Random
    random_seed = FLAGS.random_seed
    if random_seed is not None and random_seed > 0:
      utils.print_out("# Set random seed to %d" % random_seed)
      random.seed(random_seed)
      np.random.seed(random_seed)
      tf.set_random_seed(random_seed)

    # Model output directory
    output_dir = FLAGS.output_dir
    if output_dir and not tf.gfile.Exists(output_dir):
      utils.print_out("# Creating output directory %s ..." % output_dir)
      tf.gfile.MakeDirs(output_dir)

    # Load hparams.
    default_hparams = create_hparams(FLAGS)

    default_hparams.num_buckets = 5
    hparams = create_or_load_hparams(default_hparams, FLAGS.hparams_path)
    utils.print_out("training hparams:")
    utils.print_hparams(hparams)
    with tf.gfile.GFile(os.path.join(output_dir, "train_hparams.txt"), "w") as f:
      f.write(utils.serialize_hparams(hparams) + "\n")

    # The estimator model_fn is written in a way allowing train hparams to be
    # passed in infer mode.
    infer_hparams = tf.contrib.training.HParams(**hparams.values())
    infer_hparams.num_buckets = 1
    utils.print_out("infer_hparams:")
    utils.print_hparams(infer_hparams)
    with tf.gfile.GFile(os.path.join(output_dir, "infer_hparams.txt"), "w") as f:
      f.write(utils.serialize_hparams(infer_hparams) + "\n")

    epochs = 0
    should_stop = epochs >= FLAGS.max_train_epochs

    mlperf_log.gnmt_print(key=mlperf_log.TRAIN_LOOP)
    mlperf_log.gnmt_print(key=mlperf_log.EVAL_TARGET, value=hparams.target_bleu)

    while not should_stop:
      utils.print_out("Starting epoch %d" % epochs)
      mlperf_log.gnmt_print(key=mlperf_log.TRAIN_EPOCH, value=epochs)

      mlperf_log.gnmt_print(
          key=mlperf_log.INPUT_SIZE,
          value=iterator_utils.get_effective_train_epoch_size(hparams))
      mlperf_log.gnmt_print(
          key=mlperf_log.TRAIN_CHECKPOINT, value=("Under " + hparams.output_dir))
      try:
        train_start = time.time()
        estimator.train_fn(hparams)
      except tf.errors.OutOfRangeError:
        utils.print_out("training hits OutOfRangeError", f=sys.stderr)

      train_end = time.time()
      utils.print_out("training time for epoch %d: %.2f mins" %
                      (epochs, (train_end - train_start) / 60.), f=sys.stderr)

      # This is probably sub-optimal, doing eval per-epoch
      mlperf_log.gnmt_print(key=mlperf_log.EVAL_START)
      eval_start = time.time()
      bleu_score = estimator.eval_fn(infer_hparams)
      eval_end = time.time()
      utils.print_out("eval time for epoch %d: %.2f mins" %
                      (epochs, (eval_end - eval_start) / 60.), f=sys.stderr)
      mlperf_log.gnmt_print(key=mlperf_log.EVAL_ACCURACY,
                            value={"epoch": epochs, "value": bleu_score})
      mlperf_log.gnmt_print(key=mlperf_log.EVAL_STOP, value=epochs)

      if FLAGS.debug or bleu_score > FLAGS.target_bleu:
        should_stop = True
        utils.print_out(
            "Stop job since target bleu is reached at epoch %d ." % epochs,
            f=sys.stderr)
        mlperf_log.gnmt_print(mlperf_log.RUN_STOP, {"success": True})

      if epochs >= FLAGS.max_train_epochs:
        should_stop = True
        utils.print_out("Stop job since max_train_epochs is reached.",
                        f=sys.stderr)
        mlperf_log.gnmt_print(mlperf_log.RUN_STOP, {"success": False})
      epochs += 1

  mlperf_log.gnmt_print(key=mlperf_log.RUN_FINAL)

if __name__ == "__main__":
  nmt_parser = argparse.ArgumentParser()
  add_arguments(nmt_parser)
  FLAGS, unparsed = nmt_parser.parse_known_args()
  mlperf_log.gnmt_print(key=mlperf_log.RUN_START)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
