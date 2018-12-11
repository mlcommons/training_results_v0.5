# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import random

import mxnet as mx
import numpy as np
from mlperf_compliance import mlperf_log

from common import find_mxnet, dali, fit
from mlperf_log_utils import mx_resnet_print

def add_general_args(parser):
    parser.add_argument('--verbose', type=int, default=0,
                        help='turn on reporting of chosen algos for convolution, etc.')
    parser.add_argument('--seed', type=int, default=None,
                        help='set the seed for python, nd and mxnet rngs')
    parser.add_argument('--custom-bn-off', type=int, default=0,
                        help='disable use of custom batchnorm kernel')
    parser.add_argument('--fuse-bn-relu', type=int, default=0,
                        help='have batchnorm kernel perform activation relu')
    parser.add_argument('--fuse-bn-add-relu', type=int, default=0,
                        help='have batchnorm kernel perform add followed by activation relu')
    parser.add_argument('--input-layout', type=str, default='NCHW',
                        help='the layout of the input data (e.g. NCHW)')
    parser.add_argument('--conv-layout', type=str, default='NCHW',
                        help='the layout of the data assumed by the conv operation (e.g. NCHW)')
    parser.add_argument('--conv-algo', type=int, default=-1,
                        help='set the convolution algos (fwd, dgrad, wgrad)')
    parser.add_argument('--force-tensor-core', type=int, default=0,
                        help='require conv algos to be tensor core')
    parser.add_argument('--batchnorm-layout', type=str, default='NCHW',
                        help='the layout of the data assumed by the batchnorm operation (e.g. NCHW)')
    parser.add_argument('--batchnorm-eps', type=float, default=2e-5,
                        help='the amount added to the batchnorm variance to prevent output explosion.')
    parser.add_argument('--batchnorm-mom', type=float, default=0.9,
                        help='the leaky-integrator factor controling the batchnorm mean and variance.')
    parser.add_argument('--pooling-layout', type=str, default='NCHW',
                        help='the layout of the data assumed by the pooling operation (e.g. NCHW)')
    parser.add_argument('--kv-store', type=str, default='device',
                        help='key-value store type')

def _get_gpu(gpus):
    idx = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    gpu = gpus.split(",")[idx]
    return gpu



if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="MLPerf RN50v1.5 training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_general_args(parser)
    fit.add_fit_args(parser)
    dali.add_dali_args(parser)

    parser.set_defaults(
        # network
        network          = 'resnet-v1b',
        num_layers       = 50,

        # data
        resize           = 256,
        num_classes      = 1000,
        num_examples     = 1281167,
        image_shape      = '3,224,224',
        # train
        num_epochs       = 100,
        lr_step_epochs   = '30,60,80',
        dtype            = 'float32'
    )
    args = parser.parse_args()


    # select gpu for horovod process
    if 'horovod' in args.kv_store:
        args.gpus = _get_gpu(args.gpus)

    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)

    mx_resnet_print(key=mlperf_log.EVAL_EPOCH_OFFSET,
                    val=args.eval_offset)

    mx_resnet_print(key=mlperf_log.RUN_START, sync=True)
    if args.seed is None:
        args.seed = int(random.SystemRandom().randint(0, 2**16 - 1))
    
    if 'horovod' in args.kv_store:
        all_seeds = np.random.randint(2**16, size=(int(os.environ['OMPI_COMM_WORLD_SIZE'])))
        args.seed = int(all_seeds[int(os.environ['OMPI_COMM_WORLD_RANK'])])

    mx_resnet_print(key=mlperf_log.RUN_SET_RANDOM_SEED, val=args.seed, uniq=False)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    sym = net.get_symbol(**vars(args))

    fit.fit(args, kv, sym, dali.get_rec_iter)
    mx_resnet_print(key=mlperf_log.RUN_FINAL)
