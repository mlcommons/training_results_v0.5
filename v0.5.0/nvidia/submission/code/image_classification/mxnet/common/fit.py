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

""" example train fit utility """
import logging
import os
import time
import re
import math
import mxnet as mx

#### imports needed for fit monkeypatch
from mxnet.initializer import Uniform
from mxnet.context import cpu
from mxnet.model import BatchEndParam
from mxnet.initializer import Uniform
from mxnet.io import DataDesc, DataIter, DataBatch
from mxnet.base import _as_list
import copy
#####

from mlperf_compliance import mlperf_log
from mlperf_log_utils import mx_resnet_print, all_reduce

def get_epoch_size(args, kv):
    return math.ceil(int(args.num_examples / kv.num_workers) / args.batch_size)

def _get_gpu(gpus):
    idx = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    gpu = gpus.split(",")[idx]
    return gpu

def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = get_epoch_size(args, kv)
    begin_epoch = 0
    if 'pow' in args.lr_step_epochs:
        lr = args.lr
        max_up = args.num_epochs * epoch_size
        pwr = float(re.sub('pow[- ]*', '', args.lr_step_epochs))
        poly_sched = mx.lr_scheduler.PolyScheduler(max_up, lr, pwr)
        return (lr, poly_sched)
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d',
                     lr, begin_epoch)

    steps = [epoch_size * (x - begin_epoch)
             for x in step_epochs if x - begin_epoch > 0]
    if steps:
        if kv:
            num_workers = kv.num_workers
        else:
            num_workers = 1
        epoch_size = math.ceil(int(args.num_examples/num_workers)/args.batch_size)
        return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor,
                                                         base_lr=args.lr, warmup_steps=epoch_size * args.warmup_epochs,
                                                         warmup_mode=args.warmup_strategy))
    else:
        return (lr, None)


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str,
                       help='the neural network to use')
    train.add_argument('--num-layers', type=int,
                       help='number of layers in the neural network, \
                             required by some networks such as resnet')
    train.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--num-epochs', type=int, default=100,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str,
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--initializer', type=str, default='default',
                       help='the initializer type')
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd')
    train.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=128,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='model prefix')
    train.add_argument('--save-period', type=int, default=1, help='params saving period')
    train.add_argument('--eval-period', type=int, default=1, help='evaluation every N epochs')
    train.add_argument('--eval-offset', type=int, default=0, help='first evaluation on epoch N')

    train.add_argument('--top-k', type=int, default=0,
                       help='report the top-k accuracy. 0 means no report.')
    train.add_argument('--dtype', type=str, default='float32',
                       help='precision: float32 or float16')
    # additional parameters for large batch sgd
    train.add_argument('--macrobatch-size', type=int, default=0,
                       help='distributed effective batch size')
    train.add_argument('--warmup-epochs', type=int, default=5,
                       help='the epochs to ramp-up lr to scaled large-batch value')
    train.add_argument('--warmup-strategy', type=str, default='linear',
                       help='the ramping-up strategy for large batch sgd')
    train.add_argument('--logging-dir', type=str, default='logs')
    train.add_argument('--log', type=str, default='')
    train.add_argument('--bn-gamma-init0', action='store_true')
    train.add_argument('--epoch-size',type=int, default=0,
                       help='set number of batches in an epoch. useful for debugging')
    train.add_argument('--profile-worker-suffix', type=str, default='',
                       help='profile workers actions into this file. During distributed training\
                             filename saved will be rank1_ followed by this suffix')
    train.add_argument('--profile-server-suffix', type=str, default='',
                       help='profile server actions into a file with name like rank1_ followed by this suffix \
                             during distributed training')
    train.add_argument('--accuracy-threshold', default=1.0, type=float,
                       help='stop training after top1 reaches this value')
    return train


class CorrectCount(mx.metric.Accuracy):
    def __init__(self, axis=1, name='correct-count',
                 output_names=None, label_names=None):
        super(CorrectCount, self).__init__(
                name=name, axis=axis,
                output_names=output_names, label_names=label_names)
        self.axis = axis

    def get(self):
        return (self.name, self.sum_metric)


class TotalCount(mx.metric.Accuracy):
    def __init__(self, axis=1, name='total-count',
                 output_names=None, label_names=None):
        super(TotalCount, self).__init__(
                name=name, axis=axis,
                output_names=output_names, label_names=label_names)
        self.axis = axis

    def get(self):
        return (self.name, self.num_inst)

def mlperf_fit(self, train_data, eval_data=None, eval_metric='acc',
               epoch_end_callback=None, batch_end_callback=None, kvstore='local',
               optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
               eval_end_callback=None,
               eval_batch_end_callback=None, initializer=Uniform(0.01),
               arg_params=None, aux_params=None, allow_missing=False,
               force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
               validation_metric=None, monitor=None, sparse_row_id_fn=None,
               eval_offset=0, eval_period=1,
               accuracy_threshold=1.0):

    assert num_epoch is not None, 'please specify number of epochs'

    self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
              for_training=True, force_rebind=force_rebind)

    if monitor is not None:
        self.install_monitor(monitor)

    self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                     allow_missing=allow_missing, force_init=force_init)
    self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                        optimizer_params=optimizer_params)

    if validation_metric is None:
        validation_metric = eval_metric
    ###########################################################################
    # Adding Correct and Total Count metrics
    ###########################################################################
    if not isinstance(validation_metric, list):
        validation_metric = [validation_metric]

    validation_metric = mx.metric.create(validation_metric)

    if not isinstance(validation_metric, mx.metric.CompositeEvalMetric):
        vm = mx.metric.CompositeEvalMetric()
        vm.append(validation_metric)
        validation_metric = vm

    for m in [CorrectCount(), TotalCount()]:
        validation_metric.metrics.append(m)
    ###########################################################################

    if not isinstance(eval_metric, mx.metric.EvalMetric):
        eval_metric = mx.metric.create(eval_metric)

    mx_resnet_print(key=mlperf_log.TRAIN_LOOP)
    ################################################################################
    # training loop
    ################################################################################
    for epoch in range(begin_epoch, num_epoch):
        mx_resnet_print(key=mlperf_log.TRAIN_EPOCH, val=epoch)
        tic = time.time()
        eval_metric.reset()
        nbatch = 0
        data_iter = iter(train_data)
        end_of_batch = False
        next_data_batch = next(data_iter)
        while not end_of_batch:
            data_batch = next_data_batch
            if monitor is not None:
                monitor.tic()
            self.forward_backward(data_batch)
            self.update()

            if isinstance(data_batch, list):
                self.update_metric(eval_metric,
                                   [db.label for db in data_batch],
                                   pre_sliced=True)
            else:
                self.update_metric(eval_metric, data_batch.label)

            try:
                # pre fetch next batch
                next_data_batch = next(data_iter)
                self.prepare(next_data_batch, sparse_row_id_fn=sparse_row_id_fn)
            except StopIteration:
                end_of_batch = True

            if monitor is not None:
                monitor.toc_print()

            if batch_end_callback is not None:
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                for callback in _as_list(batch_end_callback):
                    callback(batch_end_params)
            nbatch += 1

        # one epoch of training is finished
        toc = time.time()
        if kvstore:
            if kvstore.rank == 0:
                self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))
        else:
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

        # sync aux params across devices
        arg_params, aux_params = self.get_params()
        self.set_params(arg_params, aux_params)

        if epoch_end_callback is not None:
            for callback in _as_list(epoch_end_callback):
                callback(epoch, self.symbol, arg_params, aux_params)

        #----------------------------------------
        # evaluation on validation set
        if eval_data and epoch % eval_period == eval_offset:
            mx_resnet_print(key=mlperf_log.EVAL_START)
            res = self.score(eval_data, validation_metric,
                             score_end_callback=eval_end_callback,
                             batch_end_callback=eval_batch_end_callback, epoch=epoch)
            #TODO: pull this into default
            if kvstore:
                if kvstore.rank == 0:
                    for name, val in res:
                        self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
            else:
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
            res = dict(res)

            acc = [res['correct-count'], res['total-count']]
            acc = all_reduce(acc)
            acc = acc[0]/acc[1]
            mx_resnet_print(key=mlperf_log.EVAL_ACCURACY, 
							val={"epoch": epoch, "value": acc})
            mx_resnet_print(key=mlperf_log.EVAL_STOP)
            if acc > accuracy_threshold:
                return epoch

        # end of 1 epoch, reset the data-iter for another epoch
        train_data.reset()

    return num_epoch

def fit(args, kv, network, data_loader, **kwargs):
    """
    train a model
    args : argparse returns
    network : the symbol definition of the nerual network
    data_loader : function that returns the train and val data iterators
    """
    if args.profile_server_suffix:
        mx.profiler.set_config(filename=args.profile_server_suffix, profile_all=True, profile_process='server')
        mx.profiler.set_state(state='run', profile_process='server')

    if args.profile_worker_suffix:
        if kv.num_workers > 1:
            filename = 'rank' + str(kv.rank) + '_' + args.profile_worker_suffix
        else:
            filename = args.profile_worker_suffix
        mx.profiler.set_config(filename=filename, profile_all=True, profile_process='worker')
        mx.profiler.set_state(state='run', profile_process='worker')

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    epoch_size = get_epoch_size(args, kv)

    # data iterators
    (train, val) = data_loader(args, kv)
    if 'dist' in args.kv_store and not 'async' in args.kv_store:
        logging.info('Resizing training data to %d batches per machine', epoch_size)
        # resize train iter to ensure each machine has same number of batches per epoch
        # if not, dist_sync can hang at the end with one machine waiting for other machines
        if not args.use_dali:
            train = mx.io.ResizeIter(train, epoch_size)

    # load model
    if 'arg_params' in kwargs and 'aux_params' in kwargs:
        arg_params = kwargs['arg_params']
        aux_params = kwargs['aux_params']
    else:
        sym, arg_params, aux_params = None, None, None
        if sym is not None:
            assert sym.tojson() == network.tojson()

    # save model
    epoch_end_callbacks = []

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus == "" else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)

    # create model
    model = mx.mod.Module(
        context=devs,
        symbol=network
    )

    optimizer_params = {
        'learning_rate': lr,
        'wd': args.wd,
        'lr_scheduler': lr_scheduler,
        'multi_precision': True}

    mx_resnet_print(
            key=mlperf_log.OPT_NAME,
            val=args.optimizer)

    mx_resnet_print(
            key=mlperf_log.OPT_LR,
            val=lr)

    mx_resnet_print(
            key=mlperf_log.OPT_MOMENTUM,
            val=args.mom)

    mx_resnet_print(
            key=mlperf_log.MODEL_L2_REGULARIZATION,
            val=args.wd)

    ##########################################################################
    # MXNet excludes BN layers from L2 Penalty by default,
    # so this won't be explicitly stated anywhere in the code
    ##########################################################################
    mx_resnet_print(key=mlperf_log.MODEL_EXCLUDE_BN_FROM_L2, val=True)

    
    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd'}
    if args.optimizer in has_momentum:
        optimizer_params['momentum'] = args.mom

    initializer = mx.init.Xavier(
                rnd_type='gaussian', factor_type="in", magnitude=2)

    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create(
            'top_k_accuracy', top_k=args.top_k))

    # callbacks that run after each batch
    batch_end_callbacks = []
    if 'horovod' in args.kv_store:
        # if using horovod, only report on rank 0 with global batch size
        if kv.rank == 0:
            batch_end_callbacks.append(mx.callback.Speedometer(
                kv.num_workers*args.batch_size, args.disp_batches))
    else:
        batch_end_callbacks.append(mx.callback.Speedometer(
            args.batch_size, args.disp_batches))

    mx_resnet_print(key=mlperf_log.EVAL_TARGET, val=args.accuracy_threshold)

    # run
    last_epoch = mlperf_fit(model,
                            train,
                            begin_epoch=0,
                            num_epoch=args.num_epochs,
                            eval_data=val,
                            eval_metric=eval_metrics,
                            kvstore=kv,
                            optimizer=args.optimizer,
                            optimizer_params=optimizer_params,
                            initializer=initializer,
                            arg_params=arg_params,
                            aux_params=aux_params,
                            batch_end_callback=batch_end_callbacks,
                            epoch_end_callback=epoch_end_callbacks, #checkpoint if args.use_dali else ,,
                            allow_missing=True, 
                            eval_offset=args.eval_offset,
                            eval_period=args.eval_period,
                            accuracy_threshold=args.accuracy_threshold)

    mx_resnet_print(key=mlperf_log.RUN_STOP, sync=True)

    if ('horovod' not in args.kv_store) or kv.rank == 0:
        model.save_checkpoint('MLPerf-RN50v15', last_epoch, save_optimizer_states=False)

    # When using horovod, ensure all ops scheduled by the engine complete before exiting
    if 'horovod' in args.kv_store:
        mx.ndarray.waitall()

    if args.profile_server_suffix:
        mx.profiler.set_state(state='run', profile_process='server')
    if args.profile_worker_suffix:
        mx.profiler.set_state(state='run', profile_process='worker')
