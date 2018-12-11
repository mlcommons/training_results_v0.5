# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import os
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection, SSDCropping
from PIL import Image
from base_model import Loss
from utils import SSDTransformer
from mlperf_logger import ssd_print, set_seeds
from ssd300 import SSD300
from sampler import GeneralDistributedSampler
from master_params import create_flat_master
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np
import io
import random
import torchvision.transforms as transforms
from mlperf_compliance import mlperf_log

# necessary pytorch imports
import torch.utils.data.distributed
import torch.distributed as dist

# Apex imports
try:
    import apex_C
    from apex.parallel.LARC import LARC
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")

from contextlib import redirect_stdout

from SSD import _C as C

# DALI import
from coco_pipeline import COCOPipeline, DALICOCOIterator



def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='/coco',
                        help='path to test and training data files')
    parser.add_argument('--epochs', '-e', type=int, default=800,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='number of examples for each iteration')
    parser.add_argument('--eval-batch-size', type=int, default=32,
                        help='number of examples for each evaluation iteration')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int, default=random.SystemRandom().randint(0, 2**32 - 1),
                        help='manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float, default=0.212,
                        help='stop training early at threshold')
    parser.add_argument('--iteration', type=int, default=0,
                        help='iteration to start from')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    parser.add_argument('--no-save', action='store_true',
                        help='save model checkpoints')
    parser.add_argument('--evaluation', nargs='*', type=int,
                        default=[120000, 160000, 180000, 200000, 220000, 240000],
                        help='iterations at which to evaluate')
    parser.add_argument('--profile', type=int, default=None)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--warmup-factor', type=int, default=1,
                        help='mlperf rule parameter for controlling warmup curve')
    parser.add_argument('--model-path', type=str, default='./vgg16n.pth')
    parser.add_argument('--backbone', type=str, choices=['vgg16', 'vgg16bn', 'resnet18', 'resnet34', 'resnet50'], default='resnet34')
    parser.add_argument('--use-larc', action='store_true')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-fp16', action='store_true')
    parser.add_argument('--print-interval', type=int, default=20)
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--nhwc', action='store_true')
    parser.add_argument('--pad-input', action='store_true')
    parser.add_argument('--lr', type=float, default=1.25e-3)
    parser.add_argument('--decay1', type=int, default=160)
    parser.add_argument('--decay2', type=int, default=200)
    parser.add_argument('--delay-allreduce', action='store_true')
    # Distributed stuff
    parser.add_argument('--local_rank', default=0, type=int,
			help='Used for multi-process training. Can either be manually set ' +
			'or automatically set by using \'python -m multiproc\'.')

    return parser.parse_args()

# make sure that arguments are all self-consistent
def validate_arguments(args):
    # nhwc can only be used with fp16
    if args.nhwc:
        assert(args.use_fp16)

    # input padding can only be used with NHWC
    if args.pad_input:
        assert(args.nhwc)

    return

def show_memusage(device=0):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))

def print_message(rank, *print_args):
    if rank == 0:
        print(*print_args)

def my_collate(batch):
    # batch is: [image (300x300) Tensor, image_id, (htot, wtot), bboxes (8732, 4) Tensor, labels (8732) Tensor]
    images = []
    image_ids = []
    image_sizes = []
    bboxes = []
    bbox_offsets = [0]
    labels = []

    for item in batch:
        images.append(item[0].view(1, *item[0].shape))
        image_ids.append(item[1])
        image_sizes.append(item[2])
        bboxes.append(item[3])
        labels.append(item[4])

        bbox_offsets.append(bbox_offsets[-1] + item[3].shape[0])

    images = torch.cat(images)
    bbox_offsets = np.array(bbox_offsets).astype(np.int32)
    return [images, torch.tensor(image_ids), image_sizes, torch.cat(bboxes), torch.cat(labels), torch.tensor(bbox_offsets)]

def generate_mean_std(args):
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]

    if args.pad_input:
        mean_val.append(0.)
        std_val.append(1.)
    mean = torch.tensor(mean_val).cuda()
    std = torch.tensor(std_val).cuda()

    if args.nhwc:
        view = [1, 1, 1, len(mean_val)]
    else:
        view = [1, len(mean_val), 1, 1]

    mean = mean.view(*view)
    std = std.view(*view)

    if args.use_fp16:
        mean = mean.half()
        std = std.half()

    return mean, std

def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    ssd_print(key=mlperf_log.FEATURE_SIZES, value=feat_size)
    steps = [8, 16, 32, 64, 100, 300]
    ssd_print(key=mlperf_log.STEPS, value=steps)
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    ssd_print(key=mlperf_log.SCALES, value=scales)
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    ssd_print(key=mlperf_log.ASPECT_RATIOS, value=aspect_ratios)
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    ssd_print(key=mlperf_log.NUM_DEFAULTS,
                         value=len(dboxes.default_boxes))
    return dboxes

def load_checkpoint(model, checkpoint):
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    # remove proceeding 'module' from checkpoint
    saved_model = od["model"]
    for k in list(saved_model.keys()):
        if k.startswith('module.'):
            saved_model[k[7:]] = saved_model.pop(k)
    model.load_state_dict(saved_model)


def coco_eval(model, coco, cocoGt, encoder, inv_map, threshold, epoch, iteration, batch_size,
              use_cuda=True, use_fp16=False, local_rank=-1, N_gpu=1,
              use_nhwc=False, pad_input=False):
    from pycocotools.cocoeval import COCOeval
    print("")

    distributed = False
    if local_rank >= 0:
        distributed = True

    ret = []
    overlap_threshold = 0.50
    nms_max_detections = 200
    ssd_print(key=mlperf_log.NMS_THRESHOLD,
                         value=overlap_threshold)
    ssd_print(key=mlperf_log.NMS_MAX_DETECTIONS,
                         value=nms_max_detections)
    ssd_print(key=mlperf_log.EVAL_START, value=epoch)
    start = time.time()
    for nbatch, (img, img_id, img_size, _, _) in enumerate(coco):
        print("Parsing batch: {}/{}".format(nbatch, len(coco)), end='\r')
        with torch.no_grad():
            inp = img.cuda()

            if pad_input:
                s = inp.shape
                inp = torch.cat([inp, torch.zeros([s[0], 1, s[2], s[3]], device=inp.device)], dim=1)

            if use_nhwc:
                inp = inp.permute(0, 2, 3 ,1).contiguous()
            if use_fp16:
                inp = inp.half()

            # Get predictions
            ploc, plabel = model(inp)
            ploc, plabel = ploc.float(), plabel.float()

            # Handle the batch of predictions produced
            # This is slow, but consistent with old implementation.
            for idx in range(ploc.shape[0]):
                # ease-of-use for specific predictions
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)

                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, overlap_threshold, nms_max_detections)[0]
                except:
                    #raise
                    print("No object detected in idx: {}".format(idx))
                    continue

                htot, wtot = img_size[0][idx].item(), img_size[1][idx].item()
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    ret.append([img_id[idx], loc_[0]*wtot, \
                                        loc_[1]*htot,
                                        (loc_[2] - loc_[0])*wtot,
                                        (loc_[3] - loc_[1])*htot,
                                        prob_,
                                        inv_map[label_]])

    # Now we have all predictions from this rank, gather them all together
    # if necessary
    ret = np.array(ret).astype(np.float32)

    # Multi-GPU eval
    if distributed:
        # NCCL backend means we can only operate on GPU tensors
        ret_copy = torch.tensor(ret).cuda()

        # Everyone exchanges the size of their results
        ret_sizes = [torch.tensor(0).cuda() for _ in range(N_gpu)]
        torch.distributed.all_gather(ret_sizes, torch.tensor(ret_copy.shape[0]).cuda())

        # Get the maximum results size, as all tensors must be the same shape for
        # the all_gather call we need to make
        max_size = 0
        sizes = []
        for s in ret_sizes:
            max_size = max(max_size, s.item())
            sizes.append(s.item())

        # Need to pad my output to max_size in order to use in all_gather
        ret_pad = torch.cat([ret_copy, torch.zeros(max_size-ret_copy.shape[0], 7, dtype=torch.float32).cuda()])

        # allocate storage for results from all other processes
        other_ret = [torch.zeros(max_size, 7, dtype=torch.float32).cuda() for i in range(N_gpu)]
        # Everyone exchanges (padded) results
        torch.distributed.all_gather(other_ret, ret_pad)

        # Now need to reconstruct the _actual_ results from the padded set using slices.
        cat_tensors = []
        for i in range(N_gpu):
            cat_tensors.append(other_ret[i][:sizes[i]][:])

        final_results = torch.cat(cat_tensors).cpu().numpy()
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda')
        # eval size per worker
        eval_tensor = torch.LongTensor([(len(coco)-1) * batch_size + ploc.shape[0]]).to(device)
        torch.distributed.all_reduce(eval_tensor)
        eval_size = eval_tensor.item()
    else:
        # Otherwise full results are just our results
        final_results = ret
        eval_size = (len(coco)-1) * batch_size + ploc.shape[0]

    print_message(local_rank, "Predicting Ended, total time: {:.2f} s".format(time.time()-start))

    cocoDt = cocoGt.loadRes(final_results)

    if local_rank == 0 or local_rank == -1:
        E = COCOeval(cocoGt, cocoDt, iouType='bbox')
        E.evaluate()
        E.accumulate()
        E.summarize()
        print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))
    else:
        # fix for cocoeval indiscriminate prints
        with redirect_stdout(io.StringIO()):
            E = COCOeval(cocoGt, cocoDt, iouType='bbox')
            E.evaluate()
            E.accumulate()
            E.summarize()

    current_accuracy = E.stats[0]

    ssd_print(key=mlperf_log.EVAL_SIZE, value=eval_size)
    ssd_print(key=mlperf_log.EVAL_ACCURACY,
                         value={"epoch": epoch,
                                "value": current_accuracy})
    ssd_print(key=mlperf_log.EVAL_ITERATION_ACCURACY,
                         value={"epoch": epoch,
                                "value": current_accuracy})
    ssd_print(key=mlperf_log.EVAL_TARGET, value=threshold)
    ssd_print(key=mlperf_log.EVAL_STOP, value=epoch)

    return current_accuracy>= threshold #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]

def lr_warmup(optim, warmup_iter, iter_num, epoch, base_lr, args):
    if iter_num < warmup_iter:
        # new_lr = 1. * base_lr / warmup_iter * iter_num

        # mlperf warmup rule
        warmup_step = base_lr / (warmup_iter * (2 ** args.warmup_factor))
        new_lr = base_lr - (warmup_iter - iter_num) * warmup_step
        ssd_print(key=mlperf_log.OPT_LR, value={"epoch": epoch, "iteration": iter_num, "value": new_lr})

        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

def train300_mlperf_coco(args):
    from coco import COCO

    # Check that GPUs are actually available
    use_cuda = not args.no_cuda

    # Setup multi-GPU if necessary
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    local_seed = set_seeds(args)
    # start timing here
    ssd_print(key=mlperf_log.RUN_START)
    if args.distributed:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    # Setup data, defaults
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    input_size = 300
    val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)
    ssd_print(key=mlperf_log.INPUT_SIZE, value=input_size)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")
    train_annotate = os.path.join(args.data, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(args.data, "train2017")

    cocoGt = COCO(annotation_file=val_annotate)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)

    if args.distributed:
        val_sampler = GeneralDistributedSampler(val_coco, pad=False)
    else:
        val_sampler = None


    train_pipe = COCOPipeline(args.batch_size, args.local_rank, train_coco_root,
                    train_annotate, N_gpu, num_threads=args.num_workers,
                    output_fp16=args.use_fp16, output_nhwc=args.nhwc,
                    pad_output=args.pad_input, seed=local_seed - 2**31)
    print_message(args.local_rank, "time_check a: {secs:.9f}".format(secs=time.time()))
    train_pipe.build()
    print_message(args.local_rank, "time_check b: {secs:.9f}".format(secs=time.time()))
    test_run = train_pipe.run()
    train_loader = DALICOCOIterator(train_pipe, 118287 / N_gpu)

    val_dataloader   = DataLoader(val_coco,
                                  batch_size=args.eval_batch_size,
                                  shuffle=False, # Note: distributed sampler is shuffled :(
                                  sampler=val_sampler,
                                  num_workers=args.num_workers)
    ssd_print(key=mlperf_log.INPUT_ORDER)
    ssd_print(key=mlperf_log.INPUT_BATCH_SIZE, value=args.batch_size)
    # Build the model
    ssd300 = SSD300(val_coco.labelnum, backbone=args.backbone, use_nhwc=args.nhwc, pad_input=args.pad_input)
    if args.checkpoint is not None:
        load_checkpoint(ssd300, args.checkpoint)

    ssd300.train()
    ssd300.cuda()
    loss_func = Loss(dboxes)
    loss_func.cuda()

    if args.distributed:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    if args.use_fp16:
        ssd300 = network_to_half(ssd300)

    # Parallelize.  Need to do this after network_to_half.
    if args.distributed:
        if args.delay_allreduce:
            print_message(args.local_rank, "Delaying allreduces to the end of backward()")
        ssd300 = DDP(ssd300,
                     delay_allreduce=args.delay_allreduce,
                     retain_allreduce_buffers=args.use_fp16)

    # Create optimizer.  This must also be done after network_to_half.
    global_batch_size = (N_gpu * args.batch_size)

    # mlperf only allows base_lr scaled by an integer
    base_lr = 1e-3
    requested_lr_multiplier = args.lr / base_lr
    adjusted_multiplier = max(1, round(requested_lr_multiplier * global_batch_size / 32))

    current_lr = base_lr * adjusted_multiplier
    current_momentum = 0.9
    current_weight_decay = 5e-4
    static_loss_scale = 128.
    if args.use_fp16:
        if args.distributed and not args.delay_allreduce:
            # We can't create the flat master params yet, because we need to
            # imitate the flattened bucket structure that DDP produces.
            optimizer_created = False
        else:
            model_buckets = [[p for p in ssd300.parameters() if p.requires_grad
                              and p.type() == "torch.cuda.HalfTensor"],
                              [p for p in ssd300.parameters() if p.requires_grad
                               and p.type() == "torch.cuda.FloatTensor"]]
            flat_master_buckets = create_flat_master(model_buckets)
            optim = torch.optim.SGD(flat_master_buckets, lr=current_lr, momentum=current_momentum,
                                    weight_decay=current_weight_decay)
            optimizer_created = True
    else:
        optim = torch.optim.SGD(ssd300.parameters(), lr=current_lr, momentum=current_momentum,
                                weight_decay=current_weight_decay)
        optimizer_created = True

    # Add LARC if desired
    if args.use_larc:
        optim = LARC(optim)

    ssd_print(key=mlperf_log.OPT_NAME, value="SGD")
    ssd_print(key=mlperf_log.OPT_LR, value=current_lr)
    ssd_print(key=mlperf_log.OPT_MOMENTUM, value=current_momentum)
    ssd_print(key=mlperf_log.OPT_WEIGHT_DECAY,
                         value=current_weight_decay)
    if args.warmup is not None:
        ssd_print(key=mlperf_log.OPT_LR_WARMUP_STEPS,
                  value=args.warmup)

    # Model is completely finished -- need to create separate copies, preserve parameters across
    # them, and jit
    ssd300_eval = SSD300(val_coco.labelnum, backbone=args.backbone, use_nhwc=args.nhwc, pad_input=args.pad_input).cuda()
    if args.use_fp16:
        ssd300_eval = network_to_half(ssd300_eval)

    # Get the existant state from the train model
    # * if we use distributed, then we want .module
    train_model = ssd300.module if args.distributed else ssd300

    ssd300_eval.load_state_dict(train_model.state_dict())

    ssd300_eval.eval()

    if args.jit:
        input_c = 4 if args.pad_input else 3
        example_shape = [args.batch_size, 300, 300, input_c] if args.nhwc else [args.batch_size, input_c, 300, 300]
        example_input = torch.randn(*example_shape).cuda()
        if args.use_fp16:
            example_input = example_input.half()
        # DDP has some Python-side control flow.  If we JIT the entire DDP-wrapped module,
        # the resulting ScriptModule will elide this control flow, resulting in allreduce
        # hooks not being called.  If we're running distributed, we need to extract and JIT
        # the wrapped .module.
        # Replacing a DDP-ed ssd300 with a script_module might also cause the AccumulateGrad hooks
        # to go out of scope, and therefore silently disappear.
        module_to_jit = ssd300.module if args.distributed else ssd300
        if args.distributed:
            ssd300.module = torch.jit.trace(module_to_jit, example_input)
        else:
            ssd300 = torch.jit.trace(module_to_jit, example_input)

    print_message(args.local_rank, "epoch", "nbatch", "loss")
    eval_points = np.array(args.evaluation) * 32 / global_batch_size
    eval_points = list(map(int, list(eval_points)))

    iter_num = args.iteration
    avg_loss = 0.0
    inv_map = {v:k for k,v in val_coco.label_map.items()}

    start_elapsed_time = time.time()
    last_printed_iter = args.iteration
    num_elapsed_samples = 0

    # Generate normalization tensors
    mean, std = generate_mean_std(args)

    def step_maybe_fp16_maybe_distributed(optim):
        if args.use_fp16:
            if args.distributed:
                for flat_master, allreduce_buffer in zip(flat_master_buckets, ssd300.allreduce_buffers):
                    if allreduce_buffer is None:
                        raise RuntimeError("allreduce_buffer is None")
                    flat_master.grad = allreduce_buffer.float()
                    flat_master.grad.data.mul_(1./static_loss_scale)
            else:
                for flat_master, model_bucket in zip(flat_master_buckets, model_buckets):
                    flat_grad = apex_C.flatten([m.grad.data for m in model_bucket])
                    flat_master.grad = flat_grad.float()
                    flat_master.grad.data.mul_(1./static_loss_scale)
        optim.step()
        if args.use_fp16:
            for model_bucket, flat_master in zip(model_buckets, flat_master_buckets):
                for model, master in zip(model_bucket, apex_C.unflatten(flat_master.data,
                                                                        model_bucket)):
                    model.data.copy_(master.data)
    ssd_print(key=mlperf_log.TRAIN_LOOP)
    for epoch in range(args.epochs):
        ssd_print(key=mlperf_log.TRAIN_EPOCH, value=epoch)
        for p in ssd300.parameters():
            p.grad = None
        for i, data in enumerate(train_loader):
            img = data[0][0][0]
            bbox = data[0][1][0]
            label = data[0][2][0]
            label = label.type(torch.cuda.LongTensor)
            bbox_offsets = data[0][3][0]

            # handle random flipping outside of DALI for now
            bbox_offsets = bbox_offsets.cuda()
            img, bbox = C.random_horiz_flip(img, bbox, bbox_offsets, 0.5, args.nhwc)
            img.sub_(mean).div_(std)

            if args.profile is not None and iter_num == args.profile:
                return
            if args.warmup is not None and optimizer_created:
                lr_warmup(optim, args.warmup, iter_num, epoch, current_lr, args)
            if iter_num == ((args.decay1 * 1000 * 32) // global_batch_size):
                print_message(args.local_rank, "lr decay step #1")
                current_lr *= 0.1
                for param_group in optim.param_groups:
                    param_group['lr'] = current_lr
                ssd_print(key=mlperf_log.OPT_LR,
                                     value=current_lr)

            if iter_num == ((args.decay2 * 1000 * 32) // global_batch_size):
                print_message(args.local_rank, "lr decay step #2")
                current_lr *= 0.1
                for param_group in optim.param_groups:
                    param_group['lr'] = current_lr
                ssd_print(key=mlperf_log.OPT_LR,
                                     value=current_lr)

            if use_cuda:
                img = img.cuda()
                # NHWC direct from DALI now if necessary
                bbox = bbox.cuda()
                label = label.cuda()
                bbox_offsets = bbox_offsets.cuda()

            # Now run the batched box encoder
            N = img.shape[0]
            if bbox_offsets[-1].item() == 0:
                print("No labels in batch")
                continue
            bbox, label = C.box_encoder(N, bbox, bbox_offsets, label, encoder.dboxes.cuda(), 0.5)

            # output is ([N*8732, 4], [N*8732], need [N, 8732, 4], [N, 8732] respectively
            M = bbox.shape[0] // N
            bbox = bbox.view(N, M, 4)
            label = label.view(N, M)
            # print(img.shape, bbox.shape, label.shape)

            ploc, plabel = ssd300(img)
            ploc, plabel = ploc.float(), plabel.float()

            trans_bbox = bbox.transpose(1,2).contiguous().cuda()
            label = label.cuda()
            gloc, glabel = Variable(trans_bbox, requires_grad=False), \
                           Variable(label, requires_grad=False)
            loss = loss_func(ploc, plabel, gloc, glabel)

            if not np.isinf(loss.item()): avg_loss = 0.999*avg_loss + 0.001*loss.item()

            num_elapsed_samples += N
            if args.local_rank == 0 and iter_num % args.print_interval == 0:
                end_elapsed_time = time.time()
                elapsed_time = end_elapsed_time - start_elapsed_time

                avg_samples_per_sec = num_elapsed_samples * N_gpu / elapsed_time

                print("Iteration: {:6d}, Loss function: {:5.3f}, Average Loss: {:.3f}, avg. samples / sec: {:.2f}"\
                            .format(iter_num, loss.item(), avg_loss, avg_samples_per_sec), end="\n")

                last_printed_iter = iter_num
                start_elapsed_time = time.time()
                num_elapsed_samples = 0

            # loss scaling
            if args.use_fp16:
                loss = loss*static_loss_scale
            loss.backward()

            if not optimizer_created:
                # Imitate the model bucket structure created by DDP.
                # These will already be split by type (float or half).
                model_buckets = []
                for bucket in ssd300.active_i_buckets:
                    model_buckets.append([])
                    for active_i in bucket:
                        model_buckets[-1].append(ssd300.active_params[active_i])
                flat_master_buckets = create_flat_master(model_buckets)
                optim = torch.optim.SGD(flat_master_buckets, lr=current_lr, momentum=current_momentum,
                                        weight_decay=current_weight_decay)
                optimizer_created = True
                # Skip this first iteration because flattened allreduce buffers are not yet created.
                # step_maybe_fp16_maybe_distributed(optim)
            else:
                step_maybe_fp16_maybe_distributed(optim)

            # Likely a decent skew here, let's take this opportunity to set the gradients to None.
            # After DALI integration, playing with the placement of this is worth trying.
            for p in ssd300.parameters():
                p.grad = None

            if iter_num in eval_points:
                if args.local_rank == 0:
                    if not args.no_save:
                        print("saving model...")
                        torch.save({"model" : ssd300.state_dict(), "label_map": val_coco.label_info},
                                    "./models/iter_{}.pt".format(iter_num))

		# Get the existant state from the train model
		# * if we use distributed, then we want .module
                train_model = ssd300.module if args.distributed else ssd300

                ssd300_eval.load_state_dict(train_model.state_dict())
                if coco_eval(ssd300_eval,
                             val_dataloader,
                             cocoGt,
                             encoder,
                             inv_map,
                             args.threshold,
                             epoch,
                             iter_num,
                             args.eval_batch_size,
                             use_fp16=args.use_fp16,
                             local_rank=args.local_rank if args.distributed else -1,
                             N_gpu=N_gpu,
                             use_nhwc=args.nhwc,
                             pad_input=args.pad_input):
                    return True

            iter_num += 1

        train_loader.reset()
    return False

def main():
    args = parse_args()
    validate_arguments(args)

    if args.local_rank == 0:
        if not os.path.isdir('./models'):
              os.mkdir('./models')

    torch.backends.cudnn.benchmark = True

    success = train300_mlperf_coco(args)

    # end timing here
    ssd_print(key=mlperf_log.RUN_STOP, value={"success": success})
    ssd_print(key=mlperf_log.RUN_FINAL)

if __name__ == "__main__":
    main()
