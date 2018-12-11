#!/bin/bash

## DL params
EXTRA_PARAMS="--min_bbox_map 0.377 --min_mask_map 0.339"
EXTRA_CONFIG=(
               "SOLVER.BASE_LR"       "0.08"
               "SOLVER.MAX_ITER"      "42000"
               "SOLVER.WARMUP_FACTOR" "0.000128"
               "SOLVER.WARMUP_ITERS"  "625"
               "SOLVER.WARMUP_METHOD" "mlperf_linear"
               "SOLVER.STEPS"         "(18000, 24000)"
               "DATALOADER.IMAGES_PER_BATCH_TRAIN"  "4"
               "MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN" "4000"
             )

## System run parms
DGXNNODES=1
DGXSYSTEM=DGX2
WALLTIME=12:00:00

## System config params
DGXNGPU=16
DGXSOCKETCORES=24
DGXHT=2         # HT is on is 2, HT off is 1
DGXIBDEVICES=''
