#!/bin/bash

## DL params
EXTRA_PARAMS="--min_bbox_map 0.377 --min_mask_map 0.339"
EXTRA_CONFIG=(
               "SOLVER.BASE_LR"       "0.04"
               "SOLVER.MAX_ITER"      "80000"
               "SOLVER.WARMUP_FACTOR" "0.000064"
               "SOLVER.WARMUP_ITERS"  "625"
               "SOLVER.WARMUP_METHOD" "mlperf_linear"
               "SOLVER.STEPS"         "(36000, 48000)"
               "DATALOADER.IMAGES_PER_BATCH_TRAIN"  "4"
               "MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN" "4000"
             )

## System run parms
DGXNNODES=1
DGXSYSTEM=DGX1
WALLTIME=12:00:00

## System config params
DGXNGPU=8
DGXSOCKETCORES=20
DGXHT=2         # HT is on is 2, HT off is 1
DGXIBDEVICES=''
