#!/bin/bash

## DL params
EXTRA_PARAMS="--min_bbox_map 0.377 --min_mask_map 0.339"
EXTRA_CONFIG=(
               "SOLVER.BASE_LR"       "0.16"
               "SOLVER.MAX_ITER"      "40000"
               "SOLVER.WARMUP_FACTOR" "0.000256"
               "SOLVER.WARMUP_ITERS"  "625"
               "SOLVER.WARMUP_METHOD" "mlperf_linear"
               "SOLVER.STEPS"         "(9000, 12000)"
               "DATALOADER.IMAGES_PER_BATCH_TRAIN"  "2"
               "MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN" "2000"
             )

## System run parms
DGXNNODES=4
DGXSYSTEM=DGX2_multi
WALLTIME=12:00:00

## System config params
DGXNGPU=16
DGXSOCKETCORES=24
DGXHT=2 	# HT is on is 2, HT off is 1
DGXIBDEVICES='--device=/dev/infiniband/ --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/ucm10 --device=/dev/infiniband/ucm9 --device=/dev/infiniband/ucm8 --device=/dev/infiniband/ucm7 --device=/dev/infiniband/ucm4 --device=/dev/infiniband/ucm3 --device=/dev/infiniband/ucm2 --device=/dev/infiniband/ucm1 --device=/dev/infiniband/uverbs10 --device=/dev/infiniband/uverbs9 --device=/dev/infiniband/uverbs8 --device=/dev/infiniband/uverbs7 --device=/dev/infiniband/uverbs4 --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/issm10 --device=/dev/infiniband/umad10 --device=/dev/infiniband/issm9 --device=/dev/infiniband/umad8 --device=/dev/infiniband/issm8 --device=/dev/infiniband/umad8 --device=/dev/infiniband/issm7 --device=/dev/infiniband/umad7 --device=/dev/infiniband/issm4 --device=/dev/infiniband/umad4 --device=/dev/infiniband/issm3 --device=/dev/infiniband/umad3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/umad2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/umad1'
