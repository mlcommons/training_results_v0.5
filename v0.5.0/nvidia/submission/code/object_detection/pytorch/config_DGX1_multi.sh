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
DGXNNODES=8
DGXSYSTEM=DGX1_multi
WALLTIME=12:00:00

## System config params
DGXNGPU=8
DGXSOCKETCORES=20
DGXHT=2 	# HT is on is 2, HT off is 1
DGXIBDEVICES='--device=/dev/infiniband --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/ucm3 --device=/dev/infiniband/ucm2 --device=/dev/infiniband/ucm1 --device=/dev/infiniband/ucm0 --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/issm3 --device=/dev/infiniband/umad3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/umad2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/umad1 --device=/dev/infiniband/issm0 --device=/dev/infiniband/umad0'
