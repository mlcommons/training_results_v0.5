#!/bin/bash

## DL params
BATCHSIZE="26"
KVSTORE="horovod"
LR="4.1"  # 4.1 is a multiple of 0.1 chosen due to batch size; intentionally different than 4.096 in DGX2_multi
WARMUP_EPOCHS="15"
EVAL_OFFSET="1"
DALI_PREFETCH_QUEUE="3"
DALI_NVJPEG_MEMPADDING="256"

## System run parms
DGXNNODES=80
#DGXNNODES=2
DGXSYSTEM=DGX1_multi
WALLTIME=12:00:00

## System config params
DGXNGPU=8
DGXSOCKETCORES=20
DGXHT=2 	# HT is on is 2, HT off is 1
DGXIBDEVICES='--device=/dev/infiniband --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/ucm3 --device=/dev/infiniband/ucm2 --device=/dev/infiniband/ucm1 --device=/dev/infiniband/ucm0 --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/issm3 --device=/dev/infiniband/umad3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/umad2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/umad1 --device=/dev/infiniband/issm0 --device=/dev/infiniband/umad0'
