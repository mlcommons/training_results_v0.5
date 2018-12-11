#!/bin/bash

## DL params
MAX_TOKENS=2560
LEARNING_RATE="2.6e-3"
WARMUP_UPDATES=290
EXTRA_PARAMS="--enable-parallel-backward-allred-opt --parallel-backward-allred-opt-threshold 94428979"

## System run parms
DGXNNODES=32
DGXSYSTEM=DGX2_multi
WALLTIME=12:00:00

## System config params
DGXNGPU=8
NVIDIA_VISIBLE_DEVICES="0,2,4,6,8,10,12,14"  # 0,2,4,6,...
DGXSOCKETCORES=24
DGXNSOCKET=2
DGXHT=2        # HT is on is 2, HT off is 1
DGXIBDEVICES='--device=/dev/infiniband/ --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/ucm10 --device=/dev/infiniband/ucm9 --device=/dev/infiniband/ucm8 --device=/dev/infiniband/ucm7 --device=/dev/infiniband/ucm4 --device=/dev/infiniband/ucm3 --device=/dev/infiniband/ucm2 --device=/dev/infiniband/ucm1 --device=/dev/infiniband/uverbs10 --device=/dev/infiniband/uverbs9 --device=/dev/infiniband/uverbs8 --device=/dev/infiniband/uverbs7 --device=/dev/infiniband/uverbs4 --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/issm10 --device=/dev/infiniband/umad10 --device=/dev/infiniband/issm9 --device=/dev/infiniband/umad8 --device=/dev/infiniband/issm8 --device=/dev/infiniband/umad8 --device=/dev/infiniband/issm7 --device=/dev/infiniband/umad7 --device=/dev/infiniband/issm4 --device=/dev/infiniband/umad4 --device=/dev/infiniband/issm3 --device=/dev/infiniband/umad3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/umad2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/umad1'
