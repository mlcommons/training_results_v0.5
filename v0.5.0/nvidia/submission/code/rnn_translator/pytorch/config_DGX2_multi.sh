#!/bin/bash

## System run parms
DGXNNODES=16
DGXSYSTEM=DGX2_multi
WALLTIME=12:00:00

## DL params
LR="1.75e-3"
BATCH=32
TEST_BATCH_SIZE=16
WARMUP_ITERS=100
REMAIN_STEPS=1450
DECAY_STEPS=40

## System config params
DGXNGPU=16
DGXSOCKETCORES=24
DGXHT=2 	# HT is on is 2, HT off is 1
DGXIBDEVICES='--device=/dev/infiniband/ --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/ucm10 --device=/dev/infiniband/ucm9 --device=/dev/infiniband/ucm8 --device=/dev/infiniband/ucm7 --device=/dev/infiniband/ucm4 --device=/dev/infiniband/ucm3 --device=/dev/infiniband/ucm2 --device=/dev/infiniband/ucm1 --device=/dev/infiniband/uverbs10 --device=/dev/infiniband/uverbs9 --device=/dev/infiniband/uverbs8 --device=/dev/infiniband/uverbs7 --device=/dev/infiniband/uverbs4 --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/issm10 --device=/dev/infiniband/umad10 --device=/dev/infiniband/issm9 --device=/dev/infiniband/umad8 --device=/dev/infiniband/issm8 --device=/dev/infiniband/umad8 --device=/dev/infiniband/issm7 --device=/dev/infiniband/umad7 --device=/dev/infiniband/issm4 --device=/dev/infiniband/umad4 --device=/dev/infiniband/issm3 --device=/dev/infiniband/umad3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/umad2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/umad1'
