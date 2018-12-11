#!/bin/bash

## System run parms
DGXNNODES=32
DGXSYSTEM=DGX1_multi
WALLTIME=12:00:00

## DL params
LR="1.75e-3"
BATCH=32
TEST_BATCH_SIZE=16
WARMUP_ITERS=100
REMAIN_STEPS=1450
DECAY_STEPS=40

## System config params
DGXNGPU=8
DGXSOCKETCORES=20
DGXHT=2 	# HT is on is 2, HT off is 1
DGXIBDEVICES='--device=/dev/infiniband --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/ucm3 --device=/dev/infiniband/ucm2 --device=/dev/infiniband/ucm1 --device=/dev/infiniband/ucm0 --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/issm3 --device=/dev/infiniband/umad3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/umad2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/umad1 --device=/dev/infiniband/issm0 --device=/dev/infiniband/umad0'
