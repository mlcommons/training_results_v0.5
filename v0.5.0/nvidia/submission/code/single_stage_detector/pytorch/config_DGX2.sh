#!/bin/bash

## DL params
EXTRA_PARAMS=(
               --batch-size      "128"
               --warmup          "900"
               --num-workers     "3"
               --nhwc
               --pad-input
             )

## System run parms
DGXNNODES=1
DGXSYSTEM=DGX2
WALLTIME=12:00:00

## System config params
DGXNGPU=16
DGXSOCKETCORES=24
DGXNSOCKET=2
DGXHT=2         # HT is on is 2, HT off is 1
DGXIBDEVICES=''
