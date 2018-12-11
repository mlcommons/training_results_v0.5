#!/bin/bash

## DL params
EXTRA_PARAMS=( )

## System run parms
DGXNNODES=1
DGXSYSTEM=DGX1
WALLTIME=12:00:00

## System config params
DGXNGPU=8
DGXSOCKETCORES=20
DGXNSOCKET=2
DGXHT=2         # HT is on is 2, HT off is 1
DGXIBDEVICES=''
