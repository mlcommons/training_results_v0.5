#!/bin/bash
set -e
set -o pipefail

MLP_HOST_OUTPUT_DIR=`pwd`/output
mkdir -p $MLP_HOST_OUTPUT_DIR

sudo nvidia-docker build . -t foo
sudo nvidia-docker run -v $MLP_HOST_DATA_DIR:/data \
-v $MLP_HOST_OUTPUT_DIR:/output -v /proc:/host_proc \
-t foo:latest /root/run_helper_8xV100.sh 2>&1 | tee output.txt
