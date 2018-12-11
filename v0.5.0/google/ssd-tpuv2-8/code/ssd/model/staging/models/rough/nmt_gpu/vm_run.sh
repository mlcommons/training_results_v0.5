#!/bin/bash

# Run this on the vm to start job.

MLPERF_HOME=${HOME}/mlperf
DATA_DIR=/data/

# mount
echo "Mount disk to ${DATA_DIR} ..."
if [ ! -d "${DATA_DIR}" ]; then
  sudo mkdir ${DATA_DIR}
fi

sudo chmod a+r ${DATA_DIR}
sudo mount -o ro,noload /dev/sdb /data

nvidia-docker run -it -v $HOME:$HOME -v /data:/data -v /tmp:/tmp \
    tensorflow/tensorflow:nightly-gpu-py3 bash ${MLPERF_HOME}/nmt/oss_8gpu.sh \
    ${MLPERF_HOME} ${DATA_DIR} ${HOME}/nmt_model

