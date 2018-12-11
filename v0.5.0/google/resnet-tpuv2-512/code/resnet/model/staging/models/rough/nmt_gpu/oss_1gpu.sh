#!/bin/bash

# Enter the parent dir of nmt
if [ "$#" -lt 3 ]; then
  echo "Usage: oss_run.sh <nmt_dir> <data_dir> <out_dir>"
  exit
fi


NMT_DIR=$1
DATA_DIR=$2
OUT_DIR=$3
BATCH_SIZE=128
NUM_GPUS=1

BLEU_PATH=${3:-${HOME}/.local/bin}
export PATH=${PATH}:${BLEU_PATH}

echo "Install pip3 ..."
apt-get install --assume-yes python3-pip

# Install sacrebleu if not already installed.
if ! which sacrebleu > /dev/null; then
  echo "sacrebleu not found, installing..."
  pip3 install sacrebleu
  echo "Done installing sacrebleu"
else
  echo "sacrebleu already installed"
fi

cd ${NMT_DIR}
echo "In ${NMT_DIR}"

echo "Starting nmt task ..."

rm -rf ${OUT_DIR} && python3 -m nmt.nmt \
  --data_dir=${DATA_DIR} \
  --out_dir=${OUT_DIR} \
  --batch_size=${BATCH_SIZE} \
  --num_gpus=${NUM_GPUS} \
  --use_fp32_batch_matmul=true &>/tmp/log
