#!/bin/bash

#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DGXSYSTEM=${DGXSYSTEM:-"DGX1"}
if [[ -f config_${DGXSYSTEM}.sh ]]; then
  source config_${DGXSYSTEM}.sh
else
  source config_DGX1.sh
  echo "Unknown system, assuming DGX1"
fi
SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$DGXNGPU}
SLURM_JOB_ID=${SLURM_JOB_ID:-$RANDOM}
MULTI_NODE=${MULTI_NODE:-''}
echo "Run vars: id $SLURM_JOB_ID gpus $SLURM_NTASKS_PER_NODE mparams $MULTI_NODE"


# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
set -x

echo "running benchmark"

DATASET_DIR='/data/ml-20m'
SEED=${SEED:-${RANDOM}}

# run training
python -m bind_launch --nsockets_per_node ${DGXNSOCKET} \
                      --ncores_per_socket ${DGXSOCKETCORES} \
                      --nproc_per_node=$SLURM_NTASKS_PER_NODE \
 ncf.py $DATASET_DIR \
  -l  "0.0045" \
  -b  "1048576" \
  -b1 "0.25" \
  -b2 "0.5" \
  --eps "1e-8" \
  --valid-batch-size 1048576 \
  --loss-scale 8192 \
  --layers 256 256 128 64 -f 64 \
  --seed ${SEED} \
  --threshold 0.635 \
  --fp16 \
  "${EXTRA_PARAMS[@]}" ; ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="RECOMMENDATION"

echo "RESULT,$result_name,$SEED,$result,nvidia,$start_fmt"
