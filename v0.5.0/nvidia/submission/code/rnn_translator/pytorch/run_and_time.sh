#!/bin/bash

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
DATASET_DIR='/data'
RESULTS_DIR='gnmt_wmt16'
BATCH=${BATCH:-32}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-128}
LR=${LR:-"1.75e-3"}
TARGET=21.80
WARMUP_ITERS=${WARMUP_ITERS:-100}
REMAIN_STEPS=${REMAIN_STEPS:-1450}
DECAY_STEPS=${DECAY_STEPS:-40}

echo "running benchmark"

# run training
python -m torch.distributed.launch --nproc_per_node $SLURM_NTASKS_PER_NODE $MULTI_NODE train.py \
  --save ${RESULTS_DIR} \
  --dataset-dir ${DATASET_DIR} \
  --target-bleu $TARGET \
  --epochs 20 \
  --math fp16 \
  --print-freq 10 \
  --batch-size $BATCH \
  --test-batch-size $TEST_BATCH_SIZE \
  --model-config "{'num_layers': 4, 'hidden_size': 1024, 'dropout':0.2, 'share_embedding': True}" \
  --optimization-config "{'optimizer': 'FusedAdam', 'lr': $LR}" \
  --scheduler-config "{'lr_method':'mlperf', 'warmup_iters':$WARMUP_ITERS, 'remain_steps':$REMAIN_STEPS, 'decay_steps':$DECAY_STEPS}" ; ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="RNN_TRANSLATOR"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"

