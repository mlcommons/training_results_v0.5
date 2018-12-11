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

echo "running benchmark"

DATASET_DIR='/data'
[ ! -f /coco ] && ln -sf ${DATASET_DIR} /coco
export SHARED_DIR='/shared'

# test that we have permissions to create files in SHARED_DIR
# or else training will fail at first eval
TMPFILE=$(mktemp --tmpdir=$SHARED_DIR)
touch $TMPFILE || exit 1
rm -f $TMPFILE

# run training
python -m torch.distributed.launch --nproc_per_node $SLURM_NTASKS_PER_NODE $MULTI_NODE tools/train_net.py \
  --fp16 \
  ${EXTRA_PARAMS} \
  --config-file 'configs/e2e_mask_rcnn_R_50_FPN_1x.yaml' \
  "${EXTRA_CONFIG[@]}" ; ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="OBJECT_DETECTION"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"

