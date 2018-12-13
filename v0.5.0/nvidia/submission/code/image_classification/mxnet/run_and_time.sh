#!/bin/bash
#
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh
set -x

DGXSYSTEM=${DGXSYSTEM:-"DGX1"}
if [[ -f config_${DGXSYSTEM}.sh ]]; then
  source config_${DGXSYSTEM}.sh
else
  source config_DGX1.sh
  echo "Unknown system, assuming DGX1"
fi

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark


SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$DGXNGPU}
BATCHSIZE=${BATCHSIZE:-1664}
KVSTORE=${KVSTORE:-"device"}
LR=${LR:-"0.6"}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
EVAL_OFFSET=${EVAL_OFFSET:-2}
DALI_PREFETCH_QUEUE=${DALI_PREFETCH_QUEUE:-2}
DALI_NVJPEG_MEMPADDING=${DALI_NVJPEG_MEMPADDING:-64}

DATAROOT="/data"

echo "running benchmark"
export NGPUS=$SLURM_NTASKS_PER_NODE
GPUS=$(seq 0 $(($NGPUS - 1)) | tr "\n" "," | sed 's/,$//')
PARAMS=(
      --gpus               "${GPUS}"
      --batch-size         "${BATCHSIZE}"
      --kv-store           "${KVSTORE}"
      --lr                 "${LR}"
      --lr-step-epochs     "30,60,80"
      --warmup-epochs      "${WARMUP_EPOCHS}"
      --eval-period        "4"
      --eval-offset        "${EVAL_OFFSET}"
      --optimizer          "sgd"
      --network            "resnet-v1b-fl"
      --num-layers         "50"
      --num-epochs         "100"
      --accuracy-threshold "0.749"
      --dtype              "float16"
      --use-dali 
      --disp-batches       "20"
      --image-shape        "4,224,224"
      --fuse-bn-relu       "1"
      --fuse-bn-add-relu   "1"
      --min-random-area    "0.05"
      --max-random-area    "1.0"
      --conv-algo          "1"
      --force-tensor-core  "1"
      --input-layout       "NHWC"
      --conv-layout        "NHWC"
      --batchnorm-layout   "NHWC"
      --pooling-layout     "NHWC"
      --batchnorm-mom      "0.9"
      --batchnorm-eps      "1e-5"
      --data-train         "${DATAROOT}/train.rec"
      --data-train-idx     "${DATAROOT}/train.idx"
      --data-val           "${DATAROOT}/val.rec"
      --data-val-idx       "${DATAROOT}/val.idx"
      --dali-prefetch-queue        "${DALI_PREFETCH_QUEUE}"
      --dali-nvjpeg-memory-padding "${DALI_NVJPEG_MEMPADDING}"
)

if [[ "${KVSTORE}" == "horovod" ]]; then
   DGXSYSTEM=${DGXSYSTEM:-DGX1}
   BIND="./ompi_bind_${DGXSYSTEM/_multi*/}.sh"
fi
${BIND} python train_imagenet.py "${PARAMS[@]}"; ret_code=$?

set +x

sleep 3

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="IMAGE_CLASSIFICATION"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"
