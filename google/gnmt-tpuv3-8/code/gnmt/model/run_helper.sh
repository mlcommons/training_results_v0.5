#!/bin/bash

set -e


cd staging/models/rough/nmt/

#source /tmp/nmt_env/bin/activate
pip3 install $MLP_TF_PIP_LINE

# start timing 
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)


echo Data Dir $MLP_PATH_GCS_NMT
CMD="python3 nmt.py \
  --data_dir=$MLP_PATH_GCS_NMT \
  --tpu_name=$MLP_TPU_NAME \
  --out_dir=$MLP_GCS_MODEL_DIR \
  \
--activation_dtype=bfloat16 \
--batch_size=2048 \
--mode=train_and_eval \
--skip_host_call=true \
--learning_rate=0.002 \
--max_train_epochs=2 \
--warmup_steps=200 \
--decay_scheme=luong234 \
"

$CMD


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"


# report result 
result=$(( $end - $start )) 
result_name="nmt"


echo "RESULT,$result_name,0,$result,$USER,$start_fmt"

