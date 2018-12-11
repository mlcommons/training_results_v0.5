#!/bin/bash

set -e


cd staging/models/rough/nmt/

#source /tmp/nmt_env/bin/activate
pip install $MLP_TF_PIP_LINE

# start timing 
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)

echo Data Dir $MLP_PATH_GCS_EUW_NMT
CMD="python3 nmt.py \
  --batch_size=4096 \
  --data_dir=$MLP_PATH_GCS_EUW_NMT \
  --tpu_name=$MLP_TPU_NAME \
  --out_dir=$MLP_GCS_EUW_MODEL_DIR \
  --use_tpu=true \
     \
     --activation_dtype=bfloat16   \
     --batch_size=4096  \
     --learning_rate=0.003   \
     --max_train_epochs=2  \
     --warmup_steps=200  \
     --decay_scheme=luong234  \
     --mode=train   \
     --warmup_steps=200  \
     --num_tpu_workers=4 \
     --num_shards=32 \
     --num_shards_per_host=8 \
  "


EVAL_CMD="python3 nmt.py \
	--activation_dtype=bfloat16 \
	  --data_dir=$MLP_PATH_GCS_EUW_NMT \
	 --tpu_name=$MLP_TPU_SIDECAR_NAME \
	  --out_dir=$MLP_GCS_EUW_MODEL_DIR \
	\
     --activation_dtype=bfloat16   \
     --mode=infer   \
     --num_buckets=1   \
     --target_bleu=21.8   \
	"

echo $CMD
echo $EVAL_CMD

timeout 2h $CMD &
timeout 2h $EVAL_CMD

wait
STAT=$?

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"


# report result 
result=$(( $end - $start )) 
result_name="ssd"


echo "RESULT,$result_name,0,$result,$USER,$start_fmt"
exit $STAT
