#!/bin/bash

set -e


PYTHONPATH=""


export PYTHONPATH="$(pwd)/tpu/models/official/retinanet:${PYTHONPATH}"
echo 'Retinanet contains:'
echo $(pwd)/tpu/models/official/retinanet
ls -lah $(pwd)/tpu/models/official/retinanet
echo
echo
cd staging/models/rough/ssd/

sudo pip install $MLP_TF_PIP_LINE


# start timing 
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)

python3 ssd_main.py  --use_tpu=True \
                     --tpu_name=${MLP_TPU_NAME} \
                     --device=tpu \
                     --num_shards=8 \
                     --mode=train_and_eval \
                     --train_and_eval \
                     --train_batch_size=1024 \
                     --training_file_pattern="${MLP_PATH_GCS_SSD}/train-*" \
                     --eval_batch_size=1000 \
                     --validation_file_pattern="${MLP_PATH_GCS_SSD}/val-*" \
                     --val_json_file="${MLP_PATH_GCS_SSD}/raw-data/annotations/instances_val2017.json" \
                     --resnet_checkpoint=${MLP_GCS_RESNET_CHECKPOINT} \
                     --model_dir=${MLP_GCS_MODEL_DIR} \
                     --num_epochs=64 \
                     --hparams=use_bfloat16=true,lr_warmup_epoch=3.0,base_learning_rate=2.5e-3 \
                     --iterations_per_loop=625 \
                     --transpose_tpu_infeed=true \
                     --nouse_async_checkpoint


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"


# report result 
result=$(( $end - $start )) 
result_name="ssd"


echo "RESULT,$result_name,0,$result,$USER,$start_fmt"

