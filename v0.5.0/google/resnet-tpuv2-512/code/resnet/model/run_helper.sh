
set -e

export PYTHONPATH=`pwd`/models:$PYTHONPATH
export PYTHONPATH=`pwd`staging/models/rough/:$PYTHONPATH

# start timing 
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)


echo "STARTING TIMING RUN AT $start_fmt"
#python3 tpu/models/official/resnet/resnet_main.py --tpu=$MLP_TPU_NAME --data_dir=$MLP_PATH_GCS_IMAGENET --model_dir=${MLP_GCS_MODEL_DIR} --train_batch_size=1024 --iterations_per_loop=112603 --mode=train --eval_batch_size=1000 --tpu_zone=us-central1-b --num_cores=8 --train_steps=112603
# Decreased iters per look to debug
#python3 tpu/models/official/resnet/resnet_main.py --tpu=$MLP_TPU_NAME --data_dir=$MLP_PATH_GCS_IMAGENET --model_dir=${MLP_GCS_MODEL_DIR} --train_batch_size=1024 --iterations_per_loop=10000 --mode=train --eval_batch_size=1000 --tpu_zone=us-central1-b --num_cores=8 --train_steps=112603


cd staging/models/rough/

EVAL_CMD="python3 resnet/resnet_main.py \
	--data_dir=$MLP_PATH_GCS_EUW_IMAGENET \
	--eval_batch_size=1024 \
	--iterations_per_loop=312 \
	--mode=eval \
	--model_dir=${MLP_GCS_EUW_MODEL_DIR} \
	--num_cores=8 \
	--resnet_depth=50 \
	--steps_per_eval=312 \
	--tpu=$MLP_TPU_SIDECAR_NAME \
	--train_batch_size=16384 \
	--train_steps=7038 "

# The training job emits a checkpoint every 4 epochs, which is determined by
# --steps_per_eval, --train_batch_size, and --num_train_images.
python -c "import mlperf_compliance;mlperf_compliance.mlperf_log.resnet_print(key=mlperf_compliance.mlperf_log.EVAL_EPOCH_OFFSET, value=3)"
echo 'WARNING: Using hard coded fulltime_pod'
timeout 1h python3 resnet/resnet_main.py \
	--data_dir=$MLP_PATH_GCS_EUW_IMAGENET \
	--eval_batch_size=1024 \
	--iterations_per_loop=312 \
	--mode=train \
	--model_dir=${MLP_GCS_EUW_MODEL_DIR} \
	--num_cores=512 \
	--resnet_depth=50 \
	--skip_host_call \
	--steps_per_eval=312 \
	--tpu=$MLP_TPU_NAME \
	--train_batch_size=16384 \
	--train_steps=7038 \
        --use_train_runner=True \
	--use_async_checkpointing &


timeout 1h $EVAL_CMD
wait
STAT=$?


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"


# report result 
result=$(( $end - $start )) 
result_name="resnet"


echo "RESULT,$result_name,0,$result,$USER,$start_fmt"

exit $STAT
