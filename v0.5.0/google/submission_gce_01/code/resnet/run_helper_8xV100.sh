set -e

###
# Note: This trains to 74.9+ top_1 at 61 epochs most of the time.  Setup to
# run a few extra just in case until 100% sure.
########################
echo "Clearing caches."
sync && echo 3 | tee /host_proc/sys/vm/drop_caches
STAT=$?
if [ $STAT -eq 0 ]; then
  echo "Cache clear successful."
  python -c "import mlperf_compliance;mlperf_compliance.mlperf_log.resnet_print(key='run_clear_caches')"
fi

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
cd benchmarks/scripts/
python3 -c 'import tensorflow; print(tensorflow.__version__)'
python3 -c 'import tensorflow; print(tensorflow.__git_version__)'

python3 tf_cnn_benchmarks/tf_cnn_benchmarks.py --data_format=NCHW \
  --batch_size=256 \
  --model=resnet50_v1.5 \
  --data_dir=/data/imagenet/combined/ \
  --optimizer=momentum \
  --weight_decay=1e-4 \
  --variable_update=replicated \
  --num_warmup_batches=0 \
  --all_reduce_spec=nccl \
  --use_fp16=True \
  --nodistortions \
  --gradient_repacking=2 \
  --num_epochs=64 \
  --ml_perf \
  --local_parameter_device=gpu \
  --num_gpus=8 \
  --display_every=100 \
  --xla_compile=True \
  --eval_during_training_at_specified_epochs='1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,62,63' \
  --num_eval_batches=25 \
  --eval_batch_size=250 \
  --loss_type_to_report=base_loss \
  --single_l2_loss_op \
  --compute_lr_on_cpu \
  --resnet_base_lr=0.06 \
  --ml_perf_compliance_logging=True \
  --datasets_parallel_interleave_cycle_length=20 --datasets_sloppy_parallel_interleave \
  --stop_at_top_1_accuracy=0.749

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"


# report result
result=$(( $end - $start ))
result_name="resnet"


echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
