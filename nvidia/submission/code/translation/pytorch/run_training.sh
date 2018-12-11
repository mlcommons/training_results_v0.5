# Start timing
START=$(date +%s)
START_FMT=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT ${START_FMT}"

# Includes online scoring
python -m bind_launch --nsockets_per_node ${DGXNSOCKET} \
                      --ncores_per_socket ${DGXSOCKETCORES} \
                      --nproc_per_node $SLURM_NTASKS_PER_NODE $MULTI_NODE \
 train.py ${DATASET_DIR} \
  --seed ${SEED} \
  --arch transformer_wmt_en_de_big_t2t \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.997)' \
  --adam-eps "1e-9" \
  --clip-norm "0.0" \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr "0.0" \
  --warmup-updates ${WARMUP_UPDATES} \
  --lr ${LEARNING_RATE} \
  --min-lr "0.0" \
  --dropout "0.1" \
  --weight-decay "0.0" \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing "0.1" \
  --max-tokens ${MAX_TOKENS} \
  --max-epoch 30 \
  --target-bleu "25.0" \
  --ignore-case \
  --no-save \
  --update-freq 1 \
  --fp16 \
  --distributed-init-method "env://" \
  ${EXTRA_PARAMS}

# End timing
END=$(date +%s)
END_FMT=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT ${END_FMT}"

# Report result
RESULT=$(( ${END} - ${START} ))
RESULT_NAME="transformer"
echo "RESULT,${RESULT_NAME},${SEED},${RESULT},${USER},${START_FMT}"
