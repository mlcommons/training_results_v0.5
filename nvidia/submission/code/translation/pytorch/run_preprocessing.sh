
TEXT=examples/translation/wmt14_en_de

(
  cd examples/translation
  bash prepare-wmt14en2de.sh --scaling18
)

python preprocess.py \
  --source-lang en \
  --target-lang de \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test \
  --destdir ${DATASET_DIR} \
  --nwordssrc 33712 \
  --nwordstgt 33712 \
  --joined-dictionary
