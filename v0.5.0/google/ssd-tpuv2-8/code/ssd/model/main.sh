#!/bin/bash
set -e

export MLP_TPU_TF_VERSION=1.12
export MLP_TF_PIP_LINE=tensorflow==1.12.0
export MLP_CIDR_SIZE=29
export MLP_TPU_VERSION=v2-8
export MLP_TPU_SIDECAR=N
export MLP_TPU_PREEMPT=
export BENCHMARK=ssd
export MLP_OVERRIDE_TPU=N

echo MLP_OVERRIDE_TPU = ${MLP_OVERRIDE_TPU:?}

SECONDS=`date +%s`
DAY_OF_MONTH=`date -d "$D" '+%d'`
export MLP_GCP_HOST=`hostname`
# export MLP_GCS_EUW_MODEL_DIR=gs://.../model-dirs/${MLP_GCP_HOST:?}-${SECONDS}
# export MLP_GCS_MODEL_DIR=gs://.../tests/${MLP_GCP_HOST:?}-${SECONDS}
export MLP_GCP_ZONE=`gcloud compute instances list ${MLP_GCP_HOST:?} --format 'csv[no-heading](zone)' 2>/dev/null`
export MLP_TPU_NAME=${MLP_GCP_HOST:?}_TPU_${BENCHMARK}_${DAY_OF_MONTH}_${SECONDS}

export MLP_TPU_SIDECAR_NAME=${MLP_GCP_HOST:?}_TPU_SIDECAR_${BENCHMARK}_${DAY_OF_MONTH}_${SECONDS}

# export MLP_PATH_GCS_IMAGENET=gs://.../imagenet/combined
# export MLP_PATH_GCS_TRANSFORMER=gs://.../benchmark_data/transformer_v2
# export MLP_PATH_GCS_SSD=gs://.../benchmark_data/ssd_coco
# export MLP_PATH_GCS_MASKRCNN=gs://.../benchmark_data/maskrcnn_coco
# export MLP_PATH_GCS_NMT=gs://.../benchmark_data/nmt/wmt16_de_en/  # Note: This path needs trailing forward slash
# export MLP_PATH_GCS_NCF=gs://.../benchmark_data/ncf_tpu

# export MLP_PATH_GCS_EUW_IMAGENET=gs://.../garden-imgnet/imagenet/combined
# export MLP_PATH_GCS_EUW_TRANSFORMER=gs://.../benchmark_data/transformer_v2
# export MLP_PATH_GCS_EUW_SSD=gs://.../benchmark_data/ssd_coco
# export MLP_PATH_GCS_EUW_MASKRCNN=gs://.../benchmark_data/maskrcnn_coco
# export MLP_PATH_GCS_EUW_NMT=gs://.../benchmark_data/nmt/wmt16_de_en/  # Note: This path needs trailing forward slash
# export MLP_PATH_GCS_EUW_NCF=gs://.../benchmark_data/ncf_tpu

# export MLP_GCS_RESNET_CHECKPOINT=gs://.../benchmark_data/resnet34_ssd_checkpoint
# export MLP_GCS_EUW_RESNET_CHECKPOINT=gs://.../benchmark_data/resnet34_ssd_checkpoint


# gcloud compute instances list ${MLP_GCP_HOST:?} --format 'csv[no-heading](zone)'

TPU_PREEMPT=${MLP_TPU_PREEMPT}

echo MLP_TPU_TF_VERSION ${MLP_TPU_TF_VERSION:?}
echo MLP_GCP_HOST ${MLP_GCP_HOST:?}
echo MLP_GCP_ZONE ${MLP_GCP_ZONE:?}
echo MLP_TPU_NAME ${MLP_TPU_NAME:?}

gcloud auth list


if [[ ${MLP_OVERRIDE_TPU:?} =~ "N"$ ]]; then

for x in {0..255}; do
BASE_IP=$((1 + RANDOM % 255))
echo gcloud alpha compute tpus create ${MLP_TPU_NAME:?} --range=10.$BASE_IP.$x.0/${MLP_CIDR_SIZE:?} $TPU_PREEMPT --version=${MLP_TPU_TF_VERSION:?} --network=default --accelerator-type=${MLP_TPU_VERSION:?} --zone ${MLP_GCP_ZONE:?}
gcloud alpha compute tpus create ${MLP_TPU_NAME:?} --range=10.$BASE_IP.$x.0/${MLP_CIDR_SIZE:?} $TPU_PREEMPT --version=${MLP_TPU_TF_VERSION:?} --network=default --accelerator-type=${MLP_TPU_VERSION:?} --zone ${MLP_GCP_ZONE:?} 2>&1 | tee /tmp/create_tpu_log.txt

STATUS=$?

if grep -q "Try a different range" /tmp/create_tpu_log.txt; then
  # In this case, the network address is taken adn we should re-try this action, incrementing x
  echo "Trying a different range...";
elif grep -q "capacity" /tmp/create_tpu_log.txt; then
  echo "No Capacity for TPUs!";
  exit 1
elif grep -q "Quota limit" /tmp/create_tpu_log.txt; then
  echo "Out of Quota for TPUs!";
  exit 1
elif grep -q "CIDR" /tmp/create_tpu_log.txt; then
  # In this case, the network address is taken adn we should re-try this action, incrementing x
  echo "Trying a different range (CIDR error)...";
elif grep -q "Invalid" /tmp/create_tpu_log.txt; then
  # In this case, the network address is taken adn we should re-try this action, incrementing x
  echo "Trying a different range...";
else
  break
  if [ $? -ne 0 ]
  then
     echo "Failed to start TPU"
     exit 1
 fi
fi
done

fi

echo Done setting up TPUs

# Start the TPU Sidecar
if [[ ${MLP_TPU_SIDECAR:?} =~ "Y"$ ]]; then
    for x in {0..255}; do
    BASE_IP=$((1 + RANDOM % 255))
    echo gcloud alpha compute tpus create ${MLP_TPU_SIDECAR_NAME:?} --range=10.$BASE_IP.$x.0/29 --version=${MLP_TPU_TF_VERSION:?} --network=default --accelerator-type=v2-8 --zone ${MLP_GCP_ZONE:?}
    gcloud alpha compute tpus create ${MLP_TPU_SIDECAR_NAME:?} --range=10.$BASE_IP.$x.0/29 --version=${MLP_TPU_TF_VERSION:?} --network=default --accelerator-type=v2-8 --zone ${MLP_GCP_ZONE:?} 2>&1 | tee /tmp/create_tpu_log.txt

    STATUS=$?

    if grep -q "Try a different range" /tmp/create_tpu_log.txt; then
      # In this case, the network address is taken adn we should re-try this action, incrementing x
      echo "Trying a different range...";
    elif grep -q "CIDR" /tmp/create_tpu_log.txt; then
      # In this case, the network address is taken adn we should re-try this action, incrementing x
      echo "Trying a different range (CIDR error)...";
    elif grep -q "Invalid" /tmp/create_tpu_log.txt; then
      # In this case, the network address is taken adn we should re-try this action, incrementing x
      echo "Trying a different range...";
    else
      break
      if [ $? -ne 0 ]
      then
         echo "Failed to start Sidecar TPU"
         echo  gcloud alpha compute tpus delete ${MLP_TPU_NAME:?} --zone ${MLP_GCP_ZONE:?}
         yes | gcloud alpha compute tpus delete ${MLP_TPU_NAME:?} --zone ${MLP_GCP_ZONE:?}
         exit 1
     fi
    fi
    done
fi

set +e

pip install mlperf_compliance==0.0.10
if [ $BENCHMARK ]; then
  echo "Clearing caches. (Fresh TPUs have already been started.)"
  sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
  STAT=$?
  if [ $STAT -eq 0 ]; then
    echo "Cache clear successful."
    python -c "import mlperf_compliance;mlperf_compliance.mlperf_log.${BENCHMARK}_print(key='run_clear_caches')"
  fi
fi


# Place certain environment variables in a file in /tmp to make it easier to SSH and collect traces.
PROFILER_PREP="/tmp/prep_profiler.sh"
echo "source ${RUN_VENV}/bin/activate" >> ${PROFILER_PREP}
echo "export PATH="\$PATH:\`python -m site --user-base\`/bin"" >> ${PROFILER_PREP}
echo "export MLP_GCP_HOST=${MLP_GCP_HOST:?}" >> ${PROFILER_PREP}
echo "export MLP_GCS_MODEL_DIR=${MLP_GCS_MODEL_DIR:?}" >> ${PROFILER_PREP}
echo "export MLP_GCP_ZONE=${MLP_GCP_ZONE:?}" >> ${PROFILER_PREP}
echo "export MLP_TPU_NAME=${MLP_TPU_NAME:?}" >> ${PROFILER_PREP}
echo "export MLP_TPU_SIDECAR_NAME=${MLP_TPU_SIDECAR_NAME:?}" >> ${PROFILER_PREP}

sudo bash upgrade_gcc.sh
export PATH="$PATH:`python3 -m site --user-base`/bin"


echo RUNNING ON TPU: ${MLP_TPU_NAME:?}
bash run_helper.sh

BENCHMARK_EXIT_CODE=$?

set -e


if [[ ${MLP_OVERRIDE_TPU:?} =~ "N"$ ]]; then

echo  gcloud alpha compute tpus delete ${MLP_TPU_NAME:?} --zone ${MLP_GCP_ZONE:?}
yes | gcloud alpha compute tpus delete ${MLP_TPU_NAME:?} --zone ${MLP_GCP_ZONE:?}

fi


# Stop the TPU Sidecar
if [[ ${MLP_TPU_SIDECAR:?} =~ "Y"$ ]]; then
    echo  gcloud alpha compute tpus delete ${MLP_TPU_SIDECAR_NAME:?} --zone ${MLP_GCP_ZONE:?}
    yes | gcloud alpha compute tpus delete ${MLP_TPU_SIDECAR_NAME:?} --zone ${MLP_GCP_ZONE:?}
fi

exit $BENCHMARK_EXIT_CODE
