#!/bin/bash

echo Updating gsutil

# This is the suggested fix:
sudo apt-get update && sudo apt-get --only-upgrade install kubectl google-cloud-sdk google-cloud-sdk-app-engine-grpc google-cloud-sdk-pubsub-emulator google-cloud-sdk-app-engine-go google-cloud-sdk-cloud-build-local google-cloud-sdk-datastore-emulator google-cloud-sdk-app-engine-python google-cloud-sdk-cbt google-cloud-sdk-bigtable-emulator google-cloud-sdk-app-engine-python-extras google-cloud-sdk-datalab google-cloud-sdk-app-engine-java
# Seems to be broken right now, preventing us from starting preemptible TPUs
gcloud components update --version 223.0.0
echo Done Updating gsutil

set -e

# Not sure why this happens... but it sometimes causes errors if not...
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -


lsb_release -a

sudo apt-get update
sudo apt-get install -y --quiet expect
sudo apt-get install -y --quiet python3-pip virtualenv python-virtualenv

which python3
python3 --version

virtualenv -p python3 ${RUN_VENV:?}
source ${RUN_VENV:?}/bin/activate
pip --version

pip install --upgrade --progress-bar=off pyyaml==3.13 oauth2client==4.1.3 google-api-python-client==1.7.4 google-cloud==0.34.0
pip install --progress-bar=off mlperf_compliance==0.0.10
pip install --progress-bar=off cloud-tpu-profiler==1.12


# Note: this could be over-ridden later
TF_TO_INSTALL=${MLP_TF_PIP_LINE:?}
pip install --progress-bar=off $TF_TO_INSTALL

echo 'TPU Host Freeze pip'
pip freeze
echo
echo
