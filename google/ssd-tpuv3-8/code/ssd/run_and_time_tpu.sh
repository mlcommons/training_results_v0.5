#!/bin/bash

set -e
set -o pipefail

cd model

export RUN_VENV="/tmp/tpu_run_env"
bash setup_environment.sh

echo "switching to virtual environment"
source ${RUN_VENV}/bin/activate

printf "\n\n\nCalling: bootstrap.sh\n"
bash bootstrap.sh

printf "\n\n\nCalling: setup.sh\n"
bash setup.sh

printf "\n\n\nCalling: main.sh\n"
unbuffer bash main.sh 2>&1 | tee output.txt
