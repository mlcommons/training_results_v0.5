#!/bin/bash

set -e

sudo apt-get update
sudo apt-get install -y python3-tk unzip

pip install sacrebleu==1.2.11

