#!/bin/bash

set -e

echo "Upgrading GCC for cloud_tpu_profiler"

export DEBIAN_FRONTEND=noninteractive
apt-get update --quiet && \
apt-get install --quiet build-essential software-properties-common -y && \
apt-get update --quiet && \
add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
apt-get update --quiet && \
apt-get install --quiet gcc-snapshot -y && \
apt-get update --quiet && \
apt-get install --quiet gcc-6 g++-6 -y && \
apt-get install --quiet gcc-4.8 g++-4.8 -y && \
apt-get upgrade --quiet -y libstdc++6
