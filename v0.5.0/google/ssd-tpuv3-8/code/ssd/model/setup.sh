#!/bin/bash

set -e

sudo apt-get update
sudo apt-get install -y python3-tk unzip

pip install --upgrade pip==9.0.1
pip install Cython==0.28.4 matplotlib==2.2.2
pip install pycocotools==2.0.0
pip install Pillow==5.2.0
alias protoc="/usr/local/bin/protoc"
INSTALL_PROTO="yes"
if protoc --version | grep -q -E --regexp="3.6.1$"; then
  INSTALL_PROTO=""
fi
if [ ! -z $INSTALL_PROTO ]; then
  pushd /tmp
  curl -OL https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
  unzip protoc-3.6.1-linux-x86_64.zip -d protoc3
  # Move protoc to /usr/local/bin/
  sudo mv protoc3/bin/* /usr/local/bin/
  # Move protoc3/include to /usr/local/include/
  if [ -d /usr/local/include/google/protobuf ]; then
    sudo rm -r /usr/local/include/google/protobuf
  fi
  sudo mv protoc3/include/* /usr/local/include/

  # Optional: change owner
  sudo chown $USER /usr/local/bin/protoc
  sudo chown -R $USER /usr/local/include/google
  popd
fi

pushd tpu
  CLOUD_TPU_SHA="665b36dfc8bc1789f5b879db8355e15d13ca400a"
  echo "Checking out cloud tpu SHA: ${CLOUD_TPU_SHA}"
  git checkout ${CLOUD_TPU_SHA}
popd
