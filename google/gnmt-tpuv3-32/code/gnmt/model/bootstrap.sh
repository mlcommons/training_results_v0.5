#!/bin/bash


set -e

git clone https://bitfort:$MLP_GITHUB_KEY@github.com/tensorflow/staging.git --branch in_progress staging

#PINNED_SHA="00005aecab27d906fff6465e05d66ff5c5cc99d5"
pushd staging
#echo "Pinning to: ${PINNED_SHA}"
echo "Pinning to HEAD"
#git checkout ${PINNED_SHA}
popd
