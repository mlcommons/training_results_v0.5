#!/bin/bash

set -e

git clone https://bitfort:$MLP_GITHUB_KEY@github.com/tensorflow/staging.git --branch in_progress staging

PINNED_SHA="e418e34e3f82f1c99e55cd579db9f81c94d78b75"
pushd staging
echo "Pinning to: ${PINNED_SHA}"
git checkout ${PINNED_SHA}
popd
