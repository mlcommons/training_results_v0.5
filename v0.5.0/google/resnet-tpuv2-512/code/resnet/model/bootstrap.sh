#!/bin/bash

set -e

git clone https://bitfort:$MLP_GITHUB_KEY@github.com/tensorflow/staging.git --branch in_progress staging

PINNED_SHA="24159824e28f06d135b363c1c522c22b1710731b"
pushd staging
#echo "Pinning to: ${PINNED_SHA}"
echo "Pinning to: HEAD"
#git checkout ${PINNED_SHA}
popd
