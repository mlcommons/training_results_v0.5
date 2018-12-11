# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/mxnet:18.11-py3
FROM ${FROM_IMAGE_NAME}

# Install Python dependencies
WORKDIR /workspace/image_classification

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy ResNet-50 code and configure
COPY . .

ENV MXNET_UPDATE_ON_KVSTORE=0      \
    MXNET_EXEC_ENABLE_ADDTO=1      \
    MXNET_USE_TENSORRT=0           \
    MXNET_GPU_WORKER_NTHREADS=1    \
    MXNET_GPU_COPY_NTHREADS=1      \
    MXNET_CUDNN_AUTOTUNE_DEFAULT=0 \
    MXNET_OPTIMIZER_AGGREGATION_SIZE=54 \
    NCCL_SOCKET_IFNAME=^docker0,bond0,lo \
    NCCL_BUFFSIZE=2097152          \
    NCCL_NET_GDR_READ=1            \
    HOROVOD_CYCLE_TIME=0.2         \
    HOROVOD_TWO_STAGE_LOOP=1       \
    HOROVOD_ALLREDUCE_MODE=1       \
    HOROVOD_FIXED_PAYLOAD=161
