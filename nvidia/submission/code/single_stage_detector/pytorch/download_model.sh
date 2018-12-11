#!/usr/bin/env bash

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

# Get VGG model
cd ./ssd;
curl -O https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
curl -O https://download.pytorch.org/models/vgg16_bn-6c64b313.pth; mv vgg16_bn-6c64b313.pth vgg16_bn.pth
python3 base_model.py; rm vgg16_reducedfc.pth;
python3 convert_vgg_bn.py; rm vgg16_bn.pth
cd ..
