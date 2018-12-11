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

import torch.nn as nn
from conv import Conv2d_NHWC
from batch_norm import BatchNorm2d_NHWC
from max_pool import MaxPool2d_NHWC

class VggBN_NHWC(nn.Module):
    def __init__(self):
        super(VggBN_NHWC, self).__init__()
        self.conv = nn.Sequential(
            # Stage 1
            Conv2d_NHWC(3, 64, 3, padding=1),
            BatchNorm2d_NHWC(64),
            nn.ReLU(),

            Conv2d_NHWC(64,64, 3, padding=1),
            BatchNorm2d_NHWC(64),
            nn.ReLU(),

            MaxPool2d_NHWC(2, 2),
            # Stage 2
            Conv2d_NHWC(64, 128, 3, padding=1),
            BatchNorm2d_NHWC(128),
            nn.ReLU(),

            Conv2d_NHWC(128, 128, 3, padding=1),
            BatchNorm2d_NHWC(128),
            nn.ReLU(),

            MaxPool2d_NHWC(2, 2),
            # Stage 3
            Conv2d_NHWC(128, 256, 3, padding=1),
            BatchNorm2d_NHWC(256),
            nn.ReLU(),

            Conv2d_NHWC(256, 256, 3, padding=1),
            BatchNorm2d_NHWC(256),
            nn.ReLU(),

            Conv2d_NHWC(256, 256, 3, padding=1),
            BatchNorm2d_NHWC(256),
            nn.ReLU(),

            MaxPool2d_NHWC(2, 2),
            # Stage 4
            Conv2d_NHWC(256, 512, 3, padding=1),
            BatchNorm2d_NHWC(512),
            nn.ReLU(),

            Conv2d_NHWC(512, 512, 3, padding=1),
            BatchNorm2d_NHWC(512),
            nn.ReLU(),

            Conv2d_NHWC(512, 512, 3, padding=1),
            BatchNorm2d_NHWC(512),
            nn.ReLU(),

            MaxPool2d_NHWC(2, 2),
            # Stage 5
            Conv2d_NHWC(512, 512, 3, padding=1),
            BatchNorm2d_NHWC(512),
            nn.ReLU(),

            Conv2d_NHWC(512, 512, 3, padding=1),
            BatchNorm2d_NHWC(512),
            nn.ReLU(),

            Conv2d_NHWC(512, 512, 3, padding=1),
            BatchNorm2d_NHWC(512),
            nn.ReLU(),

            MaxPool2d_NHWC(2, 2)
        )
        self.fc = nn.Sequential(
            # FC
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


