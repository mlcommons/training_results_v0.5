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
import torch

from base_model import *

from collections import OrderedDict

def convert_vgg_bn(src_model, src_dict, tgt_model, tgt_dict):
    src_state = torch.load(src_dict)
    src_model.load_state_dict(src_dict)

    src_state = src_model.state_dict()

    keys1 = src_state.keys()
    keys1 = [k for k in src_state.keys() if k.startswith('features')]
    keys2 = tgt_model.state_dict.keys()

    assert(len(keys1) == len(keys2))

    state = OrderedDict()


    print(len(state.keys()), state.keys())

    for key in state.keys():
        if 'features' in key:
            print(key)

if __name__ == "__main__":
    # convert_vgg_bn('vgg16_bn.pth', 'tmp.pth')
    vggt = VGG16([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', \
                  512, 512, 512, 'M' , 512, 512, 512,], use_bn=True)

    state_dict = torch.load('vgg16_bn.pth')
    state_dict_vgg = vggt.state_dict()

    keys1 = state_dict.keys()
    keys1 = [k for k in state_dict.keys() if k.startswith('features')]
    keys2 = [k for k in state_dict_vgg.keys() if 'num_batches_tracked' not in k and k.startswith('layer')]

    #print(len(keys1), len(keys2))

    #print(keys1)
    #  print(keys2)

    assert len(keys1) == len(keys2)
    state = OrderedDict()

    for k1, k2 in zip(keys1, keys2):
        # print(k1.split('.')[-1], k2.split('.')[-1])

        print(k2, ' : ', state_dict[k1].size())

        state[k2] = state_dict[k1]

    # print(state.keys())
    for i, key in enumerate(state):
        print(i, key)
    vggt.load_state_dict(state, strict=False)
    torch.save(vggt.state_dict(), "./vgg16n_bn.pth")

