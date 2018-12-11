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

import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

from .conv import Conv2d_NHWC
from .batch_norm import BatchNorm2d_NHWC
from .max_pool import MaxPool2d_NHWC

__all__ = ['ResNet_NHWC', 'resnet18_nhwc', 'resnet34_nhwc', 'resnet50_nhwc', 'resnet101_nhwc',
           'resnet152_nhwc']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d_NHWC(in_planes, out_planes, kernel_size=3, stride=stride,
                       padding=1, bias=False)


class BasicBlock_NHWC(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_NHWC, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d_NHWC(planes, fuse_relu=True)
        self.conv2 = conv3x3(planes, planes)
        # Set to True when enabling BN-Add-Relu
        self.bn2 = BatchNorm2d_NHWC(planes, fuse_relu=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        # relu fused into bn
        # out = out.relu_()

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # Fused call
        # out = self.bn2(out, residual)
        out = self.bn2(out)
        out = out + residual
        out = out.relu_()

        return out


class Bottleneck_NHWC(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_NHWC, self).__init__()
        self.conv1 = Conv2d_NHWC(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d_NHWC(planes, fuse_relu=True)
        self.conv2 = Conv2d_NHWC(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d_NHWC(planes, fuse_relu=True)
        self.conv3 = Conv2d_NHWC(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d_NHWC(planes * self.expansion, fuse_relu=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        # Fused into BN
        # out = out.relu_()

        out = self.conv2(out)
        out = self.bn2(out)
        # Fused into BN
        # out = out.relu_()

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = out.relu_()

        return out


class ResNet_NHWC(nn.Module):

    def __init__(self, block, layers, num_classes=1000, pad_input=False):
        self.inplanes = 64
        super(ResNet_NHWC, self).__init__()
        if pad_input:
            input_channels = 4
        else:
            input_channels = 3
        self.conv1 = Conv2d_NHWC(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d_NHWC(64, fuse_relu=True)
        # self.relu = nn.ReLU(inplace=True)
        self.maxpool = MaxPool2d_NHWC(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d_NHWC(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d_NHWC(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Permute back to NCHW for AvgPool and the rest
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def _transpose_state(state, pad_input=False):
    for k in state.keys():
        if len(state[k].shape) == 4:
            if pad_input and "conv1.weight" in k and not 'layer' in k:
                s = state[k].shape
                state[k] = torch.cat([state[k], torch.zeros([s[0], 1, s[2], s[3]])], dim=1)
            state[k] = state[k].permute(0, 2, 3, 1).contiguous()
    return state

def resnet18_nhwc(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_NHWC(BasicBlock_NHWC, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34_nhwc(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_NHWC(BasicBlock_NHWC, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet34'])

        pad_input = kwargs.get('pad_input', False)
        # transpose all weights in the state dict
        model.load_state_dict(_transpose_state(state_dict, pad_input), strict=False)
    return model


def resnet50_nhwc(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_NHWC(Bottleneck_NHWC, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101_nhwc(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_NHWC(Bottleneck_NHWC, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_nhwc(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_NHWC(Bottleneck_NHWC, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
