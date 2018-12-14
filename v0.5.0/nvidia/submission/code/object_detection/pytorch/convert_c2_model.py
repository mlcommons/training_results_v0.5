#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import pickle
from collections import OrderedDict

import torch


def _rename_weights_for_R50(weights):
    original_keys = sorted(weights.keys())
    layer_keys = sorted(weights.keys())
    layer_keys = [k.replace("_", ".") for k in layer_keys]
    layer_keys = [k.replace(".w", ".weight") for k in layer_keys]
    layer_keys = [k.replace(".bn", "_bn") for k in layer_keys]
    layer_keys = [k.replace(".b", ".bias") for k in layer_keys]
    layer_keys = [k.replace("_bn.s", "_bn.scale") for k in layer_keys]
    layer_keys = [k.replace(".biasranch", ".branch") for k in layer_keys]
    layer_keys = [k.replace("bbox.pred", "bbox_pred") for k in layer_keys]
    layer_keys = [k.replace("cls.score", "cls_score") for k in layer_keys]
    layer_keys = [k.replace("res.conv1_", "conv1_") for k in layer_keys]

    # RPN / Faster RCNN
    layer_keys = [k.replace(".biasbox", ".bbox") for k in layer_keys]
    layer_keys = [k.replace("conv.rpn", "rpn.conv") for k in layer_keys]
    layer_keys = [k.replace("rpn.bbox.pred", "rpn.bbox_pred") for k in layer_keys]
    layer_keys = [k.replace("rpn.cls.logits", "rpn.cls_logits") for k in layer_keys]

    # FPN
    layer_keys = [
        k.replace("fpn.inner.res2.2.sum.lateral", "fpn_inner1") for k in layer_keys
    ]
    layer_keys = [
        k.replace("fpn.inner.res3.3.sum.lateral", "fpn_inner2") for k in layer_keys
    ]
    layer_keys = [
        k.replace("fpn.inner.res4.5.sum.lateral", "fpn_inner3") for k in layer_keys
    ]
    layer_keys = [k.replace("fpn.inner.res5.2.sum", "fpn_inner4") for k in layer_keys]

    layer_keys = [k.replace("fpn.res2.2.sum", "fpn_layer1") for k in layer_keys]
    layer_keys = [k.replace("fpn.res3.3.sum", "fpn_layer2") for k in layer_keys]
    layer_keys = [k.replace("fpn.res4.5.sum", "fpn_layer3") for k in layer_keys]
    layer_keys = [k.replace("fpn.res5.2.sum", "fpn_layer4") for k in layer_keys]

    layer_keys = [k.replace("rpn.conv.fpn2", "rpn.conv") for k in layer_keys]
    layer_keys = [k.replace("rpn.bbox_pred.fpn2", "rpn.bbox_pred") for k in layer_keys]
    layer_keys = [
        k.replace("rpn.cls_logits.fpn2", "rpn.cls_logits") for k in layer_keys
    ]

    # Mask R-CNN
    layer_keys = [k.replace("mask.fcn.logits", "mask_fcn_logits") for k in layer_keys]
    layer_keys = [k.replace(".[mask].fcn", "mask_fcn") for k in layer_keys]
    layer_keys = [k.replace("conv5.mask", "conv5_mask") for k in layer_keys]

    # Keypoint R-CNN
    layer_keys = [k.replace("kps.score.lowres", "kps_score_lowres") for k in layer_keys]
    layer_keys = [k.replace("kps.score", "kps_score") for k in layer_keys]
    layer_keys = [k.replace("conv.fcn", "conv_fcn") for k in layer_keys]

    # from IPython import embed; embed()

    key_map = {k: v for k, v in zip(original_keys, layer_keys)}

    new_weights = OrderedDict()
    for k, v in weights.items():
        if "_momentum" in k:
            continue
        # if 'fc1000' in k:
        #     continue
        # new_weights[key_map[k]] = torch.from_numpy(v)
        w = torch.from_numpy(v)
        if "bn" in k:
            w = w.view(1, -1, 1, 1)
        new_weights[key_map[k]] = w

    return new_weights


def main():
    weight_file = './R-50.pkl'

    with open(weight_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    weights = data # ['blobs']

    weights_new = _rename_weights_for_R50(weights)

    torch.save(weights_new, './R-50.pth')


if __name__ == '__main__':
    main()
