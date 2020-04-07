# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .disprcnn import DispRCNN
from .disprcnn3d import DispRCNN3D

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,
                                 "DispRCNN": DispRCNN,
                                 'DispRCNN3D': DispRCNN3D}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    model = meta_arch(cfg)
    return model
