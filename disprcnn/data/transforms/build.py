# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    ts = [
        T.Resize(min_size, max_size),
        T.RandomHorizontalFlip(flip_prob),
        T.ToTensor(),
        normalize_transform,
    ]
    if not cfg.INPUT.DO_RESIZE:
        ts.pop(0)
    transform = T.Compose(
        ts
    )
    return transform
