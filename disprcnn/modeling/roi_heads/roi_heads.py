# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from disprcnn.utils.timer import Timer
from .box_head.box_head import build_roi_box_head, ROIBoxHead
from .mask_head.mask_head import build_roi_mask_head, ROIMaskHead

boxheadtimer = Timer()
maskheadtimer = Timer()
PRINTTIME = False


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """
    mask: ROIMaskHead
    box: ROIBoxHead

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(features, proposals, targets)

        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                    self.training
                    and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)

            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                    self.training
                    and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)
        return x, detections, losses


class StereoCombinedROIHeads(torch.nn.ModuleDict):
    """
        Combines a set of individual heads (for box prediction or masks) into a single
        head.
        """
    mask: ROIMaskHead
    box: ROIBoxHead

    def __init__(self, cfg, heads):
        super().__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor

    def forward(self, left_features, right_features, left_proposals, right_proposals, left_targets=None,
                right_targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        if PRINTTIME:
            torch.cuda.synchronize()
            boxheadtimer.tic()
        x, left_detections, \
        right_detections, loss_box = self.box({'left': left_features,
                                               'right': right_features},
                                              {'left': left_proposals,
                                               'right': right_proposals},
                                              {'left': left_targets,
                                               'right': right_targets})
        if PRINTTIME:
            torch.cuda.synchronize()
            boxheadtimer.toc()
            print('box head', boxheadtimer.average_time)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = left_features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                    self.training
                    and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            if PRINTTIME:
                torch.cuda.synchronize()
                maskheadtimer.tic()
            x, left_detections, loss_mask = self.mask(mask_features, left_detections, left_targets)
            if PRINTTIME:
                torch.cuda.synchronize()
                maskheadtimer.toc()
                print('mask head', maskheadtimer.average_time)
            losses.update(loss_mask)

        if self.cfg.MODEL.SHAPE_ON:
            shape_features = left_features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                    self.training
                    and self.cfg.MODEL.ROI_SHAPE_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                shape_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, left_detections, loss_shape = self.shape(shape_features, left_detections, left_targets)
            losses.update(loss_shape)
        return x, left_detections, right_detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    # combine individual heads in a single module
    if roi_heads:
        if not cfg.MODEL.STEREO_ON:
            roi_heads = CombinedROIHeads(cfg, roi_heads)
        else:
            roi_heads = StereoCombinedROIHeads(cfg, roi_heads)
    return roi_heads
