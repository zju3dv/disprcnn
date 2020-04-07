import os
from typing import Dict, List

import torch
from torch import nn

from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.image_list import to_image_list
from disprcnn.utils.timer import Timer
from .generalized_rcnn import GeneralizedRCNN
from disprcnn.modeling.psmnet.dispmodule import DispModule
from disprcnn.modeling.pointnet_module.point_rcnn.lib.net.point_rcnn import PointRCNN


class DispRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        if cfg.MODEL.DISPNET_ON:
            self.dispnet = DispModule(cfg)
        if cfg.MODEL.DET3D_ON:
            self.pcnet = PointRCNN(cfg)

    def forward(self, lrimages: Dict[str, List[torch.Tensor]], lrtargets: Dict[str, List[BoxList]] = None):
        """
        :param lrimages:
        :param lrtargets:
        :return:result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and lrtargets is None:
            raise ValueError("In training mode, targets should be passed")
        left_images, right_images = lrimages['left'], lrimages['right']
        if lrtargets is not None:
            left_targets, right_targets = lrtargets['left'], lrtargets['right']
        else:
            left_targets, right_targets = None, None
        left_images = to_image_list(left_images)
        right_images = to_image_list(right_images)
        batch_size = left_images.tensors.shape[0]
        lr_tensors = torch.cat((left_images.tensors, right_images.tensors), dim=0)
        lr_features = self.backbone(lr_tensors)
        lr_features_splited = [torch.split(t, [batch_size, batch_size]) for t in lr_features]
        left_features = [t[0] for t in lr_features_splited]
        right_features = [t[1] for t in lr_features_splited]
        left_proposals, right_proposals, proposal_losses = self.rpn(left_images, right_images,
                                                                    left_features, right_features,
                                                                    left_targets, right_targets)
        assert self.roi_heads
        x, left_result, right_result, \
        detector_losses = self.roi_heads(left_features, right_features,
                                         left_proposals, right_proposals,
                                         left_targets, right_targets)
        # disparity network
        if self.cfg.MODEL.DISPNET_ON:
            left_result, right_result, disp_loss = self.dispnet(left_features, right_features,
                                                                left_result, right_result,
                                                                left_targets, right_targets)
        if self.cfg.MODEL.DET3D_ON:
            pcout = self.pcnet(left_result, right_result, left_targets)
            if self.pcnet.training:
                left_result, det3d_loss = pcout
            else:
                left_result, right_result, det3d_loss = pcout

        if self.training:
            losses = {}
            if self.cfg.SOLVER.TRAIN_2D:
                losses.update(detector_losses)
                # losses.update(right_detector_losses)
                losses.update(proposal_losses)
                # losses.update(right_proposal_losses)
            if self.cfg.SOLVER.TRAIN_PSM and hasattr(self, 'dispnet'):
                losses.update(disp_loss)
            if self.cfg.MODEL.DET3D_ON and self.cfg.SOLVER.TRAIN_PC:
                losses.update(det3d_loss)
            return losses
        result = {'left': left_result, 'right': right_result}
        # result = left_result
        return result
