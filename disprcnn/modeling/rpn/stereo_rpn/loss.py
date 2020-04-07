# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from disprcnn.layers import smooth_l1_loss
from disprcnn.modeling.balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from disprcnn.modeling.matcher import Matcher
from disprcnn.modeling.rpn.utils import concat_box_prediction_layers
from disprcnn.structures.boxlist_ops import boxlist_iou
from disprcnn.structures.boxlist_ops import cat_boxlist
from disprcnn.structures.bounding_box import BoxList
from disprcnn.utils.stereo_utils import expand_left_right_box


class SRPNLossComputation(object):
    """
    This class computes the SRPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = ['original_lr_bbox']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def make_expand_targets_per_image(self, left_targets_per_image: BoxList, right_targets_per_image: BoxList):
        # todo: do match, expand and put original bbox in extra_fields
        # todo: check necessity of clone
        left_targets_per_image = left_targets_per_image.copy_with_fields([])
        right_targets_per_image = right_targets_per_image.copy_with_fields([])
        left_bbox = left_targets_per_image.bbox
        right_bbox = right_targets_per_image.bbox
        expand_bbox, original_lr_bbox = expand_left_right_box(left_bbox, right_bbox)
        expand_target = BoxList(expand_bbox, left_targets_per_image.size)
        expand_target.add_field('original_lr_bbox', original_lr_bbox)
        return expand_target

    def prepare_targets(self, anchors, left_targets, right_targets):
        labels = []
        regression_targets = []
        for anchors_per_image, left_targets_per_image, right_targets_per_image in zip(anchors, left_targets,
                                                                                      right_targets):
            expand_targets_per_img = self.make_expand_targets_per_image(left_targets_per_image, right_targets_per_image)
            # print(anchors_per_image.bbox.device)
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, expand_targets_per_img, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.get_field('original_lr_bbox'), anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, left_targets, right_targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            left_targets (list[BoxList])
            right_targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, left_targets, right_targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness, box_regression = \
            concat_box_prediction_layers(objectness, box_regression, stereo=True)

        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        objectness_loss = F.cross_entropy(
            objectness[sampled_inds], labels[sampled_inds].long()
        )

        return objectness_loss, box_loss


# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_srpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = SRPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels
    )
    return loss_evaluator
