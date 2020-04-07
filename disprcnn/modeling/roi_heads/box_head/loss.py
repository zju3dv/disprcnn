# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List

import torch
from torch.nn import functional as F

from disprcnn.layers import smooth_l1_loss
from disprcnn.modeling.box_coder import BoxCoder
from disprcnn.modeling.matcher import Matcher
from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.boxlist_ops import boxlist_iou
from disprcnn.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from disprcnn.modeling.utils import cat
from disprcnn.utils.stereo_utils import expand_left_right_box, box6_to_box4s, retrive_left_right_proposals_from_joint


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
            self,
            proposal_matcher,
            fg_bg_sampler,
            box_coder,
            cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target, copy_fields=[]):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["labels"] + copy_fields)
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def prepare_double_view_targets(self, joint_proposals, joint_targets):
        joint_labels = []
        joint_regression_targets = []
        for joint_proposals_per_image, joint_targets_per_image in zip(joint_proposals, joint_targets):
            expand_proposals, original_proposal_box = expand_left_right_box(
                *box6_to_box4s(joint_proposals_per_image.bbox))
            expand_targets, original_target_box = expand_left_right_box(
                *box6_to_box4s(joint_targets_per_image.bbox)
            )
            joint_proposals_per_image = joint_proposals_per_image.clone()
            joint_proposals_per_image.bbox = expand_proposals
            joint_proposals_per_image.add_field('original_box', original_proposal_box)
            joint_targets_per_image = joint_targets_per_image.clone()
            joint_targets_per_image.bbox = expand_targets
            joint_targets_per_image.add_field('original_box', original_target_box)
            matched_targets = self.match_targets_to_proposals(
                joint_proposals_per_image, joint_targets_per_image, copy_fields=['original_box']
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.get_field('original_box'), joint_proposals_per_image.get_field('original_box')
            )

            joint_labels.append(labels_per_image)
            joint_regression_targets.append(regression_targets_per_image)

        return joint_labels, joint_regression_targets

    def subsample_single_view(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
                labels, regression_targets, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def subsample_double_view(self, left_proposals, right_proposals, left_targets, right_targets):
        joint_proposals, joint_targets = [], []
        for left_proposal, right_proposal in zip(left_proposals, right_proposals):
            joint_proposal = left_proposal.clone()
            joint_proposal.bbox = torch.cat((left_proposal.bbox, right_proposal.bbox[:, [0, 2]]), dim=1)
            joint_proposals.append(joint_proposal)
        for left_target, right_target in zip(left_targets, right_targets):
            joint_target = left_target.clone()
            joint_target.bbox = torch.cat((left_target.bbox, right_target.bbox[:, [0, 2]]), dim=1)
            joint_targets.append(joint_target)
        labels, joint_regression_targets = self.prepare_double_view_targets(joint_proposals, joint_targets)
        # labels, joint_regression_targets = self.prepare_targets(joint_proposals, joint_targets)
        # _, right_regression_targets = self.prepare_targets(right_proposals, right_targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        joint_proposals = list(joint_proposals)
        # right_proposals = list(right_proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, joint_regression_targets_per_image, joint_proposals_per_image in zip(
                labels, joint_regression_targets, joint_proposals
        ):
            joint_proposals_per_image.add_field("labels", labels_per_image)
            joint_proposals_per_image.add_field(
                "regression_targets", joint_regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            joint_proposals_per_image = joint_proposals[img_idx][img_sampled_inds]
            # right_proposals_per_image = right_proposals[img_idx][img_sampled_inds]
            joint_proposals[img_idx] = joint_proposals_per_image
            # right_proposals[img_idx] = right_proposals_per_image

        self._proposals = joint_proposals
        left_proposals, right_proposals = retrive_left_right_proposals_from_joint(joint_proposals)
        return left_proposals, right_proposals

    def subsample(self, proposals, targets):
        if not isinstance(proposals, dict):
            return self.subsample_single_view(proposals, targets)
        else:
            left_proposals, right_proposals = proposals['left'], proposals['right']
            left_targets, right_targets = targets['left'], targets['right']
            return self.subsample_double_view(left_proposals, right_proposals, left_targets, right_targets)

    def compute_single_view_loss(self, proposals, class_logits, device, box_regression):

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss

    def compute_double_view_loss(self, joint_proposals, class_logits, device, box_regression):
        # left_proposals, right_proposals = proposals['left'], proposals['right']
        # joint_proposals: List[BoxList] = []
        # for left_proposal, right_proposal in zip(left_proposals, right_proposals):
        #     left_regression_targets = left_proposal.get_field('regression_targets')
        #     right_regression_targets = right_proposal.get_field('regression_targets')
        #     joint_regression_targets = torch.cat((left_regression_targets, right_regression_targets[:, [1, 3]]), dim=1)
        #     joint_proposal = left_proposal.clone()
        #     joint_proposal.add_field('regression_targets', joint_regression_targets)
        #     joint_proposals.append(joint_proposal)

        labels = cat([proposal.get_field("labels") for proposal in joint_proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in joint_proposals], dim=0
        )

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 6 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3, 4, 5], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals
        if proposals[0].bbox.shape[1] == 4:
            # if not isinstance(proposals, dict):
            return self.compute_single_view_loss(proposals, class_logits, device, box_regression)
        else:
            return self.compute_double_view_loss(proposals, class_logits, device, box_regression)


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
