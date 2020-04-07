# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from disprcnn.modeling.box_coder import BoxCoder
from disprcnn.modeling.rpn.utils import permute_and_flatten
from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.boxlist_ops import cat_boxlist, double_view_boxlist_nms
from disprcnn.structures.boxlist_ops import boxlist_nms
from disprcnn.structures.boxlist_ops import remove_small_boxes


class SRPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
            self,
            pre_nms_top_n,
            post_nms_top_n,
            nms_thresh,
            min_size,
            box_coder=None,
            fpn_post_nms_top_n=None,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super().__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 6, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape

        # put in the same format as anchors
        objectness = permute_and_flatten(objectness, N, A, 2, H, W)[:, :, 1]
        # objectness = objectness.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 6, H, W)

        num_anchors = A * H * W

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(min(pre_nms_top_n, objectness.shape[1]), dim=1, sorted=True)

        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx]

        image_shapes = [box.size for box in anchors]
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

        proposals = self.box_coder.decode(
            box_regression.view(-1, 6), concat_anchors.view(-1, 4)
        )

        proposals = proposals.view(N, -1, 6)

        left_result, right_result = [], []
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            left_boxlist = BoxList(proposal[:, 0:4], im_shape, mode="xyxy")
            right_boxlist = BoxList(proposal[:, [4, 1, 5, 3]], im_shape, mode='xyxy')

            left_boxlist.add_field("objectness", score)
            right_boxlist.add_field("objectness", score)
            left_boxlist = left_boxlist.clip_to_image(remove_empty=False)
            right_boxlist = right_boxlist.clip_to_image(remove_empty=False)
            left_boxlist = remove_small_boxes(left_boxlist, self.min_size)
            right_boxlist = remove_small_boxes(right_boxlist, self.min_size)
            left_boxlist, right_boxlist = double_view_boxlist_nms(left_boxlist, right_boxlist, self.nms_thresh,
                                                                  max_proposals=self.post_nms_top_n,
                                                                  score_field='objectness')
            left_result.append(left_boxlist)
            right_result.append(right_boxlist)
        return left_result, right_result

    def forward(self, anchors, objectness, box_regression, left_targets=None, right_targets=None):
        device = objectness[0].device
        scores = []
        for i, score in enumerate(objectness):
            scores.append(score.permute(0, 2, 3, 1).contiguous().view(score.shape[0], -1, 2))
        scores = torch.cat(scores, 1)[:, :, 1]
        bbox_regs = []
        for i, bbox_reg in enumerate(box_regression):
            bbox_regs.append(bbox_reg.permute(0, 2, 3, 1).contiguous().view(bbox_reg.shape[0], -1, 6))
        bbox_regs = torch.cat(bbox_regs, 1)
        anchors = list(zip(*anchors))
        combined_anchors = []
        batch_size = len(anchors[0])
        for i in range(batch_size):
            combined_anchors.append(cat_boxlist([anchors[level][i] for level in range(len(anchors))]))
        num_anchors = len(combined_anchors[0])
        # pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        # scores, topk_idx = scores.topk(min(pre_nms_top_n, scores.shape[1]), dim=1, sorted=True)

        # batch_idx = torch.arange(bsz, device=device)[:, None]
        # bbox_regs = bbox_regs[batch_idx, topk_idx]

        image_shapes = [box.size for box in combined_anchors]
        # concat_anchors = torch.cat([a.bbox for a in combined_anchors], dim=0)
        # concat_anchors = concat_anchors.reshape(bsz, -1, 4)[batch_idx, topk_idx]

        proposals = self.box_coder.decode(
            bbox_regs.view(-1, 6), torch.cat([a.bbox.view(-1, 4) for a in combined_anchors]).to(device)
        )

        proposals = proposals.view(batch_size, -1, 6)
        proposals_left = proposals[:, :, 0:4]
        proposals_right = proposals[:, :, [4, 1, 5, 3]]
        proposals_left = clip_boxes(proposals_left, image_shapes, batch_size)
        proposals_right = clip_boxes(proposals_right, image_shapes, batch_size)
        scores_keep = scores
        proposals_keep_left = proposals_left
        proposals_keep_right = proposals_right

        _, order = torch.sort(scores_keep, 1, True)

        left_result, right_result = [], []
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single_left = proposals_keep_left[i]
            proposals_single_right = proposals_keep_right[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if self.pre_nms_top_n > 0 and self.pre_nms_top_n < scores_keep.numel():
                order_single = order_single[:self.pre_nms_top_n]

            proposals_single_left = proposals_single_left[order_single, :]
            proposals_single_right = proposals_single_right[order_single, :]
            scores_single = scores_single[order_single].view(-1, 1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            left_boxlist = BoxList(proposals_single_left, image_shapes[i], mode="xyxy")
            right_boxlist = BoxList(proposals_single_right, image_shapes[i], mode='xyxy')

            left_boxlist.add_field("objectness", scores_single.squeeze(1))
            right_boxlist.add_field("objectness", scores_single.squeeze(1))
            left_boxlist = left_boxlist.clip_to_image(remove_empty=False)
            right_boxlist = right_boxlist.clip_to_image(remove_empty=False)
            left_boxlist = remove_small_boxes(left_boxlist, self.min_size)
            right_boxlist = remove_small_boxes(right_boxlist, self.min_size)
            left_boxlist, right_boxlist = double_view_boxlist_nms(left_boxlist, right_boxlist, self.nms_thresh,
                                                                  max_proposals=self.post_nms_top_n,
                                                                  score_field='objectness')
            left_result.append(left_boxlist)
            right_result.append(right_boxlist)
        return left_result, right_result

        # left_sampled_boxes, right_sampled_boxes = [], []
        # # num_levels = len(objectness)
        # for a, o, b in zip(anchors, objectness, box_regression):
        #     sampled_box = self.forward_for_single_feature_map(a, o, b)
        #     left_sampled_boxes.append(sampled_box[0])
        #     right_sampled_boxes.append(sampled_box[1])
        #
        # left_boxlists = list(zip(*left_sampled_boxes))
        # right_boxlists = list(zip(*right_sampled_boxes))
        # left_boxlists = [cat_boxlist(boxlist) for boxlist in left_boxlists]
        # right_boxlists = [cat_boxlist(boxlist) for boxlist in right_boxlists]
        # for i, (l, r) in enumerate(zip(left_boxlists, right_boxlists)):
        #     l, r = double_view_boxlist_nms(l, r, self.nms_thresh,
        #                                    max_proposals=self.post_nms_top_n,
        #                                    score_field='objectness')
        #     left_boxlists[i] = l
        #     right_boxlists[i] = r
        # if num_levels > 1:
        #     left_boxlists = self.select_over_all_levels(left_boxlists)
        #     right_boxlists = self.select_over_all_levels(right_boxlists)
        #
        # # append ground-truth bboxes to proposals
        # if self.training and left_targets is not None:
        #     left_boxlists = self.add_gt_proposals(left_boxlists, left_targets)
        #     right_boxlists = self.add_gt_proposals(right_boxlists, right_targets)
        #
        # return left_boxlists, right_boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # TODO resolve this difference and make it consistent. It should be per image,
        # and not per batch
        if self.training:
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


def make_srpn_postprocessor(config, rpn_box_coder, is_train):
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST

    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = SRPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
    )
    return box_selector


def clip_boxes(boxes, im_shape, batch_size):
    """

    :param boxes:
    :param im_shape: W,H
    :param batch_size:
    :return:
    """
    for i in range(batch_size):
        boxes[i, :, 0::4].clamp_(0, im_shape[i][0] - 1)
        boxes[i, :, 1::4].clamp_(0, im_shape[i][1] - 1)
        boxes[i, :, 2::4].clamp_(0, im_shape[i][0] - 1)
        boxes[i, :, 3::4].clamp_(0, im_shape[i][1] - 1)

    return boxes
