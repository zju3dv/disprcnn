# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode_x1y1x2y2(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def encode_x1y1x2y2x1px2p_fromboxes6(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_widthsp = proposals[:, 5] - proposals[:, 4] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights
        ex_ctr_xp = proposals[:, 4] + 0.5 * ex_widthsp

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_widths_p = reference_boxes[:, 5] - reference_boxes[:, 4] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights
        gt_ctr_x_p = reference_boxes[:, 4] + 0.5 * gt_widths_p

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets_dx_p = wx * (gt_ctr_x_p - ex_ctr_xp) / ex_widthsp
        targets_dw_p = ww * torch.log(gt_widths_p / ex_widthsp)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh,
                               targets_dx_p, targets_dw_p), dim=1)
        return targets

    def encode_x1y1x2y2x1px2p_fromboxes4(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_widths_p = reference_boxes[:, 5] - reference_boxes[:, 4] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights
        gt_ctr_x_p = reference_boxes[:, 4] + 0.5 * gt_widths_p

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets_dx_p = wx * (gt_ctr_x_p - ex_ctr_x) / ex_widths
        targets_dw_p = ww * torch.log(gt_widths_p / ex_widths)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh,
                               targets_dx_p, targets_dw_p), dim=1)
        return targets

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        if reference_boxes.shape[1] == 4:
            return self.encode_x1y1x2y2(reference_boxes, proposals)
        elif reference_boxes.shape[1] == 6 and proposals.shape[1] == 4:
            return self.encode_x1y1x2y2x1px2p_fromboxes4(reference_boxes, proposals)
        elif reference_boxes.shape[1] == 6 and proposals.shape[1] == 6:
            return self.encode_x1y1x2y2x1px2p_fromboxes6(reference_boxes, proposals)
        else:
            raise ValueError('size(1) is not 4 or 6.')

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        if rel_codes.shape[1] % 6 == 0 and boxes.shape[1] == 6:
            return self.decode_x1y1x2y2x1px2p_fromboxes6(rel_codes, boxes)
        elif rel_codes.shape[1] % 6 == 0 and boxes.shape[1] == 4:
            return self.decode_x1y1x2y2x1px2p_fromboxes4(rel_codes, boxes)
        elif rel_codes.shape[1] % 4 == 0:
            return self.decode_x1y1x2y2(rel_codes, boxes)
        else:
            raise ValueError('wrong shape.')

    def decode_x1y1x2y2(self, rel_codes, boxes):
        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    def decode_x1y1x2y2x1px2p_fromboxes4(self, rel_codes, boxes):
        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::6] / wx
        dy = rel_codes[:, 1::6] / wy
        dw = rel_codes[:, 2::6] / ww
        dh = rel_codes[:, 3::6] / wh
        dxp = rel_codes[:, 4::6] / wx
        dwp = rel_codes[:, 5::6] / ww
        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        dwp = torch.clamp(dwp, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_ctr_xp = dxp * widths[:, None] + ctr_x[:, None]
        pred_wp = torch.exp(dwp) * widths[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::6] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::6] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::6] = pred_ctr_x + 0.5 * pred_w
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::6] = pred_ctr_y + 0.5 * pred_h
        # x1p
        pred_boxes[:, 4::6] = pred_ctr_xp - 0.5 * pred_wp
        # x2p
        pred_boxes[:, 5::6] = pred_ctr_xp + 0.5 * pred_wp
        return pred_boxes

    def decode_x1y1x2y2x1px2p_fromboxes6(self, rel_codes, boxes):
        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        widthsp = boxes[:, 5] - boxes[:, 4] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_xp = boxes[:, 4] + 0.5 * widthsp
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::6] / wx
        dy = rel_codes[:, 1::6] / wy
        dw = rel_codes[:, 2::6] / ww
        dh = rel_codes[:, 3::6] / wh
        dxp = rel_codes[:, 4::6] / wx
        dwp = rel_codes[:, 5::6] / ww
        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        dwp = torch.clamp(dwp, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_ctr_xp = dxp * widthsp[:, None] + ctr_xp[:, None]
        pred_wp = torch.exp(dwp) * widthsp[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::6] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::6] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::6] = pred_ctr_x + 0.5 * pred_w
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::6] = pred_ctr_y + 0.5 * pred_h
        # x1p
        pred_boxes[:, 4::6] = pred_ctr_xp - 0.5 * pred_wp
        # x2p
        pred_boxes[:, 5::6] = pred_ctr_xp + 0.5 * pred_wp
        return pred_boxes
