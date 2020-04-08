from torch.nn import functional as F

import torch.nn as nn
import torch
import numpy as np
from ..utils import loss_utils as loss_utils


class PointRCNNBox3dLossComputation(object):
    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif self.cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif self.cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(self.cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError
        h, w, l = cfg.MEAN_SIZE[0]
        self.MEAN_SIZE = np.array([h, w, l])
        self.MEAN_SIZE = torch.from_numpy(self.MEAN_SIZE).float().cuda()

    def __call__(self, end_points, proposals, labels, targets, matched_idxs=None, tb_dict={}):
        """
        :param end_points: {'rcnn_cls', 'rcnn_reg'}
        :param proposals:
        :param labels:{'sampled_pts',
                       'pts_feature',
                       'cls_label',
                       'reg_valid_mask',
                       'gt_of_rois',
                       'gt_iou',
                       'roi_boxes3d'}

        :param targets:
        :return:
        """
        if matched_idxs is not None:
            loss_mask = matched_idxs.unsqueeze(-1).repeat(1, self.cfg.RCNN.ROI_PER_IMAGE).view(-1) >= 0
        else:
            loss_mask = None
        rcnn_cls, rcnn_reg = end_points['rcnn_cls'], end_points['rcnn_reg']

        cls_label = labels['cls_label'].float()
        reg_valid_mask = labels['reg_valid_mask']
        roi_boxes3d = labels['roi_boxes3d']
        roi_size = roi_boxes3d[:, 3:6]
        gt_boxes3d_ct = labels['gt_of_rois']
        pts_input = labels['pts_input']

        cls_label_flat = cls_label.view(-1)

        if self.cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            # todo: add loss mask.
            rcnn_cls_flat = rcnn_cls.view(-1)

            cls_target = (cls_label_flat > 0).float()
            pos = (cls_label_flat > 0).float()
            neg = (cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)

            rcnn_loss_cls = self.cls_loss_func(rcnn_cls_flat, cls_target, cls_weights)
            rcnn_loss_cls_pos = (rcnn_loss_cls * pos).sum()
            rcnn_loss_cls_neg = (rcnn_loss_cls * neg).sum()
            rcnn_loss_cls = rcnn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rcnn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rcnn_loss_cls_neg.item()

        elif self.cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label, reduction='none')
            # cls_valid_mask = ((cls_label_flat >= 0) & loss_mask).float()
            cls_valid_mask = (cls_label_flat >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

        elif self.cfg.TRAIN.LOSS_CLS == 'CrossEntropy':
            rcnn_cls_reshape = rcnn_cls.view(rcnn_cls.shape[0], -1)
            cls_target = cls_label_flat.long()
            # cls_valid_mask = ((cls_label_flat >= 0) & loss_mask).float()

            batch_loss_cls = self.cls_loss_func(rcnn_cls_reshape, cls_target)
            normalizer = torch.clamp(cls_valid_mask.sum(), min=1.0)
            rcnn_loss_cls = (batch_loss_cls.mean(dim=1) * cls_valid_mask).sum() / normalizer

        else:
            raise NotImplementedError

        # rcnn regression loss
        batch_size = pts_input.shape[0]
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            all_anchor_size = roi_size
            anchor_size = all_anchor_size[fg_mask] if self.cfg.RCNN.SIZE_RES_ON_ROI else self.MEAN_SIZE
            anchor_size = anchor_size.to(all_anchor_size.device)
            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                loss_utils.get_reg_loss(rcnn_reg.view(batch_size, -1)[fg_mask],
                                        gt_boxes3d_ct.view(batch_size, 7)[fg_mask],
                                        loc_scope=self.cfg.RCNN.LOC_SCOPE,
                                        loc_bin_size=self.cfg.RCNN.LOC_BIN_SIZE,
                                        num_head_bin=self.cfg.RCNN.NUM_HEAD_BIN,
                                        anchor_size=anchor_size,
                                        get_xz_fine=True, get_y_by_bin=self.cfg.RCNN.LOC_Y_BY_BIN,
                                        loc_y_scope=self.cfg.RCNN.LOC_Y_SCOPE,
                                        loc_y_bin_size=self.cfg.RCNN.LOC_Y_BIN_SIZE,
                                        get_ry_fine=True,
                                        # loss_mask=loss_mask[fg_mask]
                                        )

            loss_size = 3 * loss_size  # consistent with old codes
            rcnn_loss_reg = loss_loc + loss_angle + loss_size
            tb_dict.update(reg_loss_dict)
        else:
            loss_loc = loss_angle = loss_size = rcnn_loss_reg = rcnn_loss_cls * 0

        rcnn_loss = rcnn_loss_cls + rcnn_loss_reg
        tb_dict['rcnn_loss_cls'] = rcnn_loss_cls.item()
        tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
        tb_dict['rcnn_loss'] = rcnn_loss.item()

        tb_dict['rcnn_loss_loc'] = loss_loc.item()
        tb_dict['rcnn_loss_angle'] = loss_angle.item()
        tb_dict['rcnn_loss_size'] = loss_size.item()

        tb_dict['rcnn_cls_fg'] = (cls_label > 0).sum().item()
        tb_dict['rcnn_cls_bg'] = (cls_label == 0).sum().item()
        tb_dict['rcnn_reg_fg'] = reg_valid_mask.sum().item()

        return rcnn_loss
