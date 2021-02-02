import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..rpn.proposal_layer import ProposalLayer
from ..pointnet2_lib.pointnet2 import pytorch_utils as pt_utils
from ..utils import loss_utils
from .rpn_loss import PointRCNNLossComputation
from ..net.pointnet2_msg import Pointnet2MSG


class RPN(nn.Module):
    def __init__(self, cfg, total_cfg, use_xyz=True):
        super().__init__()
        self.cfg = cfg
        self.total_cfg = total_cfg
        self.backbone_net = Pointnet2MSG(cfg, input_channels=0, use_xyz=use_xyz)

        # classification branch
        cls_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]
        for k in range(0, cfg.RPN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.CLS_FC[k], bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        if cfg.RPN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.rpn_cls_layer = nn.Sequential(*cls_layers)

        # regression branch
        per_loc_bin_num = int(cfg.RPN.LOC_SCOPE / cfg.RPN.LOC_BIN_SIZE) * 2
        if cfg.RPN.LOC_XZ_FINE:
            reg_channel = per_loc_bin_num * 4 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
        else:
            reg_channel = per_loc_bin_num * 2 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
        reg_channel = reg_channel + 1  # reg y

        reg_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]
        for k in range(0, cfg.RPN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.REG_FC[k], bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RPN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.rpn_reg_layer = nn.Sequential(*reg_layers)

        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            self.rpn_cls_loss_func = loss_utils.DiceLoss(ignore_target=-1)
        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            self.rpn_cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RPN.FOCAL_ALPHA[0],
                                                                               gamma=cfg.RPN.FOCAL_GAMMA)
        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            self.rpn_cls_loss_func = F.binary_cross_entropy
        else:
            raise NotImplementedError

        self.proposal_layer = ProposalLayer(cfg, total_cfg)
        self.init_weights()
        self.loss_evaluator = PointRCNNLossComputation(cfg)

    def init_weights(self):
        if self.cfg.RPN.LOSS_CLS in ['SigmoidFocalLoss']:
            pi = 0.01
            nn.init.constant_(self.rpn_cls_layer[2].conv.bias, -np.log((1 - pi) / pi))

        nn.init.normal_(self.rpn_reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _forward_train(self, pts_input, rpn_cls_label=None, rpn_reg_label=None, matched_targets=None):
        assert rpn_cls_label is not None and rpn_reg_label is not None and matched_targets is not None
        backbone_xyz, backbone_features = self.backbone_net(pts_input)  # (B, N, 3), (B, C, N)

        rpn_cls = self.rpn_cls_layer(backbone_features).transpose(1, 2).contiguous()  # (B, N, 1)
        rpn_reg = self.rpn_reg_layer(backbone_features).transpose(1, 2).contiguous()  # (B, N, C)

        ret_dict = {'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg,
                    'backbone_xyz': backbone_xyz, 'backbone_features': backbone_features}
        # training rpn stage
        if not self.cfg.RPN.FIXED:
            matched_idxs = torch.cat(
                [a.get_field('matched_idxs', default=torch.zeros((len(a)), dtype=torch.long).cuda()) for a in
                 matched_targets])
            rpn_loss = self.loss_evaluator(rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, matched_idxs)
            return ret_dict, rpn_loss
        else:
            # should not be here.
            raise NotImplementedError()

    def _forward_eval(self, pts_input):
        backbone_xyz, backbone_features = self.backbone_net(pts_input)  # (B, N, 3), (B, C, N)

        rpn_cls = self.rpn_cls_layer(backbone_features).transpose(1, 2).contiguous()  # (B, N, 1)
        rpn_reg = self.rpn_reg_layer(backbone_features).transpose(1, 2).contiguous()  # (B, N, C)

        ret_dict = {'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg,
                    'backbone_xyz': backbone_xyz, 'backbone_features': backbone_features}
        with torch.no_grad():
            rpn_scores_raw = rpn_cls[:, :, 0]
            rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
            seg_mask = (rpn_scores_norm > self.cfg.RPN.SCORE_THRESH).float()
            pts_depth = torch.norm(backbone_xyz, p=2, dim=2)

            # proposal layer
            rois, roi_scores_raw = self.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)

        rcnn_input_info = {'rpn_xyz': backbone_xyz,
                           'rpn_features': backbone_features.permute((0, 2, 1)),
                           'seg_mask': seg_mask,
                           'roi_boxes3d': rois,
                           'roi_scores_raw': roi_scores_raw,
                           'pts_depth': pts_depth}
        ret_dict.update(rcnn_input_info)

        return ret_dict, {}

    def forward(self, pts_input, rpn_cls_label=None, rpn_reg_label=None, matched_targets=None):
        if self.training:
            return self._forward_train(pts_input, rpn_cls_label, rpn_reg_label, matched_targets)
        else:
            return self._forward_eval(pts_input)
