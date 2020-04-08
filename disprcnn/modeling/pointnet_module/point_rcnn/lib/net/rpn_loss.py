import torch
import torch.nn.functional as F
import numpy as np
from ..utils.loss_utils import DiceLoss, SigmoidFocalClassificationLoss, get_reg_loss


class PointRCNNLossComputation(object):
    def __init__(self, cfg):
        # l, h, w = cfg.MEAN_SIZE[0]
        height, width, length = cfg.MEAN_SIZE[0]
        self.MEAN_SIZE = np.array([height, width, length])
        self.MEAN_SIZE = torch.from_numpy(self.MEAN_SIZE).cuda().float()

        self.cfg = cfg
        if self.cfg.RPN.LOSS_CLS == 'DiceLoss':
            self.rpn_cls_loss_func = DiceLoss(ignore_target=-1)
        elif self.cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            self.rpn_cls_loss_func = SigmoidFocalClassificationLoss(alpha=self.cfg.RPN.FOCAL_ALPHA[0],
                                                                    gamma=self.cfg.RPN.FOCAL_GAMMA)
        elif self.cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            self.rpn_cls_loss_func = F.binary_cross_entropy
        else:
            raise NotImplementedError

    def __call__(self, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, matched_idxs, tb_dict={}):
        # if isinstance(model, nn.DataParallel):
        #     rpn_cls_loss_func = model.module.rpn.rpn_cls_loss_func
        # else:
        #     rpn_cls_loss_func = model.rpn.rpn_cls_loss_func
        matched_idxs = matched_idxs.unsqueeze(-1).repeat(1, self.cfg.RPN.NPOINTS).view(-1) >= 0
        rpn_cls_label_flat = rpn_cls_label.view(-1)
        rpn_cls_flat = rpn_cls.view(-1)
        fg_mask = (rpn_cls_label_flat > 0)

        # RPN classification loss
        if self.cfg.RPN.LOSS_CLS == 'DiceLoss':
            rpn_loss_cls = self.rpn_cls_loss_func(rpn_cls, rpn_cls_label_flat)

        elif self.cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            rpn_cls_flat = rpn_cls_flat[matched_idxs]
            rpn_cls_label_flat = rpn_cls_label_flat[matched_idxs]
            rpn_cls_target = (rpn_cls_label_flat > 0).float()
            pos = (rpn_cls_label_flat > 0).float()
            neg = (rpn_cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)
            rpn_loss_cls = self.rpn_cls_loss_func(rpn_cls_flat, rpn_cls_target, cls_weights)
            # rpn_loss_cls = matched_idxs.float() * rpn_loss_cls
            rpn_loss_cls_pos = (rpn_loss_cls * pos).sum()
            rpn_loss_cls_neg = (rpn_loss_cls * neg).sum()
            rpn_loss_cls = rpn_loss_cls.sum()
            # if matched_idxs.float().sum() != 0:
            #     rpn_loss_cls = rpn_loss_cls / matched_idxs.float().sum()
            tb_dict['rpn_loss_cls_pos'] = rpn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rpn_loss_cls_neg.item()

        elif self.cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            weight = rpn_cls_flat.new(rpn_cls_flat.shape[0]).fill_(1.0)
            weight[fg_mask] = self.cfg.RPN.FG_WEIGHT
            rpn_cls_label_target = (rpn_cls_label_flat > 0).float()
            batch_loss_cls = self.rpn_cls_loss_func(torch.sigmoid(rpn_cls_flat), rpn_cls_label_target,
                                                    weight=weight, reduction='none')
            cls_valid_mask = (rpn_cls_label_flat >= 0).float()
            cls_valid_mask = cls_valid_mask * matched_idxs.float()
            rpn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        # RPN regression loss
        point_num = rpn_reg.size(0) * rpn_reg.size(1)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                get_reg_loss(rpn_reg.view(point_num, -1)[fg_mask],
                             rpn_reg_label.view(point_num, 7)[fg_mask],
                             loc_scope=self.cfg.RPN.LOC_SCOPE,
                             loc_bin_size=self.cfg.RPN.LOC_BIN_SIZE,
                             num_head_bin=self.cfg.RPN.NUM_HEAD_BIN,
                             anchor_size=self.MEAN_SIZE,
                             get_xz_fine=self.cfg.RPN.LOC_XZ_FINE,
                             get_y_by_bin=False,
                             get_ry_fine=False,
                             loss_mask=matched_idxs[fg_mask])

            loss_size = 3 * loss_size  # consistent with old codes
            rpn_loss_reg = loss_loc + loss_angle + loss_size
        else:
            loss_loc = loss_angle = loss_size = rpn_loss_reg = rpn_loss_cls * 0
        rpn_loss_cls = rpn_loss_cls * self.cfg.RPN.LOSS_WEIGHT[0]
        rpn_loss_reg = rpn_loss_reg * self.cfg.RPN.LOSS_WEIGHT[1]
        rpn_loss = rpn_loss_cls + rpn_loss_reg

        tb_dict.update({'rpn_loss_cls': rpn_loss_cls.item(),
                        'rpn_loss_reg': rpn_loss_reg.item(),
                        'rpn_loss': rpn_loss.item(),
                        'rpn_fg_sum': fg_sum,
                        'rpn_loss_loc': loss_loc.item(),
                        'rpn_loss_angle': loss_angle.item(),
                        'rpn_loss_size': loss_size.item()})
        loss_dict = {"rpn_loss_cls": rpn_loss_cls,
                     "rpn_loss_reg": rpn_loss_reg}
        # print('rpn_loss_cls',rpn_loss_cls.item(),'rpn_loss_reg',rpn_loss_reg.item())
        return loss_dict
