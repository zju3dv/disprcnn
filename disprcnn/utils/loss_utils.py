from torch import nn
import torch.nn.functional as F

class PSMLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, y):
        mask = y['mask']
        target = y['disparity']

        if isinstance(output, (list, tuple)) and len(output) == 3:
            training = True
        else:
            training = False
        if training:
            output1, output2, output3 = output
            loss1 = (F.smooth_l1_loss(output1, target, reduction='none') * mask.float()).sum()
            loss2 = (F.smooth_l1_loss(output2, target, reduction='none') * mask.float()).sum()
            loss3 = (F.smooth_l1_loss(output3, target, reduction='none') * mask.float()).sum()
            if mask.sum() != 0:
                loss1 = loss1 / mask.sum()
                loss2 = loss2 / mask.sum()
                loss3 = loss3 / mask.sum()
            loss = 0.5 * loss1 + 0.7 * loss2 + loss3
        else:
            if mask.sum() == 0:
                loss = 0
            else:
                loss = ((output - target).abs() * mask.float()).sum() / mask.sum()
        return loss
