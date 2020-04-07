import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from disprcnn.structures.boxlist_ops import boxlist_iou


def align_left_right_targets(left_targets, right_targets, thresh=0.2):
    iou_matrix = boxlist_iou(left_targets, right_targets)
    if iou_matrix.numel() > 0:
        max_ious, max_iou_idxs = iou_matrix.max(dim=1)
        high_quality = max_ious >= thresh
        matched_idxs = max_iou_idxs[high_quality]
        matched_right_targets = right_targets[matched_idxs]
        matched_left_targets = left_targets[high_quality]
    else:
        matched_left_targets = left_targets[[]]
        matched_right_targets = right_targets[[]]
    return matched_left_targets, matched_right_targets


def end_point_error(disp_target, mask, disp_pred):
    assert disp_target.shape == mask.shape == disp_pred.shape, f'{disp_target.shape, mask.shape, disp_pred.shape}'
    if mask.sum() == 0:
        return 0
    else:
        mask = mask.byte()
        if hasattr(mask, 'bool'):
            mask = mask.bool()
        return (disp_target[mask] - disp_pred[mask]).abs().mean().item()


def rmse(target, mask, pred):
    """
    :param target: HxW
    :param mask: HxW of {0,1}
    :param pred: HxW
    :return:
    """
    assert target.shape == mask.shape == pred.shape, f'{target.shape, mask.shape, pred.shape}'
    if mask.sum() == 0:
        return 0
    else:
        mask = mask.byte()
        if hasattr(mask, 'bool'):
            mask = mask.bool()
        return ((target[mask] - pred[mask]) ** 2).mean().sqrt().item()


def depth_end_point_error(disp_target, mask, disp_pred, fuxb):
    assert disp_target.shape == mask.shape == disp_pred.shape, f'{disp_target.shape, mask.shape, disp_pred.shape}'
    depth_target = fuxb / (disp_target + 1e-6)
    depth_pred = fuxb / (disp_pred + 1e-6)
    if mask.sum() == 0:
        return 0
    else:
        mask = mask.byte()
        if hasattr(mask, 'bool'):
            mask = mask.bool()
        return (depth_target[mask] - depth_pred[mask]).abs().mean().item()


def photometric_huber_loss(target, mask, pred, theta=1.0):
    assert target.shape == mask.shape == pred.shape, f'{target.shape, mask.shape, pred.shape}'
    if mask.sum() == 0:
        return 0
    else:
        mask = mask.byte()
        if hasattr(mask, 'bool'):
            mask = mask.bool()
        return huber(target[mask] - pred[mask], theta).mean().item()


def huber(x, theta=1.0):
    return x ** 2 / (x ** 2 + theta)


def expand_left_right_box(left_bbox: Tensor, right_bbox: Tensor):
    stacked_bbox = torch.stack((left_bbox, right_bbox), dim=-1)
    x1y1s, x2y2s = stacked_bbox.split((2, 2), dim=1)
    x1y1s = torch.min(x1y1s, dim=-1)[0]
    x2y2s = torch.max(x2y2s, dim=-1)[0]
    expand_bbox = torch.cat((x1y1s, x2y2s), dim=1)
    original_lr_bbox = torch.stack((left_bbox[:, 0], expand_bbox[:, 1],
                                    left_bbox[:, 2], expand_bbox[:, 3],
                                    right_bbox[:, 0], right_bbox[:, 2]), dim=1)
    return expand_bbox, original_lr_bbox


def box6_to_box4s(box6):
    return box6[:, 0:4], box6[:, [4, 1, 5, 3]]


def box4s_to_box6(box1, box2):
    return torch.cat((box1, box2[:, [0, 2]]), dim=1)


def retrive_left_right_proposals_from_joint(joint_proposals):
    left_proposals, right_proposals = [], []
    for joint_proposal in joint_proposals:
        left_proposal = joint_proposal.clone()
        right_proposal = joint_proposal.clone()
        lb, rb = box6_to_box4s(joint_proposal.bbox)
        left_proposal.bbox = lb
        right_proposal.bbox = rb
        left_proposals.append(left_proposal)
        right_proposals.append(right_proposal)
    return left_proposals, right_proposals


def disparity_mask(disparity, filtering=False):
    bs, _, h, w = disparity.shape

    grid_j = torch.arange(w).long().view(1, 1, -1).expand(bs, h, -1).cuda()  # bs x h x w
    grid_i = torch.arange(h).long().view(1, -1, 1).expand(bs, -1, w).cuda()
    grid_j_trans = grid_j - disparity.data[:, 0, :, :].long()

    grid_j_trans[grid_j_trans < 0] = 0
    grid_batch = torch.arange(bs).long().view(-1, 1, 1).expand(-1, h, w).cuda()  # bs x H x W
    grid_j_new = torch.zeros((bs, h, w)).long().cuda()

    for j in range(w):
        grid_j_new[grid_batch[:, :, j], grid_i[:, :, j], grid_j_trans[:, :, j]] = grid_j[:, :, j]

    grid_j_back = grid_j_new[grid_batch, grid_i, grid_j_trans]
    mask = (grid_j_back - grid_j).abs() <= 1
    mask = mask.unsqueeze(1)

    if filtering:
        mask = mask.float()
        mask = F.max_pool2d(mask, kernel_size=3, padding=1, stride=1)
        mask = -F.max_pool2d(-mask, kernel_size=3, padding=1, stride=1)
        mask = -F.max_pool2d(-mask, kernel_size=3, padding=1, stride=1)
        mask = F.max_pool2d(mask, kernel_size=3, padding=1, stride=1)

    return mask


class DisparityWarping(nn.Module):
    def __init__(self, occlusion_mask=0):
        super(DisparityWarping, self).__init__()
        self.occlusion_mask = occlusion_mask

    def forward(self, x, disparity):
        assert x.size()[-2:] == disparity.size()[-2:], "inconsistent shape between flow and source image"
        device = x.device
        n, c, h, w = x.size()
        x_ = torch.arange(w).view(1, -1).expand(h, -1)  # h,w
        y_ = torch.arange(h).view(-1, 1).expand(-1, w)  # h,w

        x_ = x_.unsqueeze(0).expand(1, -1, -1).float().to(device=device)  # 1,h,w
        y_ = y_.unsqueeze(0).expand(1, -1, -1).float().to(device=device)  # 1,h,w
        x_ = x_.unsqueeze(0).expand(n, -1, -1, -1)  # n,1,h,w
        y_ = y_.unsqueeze(0).expand(n, -1, -1, -1)  # n,1,h,w
        x_new = x_ - disparity  # n,1,h,w
        grid = torch.cat([x_new, y_], dim=1)  # n,2,h,w

        # if self.occlusion_mask:
        #     min_x = grid[:, 0, :, :].clone()
        #     for i in range(w - 2, -1, -1):
        #         min_x[:, :, i] = torch.min(min_x[:, :, i].clone(), min_x[:, :, i + 1].clone())
        #     mask_o = grid[:, 0, :, :] > min_x

        grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
        grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
        grid = grid.permute(0, 2, 3, 1)  # n,h,w,2
        mask = grid.abs() <= 1
        mask = mask.sum(dim=3)

        if self.occlusion_mask:
            mask_o = disparity_mask(disparity).detach()
            mask = mask + mask_o[:, 0, :, :].type_as(mask)
            mask = mask == 3  # this line performs the logical "and" operation for (x, y, occlusion)
        else:
            mask = mask == 2  # this line performs the logical "and" operation for (x, y)

        mask = mask.unsqueeze(1)
        warped_img = F.grid_sample(x, grid, padding_mode='zeros')
        return warped_img, mask


class EndPointErrorLoss(nn.Module):
    def forward(self, disp_target, disp_pred, mask=None):
        if mask is None:
            mask = torch.ones_like(disp_target).byte()
        if len(disp_pred) == 3:
            training = True
        else:
            training = False
        if training:
            output1, output2, output3 = disp_pred
            loss1 = (F.smooth_l1_loss(output1, disp_target, reduction='none') * mask.float()).sum()
            loss2 = (F.smooth_l1_loss(output2, disp_target, reduction='none') * mask.float()).sum()
            loss3 = (F.smooth_l1_loss(output3, disp_target, reduction='none') * mask.float()).sum()
            if mask.sum() != 0:
                loss1 = loss1 / mask.sum()
                loss2 = loss2 / mask.sum()
                loss3 = loss3 / mask.sum()
            loss = 0.5 * loss1 + 0.7 * loss2 + loss3
        else:
            output = disp_pred
            if mask.sum() == 0:
                loss = 0
            else:
                loss = ((output - disp_target).abs() * mask.float()).sum() / mask.sum()
        return loss


def compute_sparsity(disparity, semantic_mask=None):
    if semantic_mask is None:
        semantic_mask = torch.ones_like(disparity).byte()
    sparse_mask = disparity != 0
    s = (sparse_mask & semantic_mask).sum().item() / semantic_mask.sum().item()
    return s


def expand_box_to_integer(box):
    """
    :param box: x1,y1,x2,y2
    :return: floor(x1), floor(y1), ceil(x2), ceil(y2)
    """
    x1, y1, x2, y2 = box
    x1 = math.floor(x1)
    y1 = math.floor(y1)
    x2 = math.ceil(x2)
    y2 = math.ceil(y2)
    return x1, y1, x2, y2
