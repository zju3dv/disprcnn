import numpy as np
import torch
import torch.nn as nn

from disprcnn.modeling.matcher import Matcher
from disprcnn.modeling.roi_heads.mask_head.inference import Masker
from disprcnn.structures.bounding_box_3d import Box3DList
from disprcnn.structures.boxlist_ops import boxlist_iou
from disprcnn.structures.calib import Calib
from disprcnn.structures.disparity import DisparityMap
from disprcnn.structures.segmentation_mask import SegmentationMask
from disprcnn.utils.stereo_utils import expand_box_to_integer
from disprcnn.utils.utils_3d import rotate_pc_along_y
from .rcnn_net import RCNNNet
from .rpn import RPN


class PointRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.MODEL.POINTRCNN
        assert self.cfg.RPN.ENABLED or self.cfg.RCNN.ENABLED
        self.masker_threshold = self.cfg.MASK_THRESH
        if self.cfg.RPN.ENABLED:
            self.rpn = RPN(self.cfg, self.total_cfg)

        if self.cfg.RCNN.ENABLED:
            self.rcnn_net = RCNNNet(self.cfg, self.total_cfg)

        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
            cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
            allow_low_quality_matches=False,
        )

    def back_project(self, depth_maps, mask_pred, targets, max_depth=160, fix_seed=False):
        for i in range(len(depth_maps)):
            mask_pred_per_img = mask_pred[i]
            assert depth_maps[i].shape[0] == mask_pred_per_img.shape[0]
            for j in range(depth_maps[i].shape[0]):
                if mask_pred_per_img[j].sum() != 0 and (depth_maps[i][j] * mask_pred_per_img[j].float()).max() > 0:
                    depth_maps[i][j] = depth_maps[i][j] * mask_pred_per_img[j].float()
            assert torch.isnan(depth_maps[i]).sum() == 0
        # calib
        pts_batch = []
        for i in range(len(depth_maps)):
            calib: Calib = targets[i].get_field('calib')
            for j in range(len(depth_maps[i])):
                pts_batch.append(calib.depthmap_to_rect(depth_maps[i][j])[0])
        pt = [p.t() for p in pts_batch]
        # pt = torch.stack(pts_batch).permute(0, 2, 1)
        pts_list = []
        for pt_per_roi in pt:
            pos_indices = torch.nonzero(pt_per_roi[2, :] > 0).squeeze(1)
            # skip cases when pos_indices is empty
            RANDOM_SEED = 0
            if len(pos_indices) > 0:
                if len(pos_indices) > self.cfg.RPN.NPOINTS:
                    if fix_seed:
                        np.random.seed(RANDOM_SEED)
                    choice = np.random.choice(len(pos_indices),
                                              self.cfg.RPN.NPOINTS, replace=False)
                else:
                    # print('len(pos_indices) =', len(pos_indices))
                    if fix_seed:
                        np.random.seed(RANDOM_SEED)
                    choice = np.random.choice(len(pos_indices),
                                              self.cfg.RPN.NPOINTS - len(pos_indices), replace=True)
                    choice = np.concatenate((np.arange(len(pos_indices)), choice))
                if fix_seed:
                    np.random.seed(RANDOM_SEED)
                np.random.shuffle(choice)
                indices = pos_indices[choice]

                pt_per_roi = pt_per_roi[:, indices]
                pts_list.append(pt_per_roi)
            else:
                # pass
                raise EOFError('mask is nonvalid')
        pts = torch.stack(pts_list).cuda()
        pts = pts.permute(0, 2, 1).contiguous()
        # clip depth to reasonable number
        pts[:, :, 2] = torch.clamp(pts[:, :, 2].clone(), max=max_depth)
        return pts

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        if len(proposal) == 0:
            return target[[]]
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(["box3d", "calib"])
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def process_input(self, left_inputs, right_inputs, targets, threshold=0.7, padding=1):
        left_inputs, right_inputs = remove_empty_proposals(left_inputs, right_inputs)
        left_inputs, right_inputs = remove_too_right_proposals(left_inputs, right_inputs)
        depth_maps = []
        mask_pred_list = []

        matched_targets = []
        fus = []
        for left_prediction, right_prediction, target_per_image in zip(left_inputs, right_inputs, targets):
            if len(target_per_image) != 0:
                matched_target = self.match_targets_to_proposals(left_prediction, target_per_image)
                matched_targets.append(matched_target)
            else:
                continue
            left_bbox = left_prediction.bbox
            right_bbox = right_prediction.bbox
            disparity_or_depth_preds = left_prediction.get_field('disparity')
            masks = left_prediction.get_field('mask')
            masker = Masker(threshold=threshold, padding=padding)
            mask_pred = masker([masks], [left_prediction])[0].squeeze(1)
            num_rois = len(left_bbox)

            fus.extend([target_per_image.get_field('calib').calib.fu for _ in range(num_rois)])
            depth_maps_per_img = []
            # mask_preds_per_img = []
            if num_rois != 0:
                for left_roi, right_roi, disp_or_depth_roi, maskp in zip(left_bbox, right_bbox,
                                                                         disparity_or_depth_preds, mask_pred):
                    x1, y1, x2, y2 = expand_box_to_integer(left_roi.tolist())
                    x1p, _, x2p, _ = expand_box_to_integer(right_roi.tolist())
                    depth_map_per_roi = torch.zeros((left_prediction.height, left_prediction.width)).cuda()
                    disp_roi = DisparityMap(disp_or_depth_roi).resize(
                        (max(x2 - x1, x2p - x1p), y2 - y1)).crop(
                        (0, 0, x2 - x1, y2 - y1)).data
                    disp_roi = disp_roi + x1 - x1p
                    depth_roi = target_per_image.get_field('calib').stereo_fuxbaseline / (disp_roi + 1e-6)
                    depth_map_per_roi[y1:y2, x1:x2] = depth_roi
                    depth_maps_per_img.append(depth_map_per_roi)
                depth_maps.append(depth_maps_per_img)
                mask_pred_list.append(mask_pred.cuda())
        depth_full_image = [torch.stack(d) for d in depth_maps]
        mask_pred_all = mask_pred_list
        pts = self.back_project(depth_full_image, mask_pred_all, targets)
        fus = torch.tensor(fus).cuda()
        gt_box3d_xyzhwlry = torch.cat(
            [t.get_field('box3d').convert('xyzhwl_ry').bbox_3d.view(-1, 7) for t in matched_targets])
        # aug
        # scale
        if not self.cfg.RPN.FIXED:
            scale = np.random.uniform(0.95, 1.05)
            pts = pts * scale

            gt_box3d_xyzhwlry[:, 0:6] = gt_box3d_xyzhwlry[:, 0:6] * scale
        # flip
        if not self.cfg.RPN.FIXED:
            do_flip = np.random.random() < 0.5
        else:
            do_flip = False
        if do_flip:
            pts[:, :, 0] = -pts[:, :, 0]
            gt_box3d_xyzhwlry[:, 0] = -gt_box3d_xyzhwlry[:, 0]
            gt_box3d_xyzhwlry[:, 6] = torch.sign(gt_box3d_xyzhwlry[:, 6]) * np.pi - gt_box3d_xyzhwlry[:, 6]
            # rotate
            self.rotator = rotate_pc_along_y(left_inputs, fus)
            self.rotator.rot_angle *= -1
        else:
            # rotate
            self.rotator = rotate_pc_along_y(left_inputs, fus)

        gt_box3d_xyzhwlry_batch_splited = torch.split(gt_box3d_xyzhwlry, [len(b) for b in matched_targets])
        for i in range(len(matched_targets)):
            matched_targets[i].extra_fields['box3d'] = matched_targets[i].extra_fields['box3d'].convert('xyzhwl_ry')
            matched_targets[i].extra_fields['box3d'].bbox_3d = gt_box3d_xyzhwlry_batch_splited[i]
        # rotate
        pts = self.rotator.__call__(pts.permute(0, 2, 1)).permute(0, 2, 1)
        target_corners = self.rotator.__call__(
            torch.cat(
                [t.get_field('box3d').convert('corners').bbox_3d.view(-1, 8, 3).permute(0, 2, 1) for t in
                 matched_targets])).permute(0, 2, 1)
        # translate
        pts_mean = pts.mean(1)
        self.pts_mean = pts_mean
        pts = pts - pts_mean[:, None, :]
        target_corners = target_corners - pts_mean[:, None, :]
        target_corners_splited = torch.split(target_corners, [len(b) for b in matched_targets])
        for i in range(len(matched_targets)):
            matched_targets[i].extra_fields['box3d'] = matched_targets[i].extra_fields['box3d'].convert('corners')
            matched_targets[i].extra_fields['box3d'].bbox_3d = target_corners_splited[i].contiguous().view(-1, 24)

        cls_label, reg_label = generate_rpn_training_labels(pts, matched_targets)
        return pts, cls_label, reg_label, matched_targets

    def process_input_eval(self, left_inputs, right_inputs, targets, threshold=0.7, padding=1):
        depth_maps = []
        mask_pred_list = []
        fus = []
        for left_prediction, right_prediction, target in zip(left_inputs, right_inputs, targets):
            left_bbox = left_prediction.bbox
            right_bbox = right_prediction.bbox
            disparity_preds = left_prediction.get_field('disparity')
            masks = left_prediction.get_field('mask')
            masker = Masker(threshold=threshold, padding=padding)
            mask_pred = masker([masks], [left_prediction])[0].squeeze(1)
            # assert len(left_bbox) == len(right_bbox) == len(
            #     disparity_preds), f'{len(left_bbox), len(right_bbox), len(disparity_preds)}'
            num_rois = len(left_bbox)
            fus.extend([target.get_field('calib').calib.fu for _ in range(num_rois)])
            depth_maps_per_img = []
            disparity_maps_per_img = []
            if num_rois != 0:
                for left_roi, right_roi, disp_or_depth_roi, mask_p in zip(left_bbox, right_bbox,
                                                                          disparity_preds, mask_pred):
                    x1, y1, x2, y2 = expand_box_to_integer(left_roi.tolist())
                    x1p, _, x2p, _ = expand_box_to_integer(right_roi.tolist())
                    depth_map_per_roi = torch.zeros((left_prediction.height, left_prediction.width)).cuda()
                    disparity_map_per_roi = torch.zeros_like(depth_map_per_roi)
                    mask = mask_p.squeeze(0)
                    disp_roi = DisparityMap(disp_or_depth_roi).resize(
                        (max(x2 - x1, x2p - x1p), y2 - y1)).crop(
                        (0, 0, x2 - x1, y2 - y1)).data
                    disp_roi = disp_roi + x1 - x1p
                    depth_roi = target.get_field('calib').stereo_fuxbaseline / (disp_roi + 1e-6)
                    depth_map_per_roi[y1:y2, x1:x2] = depth_roi.clamp(min=1.0)
                    disparity_map_per_roi[y1:y2, x1:x2] = disp_roi
                    disparity_map_per_roi = disparity_map_per_roi * mask.float().cuda()
                    depth_maps_per_img.append(depth_map_per_roi)
                    disparity_maps_per_img.append(disparity_map_per_roi)
                if len(depth_maps_per_img) != 0:
                    depth_maps_per_img = torch.stack(depth_maps_per_img)
                    disparity_maps_per_img = torch.stack(disparity_maps_per_img).sum(dim=0)
                else:
                    depth_maps_per_img = torch.zeros((1, left_prediction.height, left_prediction.width))
                    disparity_maps_per_img = torch.zeros((left_prediction.height, left_prediction.width))
                depth_maps.append(depth_maps_per_img)
                mask_pred_list.append(mask_pred.cuda())
        if len(depth_maps) != 0:
            fus = torch.tensor(fus).cuda()
            self.rotator = rotate_pc_along_y(left_inputs, fus)
            pts = self.back_project(depth_maps, mask_pred_list, targets=targets, fix_seed=True)
            pts = self.rotator.__call__(pts.permute(0, 2, 1)).permute(0, 2, 1)
            pts_mean = pts.mean(1)
            self.pts_mean = pts_mean
            pts = pts - pts_mean[:, None, :]
        else:
            pts = torch.empty((0, 768, 3)).cuda()
        return pts

    def _forward_train(self, left_inputs, right_inputs, targets):
        pts_input, rpn_cls_label, \
        rpn_reg_label, matched_targets = self.process_input(
            left_inputs, right_inputs, targets, self.masker_threshold)
        # norm
        loss_dict = {}
        with torch.set_grad_enabled((not self.cfg.RPN.FIXED) and self.training):
            if self.cfg.RPN.FIXED:
                self.rpn.eval()
            proposals, proposals_losses = self.rpn(pts_input, rpn_cls_label,
                                                   rpn_reg_label, matched_targets)

            loss_dict.update(proposals_losses)
        if self.cfg.RCNN.ENABLED and self.cfg.RCNN.TRAIN:
            pts_input = pts_input + self.pts_mean[:, None, :]
            pts_input = self.rotator.rotate_back(pts_input.permute(0, 2, 1)).permute(0, 2, 1)
            matched_targets[0].extra_fields['box3d'] = matched_targets[0].extra_fields['box3d'].convert('corners')
            matched_targets[0].extra_fields['box3d'].bbox_3d = (self.rotator.rotate_back(
                (matched_targets[0].extra_fields['box3d'].bbox_3d.view(-1, 8, 3) +
                 self.pts_mean[:, None, :]).permute(0, 2, 1)
            ).permute(0, 2, 1)).reshape(-1, 24)
            proposals['backbone_xyz'] = self.rotator.rotate_back(
                (proposals['backbone_xyz'] + self.pts_mean[:, None, :]).permute(0, 2, 1)
            ).permute(0, 2, 1)
            proposals['pts_depth'] = torch.norm(proposals['backbone_xyz'], p=2, dim=2)
            proposals['rpn_xyz'] = self.rotator.rotate_back(
                (proposals['rpn_xyz'] + self.pts_mean[:, None, :]).permute(0, 2, 1)
            ).permute(0, 2, 1)
            corners = Box3DList(proposals['roi_boxes3d'].view(-1, 7), (1, 1), 'xyzhwl_ry').convert(
                'corners').bbox_3d.view(
                pts_input.shape[0], -1, 24)  # numroi,?,24
            proposals['roi_boxes3d'] = Box3DList((self.rotator.rotate_back(
                (corners.view(pts_input.shape[0], -1, 3) +
                 self.pts_mean[:, None, :]).permute(0, 2, 1)).permute(0, 2, 1)
                                                  ).contiguous(),
                                                 (1, 1),
                                                 'corners').convert('xyzhwl_ry').bbox_3d.view(pts_input.shape[0], -1, 7)
            proposals, matched_targets = filter_unmatched_idxs(proposals, matched_targets)
            proposals, rcnn_loss = self.rcnn_net(proposals, matched_targets)
            loss_dict.update(rcnn_loss)
        return proposals, loss_dict

    def _forward_val(self, left_results, right_results, targets):
        left_results, right_results = remove_empty_proposals(left_results, right_results)
        pts_input = self.process_input_eval(left_results, right_results, targets,
                                            threshold=0.5)
        if pts_input.numel() != 0:
            rpn_proposals, proposals_losses = self.rpn(pts_input)
            pts_input = self.rotator.rotate_back(
                (pts_input + self.pts_mean[:, None, :]).permute(0, 2, 1)
            ).permute(0, 2, 1).contiguous()
            if hasattr(self, 'rcnn_net'):
                rpn_proposals['backbone_xyz'] = self.rotator.rotate_back(
                    (rpn_proposals['backbone_xyz'] + self.pts_mean[:, None, :]).permute(0, 2, 1)
                ).permute(0, 2, 1)
                rpn_proposals['pts_depth'] = torch.norm(rpn_proposals['backbone_xyz'], p=2, dim=2)
                rpn_proposals['rpn_xyz'] = self.rotator.rotate_back(
                    (rpn_proposals['rpn_xyz'] + self.pts_mean[:, None, :]).permute(0, 2, 1)
                ).permute(0, 2, 1)
                corners = Box3DList(rpn_proposals['roi_boxes3d'].view(-1, 7), (1, 1), 'xyzhwl_ry').convert(
                    'corners').bbox_3d.view(
                    pts_input.shape[0], -1, 24)  # numroi,?,24
                rpn_proposals['roi_boxes3d'] = Box3DList((self.rotator.rotate_back(
                    (corners.view(pts_input.shape[0], -1, 3) + self.pts_mean[:, None, :]
                     ).permute(0, 2, 1)).permute(0, 2, 1)).contiguous(),
                                                         (1, 1),
                                                         'corners').convert('xyzhwl_ry'
                                                                            ).bbox_3d.view(pts_input.shape[0],
                                                                                           -1, 7)
                proposals, rcnn_loss = self.rcnn_net(rpn_proposals)
                left_results = combine_2d_3d(left_results, proposals)
            else:
                # inference with rpn
                box3d = rpn_proposals['roi_boxes3d']  # 64,7
                bsz = box3d.shape[0]
                box3d[:, :, 0:3] = box3d[:, :, 0:3] + self.pts_mean[:, None, :]
                corners_rot_back = self.rotator.rotate_back(
                    Box3DList(box3d.reshape(-1, 7), size=(1, 1), mode='xyzhwl_ry'
                              ).convert('corners').bbox_3d.view(bsz, -1, 3).permute(0, 2, 1)
                ).permute(0, 2, 1).view(bsz, -1, 8, 3)

                # score_3d = rpn_proposals['roi_scores_raw'][:, 0]
                score_3d = rpn_proposals['roi_scores_raw']
                ss, bs = [], []
                for crb, s3d in zip(corners_rot_back, score_3d):
                    idx = torch.argmax(s3d)
                    ss.append(s3d[idx])
                    bs.append(crb[idx])
                ss = torch.tensor(ss)
                bs = torch.stack(bs)
                bs = Box3DList(bs.reshape(-1, 24), (1, 1), 'corners').convert('xyzhwl_ry')
                left_results[0].add_field('box3d', bs)
                left_results[0].add_field('scores_3d', ss.cpu())
                rpn_proposals['roi_boxes3d'] = bs.bbox_3d.unsqueeze(0)
        else:
            for left_result in left_results:
                left_result.add_field('box3d',
                                      Box3DList(torch.empty((0, 7)), size=left_result.size, mode='ry_lhwxyz'))
                left_result.add_field('scores_3d', torch.empty((0)))
        return left_results, right_results, {}

    def forward(self, left_inputs, right_inputs, targets=None):
        if self.training and targets is None:
            raise ValueError('in training mode, targets must not be None.')
        if self.training:
            return self._forward_train(left_inputs, right_inputs, targets)
        else:
            return self._forward_val(left_inputs, right_inputs, targets)


def remove_empty_proposals(left_results, right_results):
    ret_left_results, ret_right_results = [], []
    for left_result, right_result in zip(left_results, right_results):
        left_keep = (left_result.bbox[:, 2] > left_result.bbox[:, 0] + 1) & \
                    (left_result.bbox[:, 3] > left_result.bbox[:, 1] + 1)
        right_keep = (right_result.bbox[:, 2] > right_result.bbox[:, 0] + 1) & \
                     (right_result.bbox[:, 3] > right_result.bbox[:, 1] + 1)
        keep = left_keep & right_keep
        left_result = left_result[keep]
        right_result = right_result[keep]
        ret_left_results.append(left_result)
        ret_right_results.append(right_result)
    return ret_left_results, ret_right_results


def remove_too_right_proposals(left_results, right_results):
    ret_left_results, ret_right_results = [], []
    for left_result, right_result in zip(left_results, right_results):
        keep = (left_result.bbox[:, 0] > right_result.bbox[:, 0]) | \
               (left_result.bbox[:, 0] == 0)
        left_result = left_result[keep]
        right_result = right_result[keep]
        ret_left_results.append(left_result)
        ret_right_results.append(right_result)
    return ret_left_results, ret_right_results


def filter_bbox_3d(bbox_3d, point):
    v45 = bbox_3d[5] - bbox_3d[4]
    v40 = bbox_3d[0] - bbox_3d[4]
    v47 = bbox_3d[7] - bbox_3d[4]
    point = point - bbox_3d[4]
    m0 = torch.matmul(point, v45)
    m1 = torch.matmul(point, v40)
    m2 = torch.matmul(point, v47)

    cs = []
    for m, v in zip([m0, m1, m2], [v45, v40, v47]):
        c0 = 0 < m
        c1 = m < torch.matmul(v, v)
        c = c0 & c1
        cs.append(c)
    cs = cs[0] & cs[1] & cs[2]
    passed_inds = torch.nonzero(cs).squeeze(1)
    num_passed = torch.sum(cs)
    return num_passed, passed_inds, cs


def generate_rpn_training_labels(pts, targets):
    gt_boxes3d = torch.cat([target.get_field('box3d').convert('xyzhwl_ry').bbox_3d for target in targets])
    gt_corners = torch.cat(
        [target.get_field('box3d').convert('corners').bbox_3d.view(-1, 8, 3) for target in targets])
    extend_box_corners = torch.cat(
        [target.get_field('box3d').enlarge_box3d(0.2).convert('corners').bbox_3d.view(-1, 8, 3) for target in
         targets])
    cls_label = torch.zeros((pts.shape[0], pts.shape[1])).to(pts.device)
    reg_label = torch.zeros((pts.shape[0], pts.shape[1], 7)).to(pts.device)  # dx, dy, dz, ry, h, w, l
    for k in range(gt_boxes3d.shape[0]):
        _, _, cs = filter_bbox_3d(gt_corners[k], pts[k])
        fg_pts_rect = pts[k][cs]
        cls_label[k][cs] = 1
        _, _, enlarged_cs = filter_bbox_3d(extend_box_corners[k], pts[k])
        # enlarged_cs = enlarged_cs[0]
        # enlarge the bbox3d, ignore nearby points
        ignore_flag = ~(enlarged_cs == cs)
        cls_label[k][ignore_flag] = -1
        gt_box3d = gt_boxes3d[k]  # xyzhwl_ry
        # pixel offset of object center
        center3d = gt_box3d[0:3].clone()  # (x, y, z)
        center3d[1] = center3d[1] - gt_box3d[3] / 2
        reg_label[k][cs, 0:3] = center3d.unsqueeze(0) - fg_pts_rect.squeeze(
            0)  # Now y is the true center of 3d box 20180928

        # size and angle encoding
        reg_label[k][cs, 3] = gt_box3d[3]  # h
        reg_label[k][cs, 4] = gt_box3d[4]  # w
        reg_label[k][cs, 5] = gt_box3d[5]  # l
        reg_label[k][cs, 6] = gt_box3d[6]  # ry
    return cls_label, reg_label


def filter_unmatched_idxs(proposals, matched_targets):
    ret_proposals = {}
    matched_idxs = torch.cat([target.get_field('matched_idxs') for target in matched_targets])
    keep = matched_idxs >= 0
    for k, v in proposals.items():
        ret_proposals[k] = v[keep]
    keep_splited = torch.split(keep, [len(a) for a in matched_targets])
    ret_targets = []
    for k, t in zip(keep_splited, matched_targets):
        ret_targets.append(t[k])
    return ret_proposals, ret_targets


def combine_2d_3d(left_inputs, proposals):
    box_3ds_batch, scores = [], []
    randoms = []
    for proposal in proposals:
        score = proposal.get_field('box3d_score')
        box3d = proposal.get_field('box3d')
        is_random = proposal.get_field('random')
        maxidx = score.argmax()
        box3d = box3d.bbox_3d[maxidx]
        scores.append(score[maxidx].cpu())
        box_3ds_batch.append(box3d.cpu())
        randoms.append(is_random[maxidx].cpu())
    box_3ds_batch = torch.stack(box_3ds_batch)
    scores = torch.stack(scores)
    randoms = torch.stack(randoms)
    box_3ds_batch = torch.split(box_3ds_batch, [len(a) for a in left_inputs])
    scores = torch.split(scores, [len(a) for a in left_inputs])
    randoms = torch.split(randoms, [len(a) for a in left_inputs])
    for left_input, box3d, score_3d, is_rand in zip(left_inputs, box_3ds_batch, scores, randoms):
        box3d = Box3DList(box3d, size=left_input.size, mode='ry_lhwxyz')
        left_input.add_field('box3d', box3d)
        left_input.add_field('scores_3d', score_3d)
        left_input.add_field('random', is_rand)
    return left_inputs
