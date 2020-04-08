from disprcnn.structures.bounding_box_3d import Box3DList
from disprcnn.structures.bounding_box import BoxList
# from disprcnn.utils.provider import *
import torch.nn.functional as F
from ..utils.bbox_transform import decode_bbox_target
from ..utils.kitti_utils import boxes3d_to_bev_torch
from ..utils.iou3d.iou3d_utils import nms_gpu
import torch
import numpy as np


class Box3DPointRCNNPostProcess(object):
    def __init__(self, cfg):
        self.cfg_ori = cfg
        self.cfg = cfg
        h, w, l = cfg.MEAN_SIZE[0]
        self.MEAN_SIZE = np.array([h, w, l])
        self.MEAN_SIZE = torch.from_numpy(self.MEAN_SIZE).float().cuda()

    def __call__(self, output_dict, proposals):
        """

        :param output_dict: dict_keys(['rcnn_cls', 'rcnn_reg'])
        :param proposals: dict_keys(['rpn_cls', 'rpn_reg', 'backbone_xyz',
                                     'backbone_features', 'rois', 'roi_scores_raw',
                                     'seg_result', 'rpn_xyz', 'rpn_features',
                                     'seg_mask', 'roi_boxes3d', 'pts_depth'])

        :return:
        """
        size = (1280, 720)  # todo:replace
        # batch_size = len(targets)
        roi_boxes3d = proposals['roi_boxes3d']  # (B, M, 7)
        batch_size = roi_boxes3d.shape[0]
        rcnn_cls = output_dict['rcnn_cls'].view(batch_size, -1, output_dict['rcnn_cls'].shape[1])
        rcnn_reg = output_dict['rcnn_reg'].view(batch_size, -1, output_dict['rcnn_reg'].shape[1])  # (B, M, C)

        # bounding box regression
        anchor_size = self.MEAN_SIZE
        if self.cfg.RCNN.SIZE_RES_ON_ROI:
            assert False

        pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                          anchor_size=anchor_size,
                                          loc_scope=self.cfg.RCNN.LOC_SCOPE,
                                          loc_bin_size=self.cfg.RCNN.LOC_BIN_SIZE,
                                          num_head_bin=self.cfg.RCNN.NUM_HEAD_BIN,
                                          get_xz_fine=True, get_y_by_bin=self.cfg.RCNN.LOC_Y_BY_BIN,
                                          loc_y_scope=self.cfg.RCNN.LOC_Y_SCOPE,
                                          loc_y_bin_size=self.cfg.RCNN.LOC_Y_BIN_SIZE,
                                          get_ry_fine=True).view(batch_size, -1, 7)

        # scoring
        if rcnn_cls.shape[2] == 1:
            raw_scores = rcnn_cls  # (B, M, 1)

            norm_scores = torch.sigmoid(raw_scores)
            pred_classes = (norm_scores > self.cfg.RCNN.SCORE_THRESH).long().squeeze(-1)
        else:
            pred_classes = torch.argmax(rcnn_cls, dim=1).view(-1)
            cls_norm_scores = F.softmax(rcnn_cls, dim=1)
            raw_scores = rcnn_cls[:, pred_classes]
            norm_scores = cls_norm_scores[:, pred_classes]

        inds = norm_scores > self.cfg.RCNN.SCORE_THRESH
        # inds = norm_scores > 0.05
        results = []
        for k in range(batch_size):
            cur_inds = inds[k].view(-1)
            if cur_inds.sum() == 0:
                # print('Low scores, random result.')
                use_rpn_proposals = True
                if not use_rpn_proposals:
                    bbox_3d = Box3DList(torch.rand(1, 7).float(), size, 'xyzhwl_ry').convert("ry_lhwxyz")
                    bbox = torch.Tensor([0, 0, 0, 0]).repeat(1, 1).cuda()
                    bbox = BoxList(bbox, size, mode="xyxy")
                    bbox.add_field("box3d", bbox_3d)
                    # bbox.add_field("box3d_score", torch.Tensor(1).zero_())
                    bbox.add_field("box3d_score", torch.zeros(1) * (-10))
                    bbox.add_field("labels", torch.ones(1).cuda())
                    bbox.add_field("iou_score", torch.Tensor(1).zero_())
                    bbox.add_field("random", torch.ones((len(bbox))).long())
                    if self.cfg.RPN.EARLY_INTEGRATE:
                        bbox.add_field('det_id', torch.Tensor(1).zero_())
                        bbox.add_field('box3d_backend', bbox_3d)
                        bbox.add_field('box3d_backend_ids', torch.Tensor(1).zero_())
                        bbox.add_field('box3d_backend_keep', torch.Tensor(1).zero_())
                    results.append(bbox)
                    continue
                else:
                    # print('use_rpn_proposals')
                    proposal_score = proposals['roi_scores_raw'][k]
                    select_idx = proposal_score.argmax()
                    b3d = roi_boxes3d[k][select_idx]
                    bbox_3d = Box3DList(b3d, size, 'xyzhwl_ry').convert("ry_lhwxyz")
                    bbox = torch.Tensor([0, 0, size[0], size[1]]).repeat(b3d.shape[0], 1).cuda()
                    bbox = BoxList(bbox, size, mode="xyxy")
                    bbox.add_field("box3d", bbox_3d)
                    # bbox.add_field("box3d_score", torch.Tensor([proposal_score[select_idx]]))
                    bbox.add_field("box3d_score", torch.zeros(1))
                    bbox.add_field("labels", 1)
                    bbox.add_field("random", torch.ones((len(bbox))).long())
                    # bbox.add_field('iou_score', scores_selected)
                    results.append(bbox)
                    continue

            pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
            raw_scores_selected = raw_scores[k, cur_inds]
            norm_scores_selected = norm_scores[k, cur_inds]
            pred_classes_selected = pred_classes[k, cur_inds]
            # NMS thresh
            # rotated nms
            boxes_bev_selected = boxes3d_to_bev_torch(pred_boxes3d_selected)
            keep_idx = nms_gpu(boxes_bev_selected, raw_scores_selected, self.cfg.RCNN.NMS_THRESH).view(-1)
            pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]
            scores_selected = raw_scores_selected[keep_idx].squeeze(1)
            pred_classes_selected = pred_classes_selected[keep_idx]

            bbox_3d = Box3DList(pred_boxes3d_selected, size, 'xyzhwl_ry').convert("ry_lhwxyz")
            bbox = torch.Tensor([0, 0, size[0], size[1]]).repeat(pred_boxes3d_selected.shape[0], 1).cuda()
            bbox = BoxList(bbox, size, mode="xyxy")
            bbox.add_field("box3d", bbox_3d)
            bbox.add_field("box3d_score", scores_selected)
            bbox.add_field("labels", pred_classes_selected)
            bbox.add_field('iou_score', scores_selected)
            bbox.add_field('random', torch.zeros((len(bbox))).long())
            results.append(bbox)
        return results
