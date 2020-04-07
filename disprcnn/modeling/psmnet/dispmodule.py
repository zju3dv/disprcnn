from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from disprcnn.layers import ROIAlign
from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.boxlist_ops import double_view_boxlist_nms
from disprcnn.structures.disparity import DisparityMap
from disprcnn.structures.segmentation_mask import SegmentationMask
from .stackhourglass import PSMNet
import cv2
import numpy as np


class ROILevelMapper:
    def __init__(self, resolutions: List[int]):
        self.resolutions = resolutions
        self.areas = [(r * 4) ** 2 for r in resolutions]

    def __call__(self, rois, resolution, method='nearest'):
        assert method in ['nearest', 'less']
        if method == 'nearest':
            i = self.resolutions.index(resolution)
            if rois.shape[1] == 5:
                rois = rois[:, 1:]
            area = (rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1])
            if i == 0:
                keep = area < (self.areas[i] + self.areas[i + 1]) / 2
            elif i == len(self.resolutions) - 1:  # i==1
                keep = area >= (self.areas[i] + self.areas[i - 1]) / 2
            else:
                keep = (area < (self.areas[i] + self.areas[i + 1]) / 2) & \
                       (area >= (self.areas[i] + self.areas[i - 1]) / 2)
            return keep
        else:
            i = self.resolutions.index(resolution)
            if rois.shape[1] == 5:
                rois = rois[:, 1:]
            area = (rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1])
            if i == 0:
                keep = area < self.areas[i]
            elif i == len(self.resolutions) - 1:  # i==1
                keep = area >= self.areas[i - 1]
            else:
                keep = (area < self.areas[i]) & (area >= self.areas[i - 1])
            return keep


class DispModule(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        maxdisp = cfg.MODEL.DISPNET.MAX_DISP
        mindisp = cfg.MODEL.DISPNET.MIN_DISP
        pooler_scales = cfg.MODEL.DISPNET.POOLER_SCALES
        resolutions = cfg.MODEL.DISPNET.RESOLUTIONS
        disp_resolutions = cfg.MODEL.DISPNET.DISP_RESOLUTIONS
        sampling_ratio = cfg.MODEL.DISPNET.SAMPLING_RATIO
        single_modal_weighted_average = cfg.MODEL.DISPNET.SINGLE_MODAL_WEIGHTED_AVERAGE
        conv_layers = cfg.MODEL.DISPNET.CONV_LAYERS
        self.loss_weight = cfg.MODEL.DISPNET.LOSS_WEIGHT
        self.dilate_radius = cfg.MODEL.DISPNET.DILATE_RADIUS
        self.levelmapper_method = cfg.MODEL.DISPNET.LEVELMAPMETHOD
        is_module = cfg.MODEL.DISPNET.IS_MODULE
        # deconv = cfg.MODEL.DISPNET.DECONV
        # assert resolutions == (28,) or resolutions == (14, 28)
        self.resolutions = resolutions
        self.disp_resolutions = disp_resolutions
        self.levelmapper = ROILevelMapper(self.resolutions)
        self.psmnet = PSMNet(maxdisp, mindisp, is_module,
                                 len(pooler_scales), single_modal_weighted_average,
                                 conv_layers
                                 # deconv=deconv
                                 )
        multi_level_roi_aligns = {}
        for resolution in resolutions:
            roi_aligns = []
            for pooler_scale in pooler_scales:
                roi_aligns.append(ROIAlign([resolution, resolution], pooler_scale, sampling_ratio))
            roi_aligns = nn.ModuleList(roi_aligns)
            multi_level_roi_aligns[resolution] = roi_aligns
        self.multi_level_roi_aligns = multi_level_roi_aligns
        # self.roi_align = ROIAlign([resolution, resolution], 1.0 / 4, sampling_ratio)
        self.maxdisp = maxdisp
        self.mindisp = mindisp

    def expand_left_right_freex(self, left_results: List[BoxList], right_results: List[BoxList]):
        ret_left_results, ret_right_results = [], []
        for left_result, right_result in zip(left_results, right_results):
            left_result = left_result.clone(clone_bbox=True)
            right_result = right_result.clone(clone_bbox=True)
            left_widths = left_result.widths()
            right_widths = right_result.widths()
            expand_widths = torch.max(left_widths, right_widths)
            allowed_expand_widths = left_result.width - left_result.bbox[:, 0]
            expand_widths = torch.min(expand_widths, allowed_expand_widths)
            left_result.bbox[:, 2] = left_result.bbox[:, 0] + expand_widths
            right_result.bbox[:, 2] = right_result.bbox[:, 0] + expand_widths
            ret_left_results.append(left_result)
            ret_right_results.append(right_result)
        return ret_left_results, ret_right_results

    def extract_expand_features(self, left_results: List[BoxList], right_results: List[BoxList],
                                left_features: List[Tensor], right_features: List[Tensor]):
        if len(left_results) == 0:
            return None, None
        batchids = torch.cat([torch.full((len(boxlist), 1), i) for i, boxlist in enumerate(left_results)]).to(
            left_features[0].device)
        left_roi_region = torch.cat([boxlist.bbox for boxlist in left_results], dim=0)
        right_roi_region = torch.cat([boxlist.bbox for boxlist in right_results], dim=0)
        left_roi = torch.cat([batchids, left_roi_region], dim=1)
        right_roi = torch.cat([batchids, right_roi_region], dim=1)
        if len(self.resolutions) == 1:
            left_roi_features, right_roi_features = [], []
            for left_feature, right_feature, roi_align in zip(
                    left_features, right_features,
                    self.multi_level_roi_aligns[self.resolutions[0]]
            ):
                left_roi_feature = roi_align(left_feature, left_roi)
                right_roi_feature = roi_align(right_feature, right_roi)
                left_roi_features.append(left_roi_feature)
                right_roi_features.append(right_roi_feature)
            left_roi_features = torch.cat(left_roi_features, dim=1)
            # todo:five tensors of shape bsz,256,28,28, is there any advanced method?
            right_roi_features = torch.cat(right_roi_features, dim=1)
            return left_roi_features, right_roi_features
        else:
            left_roi_features, right_roi_features = {}, {}
            for resolution in self.resolutions:
                left_roi_features[resolution] = []
                right_roi_features[resolution] = []
                for left_feature, right_feature, roi_align in zip(
                        left_features, right_features, self.multi_level_roi_aligns[resolution]
                ):
                    keep = self.levelmapper(left_roi, resolution, self.levelmapper_method)
                    left_roi_feature = roi_align(left_feature, left_roi[keep])
                    right_roi_feature = roi_align(right_feature, right_roi[keep])
                    left_roi_features[resolution].append(left_roi_feature)
                    right_roi_features[resolution].append(right_roi_feature)
                left_roi_features[resolution] = torch.cat(left_roi_features[resolution], dim=1)
                right_roi_features[resolution] = torch.cat(right_roi_features[resolution], dim=1)
            return left_roi_features, right_roi_features

    def subsample_results(self, left_results, right_results):
        left_results = [l[l.get_field('labels') == 1] for l in left_results]
        right_results = [r[r.get_field('labels') == 1] for r in right_results]
        score_field = 'scores'
        lrresults = [double_view_boxlist_nms(left_boxlist, right_boxlist, 0.5, score_field=score_field) for
                     left_boxlist, right_boxlist in zip(left_results, right_results)]
        left_results = [r[0] for r in lrresults]
        right_results = [r[1] for r in lrresults]
        return left_results, right_results

    def prepare_target(self, left_results: List[BoxList], right_results: List[BoxList],
                       left_targets: List[BoxList]):
        """
        :param left_results:
        :param right_results:
        :param left_targets:
        :return:disp_true:FloatTensor of shape(IMS_PER_BATCH,self.resolution,self.resolution), mask: ByteTensor of same shape as disp_true
        """
        batch_disparity_target, batch_mask = [], []
        for left_result, right_result, left_target in zip(left_results, right_results, left_targets):
            left_bbox_per_image = left_result.bbox
            right_bbox_per_image = right_result.bbox
            disparity_target_per_image: DisparityMap = left_target.get_map('disparity')
            mask_per_image: SegmentationMask = left_target.get_field('masks')
            for left_bbox, right_bbox in zip(left_bbox_per_image, right_bbox_per_image):
                left_bbox = left_bbox.round()
                right_bbox = right_bbox.round()
                roi_disparity = disparity_target_per_image.crop(left_bbox.tolist())
                roi_disparity.data = roi_disparity.data - (left_bbox[0] - right_bbox[0])
                roi_disparity = roi_disparity.resize((self.disp_resolutions[0], self.disp_resolutions[0]))
                roi_disparity = roi_disparity.data
                roi_mask = mask_per_image.crop(left_bbox.tolist()).resize(
                    (self.disp_resolutions[0],
                     self.disp_resolutions[0])).get_full_image_mask_tensor().byte()
                assert roi_disparity.shape == roi_mask.shape
                # filter mask where disparity is out of [mindisp,maxdisp]
                if self.dilate_radius > 0:
                    device = roi_mask.device
                    roi_mask = torch.ByteTensor(
                        cv2.dilate(roi_mask.cpu().numpy(), np.ones((
                            self.dilate_radius, self.dilate_radius)), iterations=1)
                    ).to(device)
                roi_mask[roi_disparity < self.mindisp] = 0
                roi_mask[roi_disparity > self.maxdisp] = 0
                batch_disparity_target.append(roi_disparity)
                batch_mask.append(roi_mask)
        batch_disparity_target = torch.stack(batch_disparity_target)
        batch_mask = torch.stack(batch_mask)
        return batch_disparity_target, batch_mask

    def prepare_targets_multi_resolution(self, left_results, right_results, left_targets, resolution):
        batch_disparity_target, batch_mask = [], []
        for left_result, right_result, left_target in zip(left_results, right_results, left_targets):
            left_bbox_per_image = left_result.bbox
            right_bbox_per_image = right_result.bbox
            # keep = self.levelmapper(left_bbox_per_image, resolution)
            disparity_target_per_image: DisparityMap = left_target.get_map('disparity')
            mask_per_image: SegmentationMask = left_target.get_field('masks')
            for left_bbox, right_bbox in zip(left_bbox_per_image, right_bbox_per_image):
                left_bbox = left_bbox.round()
                right_bbox = right_bbox.round()
                roi_disparity = disparity_target_per_image.crop(left_bbox.tolist())
                roi_disparity.data = roi_disparity.data - (left_bbox[0] - right_bbox[0])
                # todo: replace with disp_resolution
                roi_disparity = roi_disparity.resize((resolution * 4, resolution * 4))
                roi_disparity = roi_disparity.data
                roi_mask = mask_per_image.crop(left_bbox.tolist()).resize(
                    (resolution * 4, resolution * 4)).get_full_image_mask_tensor().byte()
                assert roi_disparity.shape == roi_mask.shape
                # filter mask where disparity is out of [mindisp,maxdisp]
                if self.dilate_radius > 0:
                    device = roi_mask.device
                    roi_mask = torch.ByteTensor(
                        cv2.dilate(roi_mask.cpu().numpy(), np.ones((
                            self.dilate_radius, self.dilate_radius)), iterations=1)
                    ).to(device)
                roi_mask[roi_disparity < self.mindisp] = 0
                roi_mask[roi_disparity > self.maxdisp] = 0
                batch_disparity_target.append(roi_disparity)
                batch_mask.append(roi_mask)
        keep = self.levelmapper(torch.cat([a.bbox for a in left_results]), resolution, self.levelmapper_method)
        batch_disparity_target = torch.stack(batch_disparity_target)[keep]
        batch_mask = torch.stack(batch_mask)[keep]
        return batch_disparity_target, batch_mask

    def _forward_train(self, out, left_results, right_results,
                       expand_left_results, expand_right_results,
                       left_targets, right_targets):
        if len(self.resolutions) == 1:
            out1, out2, out3 = out
            disp_true, mask = self.prepare_target(expand_left_results, expand_right_results, left_targets)
            mask = mask.to(out1.device)
            loss1 = (F.smooth_l1_loss(out1, disp_true, reduction='none') * mask.float()).sum()
            loss2 = (F.smooth_l1_loss(out2, disp_true, reduction='none') * mask.float()).sum()
            loss3 = (F.smooth_l1_loss(out3, disp_true, reduction='none') * mask.float()).sum()
            if mask.sum() != 0:
                loss1 = loss1 / mask.sum()
                loss2 = loss2 / mask.sum()
                loss3 = loss3 / mask.sum()
            loss = 0.5 * loss1 + 0.7 * loss2 + loss3

            return left_results, right_results, {'disp_loss': self.loss_weight * loss}
        else:
            total_loss = 0
            total_mask = 0
            for resolution in self.resolutions:
                if not (isinstance(out[resolution], Tensor) and out[resolution].numel() == 0):
                    out1, out2, out3 = out[resolution]
                    disp_true, mask = self.prepare_targets_multi_resolution(expand_left_results,
                                                                            expand_right_results,
                                                                            left_targets, resolution)
                    mask = mask.to(out1.device)
                    loss1 = (F.smooth_l1_loss(out1, disp_true, reduction='none') * mask.float()).sum()
                    loss2 = (F.smooth_l1_loss(out2, disp_true, reduction='none') * mask.float()).sum()
                    loss3 = (F.smooth_l1_loss(out3, disp_true, reduction='none') * mask.float()).sum()
                    # if mask.sum() != 0:
                    #     loss1 = loss1 / mask.sum()
                    #     loss2 = loss2 / mask.sum()
                    #     loss3 = loss3 / mask.sum()
                    total_mask += mask.sum()
                    loss = 0.5 * loss1 + 0.7 * loss2 + loss3
                    # bsz += mask.shape[0]
                    total_loss += loss
            total_loss = total_loss / total_mask
            return left_results, {'disp_loss': self.loss_weight * total_loss}

    def _forward_one_resolution(self, left_roi_feature, right_roi_feature, left_results, right_results,
                                expand_left_results, expand_right_results,
                                left_targets, right_targets):
        if left_roi_feature.numel() != 0:
            out = self.psmnet(left_roi_feature, right_roi_feature)
        else:
            # roi features could be empty in inference time.
            out = torch.empty_like(left_roi_feature[:, 0, :, :])
        out3 = out[2] if self.training else out
        out3_splited = torch.split(out3, [len(r) for r in left_results])
        for o3, left_result in zip(out3_splited, left_results):
            left_result.add_field('disparity', o3)
        if self.training:
            return self._forward_train(out, left_results, right_results,
                                       expand_left_results, expand_right_results,
                                       left_targets, right_targets)
        else:
            return left_results, right_results, {}

    def _forward_multi_resolutions(self, left_roi_feature, right_roi_feature,
                                   left_results, right_results,
                                   expand_left_results, expand_right_results,
                                   left_targets, right_targets
                                   ):
        outs = {}
        for resolution in self.resolutions:
            if left_roi_feature[resolution].numel() != 0:
                outs[resolution] = self.psmnet(left_roi_feature[resolution], right_roi_feature[resolution])
            else:
                outs[resolution] = torch.empty_like(left_roi_feature[resolution][:, 0, :, :])
        out3s = {}
        for r in self.resolutions:
            if not (isinstance(outs[r], Tensor) and outs[r].numel() == 0):
                out3s[r] = outs[r][2] if self.training else outs[r]

        # out3s = {r: out[2] if self.training else out for r, out in outs.items()}
        # assign outs to each box
        out3_splited = {}
        for resolution in self.resolutions:
            if resolution in out3s.keys():
                keep = self.levelmapper(torch.cat([a.bbox for a in expand_left_results]),
                                        resolution,
                                        self.levelmapper_method)
                out3_splited.update(list(zip(keep.nonzero().squeeze(1).tolist(), out3s[resolution])))
        tmp = []
        for i in range(len(out3_splited)):
            tmp.append(out3_splited[i])
        out3_splited = tmp
        st = 0
        for left_result in left_results:
            left_result.add_field('disparity', out3_splited[st:st + len(left_result)])
            st += len(left_result)
        if self.training:
            return self._forward_train(outs, left_results, right_results,
                                       expand_left_results, expand_right_results,
                                       left_targets, right_targets)
        else:
            return left_results, {}

    def forward(self, left_features, right_features,
                left_results, right_results,
                left_targets, right_targets):
        if self.training:
            # prepare boxes
            left_results, right_results = self.subsample_results(left_results, right_results)
        # align boxes
        expand_left_results, expand_right_results = self.expand_left_right_freex(left_results, right_results)
        # extract features
        left_roi_feature, right_roi_feature = self.extract_expand_features(
            expand_left_results, expand_right_results,
            left_features, right_features)
        if len(self.resolutions) == 1:
            return self._forward_one_resolution(left_roi_feature, right_roi_feature,
                                                left_results, right_results,
                                                expand_left_results, expand_right_results,
                                                left_targets, right_targets)
        else:

            return self._forward_multi_resolutions(left_roi_feature, right_roi_feature,
                                                   left_results, right_results,
                                                   expand_left_results, expand_right_results,
                                                   left_targets, right_targets)
