from typing import Dict, List

import torch
from torch import nn

from disprcnn.layers import interpolate, ROIAlign
from disprcnn.modeling.pointnet_module.point_rcnn.lib.net.point_rcnn import PointRCNN
from disprcnn.modeling.psmnet.stackhourglass import PSMNet
from disprcnn.modeling.roi_heads.mask_head.inference import Masker
from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.disparity import DisparityMap
from disprcnn.structures.image_list import ImageList
from disprcnn.utils.stereo_utils import EndPointErrorLoss, expand_box_to_integer


class DispRCNN3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if self.cfg.MODEL.DISPNET_ON:
            self.dispnet = PSMNet(maxdisp=cfg.MODEL.DISPNET.MAX_DISP,
                                  mindisp=cfg.MODEL.DISPNET.MIN_DISP,
                                  is_module=False,
                                  single_modal_weight_average=cfg.MODEL.DISPNET.SINGLE_MODAL_WEIGHTED_AVERAGE)
            self.dispnet_lossfn = EndPointErrorLoss()
            self.disp_resolution = self.cfg.MODEL.DISPNET.RESOLUTIONS[0]
            self.roi_align = ROIAlign((self.disp_resolution, self.disp_resolution), 1.0, 0)
            self.masker = Masker(0.7, 1)
            if self.cfg.MODEL.DISPNET.TRAINED_MODEL != '':
                self.dispnet.load_state_dict(torch.load(
                    self.cfg.MODEL.DISPNET.TRAINED_MODEL, 'cpu'
                )['model'])
                print('Loading PSMNet from', self.cfg.MODEL.DISPNET.TRAINED_MODEL)
        if cfg.MODEL.DET3D_ON:
            self.pcnet = PointRCNN(cfg)
            if self.cfg.MODEL.POINTRCNN.TRAINED_MODEL != '':
                print('loading pointrcnn from', self.cfg.MODEL.POINTRCNN.TRAINED_MODEL)
                ckpt = torch.load(
                    self.cfg.MODEL.POINTRCNN.TRAINED_MODEL, 'cpu'
                )['model']
                sd = {k[7:]: v for k, v in ckpt.items() if k.startswith('module.')}
                self.pcnet.load_state_dict(sd)

    def crop_and_transform_roi_img(self, im, rois_for_image_crop):
        rois_for_image_crop = torch.as_tensor(rois_for_image_crop, dtype=torch.float32, device=im.device)
        im = self.roi_align(im, rois_for_image_crop)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=im.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=im.device)
        im.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return im

    def prepare_psmnet_input_and_target(self, left_images: ImageList, right_images: ImageList,
                                        left_result: List[BoxList], right_result: List[BoxList],
                                        left_targets: List[BoxList],
                                        require_mask_tgts=True):
        if require_mask_tgts:
            roi_disp_targets = []
            roi_masks = []
            ims_per_batch = len(left_result)
            rois_for_image_crop_left = []
            rois_for_image_crop_right = []
            for i in range(ims_per_batch):
                left_target = left_targets[i]
                mask_gt_per_img = left_target.get_field('masks').get_full_image_mask_tensor().byte()
                disparity_map_per_img: DisparityMap = left_target.get_map('disparity')

                mask_preds_per_img = self.masker([left_result[i].get_field('mask')],
                                                 [left_result[i]])[0].squeeze(1).byte()
                if mask_preds_per_img.ndimension() == 2:
                    mask_preds_per_img = mask_preds_per_img.unsqueeze(0)
                for j, (leftbox, rightbox, mask_pred) in enumerate(zip(left_result[i].bbox.tolist(),
                                                                       right_result[i].bbox.tolist(),
                                                                       mask_preds_per_img)):
                    # 1 align left box and right box
                    x1, y1, x2, y2 = expand_box_to_integer(leftbox)
                    x1p, _, x2p, _ = expand_box_to_integer(rightbox)
                    x1 = max(0, x1)
                    x1p = max(0, x1p)
                    y1 = max(0, y1)
                    y2 = min(y2, left_result[i].height - 1)
                    x2 = min(x2, left_result[i].width - 1)
                    x2p = min(x2p, left_result[i].width - 1)
                    max_width = max(x2 - x1, x2p - x1p)
                    allow_extend_width = min(left_result[i].width - x1, left_result[i].width - x1p)
                    max_width = min(max_width, allow_extend_width)
                    rois_for_image_crop_left.append([i, x1, y1, x1 + max_width, y2])
                    rois_for_image_crop_right.append([i, x1p, y1, x1p + max_width, y2])
                    # prepare target
                    roi_disparity_map = disparity_map_per_img.crop((x1, y1, x1 + max_width, y2))
                    roi_disparity_map.data = roi_disparity_map.data - (x1 - x1p)
                    roi_disp_target = roi_disparity_map.resize((self.disp_resolution, self.disp_resolution)).data
                    mask_pred = mask_pred & mask_gt_per_img
                    roi_mask = mask_pred[y1:y2, x1:x1 + max_width]
                    roi_mask = interpolate(roi_mask[None, None].float(),
                                           (self.disp_resolution, self.disp_resolution),
                                           mode='bilinear',
                                           align_corners=True
                                           )[0, 0].byte()
                    roi_disp_targets.append(roi_disp_target)
                    roi_masks.append(roi_mask)
            # crop and resize images
            left_roi_images = self.crop_and_transform_roi_img(left_images.tensors, rois_for_image_crop_left)
            right_roi_images = self.crop_and_transform_roi_img(right_images.tensors, rois_for_image_crop_right)
            if len(left_roi_images) != 0:
                roi_disp_targets = torch.stack(roi_disp_targets)
                roi_masks = torch.stack(roi_masks).cuda()
            else:
                left_roi_images = torch.empty((0, 3, self.disp_resolution, self.disp_resolution)).cuda()
                right_roi_images = torch.empty((0, 3, self.disp_resolution, self.disp_resolution)).cuda()
                roi_disp_targets = torch.empty((0, self.disp_resolution, self.disp_resolution)).cuda()
                roi_masks = torch.empty((0, self.disp_resolution, self.disp_resolution)).cuda()
            return left_roi_images, right_roi_images, roi_disp_targets, roi_masks
        else:
            ims_per_batch = len(left_result)
            rois_for_image_crop_left = []
            rois_for_image_crop_right = []
            fxus, x1s, x1ps, x2s, x2ps = [], [], [], [], []
            for i in range(ims_per_batch):
                left_target = left_targets[i]
                calib = left_target.get_field('calib')
                fxus.extend([calib.stereo_fuxbaseline for _ in range(len(left_result[i]))])
                mask_preds_per_img = self.masker([left_result[i].get_field('mask')],
                                                 [left_result[i]])[0].squeeze(1).byte()
                if mask_preds_per_img.ndimension() == 2:
                    mask_preds_per_img = mask_preds_per_img.unsqueeze(0)
                for j, (leftbox, rightbox, mask_pred) in enumerate(zip(left_result[i].bbox.tolist(),
                                                                       right_result[i].bbox.tolist(),
                                                                       mask_preds_per_img)):
                    # 1 align left box and right box
                    x1, y1, x2, y2 = expand_box_to_integer(leftbox)
                    x1p, _, x2p, _ = expand_box_to_integer(rightbox)
                    x1 = max(0, x1)
                    x1p = max(0, x1p)
                    y1 = max(0, y1)
                    y2 = min(y2, left_result[i].height - 1)
                    x2 = min(x2, left_result[i].width - 1)
                    x2p = min(x2p, left_result[i].width - 1)
                    max_width = max(x2 - x1, x2p - x1p)
                    allow_extend_width = min(left_result[i].width - x1, left_result[i].width - x1p)
                    max_width = min(max_width, allow_extend_width)
                    rois_for_image_crop_left.append([i, x1, y1, x1 + max_width, y2])
                    rois_for_image_crop_right.append([i, x1p, y1, x1p + max_width, y2])
                    x1s.append(x1)
                    x1ps.append(x1p)
                    x2s.append(x1 + max_width)
                    x2ps.append(x1p + max_width)
            # crop and resize images
            left_roi_images = self.crop_and_transform_roi_img(left_images.tensors, rois_for_image_crop_left)
            right_roi_images = self.crop_and_transform_roi_img(right_images.tensors, rois_for_image_crop_right)
            if len(left_roi_images) != 0:
                x1s = torch.tensor(x1s).cuda()
                x1ps = torch.tensor(x1ps).cuda()
                x2s = torch.tensor(x2s).cuda()
                x2ps = torch.tensor(x2ps).cuda()
                fxus = torch.tensor(fxus).cuda()
            else:
                left_roi_images = torch.empty((0, 3, self.disp_resolution, self.disp_resolution)).cuda()
                right_roi_images = torch.empty((0, 3, self.disp_resolution, self.disp_resolution)).cuda()
            return left_roi_images, right_roi_images, fxus, x1s, x1ps, x2s, x2ps

    def roi_disp_postprocess(self, left_result: List[BoxList], right_result: List[BoxList], output: torch.Tensor):
        output_splited = torch.split(output, [len(a) for a in left_result])
        for lr, rr, out in zip(left_result, right_result, output_splited):
            # each image
            roi_disps_per_img = []
            mask_preds_per_img = self.masker([lr.get_field('mask')], [lr])[0].squeeze(1)
            if mask_preds_per_img.ndimension() == 2:
                mask_preds_per_img = mask_preds_per_img.unsqueeze(0)
            for i, (leftbox, rightbox, mask_pred) in enumerate(
                    zip(lr.bbox.tolist(), rr.bbox.tolist(), mask_preds_per_img)):
                x1, y1, x2, y2 = expand_box_to_integer(leftbox)
                x1p, _, x2p, _ = expand_box_to_integer(rightbox)
                roi_disp = DisparityMap(out[i]).resize(
                    (max(x2 - x1, x2p - x1p), y2 - y1)).crop(
                    (0, 0, x2 - x1, y2 - y1))
                disparity_map_per_roi = torch.zeros((lr.height, lr.width))
                disparity_map_per_roi[int(y1):int(y1) + roi_disp.height,
                int(x1):int(x1) + roi_disp.width] = roi_disp.data + (x1 - x1p)
                disparity_map_per_roi = disparity_map_per_roi.clone().clamp(min=0)  # clip to 0.
                disparity_map_per_roi = disparity_map_per_roi * mask_pred.float()
                roi_disps_per_img.append(disparity_map_per_roi)
            if len(roi_disps_per_img) != 0:
                roi_disps_per_img = torch.stack(roi_disps_per_img).cuda().max(dim=0)[0]
            else:
                roi_disps_per_img = torch.zeros((lr.height, lr.width))
            # print(roi_disps_per_img.max(),roi_disps_per_img.min())
            # lr.add_field('disparity_full_img_size', roi_disps_per_img)
            lr.add_map('disparity', roi_disps_per_img)

        return left_result

    def remove_low_score_rois(self, left_result: List[BoxList], right_result: List[BoxList]):
        ret_lr, ret_rr = [], []
        thresh = self.cfg.MODEL.DISPNET.ROI_MIN_SCORE
        scores = torch.cat([a.get_field('scores') for a in left_result])
        keep = scores > thresh
        if 1 < keep.sum() < 2:
            # keep at least 2 rois.
            idxs = scores.argsort(descending=True)
            keep[idxs[0]] = keep[idxs[1]] = 1
        elif keep.sum() == 1:
            keep.fill_(1)
        keep_splited = torch.split(keep, [len(a) for a in left_result])
        for lr, rr, k in zip(left_result, right_result, keep_splited):
            ret_lr.append(lr[k])
            ret_rr.append(rr[k])
        return ret_lr, ret_rr

    def _forward_train(self, left_images: ImageList, right_images: ImageList,
                       left_result: List[BoxList], right_result: List[BoxList],
                       left_targets: List[BoxList]):
        losses = {}
        # forward psmnet
        # 0. remove low score rois
        left_result, right_result = self.remove_low_score_rois(left_result, right_result)
        # 1. crop rois according to results

        if self.cfg.MODEL.DISPNET_ON:
            left_roi_images, right_roi_images, roi_disp_targets, roi_masks = \
                self.prepare_psmnet_input_and_target(left_images, right_images,
                                                     left_result, right_result, left_targets)
            # 1.5 remove rois to prevent from OOM.
            if self.dispnet.training or self.pcnet.training:
                if self.dispnet.training:
                    max_rois_to_train = self.cfg.MODEL.DISPNET.MAX_ROI_FOR_TRAINING
                else:
                    max_rois_to_train = self.cfg.MODEL.PCNET.MAX_ROI_FOR_TRAINING
                if len(left_roi_images) > max_rois_to_train:
                    # todo: maybe random?
                    left_roi_images = left_roi_images[:max_rois_to_train]
                    right_roi_images = right_roi_images[:max_rois_to_train]
                    roi_disp_targets = roi_disp_targets[:max_rois_to_train]
                    roi_masks = roi_masks[:max_rois_to_train]
                    lrs, rrs = [], []
                    s = 0
                    for lr, rr in zip(left_result, right_result):
                        if s >= max_rois_to_train:
                            lrs.append(lr[[]])
                            rrs.append(rr[[]])
                        else:
                            lrs.append(lr[range(min(max_rois_to_train - s, len(lr)))])
                            rrs.append(rr[range(min(max_rois_to_train - s, len(rr)))])
                            s += min(max_rois_to_train - s, len(lr))
                    left_result = lrs
                    right_result = rrs
            # 2. forward psmnet
            output = self.dispnet((left_roi_images, right_roi_images))
            # 3. compute loss
            if self.cfg.SOLVER.TRAIN_PSM:
                disp_loss = self.dispnet_lossfn(roi_disp_targets, output, roi_masks)
                losses.update(disp_loss=disp_loss)
            if isinstance(output, (list, tuple)):
                out3 = output[2]
            else:
                out3 = output
            out3_splited = torch.split(out3, [len(r) for r in left_result])
            for o3, lr in zip(out3_splited, left_result):
                lr.add_field('disparity', o3)
        # 4. put roi disps into full image map.
        if self.cfg.MODEL.DET3D_ON:
            proposals, pc_loss = self.pcnet(left_result, right_result, left_targets)
            if self.cfg.SOLVER.TRAIN_PC:
                losses.update(pc_loss)
        return losses

    def _forward_eval(self, left_images: ImageList, right_images: ImageList,
                      left_result: List[BoxList], right_result: List[BoxList], left_targets: List[BoxList]):
        if self.cfg.MODEL.DISPNET_ON:
            left_roi_images, right_roi_images, calib, x1s, x1ps, x2s, x2ps = \
                self.prepare_psmnet_input_and_target(left_images, right_images,
                                                     left_result, right_result, left_targets, require_mask_tgts=False)
            if len(left_roi_images) > 0:
                output = self.dispnet((left_roi_images, right_roi_images))
            else:
                output = torch.zeros((0, self.disp_resolution, self.disp_resolution)).cuda()
            # add output to extra_fields
            output_splited = torch.split(output, [len(a) for a in left_result])
            assert len(output_splited) == len(left_result)
            for lr, os in zip(left_result, output_splited):
                lr.add_field('disparity', os)
        if self.cfg.MODEL.DET3D_ON:
            left_result, right_result, _ = self.pcnet(left_result, right_result, left_targets)
        result = {'left': left_result, 'right': right_result}
        return result

    def remove_illegal_detections(self, left_result: List[BoxList], right_result: List[BoxList]):
        lrs, rrs = [], []
        for lr, rr in zip(left_result, right_result):
            lk = (lr.bbox[:, 2] > lr.bbox[:, 0] + 1) & (lr.bbox[:, 3] > lr.bbox[:, 1] + 1)
            rk = (rr.bbox[:, 2] > rr.bbox[:, 0] + 1) & (rr.bbox[:, 3] > rr.bbox[:, 1] + 1)
            keep = lk & rk
            lrs.append(lr[keep])
            rrs.append(rr[keep])
        return lrs, rrs

    def forward(self, lr_images: Dict[str, ImageList],
                lr_result: Dict[str, List[BoxList]],
                lr_targets: Dict[str, List[BoxList]] = None):
        left_images, right_images = lr_images['left'], lr_images['right']
        left_result, right_result = lr_result['left'], lr_result['right']
        left_result, right_result = self.remove_illegal_detections(left_result, right_result)
        if self.training:
            assert lr_targets is not None
            left_targets, right_targets = lr_targets['left'], lr_targets['right']
            return self._forward_train(left_images, right_images,
                                       left_result, right_result, left_targets)
        else:
            return self._forward_eval(left_images, right_images, left_result, right_result, lr_targets['left'])

    def load_state_dict(self, state_dict, strict=True):
        super(DispRCNN3D, self).load_state_dict(state_dict, strict)
        if self.cfg.MODEL.DISPNET_ON and self.cfg.MODEL.DISPNET.TRAINED_MODEL != '':
            self.dispnet.load_state_dict(torch.load(
                self.cfg.MODEL.DISPNET.TRAINED_MODEL, 'cpu'
            )['model'])
            print('Loading PSMNet from', self.cfg.MODEL.DISPNET.TRAINED_MODEL)
        if self.cfg.MODEL.POINTRCNN.TRAINED_MODEL != '':
            print('loading pointrcnn from', self.cfg.MODEL.POINTRCNN.TRAINED_MODEL)
            ckpt = torch.load(
                self.cfg.MODEL.POINTRCNN.TRAINED_MODEL, 'cpu'
            )['model']
            sd = {k[7:]: v for k, v in ckpt.items() if k.startswith('module.')}
            self.pcnet.load_state_dict(sd)
