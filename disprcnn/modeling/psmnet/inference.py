import torch

from disprcnn.modeling.roi_heads.mask_head.inference import Masker
from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.disparity import DisparityMap
from disprcnn.structures.segmentation_mask import SegmentationMask, BinaryMaskList
from disprcnn.utils.stereo_utils import expand_box_to_integer


def sanity_check(left_predictions, right_predictions):
    assert len(left_predictions) == len(right_predictions)
    assert all(isinstance(l, BoxList) for l in left_predictions)
    assert all(isinstance(r, BoxList) for r in right_predictions)
    assert all(len(l) == len(r) for l, r in zip(left_predictions, right_predictions))
    assert all(l.size == r.size for l, r in zip(left_predictions, right_predictions))


class DisparityMapProcessor:
    def _forward_single_image(self, left_prediction: BoxList, right_prediction: BoxList) -> DisparityMap:
        left_bbox = left_prediction.bbox
        right_bbox = right_prediction.bbox
        disparity_preds = left_prediction.get_field('disparity')
        mask_preds = left_prediction.get_field('mask').clone()
        # print(disparity_preds.shape)
        assert len(left_bbox) == len(right_bbox) == len(
            disparity_preds), f'{len(left_bbox), len(right_bbox), len(disparity_preds)}'
        num_rois = len(left_bbox)
        if num_rois == 0:
            disparity_full_image = torch.zeros((left_prediction.height, left_prediction.width))
        else:
            disparity_maps = []
            for left_roi, right_roi, disp_roi, mask_pred in zip(left_bbox, right_bbox, disparity_preds, mask_preds):
                x1, y1, x2, y2 = left_roi.tolist()
                x1p, _, x2p, _ = right_roi.tolist()
                x1, y1, x2, y2 = expand_box_to_integer((x1, y1, x2, y2))
                x1p, _, x2p, _ = expand_box_to_integer((x1p, y1, x2p, y2))
                disparity_map_per_roi = torch.zeros((left_prediction.height, left_prediction.width))
                # mask = mask_pred.squeeze(0)
                # mask = SegmentationMask(BinaryMaskList(mask, size=mask.shape[::-1]), size=mask.shape[::-1],
                #                         mode='mask').crop((x1, y1, x1 + max(x2 - x1, x2p - x1p), y2))
                disp_roi = DisparityMap(disp_roi).resize(
                    (max(x2 - x1, x2p - x1p), y2 - y1)).crop(
                    (0, 0, x2 - x1, y2 - y1)).data
                disp_roi = disp_roi + x1 - x1p
                disparity_map_per_roi[y1:y2, x1:x2] = disp_roi
                disparity_maps.append(disparity_map_per_roi)
            disparity_full_image = torch.stack(disparity_maps).max(dim=0)[0]
        return DisparityMap(disparity_full_image)

    def __call__(self, left_predictions, right_predictions):
        if isinstance(left_predictions, BoxList) and isinstance(right_predictions, BoxList):
            left_predictions = [left_predictions]
            right_predictions = [right_predictions]
        sanity_check(left_predictions, right_predictions)
        results = []
        for l, r in zip(left_predictions, right_predictions):
            results.append(self._forward_single_image(l, r))
        if len(results) == 1:
            results = results[0]
        return results


def clip_mask_to_minmaxdisp(mask, dispairty, leftbox, rightbox, mindisp=-48, maxdisp=48, resolution=28):
    mask = mask.clone()
    disparity_map = DisparityMap(dispairty)
    for lb, rb in zip(leftbox, rightbox):
        x1, y1, x2, y2 = lb.tolist()
        x1p, _, x2p, _ = rb.tolist()
        max_width = max(x2 - x1, x2p - x1p)
        roi_disparity = disparity_map.crop(lb.tolist()).data
        roi_disparity = roi_disparity - (x1 - x1p)
        roi_mask = mask[round(y1):round(y2), round(x1):round(x2)]
        roi_mask = roi_mask & (roi_disparity * resolution * 4 / max_width > mindisp).byte() & (
                roi_disparity * resolution * 4 / max_width < maxdisp).byte()
        # roi_mask[roi_disparity * resolution * 4 / (x2 - x1) < mindisp] = 0
        # roi_mask[roi_disparity * resolution * 4 / (x2 - x1) > maxdisp] = 0
        # mask[round(y1):round(y2), round(x1):round(x2)] = roi_mask
        mask[round(y1):round(y2), round(x1):round(x2)] = mask[round(y1):round(y2), round(x1):round(x2)] & roi_mask
    return mask


def post_process_and_resize_prediction(left_prediction: BoxList, right_prediction: BoxList, dst_size=(1280, 720),
                                       threshold=0.7,
                                       padding=1, process_disparity=True):
    left_prediction = left_prediction.clone()
    right_prediction = right_prediction.clone()
    if process_disparity and not left_prediction.has_map('disparity'):
        disparity_map_processor = DisparityMapProcessor()
        disparity_pred_full_img = disparity_map_processor(
            left_prediction, right_prediction)
        left_prediction.add_map('disparity', disparity_pred_full_img)
    left_prediction = left_prediction.resize(dst_size)
    right_prediction = right_prediction.resize(dst_size)
    mask_pred = left_prediction.get_field('mask')
    masker = Masker(threshold=threshold, padding=padding)
    mask_pred = masker([mask_pred], [left_prediction])[0].squeeze(1)
    if mask_pred.shape[0] != 0:
        # mask_preds_per_img = mask_pred.sum(dim=0)[0].clamp(max=1)
        mask_preds_per_img = mask_pred
    else:
        mask_preds_per_img = torch.zeros((1, *dst_size[::-1]))
    left_prediction.add_field('mask', mask_preds_per_img)
    return left_prediction, right_prediction
