import os.path as osp
import argparse
import os
import pickle
import torch
import zarr
from disprcnn.structures.disparity import DisparityMap
import torchvision.transforms.functional as F
from disprcnn.layers import ROIAlign
from tqdm import tqdm

from disprcnn.data.datasets.kitti_car import KITTIObjectDatasetCar
from disprcnn.data.datasets.kitti_human import KITTIObjectDatasetPedestrian
from disprcnn.data.datasets.kitti_cyclist import KITTIObjectDatasetCyclist
from disprcnn.modeling.roi_heads.mask_head.inference import Masker
from disprcnn.structures.segmentation_mask import SegmentationMask
from disprcnn.utils.stereo_utils import expand_box_to_integer

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str,
                    default='data/car_vob_roi')
parser.add_argument('--shape_prior_base', type=str, default='vob')
parser.add_argument('--prediction_template', type=str, default='')
parser.add_argument('--cls', default='car', choices=['car', 'pedestrian', 'cyclist'])
parser.add_argument('--masker_thresh', type=float, default=0.5)
parser.add_argument('--splits', type=str, default='trainval')


def main():
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    root = 'data/kitti'
    roi_align = ROIAlign((224, 224), 1.0, 0)
    if args.splits == 'trainval':
        splits = ['train', 'val']
    else:
        splits = [args.splits]
    masker = Masker(args.masker_thresh)
    for split in splits:
        prediction_pth = args.prediction_template % split
        predictions = torch.load(prediction_pth)
        left_predictions, right_predictions = predictions['left'], predictions['right']
        os.makedirs(os.path.join(output_dir, split, 'image', 'left'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'image', 'right'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'label'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'disparity'), exist_ok=True)
        if args.cls == 'car':
            ds = KITTIObjectDatasetCar(root, split, filter_empty=False, shape_prior_base=args.shape_prior_base)
        elif args.cls == 'pedestrian':
            ds = KITTIObjectDatasetPedestrian(root, split, filter_empty=False,
                                              shape_prior_base=args.shape_prior_base)
        else:  # cyclist
            ds = KITTIObjectDatasetCyclist(root, split, filter_empty=False, shape_prior_base='notused')

        wrote = 0
        assert len(left_predictions) == len(ds)
        for i, (images, targets, _) in enumerate(tqdm(ds)):
            leftimg, rightimg = images['left'], images['right']
            leftanno, rightanno = targets['left'], targets['right']
            left_prediction_per_img = left_predictions[i].resize(leftimg.size)
            right_prediction_per_img = right_predictions[i].resize(leftimg.size)

            calib = leftanno.get_field('calib')
            if len(leftanno) == 0 or len(left_prediction_per_img) == 0: continue
            imgid: int = leftanno.get_field('imgid')[0, 0].item()
            # os.makedirs(osp.join(output_dir, split, 'imgid_org_left', str(imgid)), exist_ok=True)
            masks_per_img = masker([left_prediction_per_img.get_field('mask')],
                                   [left_prediction_per_img])[0].squeeze(1)
            disparity_per_img = leftanno.get_map('disparity')
            assert len(left_prediction_per_img.bbox) == len(right_prediction_per_img.bbox) == len(masks_per_img)
            rois_for_image_crop_left = []
            rois_for_image_crop_right = []
            fxus, x1s, x1ps, x2s, x2ps, y1s, y2s = [], [], [], [], [], [], []
            roi_masks = []
            roi_disps = []
            for j, (left_bbox, right_bbox, mask) in enumerate(
                    zip(left_prediction_per_img.bbox,
                        right_prediction_per_img.bbox, masks_per_img)):
                x1, y1, x2, y2 = expand_box_to_integer(left_bbox.tolist())
                x1p, _, x2p, _ = expand_box_to_integer(right_bbox.tolist())
                max_width = max(x2 - x1, x2p - x1p)
                max_width = min(max_width, leftimg.width - x1)
                allow_extend_width = min(left_prediction_per_img.width - x1, left_prediction_per_img.width - x1p)
                max_width = min(max_width, allow_extend_width)
                rois_for_image_crop_left.append([0, x1, y1, x1 + max_width, y2])
                rois_for_image_crop_right.append([0, x1p, y1, x1p + max_width, y2])
                x1s.append(x1)
                x1ps.append(x1p)
                x2s.append(x1 + max_width)
                x2ps.append(x1p + max_width)
                y1s.append(y1)
                y2s.append(y2)

                roi_mask = mask[y1:y2, x1:x1 + max_width]
                roi_mask = SegmentationMask(roi_mask, (roi_mask.shape[1], roi_mask.shape[0]), mode='mask')
                roi_mask = roi_mask.resize((224, 224))
                # roi_masks.append(roi_mask)
                roi_disparity = disparity_per_img.crop((x1, y1, x1 + max_width, y2)).data
                dispfg_mask = SegmentationMask(roi_disparity != 0, (roi_disparity.shape[1], roi_disparity.shape[0]),
                                               mode='mask').resize((224, 224)).get_mask_tensor()

                roi_disparity = roi_disparity - (x1 - x1p)
                roi_disparity = DisparityMap(roi_disparity).resize((224, 224)).data
                # pdb.set_trace()
                roi_masks.append(roi_mask)
                roi_disps.append(roi_disparity)
            # crop and resize image
            leftimg = F.to_tensor(leftimg).unsqueeze(0)
            rightimg = F.to_tensor(rightimg).unsqueeze(0)
            rois_for_image_crop_left = torch.as_tensor(rois_for_image_crop_left).float()
            rois_for_image_crop_right = torch.as_tensor(rois_for_image_crop_right).float()
            roi_left_imgs = roi_align(leftimg, rois_for_image_crop_left)
            roi_right_imgs = roi_align(rightimg, rois_for_image_crop_right)
            for j in range(len(roi_left_imgs)):
                zarr.save(osp.join(output_dir, split, 'image/left', str(wrote) + '.zarr'), roi_left_imgs[j].numpy())
                zarr.save(osp.join(output_dir, split, 'image/right', str(wrote) + '.zarr'), roi_right_imgs[j].numpy())
                zarr.save(osp.join(output_dir, split, 'disparity', str(wrote) + '.zarr'), roi_disps[j].numpy())
                out_path = os.path.join(output_dir, split, 'label', str(wrote) + '.pkl')
                pickle.dump({'mask': roi_masks[j],
                             'x1': x1s[j],
                             'y1': y1s[j], 'x2': x2s[j], 'y2': y2s[j],
                             'x1p': x1ps[j], 'x2p': x2ps[j],
                             'fuxb': calib.stereo_fuxbaseline,
                             'imgid': imgid},
                            open(out_path, 'wb'))
                wrote += 1
        print(f'made {wrote} pairs for {split}.')


if __name__ == '__main__':
    main()
