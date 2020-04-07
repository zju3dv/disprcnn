import argparse
import os
import pickle

import torch
import zarr
from PIL import Image
from tqdm import tqdm

from disprcnn.data.datasets.kitti import KITTIObjectDataset
from disprcnn.modeling.roi_heads.mask_head.inference import Masker
from disprcnn.structures.segmentation_mask import SegmentationMask
from disprcnn.utils.stereo_utils import expand_box_to_integer

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str,
                    default=os.path.expanduser('~/Datasets/kitti/object/vob_pad_cs_dimreg_inf_roi_freex'))
parser.add_argument('--mask_disp_sub_path', type=str, default='vob')
parser.add_argument('--prediction_template', type=str,
                    default='models/kitti/vob/mask/inference/kitti_%s_vob_pad_cs_dimreg/predictions.pth')
parser.add_argument('--masker_thresh', type=float, default=0.5)


def main():
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    root = os.path.expanduser('~/Datasets/kitti')
    splits = ['train', 'val']
    masker = Masker(args.masker_thresh)
    for split in splits:
        prediction_pth = args.prediction_template % split
        predictions = torch.load(prediction_pth)
        left_predictions, right_predictions = predictions['left'], predictions['right']
        os.makedirs(os.path.join(output_dir, split, 'image', 'left'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'image', 'right'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'label'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'disparity'), exist_ok=True)
        ds = KITTIObjectDataset(root, split, filter_empty=False, mask_disp_sub_path=args.mask_disp_sub_path)
        wrote = 0
        assert len(left_predictions) == len(ds)
        for i, (images, targets, _) in enumerate(tqdm(ds)):
            leftimg, rightimg = images['left'], images['right']
            leftanno, rightanno = targets['left'], targets['right']
            left_prediction_per_img = left_predictions[i].resize(leftimg.size)
            right_prediction_per_img = right_predictions[i].resize(leftimg.size)
            calib = leftanno.get_field('calib')
            if len(leftanno) == 0 or len(left_prediction_per_img) == 0: continue
            masks_per_img = masker([left_prediction_per_img.get_field('mask')],
                                   [left_prediction_per_img])[0].squeeze(1)
            disparity_per_img = leftanno.get_map('disparity')
            assert len(left_prediction_per_img.bbox) == len(right_prediction_per_img.bbox) == len(masks_per_img)
            for j, (left_bbox, right_bbox, mask) in enumerate(
                    zip(left_prediction_per_img.bbox,
                        right_prediction_per_img.bbox, masks_per_img)):
                x1, y1, x2, y2 = expand_box_to_integer(left_bbox.tolist())
                x1p, _, x2p, _ = expand_box_to_integer(right_bbox.tolist())
                max_width = max(x2 - x1, x2p - x1p)
                max_width = min(max_width, leftimg.width - x1)
                roi_mask = mask[y1:y2, x1:x1 + max_width]
                roi_mask = SegmentationMask(roi_mask, (roi_mask.shape[1], roi_mask.shape[0]), mode='mask')
                roi_left_img: Image.Image = leftimg.crop((x1, y1, x1 + max_width, y2))
                roi_right_img: Image.Image = rightimg.crop((x1p, y1, x1p + max_width, y2))
                roi_disparity = disparity_per_img.crop((x1, y1, x1 + max_width, y2)).data
                roi_disparity = roi_disparity - (x1 - x1p)
                assert roi_left_img.size == roi_right_img.size and \
                       roi_left_img.size == roi_disparity.shape[::-1], \
                    f'{roi_left_img.size} {roi_right_img.size} {roi_disparity.shape[::-1]}' \
                        f'{x1, x1p, max_width}'
                roi_left_img.save(os.path.join(output_dir, split, 'image/left', str(wrote) + '.webp'))
                roi_right_img.save(os.path.join(output_dir, split, 'image/right', str(wrote) + '.webp'))
                zarr.convenience.save(os.path.join(output_dir, split, 'disparity', str(wrote) + '.zarr'),
                                      roi_disparity.numpy())
                pickle.dump({'mask': roi_mask,
                             'x1': x1,
                             'y1': y1, 'x2': x2, 'y2': y2,
                             'x1p': x1p, 'x2p': x2p,
                             'fuxb': calib.stereo_fuxbaseline,
                             'image_width': leftimg.width,
                             'image_height': leftimg.height},
                            open(os.path.join(output_dir, split, 'label', str(wrote) + '.pkl'), 'wb'))
                wrote += 1
        print(f'made {wrote} pairs for {split}.')


if __name__ == '__main__':
    main()
