import numpy as np
import pickle
import os
import cv2
import torch
import torch.utils.data
import zarr
from PIL import Image
from tqdm import tqdm

from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.bounding_box_3d import Box3DList
from disprcnn.structures.calib import Calib
from disprcnn.structures.disparity import DisparityMap
from disprcnn.structures.segmentation_mask import SegmentationMask
from disprcnn.utils.kitti_utils import load_calib, load_image_2, load_label_2, load_label_3
from disprcnn.utils.stereo_utils import align_left_right_targets


class KITTIObjectDatasetPedestrian(torch.utils.data.Dataset):
    CLASSES = (
        "__background__",
        "pedestrian",
        'dontcare'
    )
    NUM_TRAINING = 7481
    NUM_TRAIN = 3712
    NUM_VAL = 3769
    NUM_TESTING = 7518

    def __init__(self, root, split, transforms=None, filter_empty=False, offline_2d_predictions_path='',
                 shape_prior_base='velodyne_based', remove_ignore=True):
        # todo: fix shape_prior_base.
        """
        :param root: '.../kitti/
        :param split: ['train','val']
        :param transforms:
        :param filter_empty:
        :param offline_2d_predictions_path:
        """
        self.root = root
        self.split = split
        cls = KITTIObjectDatasetPedestrian.CLASSES
        self.remove_ignore = remove_ignore
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.transforms = transforms
        self.shape_prior_base = shape_prior_base
        # make cache or read cached annotation
        self.annotations = self.read_annotations()
        self.infos = self.read_info()
        self._imgsetpath = os.path.join(self.root, "object/split_set/%s_set.txt")
        if offline_2d_predictions_path != '':
            o2ppath = offline_2d_predictions_path % split + '.pth'
            if is_testing_split(self.split):
                s = o2ppath.split('/')[-2]
                s = '_'.join(s.split('_')[:2])
                o2ppath = '/'.join(o2ppath.split('/')[:-2] + [s] + [o2ppath.split('/')[-1]])
            self.o2dpreds = torch.load(o2ppath, 'cpu')

        with open(self._imgsetpath % self.split) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        if hasattr(self, 'o2dpreds'):
            assert len(self.ids) == len(self.o2dpreds['left'])
        if filter_empty:
            ids = []
            if hasattr(self, 'o2dpreds'):
                o2dpreds = {'left': [], 'right': []}
            for i, id in enumerate(self.ids):
                if self.annotations['left'][int(id)]['labels'].sum() != 0:
                    if hasattr(self, 'o2dpreds'):
                        if len(self.o2dpreds['left'][i]) != 0:
                            ids.append(id)
                            o2dpreds['left'].append(self.o2dpreds['left'][i])
                            o2dpreds['right'].append(self.o2dpreds['right'][i])
                    else:
                        ids.append(id)
            self.ids = ids
            if hasattr(self, 'o2dpreds'):
                self.o2dpreds = o2dpreds
        # self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.truncations_list, self.occlusions_list = self.get_truncations_occluded_list()
        if '%s' in offline_2d_predictions_path:
            self.offline_2d_predictions_dir = offline_2d_predictions_path % split
        else:
            self.offline_2d_predictions_dir = offline_2d_predictions_path

    def __getitem__(self, index):
        imgs = self.get_image(index)
        targets = self.get_ground_truth(index)
        if self.transforms is not None:
            imgs, targets = self.transforms(imgs, targets)
        if not is_testing_split(self.split):
            for view in ['left', 'right']:
                labels = targets[view].get_field('labels')
                targets[view] = targets[view][labels == 1]  # remove not cars
            l, r = align_left_right_targets(targets['left'], targets['right'], thresh=0.0)
            targets['left'] = l
            targets['right'] = r
        if self.offline_2d_predictions_dir != '':
            lp, rp = self.get_offline_prediction(index)
            lp = lp.resize(targets['left'].size)
            rp = rp.resize(targets['right'].size)
            return imgs, targets, index, lp, rp
        else:
            return imgs, targets, index

    def get_image(self, index):
        img_id = self.ids[index]
        split = 'training' if not is_testing_split(self.split) else 'testing'
        left_img = Image.open(os.path.join(self.root, 'object', split, 'image_2', img_id + '.png'))
        right_img = Image.open(os.path.join(self.root, 'object', split, 'image_3', img_id + '.png'))
        imgs = {'left': left_img, 'right': right_img}
        return imgs

    def get_ground_truth(self, index):
        img_id = self.ids[index]
        if not is_testing_split(self.split):
            left_annotation = self.annotations['left'][int(img_id)]
            right_annotation = self.annotations['right'][int(img_id)]
            info = self.get_img_info(index)
            height, width = info['height'], info['width']
            # left target
            left_target = BoxList(left_annotation["boxes"], (width, height), mode="xyxy")
            left_target.add_field("labels", left_annotation["labels"])
            # left_target.add_field("alphas", left_annotation['alphas'])
            boxes_3d = Box3DList(left_annotation["boxes_3d"], (width, height), mode='ry_lhwxyz')
            left_target.add_field("box3d", boxes_3d)
            left_target.add_map('disparity', self.get_disparity(index))
            left_target.add_field('kins_masks', self.get_kins_mask(index))
            left_target.add_field('truncation', torch.tensor(self.truncations_list[int(img_id)]))
            left_target.add_field('occlusion', torch.tensor(self.occlusions_list[int(img_id)]))
            left_target.add_field('image_size', torch.tensor([[width, height]]).repeat(len(left_target), 1))
            left_target.add_field('calib', Calib(self.get_calibration(index), (width, height)))
            left_target.add_field('index', torch.full((len(left_target), 1), index, dtype=torch.long))
            left_target.add_field('imgid', torch.full((len(left_target), 1), int(img_id), dtype=torch.long))
            left_target = left_target.clip_to_image(remove_empty=True)
            # right target
            right_target = BoxList(right_annotation["boxes"], (width, height), mode="xyxy")
            right_target.add_field("labels", right_annotation["labels"])
            right_target = right_target.clip_to_image(remove_empty=True)
            target = {'left': left_target, 'right': right_target}
            return target
        else:
            fakebox = torch.tensor([[0, 0, 0, 0]])
            info = self.get_img_info(index)
            height, width = info['height'], info['width']
            # left target
            left_target = BoxList(fakebox, (width, height), mode="xyxy")
            left_target.add_field('image_size', torch.tensor([[width, height]]).repeat(len(left_target), 1))
            left_target.add_field('calib', Calib(self.get_calibration(index), (width, height)))
            left_target.add_field('index', torch.full((len(left_target), 1), index, dtype=torch.long))
            left_target.add_field('imgid', torch.full((len(left_target), 1), int(img_id), dtype=torch.long))
            # right target
            right_target = BoxList(fakebox, (width, height), mode="xyxy")
            target = {'left': left_target, 'right': right_target}
            return target

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.ids[index]
        return self.infos[int(img_id)]

    def map_class_id_to_class_name(self, class_id):
        return KITTIObjectDatasetPedestrian.CLASSES[class_id]

    def read_annotations(self):
        double_view_annotations = {}
        if is_testing_split(self.split):
            return {'left': [], 'right': []}
        for view in [2, 3]:
            annodir = os.path.join(self.root, f"object/training/label_{view}")
            anno_cache_path = os.path.join(annodir, 'pedestrian_annotations.pkl')
            if os.path.exists(anno_cache_path):
                annotations = pickle.load(open(anno_cache_path, 'rb'))
            else:
                print('generating', anno_cache_path)
                annotations = []
                for i in tqdm(range(7481)):
                    if view == 2:
                        anno_per_img = load_label_2(self.root, 'training', i)
                    else:
                        anno_per_img = load_label_3(self.root, 'training', i)
                    num_objs = len(anno_per_img)
                    label = np.zeros((num_objs), dtype=np.int32)
                    boxes = np.zeros((num_objs, 4), dtype=np.float32)
                    boxes_3d = np.zeros((num_objs, 7), dtype=np.float32)
                    alphas = np.zeros((num_objs), dtype=np.float32)
                    ix = 0
                    for anno in anno_per_img:
                        cls, truncated, occluded, alpha, x1, \
                        y1, x2, y2, h, w, l, x, y, z, ry = anno.cls.name, anno.truncated, anno.occluded, anno.alpha, anno.x1, anno.y1, anno.x2, anno.y2, \
                                                           anno.h, anno.w, anno.l, \
                                                           anno.x, anno.y, anno.z, anno.ry
                        cls_str = cls.lower().strip()
                        if self.split == 'training':
                            # regard car and van as positive
                            cls_str = 'pedestrian' if cls_str == 'pedestrian' else '__background__'
                        else:  # val
                            # return 'dontcare' in validation phase
                            if cls_str != 'pedestrian':
                                cls_str = '__background__'
                        cls = self.class_to_ind[cls_str]
                        label[ix] = cls
                        alphas[ix] = float(alpha)
                        boxes[ix, :] = [float(x1), float(y1), float(x2), float(y2)]
                        boxes_3d[ix, :] = [ry, l, h, w, x, y, z]
                        ix += 1
                    label = label[:ix]
                    alphas = alphas[:ix]
                    boxes = boxes[:ix, :]
                    boxes_3d = boxes_3d[:ix, :]
                    P2 = load_calib(self.root, 'training', i).P2
                    annotations.append({'labels': torch.tensor(label),
                                        'boxes': torch.tensor(boxes, dtype=torch.float32),
                                        'boxes_3d': torch.tensor(boxes_3d),
                                        'alphas': torch.tensor(alphas),
                                        'P2': torch.tensor(P2).float(),
                                        })
                pickle.dump(annotations, open(anno_cache_path, 'wb'))
            if view == 2:
                double_view_annotations['left'] = annotations
            else:
                double_view_annotations['right'] = annotations
        return double_view_annotations

    def read_info(self):
        split = 'training' if not is_testing_split(self.split) else 'testing'
        infopath = os.path.join(self.root,
                                f'object/{split}/infos.pkl')
        if not os.path.exists(infopath):
            infos = []
            total = 7481 if not is_testing_split(self.split) else 7518
            for i in tqdm(range(total)):
                img = load_image_2(self.root, split, i)
                infos.append({"height": img.height, "width": img.width, 'size': img.size})
            pickle.dump(infos, open(infopath, 'wb'))
        else:
            with open(infopath, 'rb') as f:
                infos = pickle.load(f)
        return infos

    def get_truncations_occluded_list(self):
        if is_testing_split(self.split):
            return [], []
        annodir = os.path.join(self.root, f"object/training/label_2")
        truncations_occluded_cache_path = os.path.join(annodir, 'truncations_occluded.pkl')
        if os.path.exists(truncations_occluded_cache_path):
            truncations_list, occluded_list = pickle.load(open(truncations_occluded_cache_path, 'rb'))
        else:
            truncations_list, occluded_list = [], []
            print('generating', truncations_occluded_cache_path)
            for i in tqdm(range(7481)):
                anno_per_img = load_label_2(self.root, 'training', i)
                truncations_list_per_img = []
                occluded_list_per_img = []
                for anno in anno_per_img:
                    truncated, occluded = float(anno.truncated), float(anno.occluded)
                    truncations_list_per_img.append(truncated)
                    occluded_list_per_img.append(occluded)
                truncations_list.append(truncations_list_per_img)
                occluded_list.append(occluded_list_per_img)
            pickle.dump([truncations_list, occluded_list],
                        open(truncations_occluded_cache_path, 'wb'))
        return truncations_list, occluded_list

    def get_offline_prediction(self, index):
        lpmem, rpmem = self.o2dpreds['left'][index], self.o2dpreds['right'][index]
        return lpmem, rpmem

    def get_kins_mask(self, index):
        try:
            imgid = self.ids[index]
            # split = 'training' if self.split != 'test' else 'testing'
            split = 'training' if not is_testing_split(self.split) else 'testing'
            imginfo = self.get_img_info(index)
            width = imginfo['width']
            height = imginfo['height']
            if split == 'training':
                mask = zarr.load(
                    os.path.join(self.root, 'object', split, 'kins_mask_2', imgid + '.zarr')) != 0
                mask = SegmentationMask(mask, (width, height), mode='mask')
            else:
                mask = SegmentationMask(np.zeros((height, width)), (width, height), mode='mask')
        except Exception as e:
            mask = self.get_mask(index)
        return mask

    def get_mask(self, index):
        imgid = self.ids[index]
        split = 'training' if not is_testing_split(self.split) else 'testing'
        imginfo = self.get_img_info(index)
        width = imginfo['width']
        height = imginfo['height']
        if split == 'training':
            mask = zarr.load(
                os.path.join(self.root, 'object', split, self.shape_prior_base, 'mask_2', imgid + '.zarr')) != 0
            mask = SegmentationMask(mask, (width, height), mode='mask')
        else:
            mask = SegmentationMask(np.zeros((height, width)), (width, height), mode='mask')
        return mask

    def get_disparity(self, index):
        imgid = self.ids[index]
        split = 'training' if not is_testing_split(self.split) else 'testing'
        if split == 'training':
            path = os.path.join(self.root, 'object', split,
                                self.shape_prior_base, 'pedestrian_disparity_2',
                                imgid + '.png')
            disp = cv2.imread(path, 2).astype(np.float32) / 256
            disp = DisparityMap(disp)
        else:
            imginfo = self.get_img_info(index)
            width = imginfo['width']
            height = imginfo['height']
            disp = DisparityMap(np.ones((height, width)))
        return disp

    def get_calibration(self, index):
        imgid = self.ids[index]
        split = 'training' if not is_testing_split(self.split) else 'testing'
        calib = load_calib(self.root, split, imgid)
        return calib


def is_testing_split(split):
    return split in ['test', 'testmini', 'test1', 'test2']
