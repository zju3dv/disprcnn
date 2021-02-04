import matplotlib.pyplot as plt
import os.path as osp
import os
import pickle
import random

import cv2
import torch
import zarr
from PIL import Image
from disprcnn.layers import interpolate
from torch.utils.data import Dataset
from torchvision import transforms

from disprcnn.structures.disparity import DisparityMap
from disprcnn.structures.segmentation_mask import SegmentationMask

imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])


class KITTIRoiDataset(Dataset):
    def __init__(self, root, split, maxdisp=48, mindisp=-48):
        """
        :param root:
        :param split:
        :param resolution: width, height
        :param maxdisp:
        :param mindisp:
        """
        self.root = root
        self.split = split
        self.maxdisp = maxdisp
        self.mindisp = mindisp
        self.leftimgdir = os.path.join(self.root, self.split, 'image/left')
        self.rightimgdir = os.path.join(self.root, self.split, 'image/right')
        self.maskdir = os.path.join(self.root, self.split, 'mask')
        self.labeldir = os.path.join(self.root, self.split, 'label')
        self.disparitydir = os.path.join(self.root, self.split, 'disparity')
        ts = [imagenet_normalize]
        self.transform = transforms.Compose(ts)
        self.length = len(os.listdir(self.leftimgdir))
        print('using dataset of length', self.length)

    def __getitem__(self, index):
        images = self.get_image(index)
        targets = self.get_target(index)
        inputs = {**images, **targets}
        return inputs, targets

    def get_image(self, index):
        leftimg = self.get_left_img(index)
        rightimg = self.get_right_img(index)
        # transforms
        if self.transform is not None:
            leftimg = self.transform(leftimg)
            rightimg = self.transform(rightimg)
        return {'left': leftimg, 'right': rightimg}

    def get_target(self, index):
        disparity = self.get_disparity(index)
        mask = self.get_mask(index).get_mask_tensor()
        mask = mask & (disparity < self.maxdisp).byte() & (disparity > self.mindisp).byte()
        label = self.get_label(index)
        targets = {**label, 'mask': mask, 'disparity': disparity}
        return targets

    def get_left_img(self, index):
        leftimg = zarr.load(osp.join(self.leftimgdir, str(index) + '.zarr'))
        leftimg = torch.from_numpy(leftimg)
        return leftimg

    def get_right_img(self, index):
        rightimg = zarr.load(osp.join(self.rightimgdir, str(index) + '.zarr'))
        rightimg = torch.from_numpy(rightimg)
        return rightimg

    def get_disparity(self, index):
        disparity = torch.from_numpy(zarr.load(osp.join(self.disparitydir, str(index) + '.zarr')))
        return disparity

    def get_mask(self, index):
        mask: SegmentationMask = self.get_label(index)['mask']
        return mask

    def get_label(self, index):
        return pickle.load(open(os.path.join(self.labeldir, str(index) + '.pkl'), 'rb'))

    def get_x1_minux_x1p(self, index):
        return self.get_label(index)['x1-x1p']

    def __len__(self):
        return self.length
