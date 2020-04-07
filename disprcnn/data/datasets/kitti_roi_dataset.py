import os
import pickle

import zarr
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from disprcnn.structures.disparity import DisparityMap
from disprcnn.structures.segmentation_mask import SegmentationMask


class KITTIRoiDataset(Dataset):
    def __init__(self, root, split, resolution=-1, maxdisp=192, mindisp=0, ds_len=-1):
        """
        :param root:
        :param split:
        :param resolution: width, height
        :param maxdisp:
        :param mindisp:
        """
        assert split in ['train', 'val']
        self.root = root
        self.split = split
        self.resolution = self._check_resolution(resolution)
        self.maxdisp = maxdisp
        self.mindisp = mindisp
        self.leftimgdir = os.path.join(self.root, self.split, 'image/left')
        self.rightimgdir = os.path.join(self.root, self.split, 'image/right')
        self.maskdir = os.path.join(self.root, self.split, 'mask')
        self.labeldir = os.path.join(self.root, self.split, 'label')
        self.disparitydir = os.path.join(self.root, self.split, 'disparity')
        ts = []
        if all(a > 0 for a in self.resolution):
            ts.append(transforms.Resize(self.resolution))
        ts = ts + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(ts)
        if ds_len < 0:
            self.length = len(os.listdir(self.leftimgdir))
        else:
            self.length = ds_len
        # print('using dataset of length', self.length)

    def __getitem__(self, index):
        images = self.get_image(index)
        targets = self.get_target(index)
        return images, targets

    def get_image(self, index):
        leftimg = Image.open(os.path.join(self.leftimgdir, str(index) + '.webp'))
        rightimg = Image.open(os.path.join(self.rightimgdir, str(index) + '.webp'))
        assert leftimg.size == rightimg.size
        # transforms
        if self.transform is not None:
            leftimg = self.transform(leftimg)
            rightimg = self.transform(rightimg)
        return leftimg, rightimg

    def get_target(self, index):
        disparity = self.get_disparity(index)
        mask = self.get_mask(index)
        mask = mask.resize(self.resolution)
        mask = mask.get_mask_tensor()

        disp = DisparityMap(disparity)
        disparity = disp.resize(self.resolution)
        mask = mask & (disparity.data < self.maxdisp).byte() & (disparity.data > self.mindisp).byte()
        label = self.get_label(index)
        targets = {**label, 'mask': mask, 'disparity': disparity.data}
        return targets

    def get_left_img(self, index):
        leftimg = Image.open(os.path.join(self.leftimgdir, str(index) + '.webp'))
        return leftimg

    def get_right_img(self, index):
        rightimg = Image.open(os.path.join(self.rightimgdir, str(index) + '.webp'))
        return rightimg

    def get_disparity(self, index):
        disparity = zarr.convenience.load(os.path.join(self.disparitydir, str(index) + '.zarr'))
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

    def _check_resolution(self, r):
        if isinstance(r, int):
            r = (r, r)
        if isinstance(r, (tuple, list)):
            assert len(r) == 2
            assert isinstance(r[0], int) and isinstance(r[1], int)
        return r
