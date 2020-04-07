# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import numpy as np
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F

from disprcnn.utils.timer import Timer

PRINT_TIME = False


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.getsizetimer = Timer()
        self.imgtimer = Timer()
        self.tgttimer = Timer()
        # self.timer = Timer()

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def call_single_view(self, image, target):
        # h, w = self.get_size(image.shape[:-1][::-1])
        # image = cv2.resize(image, None, None, fx=h / image.shape[0],
        #                    fy=h / image.shape[0], interpolation=cv2.INTER_LINEAR)
        self.getsizetimer.tic()
        h, w = self.get_size(image.size)
        self.getsizetimer.toc()
        if PRINT_TIME:
            print('getsize', self.getsizetimer.average_time)
        self.imgtimer.tic()
        image = image.resize((w, h))
        self.imgtimer.toc()
        if PRINT_TIME:
            print('img resize', self.imgtimer.average_time)
        self.tgttimer.tic()
        target = target.resize(image.size)
        self.tgttimer.toc()
        if PRINT_TIME:
            print('tgt resize', self.tgttimer.average_time)
        return image, target

    def call_double_view(self, image, target):
        # self.timer.tic()
        left_image, right_image = image['left'], image['right']
        left_target, right_target = target['left'], target['right']
        left_image, left_target = self.call_single_view(left_image, left_target)
        right_image, right_target = self.call_single_view(right_image, right_target)
        image = {'left': left_image, 'right': right_image}
        target = {'left': left_target, 'right': right_target}
        # self.timer.toc()
        # print('resize', self.timer.average_time)
        return image, target

    def __call__(self, image, target):
        if isinstance(image, dict) and isinstance(target, dict):
            return self.call_double_view(image, target)
        else:
            return self.call_single_view(image, target)


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        # self.timer = Timer()

    def call_single_view(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target

    def call_double_view(self, image, target):
        # self.timer.tic()
        left_image, right_image = image['left'], image['right']
        left_target, right_target = target['left'], target['right']
        if random.random() < self.prob:
            left_image = F.hflip(left_image)
            right_image = F.hflip(right_image)
            left_target = left_target.transpose(0)
            right_target = right_target.transpose(0)
            image = {'left': left_image, 'right': right_image}
            target = {'left': left_target, 'right': right_target}
        # self.timer.toc()
        # print('flip', self.timer.average_time)
        return image, target

    def __call__(self, image, target):
        if isinstance(image, dict) and isinstance(target, dict):
            return self.call_double_view(image, target)
        else:
            return self.call_single_view(image, target)


class ToTensor(object):
    def __init__(self):
        self.timer = Timer()

    def call_single_view(self, image, target):
        return F.to_tensor(image), target

    def call_double_view(self, image, target):
        self.timer.tic()
        left_image, right_image = image['left'], image['right']
        left_target, right_target = target['left'], target['right']
        left_image = F.to_tensor(left_image)
        right_image = F.to_tensor(right_image)
        image = {'left': left_image, 'right': right_image}
        target = {'left': left_target, 'right': right_target}
        self.timer.toc()
        if PRINT_TIME:
            print('totensor', self.timer.average_time)
        return image, target

    def __call__(self, image, target):
        if isinstance(image, dict) and isinstance(target, dict):
            return self.call_double_view(image, target)
        else:
            return self.call_single_view(image, target)


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255
        self.timer = Timer()

    def call_single_view(self, image, target):
        # if self.to_bgr255:
        #     image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

    def call_double_view(self, image, target):
        self.timer.tic()
        left_image, right_image = image['left'], image['right']
        left_target, right_target = target['left'], target['right']
        if self.to_bgr255:
            left_image = left_image[[2, 1, 0]] * 255
            right_image = right_image[[2, 1, 0]] * 255
        # left_image -= self.mean
        # right_image -= self.mean
        left_image = F.normalize(left_image, mean=self.mean, std=self.std)
        right_image = F.normalize(right_image, mean=self.mean, std=self.std)
        image = {'left': left_image, 'right': right_image}
        target = {'left': left_target, 'right': right_target}
        self.timer.toc()
        if PRINT_TIME:
            print('normalize', self.timer.average_time)
        return image, target

    def __call__(self, image, target):
        if isinstance(image, dict) and isinstance(target, dict):
            return self.call_double_view(image, target)
        else:
            return self.call_single_view(image, target)


class ImageNetNormalize(object):
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def call_single_view(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

    def call_double_view(self, image, target):
        left_image, right_image = image['left'], image['right']
        left_target, right_target = target['left'], target['right']

        left_image = F.normalize(left_image, mean=self.mean, std=self.std)
        right_image = F.normalize(right_image, mean=self.mean, std=self.std)
        image = {'left': left_image, 'right': right_image}
        target = {'left': left_target, 'right': right_target}
        return image, target

    def __call__(self, image, target):
        if isinstance(image, dict) and isinstance(target, dict):
            return self.call_double_view(image, target)
        else:
            return self.call_single_view(image, target)


class FixSizeCrop(object):
    def __init__(self, size=(256, 512), crop_map=False):
        assert isinstance(size, (int, tuple, list))
        if isinstance(size, int):
            size = (size, size)
        assert len(size) == 2
        self.h, self.w = size
        self.crop_map = crop_map

    def call_single_view(self, image, target):
        W, H = image.size
        x1 = random.randint(0, W - self.w)
        y1 = random.randint(0, H - self.h)
        image = F.crop(image, y1, x1, self.h, self.w)
        target = target.crop((x1, y1, x1 + self.w, y1 + self.h), self.crop_map)
        return image, target

    def call_double_view(self, image, target):
        left_image, right_image = image['left'], image['right']
        left_target, right_target = target['left'], target['right']
        W, H = left_image.size
        x1 = random.randint(0, W - self.w)
        y1 = random.randint(0, H - self.h)
        left_image = F.crop(left_image, y1, x1, self.h, self.w)
        right_image = F.crop(right_image, y1, x1, self.h, self.w)
        left_target = left_target.crop((x1, y1, x1 + self.w, y1 + self.h), self.crop_map)
        right_target = right_target.crop((x1, y1, x1 + self.w, y1 + self.h), self.crop_map)
        image = {'left': left_image, 'right': right_image}
        target = {'left': left_target, 'right': right_target}
        return image, target

    def __call__(self, image, target):
        if isinstance(image, dict) and isinstance(target, dict):
            return self.call_double_view(image, target)
        else:
            return self.call_single_view(image, target)


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue, )

    def call_single_view(self, image, target):
        image = self.color_jitter(image)
        return image, target

    def call_double_view(self, image, target):
        left_image, right_image = image['left'], image['right']
        left_target, right_target = target['left'], target['right']
        left_image, right_image = self.color_jitter(left_image), self.color_jitter(right_image)
        image = {'left': left_image, 'right': right_image}
        target = {'left': left_target, 'right': right_target}
        return image, target

    def __call__(self, image, target):
        if isinstance(image, dict) and isinstance(target, dict):
            return self.call_double_view(image, target)
        else:
            return self.call_single_view(image, target)


class ToRGB255(object):

    def call_single_view(self, image, target):
        image = image[:, :, ::-1]
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.astype(np.uint8)
        return image, target

    def call_double_view(self, image, target):
        left_image, right_image = image['left'], image['right']
        left_target, right_target = target['left'], target['right']

        left_image, left_target = self.call_single_view(left_image, left_target)
        right_image, right_target = self.call_single_view(right_image, right_target)
        image = {'left': left_image, 'right': right_image}
        target = {'left': left_target, 'right': right_target}
        return image, target

    def __call__(self, image, target):
        if isinstance(image, dict) and isinstance(target, dict):
            return self.call_double_view(image, target)
        else:
            return self.call_single_view(image, target)


class ToPILImage(object):

    def call_single_view(self, image, target):
        image = F.to_pil_image(image)
        return image, target

    def call_double_view(self, image, target):
        left_image, right_image = image['left'], image['right']
        left_target, right_target = target['left'], target['right']

        left_image, left_target = self.call_single_view(left_image, left_target)
        right_image, right_target = self.call_single_view(right_image, right_target)
        image = {'left': left_image, 'right': right_image}
        target = {'left': left_target, 'right': right_target}
        return image, target

    def __call__(self, image, target):
        if isinstance(image, dict) and isinstance(target, dict):
            return self.call_double_view(image, target)
        else:
            return self.call_single_view(image, target)
