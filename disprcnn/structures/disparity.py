from typing import Union
from warnings import warn

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from disprcnn.layers import interpolate


class DisparityMap:
    data: Tensor

    def __init__(self, data: Union[np.ndarray, Tensor]):
        if isinstance(data, Tensor):
            pass
        elif isinstance(data, np.ndarray):
            data = torch.tensor(data)
        else:
            raise TypeError('type not supported.')
        self.data = data.float()

    def clone(self):
        return DisparityMap(self.data.clone())

    @property
    def size(self):
        return self.data.shape[::-1]

    @property
    def width(self):
        return self.data.shape[1]

    @property
    def height(self):
        return self.data.shape[0]

    def resize(self, dst_size, use_max_pooling=False):
        """
        :param use_max_pooling:
        :param dst_size: (width, height)
        :return:
        """
        if any(a < 0 for a in dst_size):
            warn('dst size < 0, size will not change.')
            return self.clone()
        dst_width, dst_height = map(round, dst_size)
        if not use_max_pooling:
            dst_tensor = interpolate(self.data.clone()[None, None],
                                     (dst_height, dst_width),
                                     mode='bilinear',
                                     align_corners=True)[0, 0]
        else:
            input = self.data.clone()[None, None, :, :]
            positive = (input > 0).float()
            negative = (input < 0).float()
            dst_tensor = F.adaptive_max_pool2d(input * positive, (dst_height, dst_width))[0, 0] - \
                         F.adaptive_max_pool2d(-input * negative, (dst_height, dst_width))[0, 0]
        dst_tensor = dst_tensor / self.width * dst_width
        dst = DisparityMap(dst_tensor)
        return dst

    def to(self, device):
        data = self.data.to(device)
        return DisparityMap(data)

    def crop(self, box):
        """
        :param box: The crop rectangle, as a (left, upper, right, lower)-tuple.
        :returns:
        """
        x1, y1, x2, y2 = map(round, box)
        cropped_tensor = self.data[y1:y2, x1:x2]
        dst_t = torch.zeros((y2 - y1, x2 - x1)).type_as(self.data)
        dst_t[:cropped_tensor.shape[0], :cropped_tensor.shape[1]] = cropped_tensor
        assert dst_t.shape[0] == y2 - y1 and dst_t.shape[1] == x2 - x1
        return DisparityMap(dst_t)

    def __sub__(self, other):
        d = self.clone()
        d.data = d.data - other
        return d
