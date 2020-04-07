from copy import deepcopy
from warnings import warn

import torch

from disprcnn.utils.kitti_utils import Calibration


class Calib:
    def __init__(self, calib: Calibration, image_size):
        assert isinstance(calib, Calibration)
        self.calib = calib
        self.size = image_size

    @property
    def P0(self):
        return self.calib.P0

    @property
    def P2(self):
        return self.calib.P2

    @property
    def P3(self):
        return self.calib.P3

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    @property
    def stereo_fuxbaseline(self):
        return self.P2[0, 3] - self.P3[0, 3]

    def crop(self, box):
        x1, y1, x2, y2 = box
        ret = Calib(deepcopy(self.calib), (x2 - x1, y2 - y1))
        ret.P0[0, 2] = ret.P0[0, 2] - x1
        ret.P0[1, 2] = ret.P0[1, 2] - y1
        ret.P2[0, 2] = ret.P2[0, 2] - x1
        ret.P2[1, 2] = ret.P2[1, 2] - y1
        ret.P3[0, 2] = ret.P3[0, 2] - x1
        ret.P3[1, 2] = ret.P3[1, 2] - y1
        return ret

    def resize(self, dst_size):
        """
        :param dst_size:width, height
        :return:
        """
        assert len(dst_size) == 2
        if any(a < 0 for a in dst_size):
            warn('dst size < 0, size will not change')
            return self
        width, height = dst_size
        ret = Calib(deepcopy(self.calib), (width, height))
        ret.P0[0] = ret.P0[0] / self.width * width
        ret.P0[1] = ret.P0[1] / self.height * height
        ret.P2[0] = ret.P2[0] / self.width * width
        ret.P2[1] = ret.P2[1] / self.height * height
        ret.P3[0] = ret.P3[0] / self.width * width
        ret.P3[1] = ret.P3[1] / self.height * height
        return ret

    def __getitem__(self, item):
        return self

    def cart_to_hom(self, pts):
        return self.calib.cart_to_hom(pts)

    def lidar_to_rect(self, pts_lidar):
        return self.calib.lidar_to_rect(pts_lidar)

    def rect_to_img(self, pts_rect):
        return self.calib.rect_to_img(pts_rect)

    def lidar_to_img(self, pts_lidar):
        return self.calib.lidar_to_img(pts_lidar)

    def img_to_rect(self, u, v, depth_rect):
        if isinstance(u, torch.Tensor):
            x = ((u.float() - self.calib.cu) * depth_rect) / self.calib.fu + self.calib.tx
            y = ((v.float() - self.calib.cv) * depth_rect) / self.calib.fv + self.calib.ty
            pts_rect = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), dim=1)
            return pts_rect
        else:
            return self.calib.img_to_rect(u, v, depth_rect)

    def depthmap_to_rect(self, depth_map):
        if isinstance(depth_map, torch.Tensor):
            x_range = torch.arange(0, depth_map.shape[1]).to(device=depth_map.device)
            y_range = torch.arange(0, depth_map.shape[0]).to(device=depth_map.device)
            x_idxs, y_idxs = torch.meshgrid(x_range, y_range)
            x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
            depth = depth_map[y_idxs, x_idxs]
            pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
            return pts_rect, x_idxs, y_idxs
        else:
            return self.calib.depthmap_to_rect(depth_map)

    def disparity_map_to_rect(self, disparity_map, sigma=1e-6):
        depth_map = self.stereo_fuxbaseline / (disparity_map + sigma)
        return self.depthmap_to_rect(depth_map)

    def corners3d_to_img_boxes(self, corners3d):
        if isinstance(corners3d, torch.Tensor):
            device = corners3d.device
            dtype = corners3d.dtype
            corners3d = corners3d.cpu().numpy()
            boxes, boxes_corner = self.calib.corners3d_to_img_boxes(corners3d)
            boxes = torch.from_numpy(boxes).to(device=device, dtype=dtype)
            boxes_corner = torch.from_numpy(boxes_corner).to(device=device, dtype=dtype)
            return boxes, boxes_corner
        else:
            return self.calib.corners3d_to_img_boxes(corners3d)

    def camera_dis_to_rect(self, u, v, d):
        return self.calib.camera_dis_to_rect(u, v, d)

    def transpose(self, method):
        return self
