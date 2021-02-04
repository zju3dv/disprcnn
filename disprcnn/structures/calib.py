import numpy as np
from copy import deepcopy
from warnings import warn

import torch

from disprcnn.utils.kitti_utils import Calibration


class Calib:
    def __init__(self, calib: Calibration, image_size):
        # assert isinstance(calib, Calibration)
        self.calib = calib
        self.size = image_size

    @property
    def P0(self):
        return torch.tensor(self.calib.P0).float()

    @property
    def P2(self):
        return torch.tensor(self.calib.P2).float()

    @property
    def P3(self):
        return torch.tensor(self.calib.P3).float()

    @property
    def V2C(self):
        return torch.tensor(self.calib.V2C).float()

    @property
    def R0(self):
        return torch.tensor(self.calib.R0).float()

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    @property
    def stereo_fuxbaseline(self):
        return (self.P2[0, 3] - self.P3[0, 3]).item()

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
        ret.calib.P0[0] = ret.calib.P0[0] / self.width * width
        ret.calib.P0[1] = ret.calib.P0[1] / self.height * height
        ret.calib.P2[0] = ret.calib.P2[0] / self.width * width
        ret.calib.P2[1] = ret.calib.P2[1] / self.height * height
        ret.calib.P3[0] = ret.calib.P3[0] / self.width * width
        ret.calib.P3[1] = ret.calib.P3[1] / self.height * height
        return ret

    def __getitem__(self, item):
        return self

    @staticmethod
    def cart_to_hom(pts):
        if isinstance(pts, np.ndarray):
            pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        else:
            ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=pts.device)
            pts_hom = torch.cat((pts, ones), dim=1)
        return pts_hom

    @staticmethod
    def hom_to_cart(pts):
        return pts[:, :-1] / pts[:, -1:]

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

    def uncertainty_map_to_rect(self, uncertainty_map):
        assert isinstance(uncertainty_map, torch.Tensor)
        x_range = torch.arange(0, uncertainty_map.shape[1]).to(device=uncertainty_map.device)
        y_range = torch.arange(0, uncertainty_map.shape[0]).to(device=uncertainty_map.device)
        x_idxs, y_idxs = torch.meshgrid(x_range, y_range)
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        uncertainty = uncertainty_map[y_idxs, x_idxs]
        pts_rect = self.img_to_rect(x_idxs, y_idxs, uncertainty)
        return pts_rect, x_idxs, y_idxs

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

    def to_dict(self):
        return {'P0': self.P0, 'P2': self.P2, 'P3': self.P3, 'V2C': self.V2C, 'R0': self.R0}

    def rect_to_lidar(self, pc):
        """
        @param pc: N,3
        @return: N,3
        """
        A = self.V2C.t() @ self.R0.t()
        A = A.t()
        A, b = A.split([3, 1], dim=1)  # 3,3 3,1
        pc = pc.t()  # 3,N
        pc = pc - b
        velo = (torch.inverse(A) @ pc).t()
        return velo

    def rect_to_cam2(self, pts):
        if isinstance(pts, np.ndarray):
            C0to2 = self.C0to2
            pts = self.cart_to_hom(pts)
            pts = pts @ C0to2.T
            pts = self.hom_to_cart(pts)
        else:
            C0to2 = torch.from_numpy(self.C0to2).to(device=pts.device).float()
            pts = self.cart_to_hom(pts)
            pts = pts @ C0to2.t()
            pts = self.hom_to_cart(pts)
        return pts

    @property
    def C0to2(self):
        C = np.eye(4)
        C[0, 3] = -self.tx
        C[1, 3] = -self.ty
        return C

    @property
    def tx(self):
        return self.P2[0, 3] / (-self.fu)

    @property
    def ty(self):
        return self.P2[1, 3] / (-self.fv)

    @property
    def fu(self):
        return self.P2[0, 0]

    @property
    def fv(self):
        return self.P2[1, 1]

    @property
    def cu(self):
        return self.P2[0, 2]

    @property
    def cv(self):
        return self.P2[1, 2]

