import os

import numpy as np
from PIL import Image


class Calibration:
    def __init__(self, calibs):
        self.P0 = calibs['P0']  # 3 x 4
        self.P1 = calibs['P1']  # 3 x 4
        self.P2 = calibs['P2']  # 3 x 4
        self.P3 = calibs['P3']  # 3 x 4
        self.R0 = calibs['R0_rect']  # 3 x 3
        self.V2C = calibs['Tr_velo_to_cam']  # 3 x 4
        self.I2V = calibs['Tr_imu_to_velo']  # 3 x 4

        # Camera intrinsics and extrinsics
        # self.cu = self.P2[0, 2]
        # self.cv = self.P2[1, 2]
        # self.fu = self.P2[0, 0]
        # self.fv = self.P2[1, 1]
        # self.tx = self.P2[0, 3] / (-self.fu)
        # self.ty = self.P2[1, 3] / (-self.fv)

    @property
    def cu(self):
        return self.P2[0, 2]

    @property
    def cv(self):
        return self.P2[1, 2]

    @property
    def fu(self):
        return self.P2[0, 0]

    @property
    def fv(self):
        return self.P2[1, 1]

    @property
    def tx(self):
        return self.P2[0, 3] / (-self.fu)

    @property
    def ty(self):
        return self.P2[1, 3] / (-self.fv)

    @property
    def stereo_baseline(self):
        return self.P2[0, 3] - self.P3[0, 3]

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return: pts_rect:(N, 3)
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def depthmap_to_rect(self, depth_map):
        """
        :param depth_map: (H, W), depth_map
        :return: pts_rect(H*W, 3), x_idxs(N), y_idxs(N)
        """
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs

    def disparity_map_to_rect(self, disparity_map, epsilon=1e-6):
        depth_map = self.stereo_baseline / (disparity_map + epsilon)
        return self.depthmap_to_rect(depth_map)

    def disparity_map_to_depth_map(self, disparity_map, epsilon=1e-6):
        depth_map = self.stereo_baseline / (disparity_map + epsilon)
        return depth_map

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        """
        Can only process valid u, v, d, which means u, v can not beyond the image shape, reprojection error 0.02
        :param u: (N)
        :param v: (N)
        :param d: (N), the distance between camera and 3d points, d^2 = x^2 + y^2 + z^2
        :return:
        """
        assert self.fu == self.fv, '%.8f != %.8f' % (self.fu, self.fv)
        fd = np.sqrt((u - self.cu) ** 2 + (v - self.cv) ** 2 + self.fu ** 2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d ** 2 - x ** 2 - y ** 2)
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
        return pts_rect


def load_calib(kitti_root, split, imgid):
    if isinstance(imgid, int):
        imgid = '%06d' % imgid
    calib_dir = os.path.join(kitti_root, 'object', split, 'calib')
    absolute_path = os.path.join(calib_dir, imgid + '.txt')
    with open(absolute_path) as f:
        lines = {line.strip().split(':')[0]: list(map(float, line.strip().split(':')[1].split())) for line in
                 f.readlines()[:-1]}
    calibs = {'P0': np.array(lines['P0']).reshape((3, 4)),
              'P1': np.array(lines['P1']).reshape((3, 4)),
              'P2': np.array(lines['P2']).reshape((3, 4)),
              'P3': np.array(lines['P3']).reshape((3, 4)),
              'R0_rect': np.array(lines['R0_rect']).reshape((3, 3)),
              'Tr_velo_to_cam': np.array(lines['Tr_velo_to_cam']).reshape((3, 4)),
              'Tr_imu_to_velo': np.array(lines['Tr_imu_to_velo']).reshape((3, 4))}
    return Calibration(calibs)


def load_image_2(kitti_root, split, imgid):
    imgid = '%06d' % imgid
    img_dir = os.path.join(kitti_root, 'object', split, 'image_2')
    absolute_path = os.path.join(img_dir, imgid + '.png')
    rgb = Image.open(absolute_path)
    return rgb


from enum import IntEnum


class KITTIObjectClass(IntEnum):
    Car = 1
    Van = 2
    Truck = 3
    Pedestrian = 4
    Person_sitting = 5
    Cyclist = 6
    Tram = 7
    Misc = 8
    DontCare = 9


class KITTIObject3D:
    cls: KITTIObjectClass
    truncated: float
    occluded: float
    alpha: float
    x1: float
    y1: float
    x2: float
    y2: float
    h: float
    w: float
    l: float
    x: float
    y: float
    z: float
    ry: float

    def __init__(self, cls: KITTIObjectClass, truncated: float, occluded: float, alpha: float,
                 x1: float, y1: float, x2: float, y2: float,
                 h: float, w: float, l: float,
                 x: float, y: float, z: float, ry: float) -> None:
        super().__init__()
        self.ry = ry
        self.z = z
        self.y = y
        self.x = x
        self.l = l
        self.w = w
        self.h = h
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.alpha = alpha
        self.occluded = occluded
        self.truncated = truncated
        self.cls = cls


def load_label_2(kitti_root, split, imgid):
    imgid = '%06d' % imgid
    label_2_dir = os.path.join(kitti_root, 'object', split, 'label_2')
    absolute_path = os.path.join(label_2_dir, imgid + '.txt')
    with open(absolute_path) as f:
        lines = f.read().splitlines()
    labels = []
    for l in lines:
        items = l.split()
        cls = items[0]
        truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry = map(float, items[1:])
        label = KITTIObject3D(KITTIObjectClass[cls], truncated, occluded, alpha,
                              x1, y1, x2, y2, h, w, l, x, y, z, ry)
        labels.append(label)
    return labels


def load_label_3(kitti_root, split, imgid):
    imgid = '%06d' % imgid
    label_3_dir = os.path.join(kitti_root, 'object', split, 'label_3')
    absolute_path = os.path.join(label_3_dir, imgid + '.txt')
    with open(absolute_path) as f:
        lines = f.read().splitlines()
    labels = []
    for l in lines:
        items = l.split()
        cls = items[0]
        truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry = map(float, items[1:])
        label = KITTIObject3D(KITTIObjectClass[cls], truncated, occluded, alpha,
                              x1, y1, x2, y2, h, w, l, x, y, z, ry)
        labels.append(label)
    return labels
