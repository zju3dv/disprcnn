import os
import pickle
from typing import Union
from warnings import warn

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


class Calibration:
    def __init__(self, calibs, image_size):
        """
        :param calibs:
        :param image_size: width, height
        """
        self.P0 = calibs['P0']  # 3 x 4
        self.P1 = calibs['P1']  # 3 x 4
        self.P2 = calibs['P2']  # 3 x 4
        self.P3 = calibs['P3']  # 3 x 4
        self.R0 = calibs['R0_rect']  # 3 x 3
        self.V2C = calibs['Tr_velo_to_cam']  # 3 x 4
        self.I2V = calibs['Tr_imu_to_velo']  # 3 x 4
        self.size = image_size

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
    def C0to2(self):
        C = np.eye(4)
        C[0, 3] = -self.tx
        C[1, 3] = -self.ty
        return C

    @property
    def C2to0(self):
        C = np.eye(4)
        C[0, 3] = self.tx
        C[1, 3] = self.ty
        return C

    @property
    def stereo_baseline(self):
        return self.P2[0, 3] - self.P3[0, 3]

    @staticmethod
    def cart_to_hom(pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        check_type(pts)
        if isinstance(pts, np.ndarray):
            pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        else:
            ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=pts.device)
            pts_hom = torch.cat((pts, ones), dim=1)
        return pts_hom

    @staticmethod
    def hom_to_cart(pts):
        """
        :param pts: (N, 4 or 3)
        :return pts_hom: (N, 3 or 2)
        """
        check_type(pts)
        return pts[:, :-1] / pts[:, -1:]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """

        check_type(pts_lidar)
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        if isinstance(pts_lidar_hom, np.ndarray):
            pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        else:
            device = pts_lidar_hom.device
            pts_rect = pts_lidar_hom @ torch.tensor(self.V2C).float().t().to(device=device) @ torch.tensor(
                self.R0).float().t().to(device=device)
        return pts_rect

    def rect_to_lidar(self, pts_rect):
        pts_ref = self.rect_to_ref(pts_rect)
        pts_lidar = self.ref_to_lidar(pts_ref)
        return pts_lidar

    def rect_to_ref(self, pts_rect):
        check_type(pts_rect)
        if isinstance(pts_rect, np.ndarray):
            return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_rect)))
        else:
            device = pts_rect.device
            R0 = torch.tensor(self.R0).float().to(device=device)
            return (torch.inverse(R0) @ pts_rect.t()).t()

    def ref_to_rect(self, pts_ref):
        check_type(pts_ref)
        if isinstance(pts_ref, np.ndarray):
            return np.transpose(np.dot(self.R0, np.transpose(pts_ref)))
        else:
            device = pts_ref.device
            R0 = torch.tensor(self.R0).float().to(device=device)
            return (R0 @ pts_ref.t()).t()

    def ref_to_lidar(self, pts_ref):
        check_type(pts_ref)
        pts_3d_ref = self.cart_to_hom(pts_ref)  # nx4
        if isinstance(pts_3d_ref, np.ndarray):
            return np.dot(pts_3d_ref, np.transpose(self.C2V))
        else:
            device = pts_3d_ref.device
            C2V = torch.tensor(self.C2V).float().to(device=device)
            return pts_3d_ref @ C2V.t()

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        check_type(pts_rect)
        pts_rect_hom = self.cart_to_hom(pts_rect)
        if isinstance(pts_rect_hom, np.ndarray):
            pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
            pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
            pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        else:
            device = pts_rect_hom.device
            P2 = torch.tensor(self.P2).float().to(device=device)
            pts_2d_hom = pts_rect_hom @ P2.t()
            pts_img = (pts_2d_hom[:, 0:2].t() / pts_rect_hom[:, 2]).t()
            pts_rect_depth = pts_2d_hom[:, 2] - P2.t()[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

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

    def cam2_to_rect(self, pts):
        if isinstance(pts, np.ndarray):
            C2to0 = self.C2to0
            pts = self.cart_to_hom(pts)
            pts = pts @ C2to0.T
            pts = self.hom_to_cart(pts)
        else:
            C2to0 = torch.from_numpy(self.C2to0).to(device=pts.device).float()
            pts = self.cart_to_hom(pts)
            pts = pts @ C2to0.t()
            pts = self.hom_to_cart(pts)
        return pts

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        check_type(pts_lidar)
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
        check_type(u)
        check_type(v)
        check_type(depth_rect)
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        if isinstance(depth_rect, np.ndarray):
            pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        else:
            pts_rect = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), dim=1)
        return pts_rect

    def depth_map_to_rect(self, depth_map):
        """
        :param depth_map: (H, W), depth_map
        :return: pts_rect(H*W, 3), x_idxs(N), y_idxs(N)
        """
        check_type(depth_map)
        if isinstance(depth_map, np.ndarray):
            x_range = np.arange(0, depth_map.shape[1])
            y_range = np.arange(0, depth_map.shape[0])
            x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        else:
            x_range = torch.arange(0, depth_map.shape[1]).to(device=depth_map.device)
            y_range = torch.arange(0, depth_map.shape[0]).to(device=depth_map.device)
            y_idxs, x_idxs = torch.meshgrid(y_range, x_range)
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs

    def disparity_map_to_rect(self, disparity_map, epsilon=1e-6):
        check_type(disparity_map)
        depth_map = self.stereo_baseline / (disparity_map + epsilon)
        return self.depth_map_to_rect(depth_map)

    def disparity_map_to_depth_map(self, disparity_map, epsilon=1e-6):
        check_type(disparity_map)
        depth_map = self.stereo_baseline / (disparity_map + epsilon)
        return depth_map

    def depth_map_to_disparity_map(self, depth_map, epsilon=1e-6):
        check_type(depth_map)
        disparity_map = self.stereo_baseline / (depth_map + epsilon)
        return disparity_map

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        check_type(corners3d)
        sample_num = corners3d.shape[0]
        if isinstance(corners3d, np.ndarray):
            corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

            img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

            x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
            x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
            x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

            boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
            boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)
        else:
            device = corners3d.device
            ones = torch.ones((sample_num, 8, 1), dtype=torch.float).to(device=device)
            corners3d_hom = torch.cat((corners3d, ones),
                                      dim=2)  # (N, 8, 4)
            P2 = torch.tensor(self.P2).float().to(device=device)
            img_pts = corners3d_hom @ P2.t()  # (N, 8, 3)
            x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
            x1, y1 = torch.min(x, dim=1).values, torch.min(y, dim=1).values
            x2, y2 = torch.max(x, dim=1).values, torch.max(y, dim=1).values

            boxes = torch.cat((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), dim=1)
            boxes_corner = torch.cat((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), dim=2)
        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        """
        Can only process valid u, v, d, which means u, v can not beyond the image shape, reprojection error 0.02
        :param u: (N)
        :param v: (N)
        :param d: (N), the distance between camera and 3d points, d^2 = x^2 + y^2 + z^2
        :return:
        """
        check_type(u)
        check_type(v)
        check_type(d)
        assert self.fu == self.fv, '%.8f != %.8f' % (self.fu, self.fv)
        fd = ((u - self.cu) ** 2 + (v - self.cv) ** 2 + self.fu ** 2) ** 0.5
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = (d ** 2 - x ** 2 - y ** 2) ** 0.5
        if isinstance(x, np.ndarray):
            pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
        else:
            pts_rect = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1)
        return pts_rect

    def imu_to_velo(self, pts_3d_imu):
        pts_3d_imu = self.cart_to_hom(pts_3d_imu)  # nx4
        if isinstance(pts_3d_imu, np.ndarray):
            return np.dot(pts_3d_imu, np.transpose(self.I2V))
        else:
            return pts_3d_imu @ torch.from_numpy(self.I2V).to(pts_3d_imu.device).float().t()

    def velo_to_imu(self, pts_3d_velo):
        pts_3d_velo = self.cart_to_hom(pts_3d_velo)  # nx4
        if isinstance(pts_3d_velo, np.ndarray):
            return pts_3d_velo @ self.V2I.T
        else:
            return pts_3d_velo @ torch.from_numpy(self.V2I).to(pts_3d_velo.device).float().t()

    def rect_to_imu(self, pts):
        return self.velo_to_imu(self.rect_to_lidar(pts))

    def imu_to_rect(self, pts):
        return self.lidar_to_rect(self.imu_to_velo(pts))

    def todict(self):
        calibs = {}
        calibs['P0'] = self.P0
        calibs['P1'] = self.P1
        calibs['P2'] = self.P2
        calibs['P3'] = self.P3
        calibs['R0_rect'] = self.R0
        calibs['Tr_velo_to_cam'] = self.V2C
        calibs['Tr_imu_to_velo'] = self.I2V
        d = {'calibs': calibs, 'image_size': self.size}
        return d

    def crop(self, box):
        x1, y1, x2, y2 = box
        d = self.todict()
        ret = Calibration(d['calibs'], (x2 - x1, y2 - y1))
        ret.P0[0, 2] = ret.P0[0, 2] - x1
        ret.P0[1, 2] = ret.P0[1, 2] - y1
        ret.P2[0, 2] = ret.P2[0, 2] - x1
        ret.P2[1, 2] = ret.P2[1, 2] - y1
        ret.P3[0, 2] = ret.P3[0, 2] - x1
        ret.P3[1, 2] = ret.P3[1, 2] - y1
        return ret

    def resize(self, dst_size):
        """
        :param dst_size: width, height
        :return:
        """
        assert len(dst_size) == 2
        if any(a < 0 for a in dst_size):
            warn('dst size < 0, size will not change')
            return self
        width, height = dst_size
        d = self.todict()
        ret = Calibration(d['calibs'], (width, height))
        ret.P0[0] = ret.P0[0] / self.width * width
        ret.P0[1] = ret.P0[1] / self.height * height
        ret.P2[0] = ret.P2[0] / self.width * width
        ret.P2[1] = ret.P2[1] / self.height * height
        ret.P3[0] = ret.P3[0] / self.width * width
        ret.P3[1] = ret.P3[1] / self.height * height
        return ret

    def filter_fov_pts(self, pts_rect):
        pts_img, pts_rect_depth = self.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < self.width)
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < self.height)
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        return pts_valid_flag

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]


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
    H, W, _ = load_image_info(kitti_root, split, imgid)
    image_size = (W, H)
    return Calibration(calibs, image_size)


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


def check_type(arr):
    assert isinstance(arr, (np.ndarray, torch.Tensor))


class _ImageInfoCache:
    instance = None

    def __init__(self):
        raise SyntaxError('can not instance, please use get_instance')

    @staticmethod
    def get_instance(root, split):
        if _ImageInfoCache.instance is None:
            _ImageInfoCache.instance = object.__new__(_ImageInfoCache)
            _ImageInfoCache.instance.post_init(root, split)
        return _ImageInfoCache.instance

    def post_init(self, root, split):

        cache_dir = os.path.expanduser('~/.dl_ext/vision_ext/datasets/kitti/image_info')
        self.cache_path = os.path.join(cache_dir, f'{split}.pkl')
        os.makedirs(cache_dir, exist_ok=True)
        self.root = root
        self.split = split
        self.infos = self.load_info()

    def load_info(self):
        if os.path.exists(self.cache_path):
            return pickle.load(open(self.cache_path, 'rb'))
        else:
            print('Image info cache not found. Generating...')
            img_shapes = []
            n = len(os.listdir(os.path.join(self.root, 'object', self.split, 'image_2')))
            for i in tqdm(range(n), leave=False):
                img2 = np.array(load_image_2(self.root, self.split, i))
                img_shapes.append(img2.shape)
            pickle.dump(img_shapes, open(self.cache_path, 'wb'))
            print('Done.')
            return img_shapes


def load_image_info(kitti_root: str, split: str, imgid: Union[int, str]):
    if not isinstance(imgid, int):
        imgid = int(imgid)
    img_info = _ImageInfoCache.get_instance(kitti_root, split)
    return img_info.infos[imgid]
