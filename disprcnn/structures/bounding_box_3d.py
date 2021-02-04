# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import math

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Box3DList(object):
    """
    This class represents a set of 3D bounding boxes.
    The bounding boxes are represented as a Nx7 Tensor in "ry_lhwxyz" mode or Nx24 in "corners" mode.
    In order ot uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.

    description of the "ry_lhwxyz":
        ry: the rotation around the Y-axis
        l, h, w: the length, height, width of the car
        x, y, z: the coordinate of the center of bottom face
    Draw 3d bounding box in image
    qs: (8,3) array of vertices for the 3d box in following order:
            2 -------- 6
           /|         /|
          1 -------- 5 .
          | |        | |
          . 3 -------- 7
          |/         |/
          0 -------- 4
    """

    def __init__(self, bbox_3d, size, mode="corners", ry=None, pose=None, frame="rect"):
        self.device = bbox_3d.device if isinstance(bbox_3d, torch.Tensor) else torch.device("cpu")
        bbox_3d = torch.as_tensor(bbox_3d, dtype=torch.float32, device=self.device)
        size = (int(size[0]), int(size[1]))
        # frame = 'velodyne'
        self.frame = frame
        if bbox_3d.ndimension() == 1:
            bbox_3d = bbox_3d.unsqueeze(0)
        if bbox_3d.ndimension() == 3:
            bbox_3d = bbox_3d.view(-1, 24)
        if bbox_3d.ndimension() != 2 and bbox_3d.numel() != 0:
            raise ValueError(
                "bbox_3d should have 2 dimensions, got {} {}".format(bbox_3d.ndimension(), bbox_3d.size())
            )
        if (mode == "ry_lhwxyz" or mode == "alpha_lhwxyz" or mode == "xyzhwl_ry") and bbox_3d.size(-1) != 7:
            if bbox_3d.numel() == 0:
                bbox_3d = torch.empty((0, 7))
            else:
                raise ValueError(
                    "last dimenion of bbox_3d in the ry_lhwxyz mode should have a "
                    "size of 7, got {}".format(bbox_3d.size(-1))
                )
        if mode == "corners" and bbox_3d.size(-1) != 24:
            if bbox_3d.numel() == 0:
                bbox_3d = torch.empty((0, 24))
            else:
                raise ValueError(
                    "last dimenion of bbox_3d in the corners mode should have a "
                    "size of 24, got {}".format(bbox_3d.size(-1))
                )

        if mode not in ("ry_lhwxyz", "corners", "xyzhwl_ry", "alpha_lhwxyz"):
            raise ValueError("mode should be 'ry_lhwxyz', 'alpha_lhwxyz', 'xyzhwl_ry' or 'corners'")

        self.pose = pose
        self.bbox_3d = bbox_3d
        self.ry = ry
        self.size = size  # (image_width, image_height)
        self.mode = mode

    def __getitem__(self, item):
        bbox_3d = Box3DList(self.bbox_3d[item], self.size, self.mode, pose=self.pose)
        return bbox_3d

    def __len__(self):
        return self.bbox_3d.shape[0]

    def convert(self, mode):
        if mode not in ("ry_lhwxyz", "alpha_lhwxyz", "xyzhwl_ry", "corners"):
            raise ValueError("mode should be 'ry_lhwxyz', 'alpha_lhwxyz', 'xyzhwl_ry' or 'corners'")

        if mode == self.mode:
            return self

        corners = self._split_into_corners()
        if mode == "corners":
            bbox_3d = torch.cat(corners, dim=-1)
            box_3d_list = Box3DList(bbox_3d, self.size, mode=mode)
        elif mode == "ry_lhwxyz" or mode == "xyzhwl_ry":
            # TODO calculate the mean
            dif = (corners[3] - corners[0])
            if self.ry is None:
                if self.frame == "velodyne":
                    ry = torch.atan2(dif[:, 1], dif[:, 0]).view(-1, 1)
                else:
                    ry = -(torch.atan2(dif[:, 2], dif[:, 0])).view(-1, 1)
            else:
                ry = self.ry
            xyz = ((corners[7] + corners[0]) / 2).view(-1, 3)
            l = torch.norm((corners[0] - corners[3]), dim=1).view(-1, 1)
            h = torch.norm((corners[0] - corners[1]), dim=1).view(-1, 1)
            w = torch.norm((corners[0] - corners[4]), dim=1).view(-1, 1)
            if mode == "xyzhwl_ry":
                bbox_3d = torch.cat((xyz, h, w, l, ry), dim=-1)
            else:
                bbox_3d = torch.cat((ry, l, h, w, xyz), dim=-1)

            box_3d_list = Box3DList(bbox_3d, self.size, mode=mode)

        return box_3d_list

    def _split_into_corners(self):
        """
        :return: a list of 8 tensor, with shape of (N, 3), each row is a point of corners
        """
        if self.mode == "corners":
            corners = self.bbox_3d.split(3, dim=-1)
            return corners

        elif (self.mode == "ry_lhwxyz" or self.mode == "xyzhwl_ry") and self.frame == "velodyne":
            if self.mode == "xyzhwl_ry":
                x, y, z, h, w, l, ry = self.bbox_3d.split(1, dim=-1)
            else:
                ry, l, h, w, x, y, z = self.bbox_3d.split(1, dim=-1)

            zero_col = torch.zeros(ry.shape).to(self.device)
            ones_col = torch.ones(ry.shape).to(self.device)
            y_corners = torch.cat((w / 2, w / 2, w / 2, w / 2, -w / 2, -w / 2, -w / 2, -w / 2), dim=1)
            z_corners = torch.cat((zero_col, h, h, zero_col, zero_col, h, h, zero_col), dim=1)
            x_corners = torch.cat((-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2), dim=1)
            corners_obj = (torch.stack((x_corners, y_corners, z_corners), dim=1))

            R = torch.cat([torch.cos(ry), -torch.sin(ry), zero_col, torch.sin(ry), torch.cos(ry), zero_col,
                           zero_col, zero_col, ones_col], dim=1)

            R = R.view(-1, 3, 3)

            corners_cam = torch.matmul(R, corners_obj) + torch.cat((x, y, z), dim=-1).view(-1, 3, 1)
            corners_cam = corners_cam.transpose(1, 2).reshape(-1, 24)
            return corners_cam.split(3, dim=-1)

        elif (self.mode == "ry_lhwxyz" or self.mode == "xyzhwl_ry") and self.frame == "rect":
            if self.mode == "xyzhwl_ry":
                x, y, z, h, w, l, ry = self.bbox_3d.split(1, dim=-1)
            else:
                ry, l, h, w, x, y, z = self.bbox_3d.split(1, dim=-1)

            zero_col = torch.zeros(ry.shape).to(self.device)
            ones_col = torch.ones(ry.shape).to(self.device)
            x_corners = torch.cat((-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2), dim=1)
            y_corners = torch.cat((zero_col, -h, -h, zero_col, zero_col, -h, -h, zero_col), dim=1)
            z_corners = torch.cat((w / 2, w / 2, w / 2, w / 2, -w / 2, -w / 2, -w / 2, -w / 2), dim=1)
            corners_obj = (torch.stack((x_corners, y_corners, z_corners), dim=1))

            R = torch.cat([torch.cos(ry), zero_col, torch.sin(ry), zero_col, ones_col, zero_col,
                           -torch.sin(ry), zero_col, torch.cos(ry)], dim=1)

            R = R.view(-1, 3, 3)
            corners_cam = torch.matmul(R, corners_obj) + torch.cat((x, y, z), dim=-1).view(-1, 3, 1)
            corners_cam = corners_cam.transpose(1, 2).reshape(-1, 24)
            return corners_cam.split(3, dim=-1)

        else:
            raise RuntimeError("Should not be here")

    def _split_into_ry_lhwxyz(self):
        box_3d_list = self.convert("ry_lhwxyz")
        return box_3d_list.box_3d.split(1, dim=-1)

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        pts_3d = pts_3d.view(-1, 3)
        ones = torch.ones(pts_3d.shape[0]).unsqueeze(1)
        pts_3d = torch.cat([pts_3d, ones], dim=1)
        return pts_3d

    def project_world_to_rect(self, boxes_world, pose):
        pts_3d_world = self.cart2hom(boxes_world)  # nx4
        box3d_rect = torch.inverse(pose) @ pts_3d_world.transpose(0, 1)
        return box3d_rect.transpose(0, 1)[:, :3]

    def project_rect_to_world(self, boxes_rect, pose):
        pts_3d_rect = self.cart2hom(boxes_rect)  # nx4
        box3d_world = pose @ pts_3d_rect.transpose(0, 1)
        return box3d_world.transpose(0, 1)[:, :3]

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )
        corners = self._split_into_corners()
        if method == FLIP_LEFT_RIGHT:
            for corner in corners:
                if self.frame == "velodyne":
                    corner[:, :] = self.project_world_to_rect(corner, self.pose)
                    corner[:, 0] = -corner[:, 0]
                    corner[:, :] = self.project_rect_to_world(corner, self.pose)
                else:
                    corner[:, 0] = -corner[:, 0]

        if method == FLIP_TOP_BOTTOM:
            for corner in corners:
                corner[:, 1] = -corner[:, 1]

        bbox_3d = torch.cat(corners, dim=-1)
        box_3d_list = Box3DList(bbox_3d, self.size, mode="corners")

        return box_3d_list

    # Tensor-like methods
    def to(self, device):
        bbox_3d = Box3DList(self.bbox_3d.to(device), self.size, self.mode)
        return bbox_3d

    def cuda(self):
        return self.to(torch.device('cuda'))

    def resize(self, size, *args, **kwargs):
        # TODO make sure image size is not changed before finished this part
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """
        return self

    def crop(self, box):
        return self

    def clip_to_image(self, remove_empty=True):
        return self

    def area(self):
        bbox_3d = self.bbox_3d
        if self.mode == "corners":
            corners = self._split_into_corners()
            l = torch.norm((corners[0] - corners[4]), dim=1).reshape(-1, 1)
            h = torch.norm((corners[0] - corners[1]), dim=1).reshape(-1, 1)
            w = torch.norm((corners[0] - corners[3]), dim=1).reshape(-1, 1)
            area = l * h * w
        else:
            _, l, h, w, _, _, _ = self._split_into_ry_lhwxyz()
            area = l * h * w

        return area

    def project_to_2d(self, P):
        """
        project the 3d bbox to camera plane using the intrinsic
        :param P: the intrinsic and extrinsics of camera. shape: (3, 4)
        :return bbox_2d: the projected bbox in camera plane, shape: (N, 8, 2)
        """
        box_3d_list = self.convert("corners")
        bbox_3d = box_3d_list.bbox_3d
        bbox_3d = bbox_3d.view(-1, 8, 3)

        n = bbox_3d.shape[0]
        ones = torch.ones((n, 8, 1)).to(self.bbox_3d.device)
        bbox_3d = torch.cat([bbox_3d, ones], dim=-1)
        bbox_2d = torch.matmul(P, bbox_3d.permute(0, 2, 1)).permute(0, 2, 1)

        # (N, 8, 2)
        bbox_2d = torch.stack((bbox_2d[:, :, 0] / bbox_2d[:, :, 2], bbox_2d[:, :, 1] / bbox_2d[:, :, 2]), dim=2)
        return bbox_2d

    def enlarge_box3d(self, extra_width):
        boxes3d = self.convert("xyzhwl_ry").bbox_3d
        large_boxes3d = boxes3d.clone()
        large_boxes3d[:, 3:6] += extra_width * 2
        large_boxes3d[:, 1] += extra_width
        box_3d_list = Box3DList(large_boxes3d, self.size, mode="xyzhwl_ry").convert(self.mode)
        return box_3d_list

    @staticmethod
    def get_faces():
        return [[2, 3, 7, 6],
                [2, 3, 0, 1],
                [6, 7, 4, 5],
                [0, 1, 5, 4],
                [0, 4, 5, 1],
                [5, 6, 2, 1]]

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes_3d={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s

    @property
    def center(self):
        b3d = self.convert('xyzhwl_ry').bbox_3d.clone()
        xyz = b3d[:, :3]
        h = b3d[:, 3]
        xyz[:, 1] = xyz[:, 1] - h / 2
        return xyz

    @property
    def dimension(self):
        b3d = self.convert('xyzhwl_ry').bbox_3d.clone()
        hwl = b3d[:, 3:6]
        return hwl

    def filter_points(self, pts):
        """
        return points in bounding boxes
        @param pts:
        @return:
        """
        from disprcnn.utils.utils_3d import filter_bbox_3d
        corners = self.convert('corners').bbox_3d.reshape(-1,8,3)
        filtered_pts = []
        for c in corners:
            _, _, keep = filter_bbox_3d(c, pts)
            p = pts[keep]
            filtered_pts.append(p)
        return filtered_pts

    def filter_out_points(self, pts):
        """
        return points not in any bounding box.
        @param pts:
        @return:
        """
        from disprcnn.utils.utils_3d import filter_bbox_3d
        corners = self.convert('corners').bbox_3d.reshape(-1,8,3)
        keep = torch.zeros((pts.shape[0])).bool()
        for c in corners:
            _, _, k = filter_bbox_3d(c, pts)
            keep = keep | k
        keep = ~keep
        filtered_pts = pts[keep]
        return filtered_pts

    # def from_objects(self, objects):
    #     """
    #     @param objects: list of ObjectLabelRecord
    #     @return:
    #     """
    #     box3d=[]
    #     for o in objects:
    #         box3d.append([o.])
    # @property
    # def x(self):
    #     b3d = self.convert('xyzhwl_ry')
    #     return b3d.bbox_3d[:, 0]
    #
    # @property
    # def y(self):
    #     b3d = self.convert('xyzhwl_ry')
    #     return b3d.bbox_3d[:, 1]
    #
    # @property
    # def z(self):
    #     b3d = self.convert('xyzhwl_ry')
    #     return b3d.bbox_3d[:, 2]
    #
    # @property
    # def h(self):
    #     b3d = self.convert('xyzhwl_ry')
    #     return b3d.bbox_3d[:, 3]
    #
    # @property
    # def w(self):
    #     b3d = self.convert('xyzhwl_ry')
    #     return b3d.bbox_3d[:, 4]
    #
    # @property
    # def l(self):
    #     b3d = self.convert('xyzhwl_ry')
    #     return b3d.bbox_3d[:, 5]
    #
    # @property
    # def ry(self):
    #     b3d = self.convert('xyzhwl_ry')
    #     return b3d.bbox_3d[:, 6]
    #
    # @property
    # def xyz(self):
    #     b3d = self.convert('xyzhwl_ry')
    #     return b3d.bbox_3d[:, 0:3]
    #
    # @property
    # def alpha(self):
    #     return self.ry + torch.atan(-self.x / self.z)


if __name__ == "__main__":
    # euler = (-0.06113487224401537, -0.010398221352184765, 0.35926017719345693)
    bbox_3d = torch.tensor([[-39.5482, 1.0015, 72.5878],
                            [-39.5280, -0.9614, 72.4898],
                            [-44.6086, -1.0026, 72.2687],
                            [-44.6288, 0.9603, 72.3666],
                            [-39.6350, 0.9002, 74.5999],
                            [-39.6148, -1.0627, 74.5020],
                            [-44.6954, -1.1039, 74.2808],
                            [-44.7155, 0.8590, 74.3788]], dtype=torch.float32)

    bbox_3d = bbox_3d.view(1, -1)

    box_3d_list = Box3DList(bbox_3d, (1280, 720))
    box_3d_list = box_3d_list.convert("ry_lhwxyz")
    # print("-----------convert to ry_lhwxyz-----------")
    # print("after convert: {}, annotation: {}".format(box_3d_list.bbox_3d[0, 0], euler[2]))
    # print("dif: {}".format(torch.norm(box_3d_list.bbox_3d[0, 0] - euler[2])))

    box_3d_list = box_3d_list.convert("corners")
    print("-----------convert to corners-----------")
    print("after convert: {}".format(box_3d_list.bbox_3d))
    print("annotation: {}".format(bbox_3d))
    print("dif: {}".format(torch.norm(box_3d_list.bbox_3d - bbox_3d)))
