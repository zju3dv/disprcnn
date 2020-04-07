import numpy as np
import torch


def pose_to_xyzr(pose):
    t = pose[:, 3]
    x, y, z = t
    sin_theta = pose[0, 2]
    cos_theta = pose[0, 0]
    theta = np.round(np.arctan2(sin_theta, cos_theta), 2)
    return x, y, z, theta


def xyzr_to_pose(x, y, z, r):
    pose = np.array([[np.cos(r), 0, np.sin(r), x],
                     [0, 1, 0, y],
                     [-np.sin(r), 0, np.cos(r), z]])
    return pose


def make_xyzcorners(l, h, w, num_vertex=8):
    if num_vertex == 8:
        xyzcorners = np.array([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                               [0, 0, 0, 0, -h, -h, -h, -h],
                               [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]]).T

    elif num_vertex == 15:
        xyzcorners = np.array([[l / 2, 0, w / 2],
                               [l / 2, 0, -w / 2],
                               [-l / 2, 0, -w / 2],
                               [-l / 2, 0, w / 2],
                               [l / 2, -h, w / 2],
                               [l / 2, -h, -w / 2],
                               [-l / 2, -h, -w / 2],
                               [-l / 2, -h, w / 2],
                               [0, -h / 2, 0],
                               [0, 0, 0],
                               [0, -h, 0],
                               [0, -h / 2, -w / 2],
                               [0, -h / 2, w / 2],
                               [l / 2, -h / 2, 0],
                               [-l / 2, -h / 2, 0]])
    elif num_vertex == 9:
        xyzcorners = np.array([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, 0],
                               [0, 0, 0, 0, -h, -h, -h, -h, -h / 2],
                               [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, 0]]).T
    else:
        raise ValueError()
    return xyzcorners


def filter_bbox_3d(bbox_3d, point):
    v45 = bbox_3d[5] - bbox_3d[4]
    v40 = bbox_3d[0] - bbox_3d[4]
    v47 = bbox_3d[7] - bbox_3d[4]
    # point -= bbox_3d[4]
    point = point - bbox_3d[4]
    m0 = torch.matmul(point, v45)
    m1 = torch.matmul(point, v40)
    m2 = torch.matmul(point, v47)

    cs = []
    for m, v in zip([m0, m1, m2], [v45, v40, v47]):
        c0 = 0 < m
        c1 = m < torch.matmul(v, v)
        c = c0 & c1
        cs.append(c)
    cs = cs[0] & cs[1] & cs[2]
    passed_inds = torch.nonzero(cs).squeeze(1)
    num_passed = torch.sum(cs)
    return num_passed, passed_inds, cs


class rotate_pc_along_y(object):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        f: focal length
    Output:
        pc: updated pc with XYZ rotated
    '''

    def __init__(self, proposals, f, rot_angle=None):
        if rot_angle is not None:
            self.rot_angle = rot_angle
        else:
            if isinstance(proposals, list):
                w = proposals[0].size[0] / 2
                bbox2d = torch.cat([proposal.bbox for proposal in proposals], dim=0)
            else:
                w = proposals.size[0] / 2
                bbox2d = proposals.bbox
            bbox_center_w = (bbox2d[:, 0] + bbox2d[:, 2]) / 2
            self.rot_angle = torch.atan2(bbox_center_w - w, f)

    def __call__(self, pc):
        cosval = torch.cos(self.rot_angle).unsqueeze(dim=1)
        sinval = torch.sin(self.rot_angle).unsqueeze(dim=1)
        rotmat = torch.cat([cosval, -sinval, sinval, cosval], dim=1)
        rotmat = rotmat.view(-1, 2, 2)
        pc = pc.permute(0, 2, 1)
        pc[:, :, [0, 2]] = torch.bmm(pc[:, :, [0, 2]], torch.transpose(rotmat, 1, 2).float())
        return pc.permute(0, 2, 1).contiguous()

    def rotate_back(self, pc):
        rot_angle = -self.rot_angle
        cosval = torch.cos(rot_angle).unsqueeze(dim=1)
        sinval = torch.sin(rot_angle).unsqueeze(dim=1)
        rotmat = torch.cat([cosval, -sinval, sinval, cosval], dim=1)
        rotmat = rotmat.view(-1, 2, 2)
        pc = pc.permute(0, 2, 1)
        pc[:, :, [0, 2]] = torch.bmm(pc[:, :, [0, 2]], torch.transpose(rotmat, 1, 2).float())
        return pc.permute(0, 2, 1).contiguous()
