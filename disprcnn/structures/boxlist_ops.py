# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from disprcnn.structures.bounding_box_3d import Box3DList
from .bounding_box import BoxList

from disprcnn.layers import nms as _box_nms
import numpy as np


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def intersect_pytorch(a, b):
    a = torch.unique(a)
    b = torch.unique(b)

    aux = torch.cat((a, b))
    aux = aux.sort().values

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    return int1d


def double_view_boxlist_nms(left_boxlist, right_boxlist,
                            nms_thresh, max_proposals=-1, score_field='scores',
                            use_keep='joint'):
    assert use_keep in ['joint', 'left', 'right']
    if nms_thresh <= 0:
        return left_boxlist, right_boxlist
    mode = left_boxlist.mode
    left_boxlist = left_boxlist.convert("xyxy")
    right_boxlist = right_boxlist.convert("xyxy")
    left_boxes = left_boxlist.bbox
    right_boxes = right_boxlist.bbox
    left_score = left_boxlist.get_field(score_field)
    right_score = right_boxlist.get_field(score_field)
    left_keep = _box_nms(left_boxes, left_score, nms_thresh)
    right_keep = _box_nms(right_boxes, right_score, nms_thresh)
    if use_keep == 'joint':
        # keep_np = torch.from_numpy(np.intersect1d(left_keep.cpu().numpy(),
        #                                           right_keep.cpu().numpy())).to(left_keep.device)
        keep = intersect_pytorch(left_keep, right_keep)
        # assert torch.allclose(keep, keep_np)
    elif use_keep == 'left':
        keep = left_keep
    else:
        keep = right_keep
    if max_proposals > 0:
        keep = keep[: max_proposals]
    left_boxlist = left_boxlist[keep]
    right_boxlist = right_boxlist[keep]
    left_boxlist = left_boxlist.convert(mode)
    right_boxlist = right_boxlist.convert(mode)
    return left_boxlist, right_boxlist


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
            (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def box3dlist_iou3d(box3dlist1, box3dlist2):
    from disprcnn.modeling.pointnet_module.point_rcnn.lib.utils.iou3d.iou3d_utils import boxes_iou3d_gpu
    box3d1 = box3dlist1.convert('xyzhwl_ry').bbox_3d.cuda().float()
    box3d2 = box3dlist2.convert('xyzhwl_ry').bbox_3d.cuda().float()
    if len(box3dlist1) == 0 or len(box3dlist2) == 0:
        return torch.empty((len(box3dlist1), len(box3dlist2)))
    return boxes_iou3d_gpu(box3d1, box3d2)


def box3dlist_ioubev(box3dlist1, box3dlist2):
    from disprcnn.modeling.pointnet_module.point_rcnn.lib.utils.iou3d.iou3d_utils import boxes_iou_bev
    from disprcnn.modeling.pointnet_module.point_rcnn.lib.utils.kitti_utils import boxes3d_to_bev_torch
    if len(box3dlist1) == 0 or len(box3dlist2) == 0:
        return torch.empty((len(box3dlist1), len(box3dlist2)))
    ioubev = boxes_iou_bev(
        boxes3d_to_bev_torch(box3dlist1.convert('xyzhwl_ry').bbox_3d.cuda()),
        boxes3d_to_bev_torch(box3dlist2.convert('xyzhwl_ry').bbox_3d.cuda()))
    return ioubev


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    if isinstance(tensors[0], Box3DList):
        return cat_boxlist3d(tensors)
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    maps = set(bboxes[0].maps())
    assert all(set(bbox.maps()) == maps for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)
    #     todo: fix map
    # for map in maps:
    #     data = _cat([bbox.get_map(map) for bbox in bboxes], dim=0)
    #     cat_boxes.add_map(map, data)
    return cat_boxes


def cat_boxlist3d(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[Box3DList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, Box3DList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    cat_boxes = Box3DList(_cat([bbox.bbox_3d for bbox in bboxes], dim=0), size, mode)

    return cat_boxes
