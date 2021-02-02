# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) not in [4, 6]:
            raise ValueError(
                "last dimenion of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}
        self.PixelWise_map = {}
        self.mask_thresh = 0.5

    @property
    def shape(self):
        return self.size[::-1]

    @property
    def height(self):
        return self.size[1]

    @property
    def width(self):
        return self.size[0]

    @property
    def cx(self):
        bl = self.convert('xyxy').bbox.float()
        return (bl[:, 0] + bl[:, 2]) / 2

    @property
    def cy(self):
        bl = self.convert('xyxy').bbox.float()
        return (bl[:, 1] + bl[:, 3]) / 2

    @property
    def x1(self):
        return self.convert('xyxy').bbox[:, 0]

    @property
    def x2(self):
        return self.convert('xyxy').bbox[:, 2]

    @property
    def y1(self):
        return self.convert('xyxy').bbox[:, 1]

    @property
    def y2(self):
        return self.convert('xyxy').bbox[:, 3]

    def add_map(self, map, map_data):
        self.PixelWise_map[map] = map_data

    def get_map(self, map):
        return self.PixelWise_map[map]

    def has_map(self, map):
        return map in self.PixelWise_map

    def remove_map_if_exist(self, key):
        if self.has_map(key):
            self.PixelWise_map.pop(key)

    def maps(self):
        return list(self.PixelWise_map.keys())

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field, default=None):
        return self.extra_fields.get(field, default)

    def pop_field(self, field):
        return self.extra_fields.pop(field)

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def _copy_map(self, bbox):
        for k, v in bbox.PixelWise_map.items():
            self.PixelWise_map[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        bbox._copy_map(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box
        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            # bbox._copy_map(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor) and hasattr(v, 'resize'):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            for k, v in self.PixelWise_map.items():
                if not isinstance(v, torch.Tensor) and hasattr(v, 'resize'):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_map(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxList(scaled_box, size, mode="xyxy")
        bbox._copy_map(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor) and hasattr(v, 'resize'):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)
        for k, v in self.PixelWise_map.items():
            if not isinstance(v, torch.Tensor) and hasattr(v, 'resize'):
                v = v.resize(size, *args, **kwargs)
            bbox.add_map(k, v)

        return bbox.convert(self.mode)

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

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        bbox._copy_map(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        # for k, v in self.PixelWise_map.items():
        #     if not isinstance(v, torch.Tensor):
        #         v = v.transpose(method)
        #     bbox.add_map(k, v)
        return bbox.convert(self.mode)

    def crop(self, box, crop_map=False):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        if not crop_map:
            bbox._copy_map(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        if crop_map:
            for k, v in self.PixelWise_map.items():
                if not isinstance(v, torch.Tensor):
                    v = v.crop(box)
                bbox.add_map(k, v)
        return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        for k, v in self.PixelWise_map.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_map(k, v)
        return bbox

    def cpu(self):
        return self.to(torch.device('cpu'))

    def cuda(self):
        bbox = BoxList(self.bbox.cuda(), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "cuda"):
                v = v.cuda()
            bbox.add_field(k, v)
        for k, v in self.PixelWise_map.items():
            if hasattr(v, "cuda"):
                v = v.cuda()
            bbox.add_map(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        bbox._copy_map(self)
        for k, v in self.extra_fields.items():
            # print(k)
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            return self.remove_empty()
        return self

    def remove_empty(self):
        box = self.bbox
        keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
        return self[keep]

    def remove_small_area(self, thresh=100):
        if not self.has_field('masks'):
            return self
        masks = self.extra_fields['masks']
        if len(masks) == 0:
            return self
        keep = []
        max_area = 0
        # print('len(masks) =', len(masks))
        for i, mask in enumerate(masks):
            # mask = np.ascontiguousarray(polygon.convert(mode='mask').numpy().astype(np.uint8)[:, :, None])
            """
            refactor this function to adjust new segmentation structure.
            """
            mask = mask.convert('mask').instances.masks.numpy()[0]
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.findContours(,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(contour) for contour in contours]
            if len(areas) > 0:
                if max(areas) > max_area:
                    max_keep = i
                if max(areas) >= thresh:
                    keep.append(i)
        if len(keep) == 0:
            keep = [max_keep]
        return self[keep]

    def remove_small_side_length(self, thresh=20):
        bboxes = self.bbox
        if len(bboxes) == 0:
            return self
        keep = []
        # print('before removing,', len(bboxes))
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox.tolist()
            width, height = x2 - x1, y2 - y1
            if min(width, height) >= thresh:
                keep.append(i)
        # print('after removing', len(keep))
        return self[keep]

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")
        return area

    @property
    def widths(self):
        box = self.bbox
        if self.mode == 'xyxy':
            TO_REMOVE = 1
            widths = box[:, 2] - box[:, 0] + TO_REMOVE
        elif self.mode == 'xywh':
            widths = bbox[:, 2]
        else:
            raise RuntimeError("Should not be here")
        return widths

    @property
    def heights(self):
        box = self.bbox
        if self.mode == 'xyxy':
            TO_REMOVE = 1
            heights = box[:, 3] - box[:, 1] + TO_REMOVE
        elif self.mode == 'xywh':
            heights = bbox[:, 3]
        else:
            raise RuntimeError("Should not be here")
        return heights

    def copy_with_fields(self, fields, maps=None, clone_bbox=False):
        bbox = BoxList(self.bbox if not clone_bbox else self.bbox.clone(), self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            bbox.add_field(field, self.get_field(field))
        if maps is not None:
            if not isinstance(maps, (list, tuple)):
                maps = [maps]
            for map in maps:
                bbox.add_map(map, self.get_map(map))
        return bbox

    def clone(self, clone_bbox=False):
        return self.copy_with_fields(list(self.extra_fields.keys()), list(self.PixelWise_map.keys()), clone_bbox)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s

    def plot(self, **kwargs):
        import matplotlib.pyplot as plt
        for i, box in enumerate(self.convert('xywh').bbox.tolist()):
            x, y, w, h = box
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, **kwargs))
            if self.has_field('scores'):
                plt.text(x, y, '%.2f' % self.get_field('scores').tolist()[i], **kwargs)

    def sort_by_depth(self):
        assert self.has_field('box3d')
        z = self.get_field('box3d').convert('xyzhwl_ry').bbox_3d[:, 2]
        idx = z.argsort().tolist()
        return self[idx]


if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)
