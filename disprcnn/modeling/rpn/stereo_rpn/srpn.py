import torch
from torch import nn, Tensor
import torch.nn.functional as F
from disprcnn.modeling.box_coder import BoxCoder
from disprcnn.modeling.rpn.anchor_generator import make_anchor_generator
from .inference import make_srpn_postprocessor
from .loss import make_srpn_loss_evaluator


def build_stereorpn(cfg, in_channels):
    return StereoRPN(cfg, in_channels)


class SRPNHead(nn.Module):
    """
    Adds a Stereo RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels * 4, num_anchors * 2, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels * 4, num_anchors * 6, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, left_features, right_features):
        logits = []
        bbox_reg = []
        for left_feature, right_feature in zip(left_features, right_features):
            lt = F.relu(self.conv(left_feature))
            rt = F.relu(self.conv(right_feature))
            t = torch.cat((lt, rt), dim=1)
            score = self.cls_logits(t)
            logit = score.view(score.shape[0], 2, -1, score.shape[3]).softmax(1).view(*score.shape)
            logits.append(logit)
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class StereoRPN(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()

        self.cfg = cfg.clone()

        self.anchor_generator = make_anchor_generator(cfg)

        self.head = SRPNHead(
            cfg, in_channels, self.anchor_generator.num_anchors_per_location()[0]
        )

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        self.box_selector_train = make_srpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        self.box_selector_test = make_srpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        self.loss_evaluator = make_srpn_loss_evaluator(cfg, rpn_box_coder)

    def forward(self, left_images, right_images, left_features, right_features, left_targets=None, right_targets=None):
        """
        Arguments:
            left_images (ImageList): images for which we want to compute the predictions
            right_images (ImageList): images for which we want to compute the predictions
            left_features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            right_features (list[Tensor]): features computed from the images that are
            used for computing the predictions. Each tensor in the list
            correspond to different feature levels
            left_targets (list[BoxList): ground-truth boxes present in the image (optional)
            right_targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression = self.head(left_features, right_features)
        anchors = self.anchor_generator(left_images, left_features)

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, left_targets, right_targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, left_targets, right_targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                left_boxes, right_boxes = self.box_selector_train(anchors, objectness, rpn_box_regression,
                                                                  left_targets, right_targets)
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, left_targets, right_targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return left_boxes, right_boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        left_boxes, right_boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in left_boxes
            ]
            left_boxes = [box[ind] for box, ind in zip(left_boxes, inds)]
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in right_boxes
            ]
            right_boxes = [box[ind] for box, ind in zip(right_boxes, inds)]
        return left_boxes, right_boxes, {}
