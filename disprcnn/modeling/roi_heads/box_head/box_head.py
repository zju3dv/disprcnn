# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from disprcnn.utils.timer import Timer
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

featuretimer = Timer()
predtimer = Timer()
pptimer = Timer()
PRINTTIME = False


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward_single_view(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads

        x = self.feature_extractor(features, proposals)


        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:

            result = self.post_processor((class_logits, box_regression), proposals)

            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )

    def forward_double_view(self, left_features, right_features, left_proposals, right_proposals, left_targets=None,
                            right_targets=None):
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                left_proposals, right_proposals = self.loss_evaluator.subsample(
                    {'left': left_proposals, 'right': right_proposals},
                    {'left': left_targets, 'right': right_targets})

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        if PRINTTIME:
            torch.cuda.synchronize()
            featuretimer.tic()
        x = self.feature_extractor({'left': left_features, 'right': right_features},
                                   {'left': left_proposals, 'right': right_proposals})
        if PRINTTIME:
            torch.cuda.synchronize()
            featuretimer.toc()
            print('feature', featuretimer.average_time)
        # xl = self.feature_extractor(left_features, left_proposals)
        # xr = self.feature_extractor(right_features, right_proposals)
        # x = torch.cat((xl, xr), dim=1)
        if PRINTTIME:
            torch.cuda.synchronize()
            predtimer.tic()
        class_logits, box_regression = self.predictor(x)
        if PRINTTIME:
            torch.cuda.synchronize()
            predtimer.toc()
            print('pred', predtimer.avg_time_str())
        if not self.training:
            if PRINTTIME:
                torch.cuda.synchronize()
                pptimer.tic()
            left_result, right_result = self.post_processor((class_logits, box_regression),
                                                            {'left': left_proposals, 'right': right_proposals})
            if PRINTTIME:
                torch.cuda.synchronize()
                pptimer.toc()
                print('pp', pptimer.average_time)
            return x, left_result, right_result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            left_proposals,
            right_proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )

    def forward(self, features, proposals, targets=None):
        if not isinstance(features, dict):
            return self.forward_single_view(features, proposals, targets)
        else:
            return self.forward_double_view(features['left'], features['right'],
                                            proposals['left'], proposals['right'],
                                            targets['left'], targets['right'])


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
