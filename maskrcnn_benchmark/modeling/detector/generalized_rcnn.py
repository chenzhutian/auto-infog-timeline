# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone, ResNetXFPN
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..classifier.classifier import build_classifier

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        if cfg.MODEL.CLASSIFIER_CLS_ON:
            self.classifier = build_classifier('image_class', 0.15, cfg)
        if cfg.MODEL.CLASSIFIER_ORIENT_ON:
            self.classifier2 =  build_classifier('image_orientation', 0.15, cfg)
        self.cfg = cfg.clone()

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)

        if isinstance(self.backbone, ResNetXFPN): 
            features, image_feature = self.backbone(images.tensors)
        else:
            features = self.backbone(images.tensors)

        # the image_class should be return in the testing mode
        if self.cfg.MODEL.CLASSIFIER_CLS_ON:
            # when not trainning, image_classes: Bx2 (class_prob, class_label)
            # when trainning, image_classes: Bx1 class_label
            _, image_classes, classifier_losses = self.classifier(features[-1], targets)
        
        if self.cfg.MODEL.CLASSIFIER_ORIENT_ON:
            # when not trainning, image_classes: Bx2 (class_prob, class_label)
            # when trainning, image_classes: Bx1 class_label
            _, image_orientations, classifier2_losses = self.classifier2(features[-1], targets)
        
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            if self.cfg.MODEL.CLASSIFIER_CLS_ON:
                losses.update(classifier_losses)
            if self.cfg.MODEL.CLASSIFIER_ORIENT_ON:
                losses.update(classifier2_losses)
            return losses

        # if is not trainning
        if self.cfg.MODEL.CLASSIFIER_CLS_ON:
            for image_class_prob, image_class_pred, boxlist in zip(*image_classes, result):
                boxlist.add_field("image_class_pred", (image_class_prob, image_class_pred))
        
        # if is not trainning
        if self.cfg.MODEL.CLASSIFIER_ORIENT_ON:
            for image_orientation_prob, image_orientation_pred, boxlist in zip(*image_orientations, result):
                boxlist.add_field("image_orientation_pred", (image_orientation_prob, image_orientation_pred))

        return result
