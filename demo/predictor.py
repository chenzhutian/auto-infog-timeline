# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
import numpy as np
from torchvision import transforms as T

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util

class COCODemo(object):

    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)

        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

CATEGORIES = [
    "__background",
    "event_image", #1
    "event_text", #2
    "annotation_image", #3
    "annotation_text", #4
    "icon", # 5
    "@deprecated", #6
    "main_body" # 7
]

DRAW_ORDER = [
    1,
    7,
    3,
    2,
    4,
    5,
    6,
]

CATEGORIES_COLOR = [
    [0, 0, 0], # 0
    [255, 192, 0], # 1
    [52, 73, 94], # 2
    [91, 155, 213], # 3
    [175, 122, 196], # 4
    [252, 104, 104], # 5
    [255, 192, 0], # 6
    [26, 188, 156] # 7
]

def draw_on_image(image, top_predictions, mask_on = True):
    result = image.copy()
    result = overlay_boxes(result, top_predictions)
    if mask_on:
        result = overlay_mask(result, top_predictions)
    result = overlay_class_names(result, top_predictions)

    return result

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    for c_label in DRAW_ORDER:
        for box, label in zip(boxes, labels):
            if label == c_label:
                color = CATEGORIES_COLOR[label] # [int(color[0]), int(color[1]), int(color[2])]
                box = box.to(torch.int64)
                top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
                image = cv2.rectangle(
                    image, tuple(top_left), tuple(bottom_right), tuple(color), 3
                )

    return image

def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy

def overlay_mask(image, predictions):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    masks = predictions.get_field("mask").numpy()
    labels = predictions.get_field("labels")

    for c_label in DRAW_ORDER:
        for mask, label in zip(masks, labels):
            if label != c_label:
                continue
            color = CATEGORIES_COLOR[label]
            thresh = mask[0, :, :, None]
            contours, _ = findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            overlay = image.copy()
            overlay = cv2.drawContours(overlay, contours, -1, color, -1)
            alpha = 0.4  # Transparency factor.
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            image = cv2.drawContours(image, contours, -1, color, 3)

    composite = image

    return composite

def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    labels_text = [CATEGORIES[i] for i in labels]
    boxes = predictions.bbox

    img_h, img_w, _ = image.shape

    template = "{} {}: {:.2f}"
    abv_map = {
        'annotation_image': 'am',
        'annotation_text': 'at',
        'event_image': 'em',
        'event_text': 'et',
        'main_body': 'mb'
    }
    for c_label in DRAW_ORDER:
        for b_id, (box, score, label, label_text) in enumerate(zip(boxes, scores, labels, labels_text)):
            if label != c_label:
                continue
            x, y = box[:2]
            color = CATEGORIES_COLOR[label]
            label_text = abv_map.get(label_text, label_text)
            s = template.format(b_id, label_text, score)
            fontScale = img_w / 2000.0
            (text_width, text_height) = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, thickness=1)[0]
            box_coords = ((x, y), (x + text_width - 2, y - text_height - 2))
            cv2.rectangle(image, box_coords[0], box_coords[1], color, cv2.FILLED)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 1
            )

    return image
