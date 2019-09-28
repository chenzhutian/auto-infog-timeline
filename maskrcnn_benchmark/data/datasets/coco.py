# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from os import path

import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

            ids_to_remove = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anno = self.coco.loadAnns(ann_ids)
                if all(
                    any(o <= 1 for o in obj["bbox"][2:])
                    for obj in anno
                    if obj["iscrowd"] == 0
                ):
                    ids_to_remove.append(img_id)

            self.ids = [
                img_id for img_id in self.ids if img_id not in ids_to_remove
            ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

        # self.combination_to_class = {
        #     'Linear_Chronological_Segmented': 0,
        #     'Linear_Chronological_Unified': 1,
        #     'Linear_Chronological_Faceted': 2,
        #     'Linear_Relative_Faceted': 3,
        #     'Linear_Log_Unified': 4,
        #     'Linear_Log_Faceted': 5,
        #     'Linear_Sequential_Unified': 6,
        #     'Linear_Sequential_Faceted': 7,
        #     'Linear_Collapsed_Unified': 8,
        # }
        # self.class_to_combination = {
        #     v: k for k, v in self.combination_to_class.items()
        # }

    def parseFilename(self, filename):
        try:
            created_date, created_time, representation, scale, layout, id = filename.split('_')
            return representation+'_'+scale+'_'+layout
        except Exception as ex:
            # print(ex)
            return None

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # masks = [obj["segmentation"] if obj["category_id"] != 3 else [] for obj in anno]
        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        image_info = self.get_img_info(idx)
        image_class = image_info.get('image_class', -1)
        if image_class != -1:
            target.add_field("image_class", torch.tensor(image_class))

        image_orientation = image_info.get('image_orientation', -2)
        if image_orientation != -2:
            if image_orientation == -1:
                image_orientation = 2
            target.add_field("image_orientation", torch.tensor(image_orientation))

        # class_name = self.get_img_class_by_idx(idx)
        # if class_name is not None:
        #     image_class = torch.tensor(self.combination_to_class.get(class_name, -1))
        #     target.add_field("image_class_label", image_class)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_orientation_by_id(self, img_id):
        img_data = self.coco.imgs[img_id]
        return img_data['image_orientation']

    def get_img_class_by_id(self, img_id):
        img_data = self.coco.imgs[img_id]
        # file_name, _ = path.splitext(img_data['file_name'])
        return img_data['image_class']

    def get_img_class_by_idx(self, index):
        file_name, _ = path.splitext(self.get_img_info(index)['file_name'])
        return self.parseFilename(file_name)

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def get_img_info_by_id(self, img_id):
        img_data = self.coco.imgs[img_id]
        return img_data