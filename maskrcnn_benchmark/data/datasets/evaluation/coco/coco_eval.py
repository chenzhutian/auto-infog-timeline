import logging
import tempfile
import os
import torch
from collections import OrderedDict
from tqdm import tqdm

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


def do_coco_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    if box_only:
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    predictions, dataset, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        check_expected_results(res, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return
    logger.info("Preparing results for COCO format")
    coco_results = {}
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset)
    if "segm" in iou_types:
        logger.info("Preparing segm results")
        coco_results["segm"] = prepare_for_coco_segmentation(predictions, dataset)
    if "image_class" in iou_types:
        logger.info("Preparing image_class results")
        coco_results["image_class"] = prepare_for_image_classification(predictions, dataset)
    if "image_orientation" in iou_types:
        logger.info("Preparing image_orientation results")
        coco_results["image_orientation"] = prepare_for_image_orientation(predictions, dataset)

    # it seems that the results is the evaluating restuls
    # and the coco_results is the predecting results
    results = COCOResults(*iou_types) # init the package
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            # evaluate the data
            if iou_type == 'image_class':
                res = evaluate_predictions_on_image_class(
                    dataset, coco_results[iou_type]
                )
                results.update_image_class_acc(res)
            elif iou_type == 'image_orientation':
                res = evaluate_predictions_on_image_orientation(
                    dataset, coco_results[iou_type]
                )
                results.update_image_orientation_acc(res)
            else:
                res = evaluate_predictions_on_coco(
                    dataset.coco, coco_results[iou_type], file_path, iou_type
                )
                results.update(res) # store in package
    # how to add the classification results

    logger.info(results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))
    return results, coco_results

def prepare_for_image_orientation(predictions, dataset):
    classification_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        score, image_orientation = [tensor.item() for tensor in prediction.get_field("image_orientation_pred")]

        classification_results.append({
            "image_id": original_id,
            "image_orientation": image_orientation,
            "score": score
        })

    return classification_results

def prepare_for_image_classification(predictions, dataset):
    classification_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        score, image_class = [tensor.item() for tensor in prediction.get_field("image_class_pred")]
        # class_name = dataset.class_to_combination.get(image_class)
        
        classification_results.append({
            "image_id": original_id,
            "image_class": image_class,
            "score": score
        })

    return classification_results

def prepare_for_coco_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        # check the number of bbox
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        # image_width = dataset.coco.imgs[original_id]["width"]
        # image_height = dataset.coco.imgs[original_id]["height"]
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def prepare_for_coco_segmentation(predictions, dataset):
    import pycocotools.mask as mask_util
    import numpy as np

    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        # check the number of bbox
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        # image_width = dataset.coco.imgs[original_id]["width"]
        # image_height = dataset.coco.imgs[original_id]["height"]
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        # t = time.time()
        # Masker is necessary only if masks haven't been already resized.
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        # rles = prediction.get_field('mask')

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results


# inspired from Detectron
def evaluate_box_proposals(predictions, dataset, thresholds=None, area="all", limit=None):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        # TODO replace with get_img_info?
        # image_width = dataset.coco.imgs[original_id]["width"]
        # image_height = dataset.coco.imgs[original_id]["height"]
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("objectness").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }

def evaluate_predictions_on_image_orientation(coco_gt, coco_results):
    correct_count = 0
    image_count = len(coco_results)
    print('Evaluate annotation type *image_orientation*')
    T1 = {}
    F1 = {}
    T2 = {}

    for pred in coco_results:
        orientation_gt = coco_gt.get_img_orientation_by_id(pred['image_id'])
        T2[orientation_gt] = T2.get(orientation_gt, 0) + 1
        if orientation_gt == pred["image_orientation"]:
            T1[orientation_gt] = T1.get(orientation_gt, 0) + 1
            correct_count += 1
        else:
            print('Wrong orientation at ' + str(pred['image_orientation']), ', gt ' + str(orientation_gt), coco_gt.get_img_info_by_id(pred['image_id'])['file_name'])
            F1[pred["image_orientation"]] = F1.get(pred["image_orientation"], 0) + 1
    classes = []
    mF1 = 0
    mPre = 0
    mRec = 0
    for k in T2:
        pre = 0 if T1.get(k, 0) == 0 else T1.get(k, 0) / (T1.get(k, 0) + F1.get(k, 0))
        rec = T1.get(k, 0) / T2.get(k, 0)
        f1 = 0 if pre == 0 and rec == 0 else (2 * pre * rec) / (pre + rec)
        classes.append({
            'pre': pre,
            'rec': rec,
            'f1': f1
        })
        mF1 += (f1 * (T2.get(k, 0) / float(image_count)))
        mPre += (pre * (T2.get(k, 0) / float(image_count)))
        mRec += (rec * (T2.get(k, 0) / float(image_count)))
    accs = [float(correct_count) / float(image_count), classes, mF1, mPre, mRec] # maybe in the future we will have accuracy in per class
    print('DONE')
    for acc in accs:
        # TODO, re-fine the name in the future
        print('Global Image Orientation Accuracy: ', acc)
    return accs

def evaluate_predictions_on_image_class(coco_gt, coco_results):
    correct_count = 0
    image_count = len(coco_results)
    print('Evaluate annotation type *image_class*')
    T1 = {}
    F1 = {}
    T2 = {}

    for pred in coco_results:
        class_name_gt = coco_gt.get_img_class_by_id(pred['image_id'])
        T2[class_name_gt] = T2.get(class_name_gt, 0) + 1
        if class_name_gt == pred["image_class"]:
            T1[class_name_gt] = T1.get(class_name_gt, 0) + 1
            correct_count += 1
        else:
            print('Wrong class at ' + str(pred['image_class']), ', gt ' + str(class_name_gt), coco_gt.get_img_info_by_id(pred['image_id'])['file_name'])
            F1[pred["image_class"]] = F1.get(pred["image_class"], 0) + 1
    classes = []
    mF1 = 0
    mPre = 0
    mRec = 0
    for k in T2:
        pre = 0 if T1.get(k, 0) == 0 else T1.get(k, 0) / (T1.get(k, 0) + F1.get(k, 0))
        rec = T1.get(k, 0) / T2.get(k, 0)
        f1 = 0 if pre == 0 and rec == 0 else (2 * pre * rec) / (pre + rec)
        classes.append({
            'cls': k,
            'pre': pre,
            'rec': rec,
            'f1': f1
        })
        mF1 +=  (f1 *  (T2.get(k, 0) / float(image_count)))
        mPre += (pre * (T2.get(k, 0) / float(image_count)))
        mRec += (rec * (T2.get(k, 0) / float(image_count)))
        print(mF1, mPre, mRec)
    accs = [float(correct_count) / float(image_count), classes, mF1, mPre, mRec] # maybe in the future we will have accuracy in per class
    print('DONE')
    for acc in accs:
        # TODO, re-fine the name in the future
        print('Global Image Class Accuracy: ', acc)
    return accs

def evaluate_predictions_on_coco(coco_gt, coco_results, json_result_file, iou_type="bbox"):
    import json

    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(str(json_result_file))
    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.catIds = [1, 2, 3, 4, 5, 7] if iou_type == 'bbox' else [1, 3, 7]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval

class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoint": ["AP", "AP50", "AP75", "APm", "APl"],
        "image_class": ["ACC", 'PRE_REC', 'mF1', 'mPre', 'mRec'],
        "image_orientation": ["ACC", 'PRE_REC', 'mF1', 'mPre', 'mRec']
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "image_class", "image_orientation")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update_image_class_acc(self, acc_value):
        if acc_value is None:
            return
        res = self.results['image_class']
        metrics = COCOResults.METRICS['image_class']
        for idx, metric in enumerate(metrics):
            res[metric] = acc_value[idx]

    def update_image_orientation_acc(self, acc_value):
        if acc_value is None:
            return
        res = self.results['image_orientation']
        metrics = COCOResults.METRICS['image_orientation']
        for idx, metric in enumerate(metrics):
            res[metric] = acc_value[idx]

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)
