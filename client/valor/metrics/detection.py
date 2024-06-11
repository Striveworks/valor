import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional

from valor.coretypes import Annotation, GroundTruth, Prediction
from valor.schemas import Box

# TODO: all of these assume the box is axis aligned


def area(rect: Box) -> float:
    """Computes the area of a rectangle"""
    return (rect.xmax - rect.xmin) * (rect.ymax - rect.ymin)


def intersection_area(rect1: Box, rect2: Box) -> float:
    """Computes the intersection area of two rectangles"""
    inter_xmin = max(rect1.xmin, rect2.xmin)
    inter_xmax = min(rect1.xmax, rect2.xmax)

    inter_ymin = max(rect1.ymin, rect2.ymin)
    inter_ymax = min(rect1.ymax, rect2.ymax)

    inter_width = max(inter_xmax - inter_xmin, 0)
    inter_height = max(inter_ymax - inter_ymin, 0)

    return inter_width * inter_height


def iou(rect1: Box, rect2: Box) -> float:
    """Computes the "intersection over union" of two rectangles"""
    inter_area = intersection_area(rect1, rect2)
    return inter_area / (area(rect1) + area(rect2) - inter_area)


def _match_array(
    ious: List[List[float]],
    iou_thres: float,
) -> List[Optional[int]]:
    """
    iou[i][j] should be the iou between predicted detection i and groundtruth detection j

    important assumption is that the predicted detections are sorted by confidence

    Returns
    -------
    array of match indices. value at index i is the index of the groundtruth object that
    best matches the ith detection, or None if that deteciton has no match
    """
    matches = []
    for i in range(len(ious)):
        best_match_idx, current_max_iou = None, None
        for j, iou in enumerate(ious[i]):
            if iou >= iou_thres and j not in matches:
                if best_match_idx is None or iou > current_max_iou:
                    best_match_idx = j
                    current_max_iou = iou

        matches.append(best_match_idx)

    return matches


def iou_matrix(
    predictions: List[Prediction],
    groundtruths: List[GroundTruth],
) -> List[List[float]]:
    """Returns a list of lists where the entry at [i][j]
    is the iou between `predictions[i]` and `groundtruths[j]`.
    """
    return [
        [iou(p.boundary, g.boundary) for g in groundtruths]
        for p in predictions
    ]


def _cumsum(a: list) -> list:
    ret = [0]
    for x in a:
        ret.append(ret[-1] + x)

    return ret[1:]


@dataclass
class DetectionMatchInfo:
    tp: bool
    score: float


def _get_tp_fp_single_image_single_class(
    predictions: List[Annotation],
    groundtruths: List[Annotation],
    iou_threshold: float,
) -> List[DetectionMatchInfo]:
    predictions = sorted(
        predictions,
        key=lambda p: p.score,
        reverse=True,
    )

    ious = iou_matrix(groundtruths=groundtruths, predictions=predictions)

    matches = _match_array(ious, iou_threshold)

    return [
        DetectionMatchInfo(tp=(match is not None), score=prediction.score)
        for match, prediction in zip(matches, predictions)
    ]


def ap(
    predictions: List[List[Prediction]],
    groundtruths: List[List[GroundTruth]],
    class_label: str,
    iou_thresholds: list[float],
) -> Dict[float, float]:
    """Computes the average precision. Return is a dict with keys
    `f"IoU={iou_thres}"` for each `iou_thres` in `iou_thresholds` as well as
    `f"IoU={min(iou_thresholds)}:{max(iou_thresholds)}"` which is the average
    of the scores across all of the IoU thresholds.
    """
    if len(predictions) != len(groundtruths):
        raise ValueError(
            "Number of predictions and groundtruths must be the same"
        )

    predictions = [
        [p for p in preds if p.class_label == class_label]
        for preds in predictions
    ]
    groundtruths = [
        [gt for gt in gts if gt.class_label == class_label]
        for gts in groundtruths
    ]

    # total number of groundtruth objects across all images
    n_gt = sum([len(gts) for gts in groundtruths])

    ret = {}
    if n_gt == 0:
        ret.update({f"IoU={iou_thres}": -1.0 for iou_thres in iou_thresholds})
    else:
        for iou_thres in iou_thresholds:
            match_infos = []
            for preds, gts in zip(predictions, groundtruths):
                match_infos.extend(
                    _get_tp_fp_single_image_single_class(
                        predictions=preds,
                        groundtruths=gts,
                        iou_threshold=iou_thres,
                    )
                )

            match_infos = sorted(
                match_infos, key=lambda m: m.score, reverse=True
            )

            tp = [float(m.tp) for m in match_infos]
            fp = [float(not m.tp) for m in match_infos]

            cum_tp = _cumsum(tp)
            cum_fp = _cumsum(fp)

            precisions = [
                ctp / (ctp + cfp) for ctp, cfp in zip(cum_tp, cum_fp)
            ]
            recalls = [ctp / n_gt for ctp in cum_tp]

            ret[f"IoU={iou_thres}"] = calculate_ap_101_pt_interp(
                precisions=precisions, recalls=recalls
            )

    # compute average over all IoUs
    ret[f"IoU={min(iou_thresholds)}:{max(iou_thresholds)}"] = sum(
        [ret[f"IoU={iou_thres}"] for iou_thres in iou_thresholds]
    ) / len(iou_thresholds)

    return ret


def compute_ap_metrics(
    predictions: List[List[Prediction]],
    groundtruths: List[List[GroundTruth]],
    iou_thresholds: List[float],
) -> dict:
    """Computes average precision metrics. Note that this is not an optimized method
    and is here for toy/test purposes. Will likely be (re)moved in future versions.

    The return dictionary, `d` indexes AP scores by `d["AP"][class_label][iou_key]`
    and mAP scores by `d["mAP"][iou_key]` where `iou_key` is `f"IoU={iou_thres}"` or
    `f"IoU={min(iou_thresholds)}:{max(iou_thresholds)}"`
    """
    class_labels = set(
        [pred.class_label for preds in predictions for pred in preds]
    ).union([gt.class_label for gts in groundtruths for gt in gts])

    ret = {
        "AP": {
            class_label: ap(
                predictions=predictions,
                groundtruths=groundtruths,
                class_label=class_label,
                iou_thresholds=iou_thresholds,
            )
            for class_label in class_labels
        }
    }

    def _ave_ignore_minus_one(a):
        num, denom = 0.0, 0.0
        for x in a:
            if x != -1:
                num += x
                denom += 1
        return num / denom

    keys_to_avg_over = [f"IoU={iou_thres}" for iou_thres in iou_thresholds] + [
        f"IoU={min(iou_thresholds)}:{max(iou_thresholds)}"
    ]
    ret["mAP"] = {
        k: _ave_ignore_minus_one(
            [ret["AP"][class_label][k] for class_label in class_labels]
        )
        for k in keys_to_avg_over
    }

    return ret


def calculate_ap_101_pt_interp(precisions, recalls):
    """Use the 101 point interpolation method (following torchmetrics)"""
    assert len(precisions) == len(recalls)

    if len(precisions) == 0:
        return 0

    data = list(zip(precisions, recalls))
    data.sort(key=lambda l: l[1])
    # negative is because we want a max heap
    prec_heap = [[-precision, i] for i, (precision, _) in enumerate(data)]
    prec_heap.sort()

    cutoff_idx = 0

    ret = 0

    for r in [0.01 * i for i in range(101)]:
        while cutoff_idx < len(data) and data[cutoff_idx][1] < r:
            cutoff_idx += 1
        while prec_heap and prec_heap[0][1] < cutoff_idx:
            heapq.heappop(prec_heap)
        if cutoff_idx >= len(data):
            continue
        ret -= prec_heap[0][0]
    return ret / 101
