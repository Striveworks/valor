from dataclasses import dataclass
from typing import Dict, List

from velour.data_types import (
    BoundingPolygon,
    GroundTruthDetection,
    PredictedDetection,
)


def area(rect: BoundingPolygon) -> float:
    """Computes the area of a rectangle"""
    assert len(rect.points) == 4
    xs = [pt.x for pt in rect.points]
    ys = [pt.y for pt in rect.points]

    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def intersection_area(rect1: BoundingPolygon, rect2: BoundingPolygon) -> float:
    """Computes the intersection area of two rectangles"""
    assert len(rect1.points) == len(rect2.points) == 4

    xs1 = [pt.x for pt in rect1.points]
    xs2 = [pt.x for pt in rect2.points]

    ys1 = [pt.y for pt in rect1.points]
    ys2 = [pt.y for pt in rect2.points]

    inter_xmin = max(min(xs1), min(xs2))
    inter_xmax = min(max(xs1), max(xs2))

    inter_ymin = max(min(ys1), min(ys2))
    inter_ymax = min(max(ys1), max(ys2))

    inter_width = max(inter_xmax - inter_xmin, 0)
    inter_height = max(inter_ymax - inter_ymin, 0)

    return inter_width * inter_height


def iou(rect1: BoundingPolygon, rect2: BoundingPolygon) -> float:
    """Computes the "intersection over union" of two rectangles"""
    inter_area = intersection_area(rect1, rect2)
    return inter_area / (area(rect1) + area(rect2) - inter_area)


def _match_array(
    ious: List[List[float]],
    iou_thres: float,
) -> List[int | None]:
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
    predictions: List[PredictedDetection],
    groundtruths: List[GroundTruthDetection],
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
    predictions: List[PredictedDetection],
    groundtruths: List[GroundTruthDetection],
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
    predictions: List[List[PredictedDetection]],
    groundtruths: List[List[GroundTruthDetection]],
    class_label: str,
    iou_thresholds: list[float],
) -> Dict[float, float]:
    """Computes the average precision. Return is a dict with keys
    `f"IoU={iou_thres}"` for each `iou_thres` in `iou_thresholds` as well as
    `f"IoU={min(iou_thresholds)}:{max(iou_thresholds)}"` which is the average
    of the scores across all of the IoU thresholds.
    """
    assert len(predictions) == len(groundtruths)

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
    predictions: List[List[PredictedDetection]],
    groundtruths: List[List[GroundTruthDetection]],
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

    ret = 0
    # TODO: should be able to make this part more efficient.
    for r in [0.01 * i for i in range(101)]:
        precs = [
            prec for prec, recall in zip(precisions, recalls) if recall >= r
        ]
        ret += max(precs) if len(precs) > 0 else 0.0

    return ret / 101
