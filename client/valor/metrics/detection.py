import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from valor.coretypes import GroundTruth, Label, Prediction
from valor.metrics.classification import combine_tps_fps_thresholds
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
                if best_match_idx is None or iou > current_max_iou:  # type: ignore
                    best_match_idx = j
                    current_max_iou = iou

        matches.append(best_match_idx)

    return matches


def iou_matrix(
    predicted_boxes_and_scores: List[Tuple[Box, float]],
    groundtruth_boxes: List[Box],
) -> List[List[float]]:
    """Returns a list of lists where the entry at [i][j]
    is the iou between `predictions[i]` and `groundtruths[j]`.
    """
    if len(predicted_boxes_and_scores) == 0:
        return [[]]
    return [
        [iou(p[0], g) for g in groundtruth_boxes]
        for p in predicted_boxes_and_scores
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
    predicted_boxes_and_scores: List[Tuple[Box, float]],
    groundtruth_boxes: List[Box],
    iou_threshold: float,
) -> List[DetectionMatchInfo]:
    predicted_boxes_and_scores = sorted(
        predicted_boxes_and_scores, key=lambda p: p[1], reverse=True
    )

    ious = iou_matrix(
        groundtruth_boxes=groundtruth_boxes,
        predicted_boxes_and_scores=predicted_boxes_and_scores,
    )

    matches = _match_array(ious, iou_threshold)

    return [
        DetectionMatchInfo(tp=(match is not None), score=p[1])
        for match, p in zip(matches, predicted_boxes_and_scores)
    ]


@dataclass
class IntermediateMetricData:
    iou_threshold: float
    n_gt: int
    tp: Union[List[float], None] = None
    fp: Union[List[float], None] = None
    scores: Union[List[float], None] = None

    def copy(self):
        return IntermediateMetricData(
            iou_threshold=self.iou_threshold,
            n_gt=self.n_gt,
            tp=self.tp.copy() if self.tp is not None else None,
            fp=self.fp.copy() if self.fp is not None else None,
            scores=self.scores.copy() if self.scores is not None else None,
        )


def combine_intermediate_metric_data(
    imd1: IntermediateMetricData, imd2: IntermediateMetricData
) -> IntermediateMetricData:
    if imd1.iou_threshold != imd2.iou_threshold:
        raise ValueError("iou thresholds must be the same")
    n_gt = imd1.n_gt + imd2.n_gt
    if imd1.tp is None or imd1.tp is None:
        return imd2.copy()
    if imd2.tp is None or imd2.tp is None:
        return imd1.copy()

    tps, fps, scores = combine_tps_fps_thresholds(
        tps1=imd1.tp,
        fps1=imd1.fp,
        thresholds1=imd1.scores,
        tps2=imd2.tp,
        fps2=imd2.fp,
        thresholds2=imd2.scores,
    )

    return IntermediateMetricData(
        iou_threshold=imd1.iou_threshold,
        n_gt=n_gt,
        tp=tps,
        fp=fps,
        scores=scores,
    )


def get_intermediate_metric_data(
    predictions: List[Prediction],
    groundtruths: List[GroundTruth],
    label: Label,
    iou_thresholds: List[float],
) -> List[IntermediateMetricData]:
    if len(predictions) != len(groundtruths):
        raise ValueError(
            "Number of predictions and groundtruths must be the same"
        )

    predicted_boxes_and_scores = [
        [
            (ann.bounding_box, la.score)
            for ann in pred.annotations
            for la in ann.labels
            if la.key == label.key and la.value == label.value
        ]
        for pred in predictions
    ]
    groundtruth_boxes = [
        [
            ann.bounding_box
            for ann in gt.annotations
            for la in ann.labels
            if la.key == label.key and la.value == label.value
        ]
        for gt in groundtruths
    ]

    # total number of groundtruth objects across all images
    n_gt = sum([len(gt_box) for gt_box in groundtruth_boxes])

    if n_gt == 0:
        return [
            IntermediateMetricData(iou_threshold=iou_thres, n_gt=0)
            for iou_thres in iou_thresholds
        ]

    ret = []
    for iou_thres in iou_thresholds:
        match_infos = []
        for preds, gts in zip(predicted_boxes_and_scores, groundtruth_boxes):
            match_infos.extend(
                _get_tp_fp_single_image_single_class(
                    predicted_boxes_and_scores=preds,  # type: ignore
                    groundtruth_boxes=gts,
                    iou_threshold=iou_thres,
                )
            )

        match_infos = sorted(match_infos, key=lambda m: m.score, reverse=True)

        tp = [float(m.tp) for m in match_infos]
        fp = [float(not m.tp) for m in match_infos]

        cum_tp = _cumsum(tp)
        cum_fp = _cumsum(fp)
        scores = [m.score for m in match_infos]

        ret.append(
            IntermediateMetricData(
                iou_threshold=iou_thres,
                tp=cum_tp,
                fp=cum_fp,
                scores=scores,
                n_gt=n_gt,
            )
        )
    return ret


def ap_from_intermediate_metric_data(
    *intermediate_metric_data: IntermediateMetricData,
):
    imd = intermediate_metric_data[0]
    for other_imd in intermediate_metric_data[1:]:
        imd = combine_intermediate_metric_data(imd, other_imd)

    # total number of groundtruth objects across all images
    ret = {}
    iou_thresholds = []
    for tp_fp_threshold in imd:
        iou_thresholds.append(tp_fp_threshold.iou_threshold)
        if tp_fp_threshold.fp is None or tp_fp_threshold.tp is None:
            ret[f"IoU={tp_fp_threshold.iou_threshold}"] = -1.0
        else:
            precisions = [
                tp / (tp + fp)
                for tp, fp in zip(tp_fp_threshold.tp, tp_fp_threshold.fp)
            ]
            recalls = [tp / tp_fp_threshold.n_gt for tp in tp_fp_threshold.tp]

            ret[
                f"IoU={tp_fp_threshold.iou_threshold}"
            ] = calculate_ap_101_pt_interp(
                precisions=precisions, recalls=recalls
            )
    ret[f"IoU={min(iou_thresholds)}:{max(iou_thresholds)}"] = sum(
        [ret[f"IoU={iou_thres}"] for iou_thres in iou_thresholds]
    ) / len(iou_thresholds)
    return ret


def ap(
    predictions: List[Prediction],
    groundtruths: List[GroundTruth],
    label: Label,
    iou_thresholds: List[float],
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

    tp_fp_thresholds = get_intermediate_metric_data(
        predictions=predictions,
        groundtruths=groundtruths,
        label=label,
        iou_thresholds=iou_thresholds,
    )

    ret = ap_from_intermediate_metric_data(tp_fp_thresholds)

    return ret


def compute_ap_metrics(
    predictions: List[Prediction],
    groundtruths: List[GroundTruth],
    iou_thresholds: List[float],
) -> dict:
    """Computes average precision metrics. Note that this is not an optimized method
    and is here for toy/test purposes. Will likely be (re)moved in future versions.

    The return dictionary, `d` indexes AP scores by `d["AP"][class_label][iou_key]`
    and mAP scores by `d["mAP"][iou_key]` where `iou_key` is `f"IoU={iou_thres}"` or
    `f"IoU={min(iou_thresholds)}:{max(iou_thresholds)}"`
    """
    class_label_tuples = set(
        (la.key, la.value)
        for pred in predictions
        for ann in pred.annotations
        for la in ann.labels
    ).union(
        (la.key, la.value)
        for gt in groundtruths
        for ann in gt.annotations
        for la in ann.labels
    )

    ret = {
        "AP": {
            (la_key, la_value): ap(
                predictions=predictions,
                groundtruths=groundtruths,
                label=Label(key=la_key, value=la_value),  # type: ignore
                iou_thresholds=iou_thresholds,
            )
            for la_key, la_value in class_label_tuples
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
    ret["mAP"] = {  # type: ignore
        k: _ave_ignore_minus_one(
            [ret["AP"][class_label][k] for class_label in class_label_tuples]  # type: ignore
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
