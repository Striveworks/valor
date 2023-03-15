import heapq
from dataclasses import dataclass

from sqlalchemy.orm import Session

from velour_api import ops, schemas
from velour_api.models import (
    Label,
    LabeledGroundTruthDetection,
    LabeledPredictedDetection,
)


def _match_array(
    ious: list[list[float]],
    iou_thres: float,
) -> list[int | None]:
    """
    Parameters
    ----------
    ious
        ious[i][j] should be the iou between predicted detection i and groundtruth detection j
        an important assumption is that the predicted detections are sorted by confidence

    Returns
    -------
    array of match indices. value at index i is the index of the groundtruth object that
    best matches the ith detection, or None if that detection has no match
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


def _cumsum(a: list) -> list:
    ret = [0]
    for x in a:
        ret.append(ret[-1] + x)

    return ret[1:]


@dataclass
class DetectionMatchInfo:
    tp: bool
    score: float
    # add ids and iou here and then store this informatoin?


def _get_tp_fp_single_image_single_class(
    predictions: list[LabeledPredictedDetection],
    iou_threshold: float,
    ious: list[list[float]],
) -> list[DetectionMatchInfo]:
    matches = _match_array(ious, iou_threshold)

    return [
        DetectionMatchInfo(tp=(match is not None), score=prediction.score)
        for match, prediction in zip(matches, predictions)
    ]


def iou_matrix(
    db: Session,
    predictions: list[LabeledPredictedDetection],
    groundtruths: list[LabeledGroundTruthDetection],
) -> list[list[float]]:
    """Returns a list of lists where the entry at [i][j]
    is the iou between `predictions[i]` and `groundtruths[j]`.
    """
    return [
        [ops.iou(db, p.detection, g.detection) for g in groundtruths]
        for p in predictions
    ]


def _db_label_to_pydantic_label(label: Label):
    return schemas.Label(key=label.key, value=label.value)


def ap(
    db: Session,
    predictions: list[list[LabeledPredictedDetection]],
    groundtruths: list[list[LabeledGroundTruthDetection]],
    label: Label,
    iou_thresholds: list[float],
) -> list[schemas.APAtIOU]:
    """Computes the average precision. Return is a dict with keys
    `f"IoU={iou_thres}"` for each `iou_thres` in `iou_thresholds` as well as
    `f"IoU={min(iou_thresholds)}:{max(iou_thresholds)}"` which is the average
    of the scores across all of the IoU thresholds.
    """
    assert len(predictions) == len(groundtruths)

    predictions = [
        [p for p in preds if p.label == label] for preds in predictions
    ]
    groundtruths = [
        [gt for gt in gts if gt.label == label] for gts in groundtruths
    ]

    predictions = [
        sorted(
            preds,
            key=lambda p: p.score,
            reverse=True,
        )
        for preds in predictions
    ]

    # total number of groundtruth objects across all images
    n_gt = sum([len(gts) for gts in groundtruths])

    ret = []
    if n_gt == 0:
        ret.extend(
            [
                schemas.APAtIOU(
                    iou=iou_thres,
                    value=-1.0,
                    label=_db_label_to_pydantic_label(label),
                )
                for iou_thres in iou_thresholds
            ]
        )
    else:
        # TODO: for large datasets is this going to be a memory problem?
        # these should often be sparse so maybe use sparse matrices?
        # or use something like redis?
        iou_matrices = [
            iou_matrix(db=db, groundtruths=gts, predictions=preds)
            for preds, gts in zip(predictions, groundtruths)
        ]
        for iou_thres in iou_thresholds:
            match_infos = []
            for preds, ious in zip(predictions, iou_matrices):
                match_infos.extend(
                    _get_tp_fp_single_image_single_class(
                        predictions=preds,
                        iou_threshold=iou_thres,
                        ious=ious,
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

            ret.append(
                schemas.APAtIOU(
                    iou=iou_thres,
                    value=calculate_ap_101_pt_interp(
                        precisions=precisions, recalls=recalls
                    ),
                    label=_db_label_to_pydantic_label(label),
                )
            )

    # compute average over all IoUs
    ret.append(
        schemas.APAtIOU(
            iou=iou_thresholds,
            value=sum([a.value for a in ret]) / len(iou_thresholds),
            label=_db_label_to_pydantic_label(label),
        )
    )

    return ret


def calculate_ap_101_pt_interp(precisions, recalls) -> float:
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


def compute_ap_metrics(
    db: Session,
    predictions: list[list[LabeledPredictedDetection]],
    groundtruths: list[list[LabeledGroundTruthDetection]],
    iou_thresholds: list[float],
) -> list[schemas.APAtIOU]:
    """Computes average precision metrics."""
    iou_thresholds = sorted(iou_thresholds)
    labels = set(
        [pred.label for preds in predictions for pred in preds]
    ).union([gt.label for gts in groundtruths for gt in gts])

    ap_metrics = []

    for label in labels:
        ap_metrics.extend(
            ap(
                db=db,
                predictions=predictions,
                groundtruths=groundtruths,
                label=label,
                iou_thresholds=iou_thresholds,
            )
        )

    return ap_metrics


def compute_map_metrics_from_aps(
    ap_scores: list[schemas.APAtIOU],
) -> list[schemas.MAPAtIOU]:
    """
    Parameters
    ----------
    ap_scores
        list of AP scores. this should be output from the method `compute_ap_metrics`
    """

    def _ave_ignore_minus_one(a):
        num, denom = 0.0, 0.0
        for x in a:
            if x != -1:
                num += x
                denom += 1
        return num / denom

    # dictionary for mapping an iou threshold to set of APs
    vals: dict[float, list] = {}
    # for storing APs that are averaged over IOU thresholds
    average_over_iou_vals: list[float] = []
    labels: list[schemas.Label] = []
    for ap in ap_scores:
        # see if metric is AP at single value or averaged
        if isinstance(ap.iou, list):
            average_over_iou_vals.append(ap.value)
        else:
            if ap.iou not in vals:
                vals[ap.iou] = []
            vals[ap.iou].append(ap.value)

        if ap.label not in labels:
            labels.append(ap.label)

    iou_thresholds = list(vals.keys())

    # get mAP metrics at the individual IOUs
    map_metrics = [
        schemas.MAPAtIOU(
            iou=iou, value=_ave_ignore_minus_one(vals[iou]), labels=labels
        )
        for iou in iou_thresholds
    ]

    # get mAP metric averaged over all IOUs
    map_metrics.append(
        schemas.MAPAtIOU(
            iou=iou_thresholds,
            value=_ave_ignore_minus_one(average_over_iou_vals),
            labels=labels,
        )
    )

    return map_metrics
