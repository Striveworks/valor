import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.enums import AnnotationType
from velour_api.sql import (
    compute_iou,
    convert_polygons_to_bbox,
    convert_raster_to_bbox,
    convert_raster_to_polygons,
    function_find_ranked_pairs,
    get_labels,
    get_number_of_ground_truths,
    get_sorted_ranked_pairs,
    join_labels,
    join_tables,
)


@dataclass
class RankedPair:
    gt_id: int
    pd_id: int
    score: float
    iou: float


def _ap(
    sorted_ranked_pairs: Dict[int, List[RankedPair]],
    number_of_ground_truths: int,
    labels: Dict[int, schemas.Label],
    iou_thresholds: list[float],
    score_threshold: float,
) -> list[schemas.APMetric]:
    """Computes the average precision. Return is a dict with keys
    `f"IoU={iou_thres}"` for each `iou_thres` in `iou_thresholds` as well as
    `f"IoU={min(iou_thresholds)}:{max(iou_thresholds)}"` which is the average
    of the scores across all of the IoU thresholds.
    """

    ap_metrics = []
    for iou_threshold in iou_thresholds:
        for label_id in sorted_ranked_pairs:

            if not number_of_ground_truths[label_id]:
                ap_metrics.extend(
                    [
                        schemas.APMetric(
                            iou=iou_thres,
                            value=-1.0,
                            label=labels[label_id],
                        )
                        for iou_thres in iou_thresholds
                    ]
                )
            else:
                precisions = []
                recalls = []
                cnt_tp = 0
                cnt_fp = 0
                for row in sorted_ranked_pairs[label_id]:
                    if (
                        row.score >= score_threshold
                        and row.iou >= iou_threshold
                    ):
                        cnt_tp += 1
                    else:
                        cnt_fp += 1
                    cnt_fn = number_of_ground_truths[label_id] - cnt_tp

                    precisions.append(cnt_tp / (cnt_tp + cnt_fp))
                    recalls.append(cnt_tp / (cnt_tp + cnt_fn))

                ap_metrics.append(
                    schemas.APMetric(
                        iou=iou_threshold,
                        value=calculate_ap_101_pt_interp(
                            precisions=precisions, recalls=recalls
                        ),
                        label=labels[label_id],
                    )
                )
    return ap_metrics


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
    dataset_id: int,
    model_id: int,
    gt_type: schemas.Task,
    pd_type: schemas.Task,
    label_key: str,
    iou_thresholds: list[float],
    ious_to_keep: list[float],
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
    score_threshold: float = 0.0,
) -> list[
    schemas.APMetric
    | schemas.APMetricAveragedOverIOUs
    | schemas.mAPMetric
    | schemas.mAPMetricAveragedOverIOUs
]:
    """Computes average precision metrics."""

    taskTypes = {
        schemas.Task.BBOX_OBJECT_DETECTION: "detection",
        schemas.Task.POLY_OBJECT_DETECTION: "detection",
        schemas.Task.INSTANCE_SEGMENTATION: "segmentation",
    }

    # Generate select
    gt_select = f"select * from ground_truth_{taskTypes[gt_type]}"
    pd_select = f"select * from predicted_{taskTypes[pd_type]}"

    # Apply type conversion query (if applicable)
    if schemas.Task.BBOX_OBJECT_DETECTION in [gt_type, pd_type]:
        common_task = schemas.Task.BBOX_OBJECT_DETECTION

        if gt_type in [
            schemas.Task.BBOX_OBJECT_DETECTION,
            schemas.Task.POLY_OBJECT_DETECTION,
        ]:
            gt_select = convert_polygons_to_bbox(
                "ground_truth_detection",
                dataset_id=dataset_id,
                min_area=min_area,
                max_area=max_area,
            )
        elif gt_type == schemas.Task.INSTANCE_SEGMENTATION:
            gt_select = convert_raster_to_bbox(
                "ground_truth_segmentation",
                dataset_id=dataset_id,
                min_area=min_area,
                max_area=max_area,
            )
        else:
            raise ValueError("Ground Truth data is of a unsupported type.")

        if pd_type in [
            schemas.Task.BBOX_OBJECT_DETECTION,
            schemas.Task.POLY_OBJECT_DETECTION,
        ]:
            pd_select = convert_polygons_to_bbox(
                "predicted_detection",
                model_id=model_id,
                min_area=min_area,
                max_area=max_area,
            )
        elif pd_type == schemas.Task.INSTANCE_SEGMENTATION:
            pd_select = convert_raster_to_bbox(
                "predicted_segmentation",
                model_id=model_id,
                min_area=min_area,
                max_area=max_area,
            )
        else:
            raise ValueError("Predicted data is of a unsupported type.")

    elif schemas.Task.POLY_OBJECT_DETECTION in [gt_type, pd_type]:
        common_task = schemas.Task.POLY_OBJECT_DETECTION

        if gt_type == schemas.Task.INSTANCE_SEGMENTATION:
            gt_select = convert_raster_to_polygons(
                "ground_truth_segmentation",
                dataset_id=dataset_id,
                min_area=min_area,
                max_area=max_area,
            )
        else:
            raise ValueError("Ground Truth data is of a unsupported type.")

        if pd_type == schemas.Task.INSTANCE_SEGMENTATION:
            pd_select = convert_raster_to_polygons(
                "predicted_segmentation",
                model_id=model_id,
                min_area=min_area,
                max_area=max_area,
            )
        else:
            raise ValueError("Predicted data is of a unsupported type.")

    else:
        common_task = schemas.Task.INSTANCE_SEGMENTATION

    # Join labels
    labeled_gt_select = join_labels(
        subquery=gt_select,
        label_table=f"labeled_ground_truth_{taskTypes[gt_type]}",
        column=f"{taskTypes[gt_type]}_id",
        label_key=label_key,
        is_prediction=False,
    )
    labeled_pd_select = join_labels(
        subquery=pd_select,
        label_table=f"labeled_predicted_{taskTypes[pd_type]}",
        column=f"{taskTypes[pd_type]}_id",
        label_key=label_key,
        is_prediction=True,
    )

    # Join gt with pd
    annotationType = {
        schemas.Task.BBOX_OBJECT_DETECTION: AnnotationType.BBOX,
        schemas.Task.POLY_OBJECT_DETECTION: AnnotationType.BOUNDARY,
        schemas.Task.INSTANCE_SEGMENTATION: AnnotationType.RASTER,
    }
    joint_table = join_tables(
        labeled_gt_select, labeled_pd_select, annotationType[common_task]
    )

    # Compute IOU's
    ious = compute_iou(joint_table, annotationType[common_task])

    # Load IOU's into a temporary table
    ious_table = f"create temporary table iou as ({ious})"
    try:
        db.execute(text("drop table iou"))
    except ProgrammingError:
        db.rollback()
    db.execute(text(ious_table))

    # Create 'find_ranked_pairs' function
    db.execute(text(function_find_ranked_pairs()))

    # Get params
    labels = {
        row[0]: schemas.Label(key=row[1], value=row[2])
        for row in db.execute(text(get_labels())).fetchall()
    }
    number_of_ground_truths = {
        row[0]: row[1]
        for row in db.execute(text(get_number_of_ground_truths()))
    }

    # Load ranked_pairs
    pairs = {}
    for row in db.execute(text(get_sorted_ranked_pairs())).fetchall():
        label_id = row[0]
        if label_id not in pairs:
            pairs[label_id] = []
        pairs[label_id].append(
            RankedPair(gt_id=row[1], pd_id=row[2], score=row[3], iou=row[4])
        )

    # Clear the session
    db.execute(text("DROP TABLE iou"))
    # db.commit()

    # Compute AP
    ap_metrics = _ap(
        sorted_ranked_pairs=pairs,
        number_of_ground_truths=number_of_ground_truths,
        labels=labels,
        iou_thresholds=iou_thresholds,
        score_threshold=score_threshold,
    )

    # now extend to the averaged AP metrics and mAP metric
    map_metrics = compute_map_metrics_from_aps(ap_metrics)
    ap_metrics_ave_over_ious = compute_ap_metrics_ave_over_ious_from_aps(
        ap_metrics
    )
    map_metrics_ave_over_ious = compute_map_metrics_from_aps(
        ap_metrics_ave_over_ious
    )

    # filter out only specified ious
    ap_metrics = [m for m in ap_metrics if m.iou in ious_to_keep]
    map_metrics = [m for m in map_metrics if m.iou in ious_to_keep]

    return (
        ap_metrics
        + map_metrics
        + ap_metrics_ave_over_ious
        + map_metrics_ave_over_ious
    )


def compute_ap_metrics_ave_over_ious_from_aps(
    ap_scores: list[schemas.APMetric],
) -> list[schemas.APMetricAveragedOverIOUs]:
    label_tuple_to_values = {}
    label_tuple_to_ious = {}
    for ap_score in ap_scores:
        label_tuple = (ap_score.label.key, ap_score.label.value)
        if label_tuple not in label_tuple_to_values:
            label_tuple_to_values[label_tuple] = 0
            label_tuple_to_ious[label_tuple] = []
        label_tuple_to_values[label_tuple] += ap_score.value
        label_tuple_to_ious[label_tuple].append(ap_score.iou)

    ret = []
    for label_tuple, value in label_tuple_to_values.items():
        ious = label_tuple_to_ious[label_tuple]
        ret.append(
            schemas.APMetricAveragedOverIOUs(
                ious=set(ious),
                value=value / len(ious),
                label=schemas.Label(key=label_tuple[0], value=label_tuple[1]),
            )
        )

    return ret


def compute_map_metrics_from_aps(
    ap_scores: list[schemas.APMetric | schemas.APMetricAveragedOverIOUs],
) -> list[schemas.mAPMetric]:
    """
    Parameters
    ----------
    ap_scores
        list of AP scores.
    """

    if len(ap_scores) == 0:
        return []

    def _ave_ignore_minus_one(a):
        num, denom = 0.0, 0.0
        for x in a:
            if x != -1:
                num += x
                denom += 1
        return num / denom

    # dictionary for mapping an iou threshold to set of APs
    vals: dict[float | set[float], list] = {}
    labels: list[schemas.Label] = []
    for ap in ap_scores:
        if hasattr(ap, "iou"):
            iou = ap.iou
        else:
            iou = frozenset(ap.ious)
        if iou not in vals:
            vals[iou] = []
        vals[iou].append(ap.value)

        if ap.label not in labels:
            labels.append(ap.label)

    # get mAP metrics at the individual IOUs
    map_metrics = [
        schemas.mAPMetric(iou=iou, value=_ave_ignore_minus_one(vals[iou]))
        if isinstance(iou, float)
        else schemas.mAPMetricAveragedOverIOUs(
            ious=iou, value=_ave_ignore_minus_one(vals[iou]), labels=labels
        )
        for iou in vals.keys()
    ]

    return map_metrics
