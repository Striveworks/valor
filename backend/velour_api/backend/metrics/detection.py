import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional

from sqlalchemy import text, select, and_, or_, func
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session
from geoalchemy2 import func as gfunc

from velour_api import schemas, enums
from velour_api.backend import models, core, query
from velour_api.backend.metrics.ap import (
    compute_iou,
    function_find_ranked_pairs,
    get_labels,
    get_number_of_ground_truths,
    get_sorted_ranked_pairs,
    join_labels,
    join_tables,
)
from velour_api.backend.core.geometry import (
    convert_polygon_to_box,
    convert_raster_to_box,
    convert_raster_to_polygon,
)
from velour_api.enums import AnnotationType


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
) -> list[schemas.APMetric]:
    """Computes the average precision. Return is a dict with keys
    `f"IoU={iou_thres}"` for each `iou_thres` in `iou_thresholds` as well as
    `f"IoU={min(iou_thresholds)}:{max(iou_thresholds)}"` which is the average
    of the scores across all of the IoU thresholds.
    """

    ap_metrics = []
    for iou_threshold in iou_thresholds:
        for label_id in sorted_ranked_pairs:
            precisions = []
            recalls = []
            cnt_tp = 0
            cnt_fp = 0

            for row in sorted_ranked_pairs[label_id]:
                if row.score > 0 and row.iou >= iou_threshold:
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
    dataset_name: str,
    model_name: str,
    label_key: str,
    iou_thresholds: list[float],
    ious_to_keep: list[float],
    target_type: enums.AnnotationType,
    dataset_type: enums.AnnotationType,
    model_type: enums.AnnotationType,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
) -> list[
    schemas.APMetric
    | schemas.APMetricAveragedOverIOUs
    | schemas.mAPMetric
    | schemas.mAPMetricAveragedOverIOUs
]:
    """Computes average precision metrics."""

    # Retrieve sql models
    dataset = core.get_dataset(db, dataset_name)
    model = core.get_model(db, model_name)
    
    # Convert geometries to target type (if required)
    core.convert_geometry(
        db,
        dataset=dataset,
        model=model,
        dataset_source_type=dataset_type,
        model_source_type=model_type,
        evaluation_target_type=target_type,
    )

    # Select annotations column
    geometry = {
        AnnotationType.BOX: models.Annotation.box,
        AnnotationType.POLYGON: models.Annotation.polygon,
        AnnotationType.MULTIPOLYGON: models.Annotation.multipolygon,
        AnnotationType.RASTER: models.Annotation.raster,
    }

    # Join gt, datum, annotation, label
    gt = (
        select(
            models.GroundTruth.id.label("id"),
            models.Datum.id.label("datum_id"),
            geometry[target_type].label("geom"),
            models.Label.id.label("label_id"),
        )
        .select_from(models.GroundTruth)
        .join(models.Annotation, models.Annotation.id == models.GroundTruth.annotation_id)
        .join(models.Label, models.Label.id == models.GroundTruth.label_id)
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            and_(
                models.Datum.dataset_id == dataset.id,
                models.Annotation.model_id.is_(None),
            )
        )
    )

    # Join pd, datum, annotation, label
    pd = (
        select(
            models.Prediction.id.label("id"),
            models.Datum.id.label("datum_id"),
            geometry[target_type].label("geom"),
            models.Label.id.label("label_id"),
            models.Prediction.score.label("score"),
        )
        .select_from(models.Prediction)
        .join(models.Annotation, models.Annotation.id == models.Prediction.annotation_id)
        .join(models.Label, models.Label.id == models.Prediction.label_id)
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            and_(
                models.Datum.dataset_id == dataset.id,
                models.Annotation.model_id == model.id,
            )
        )
        .subquery()
    )

    # join gt with pd
    joint = (
        select(
            func.coalesce(gt.c.datum_id, pd.c.datum_id).label("datum_id"),
            gt.c.id.label("gt_id"),
            pd.c.id.label("pd_id"),
            gt.c.label_id.label("gt_label_id"),
            pd.c.label_id.label("pd_label_id"),
            gt.c.geom.label("gt_geom"),
            pd.c.geom.label("pd_geom"),
            pd.c.score.label("score"),
        )
        .select_from(gt)
        .join(
            pd, 
            and_(
                pd.c.datum_id == gt.c.datum_id,
                pd.c.label_id == gt.c.label_id,
            ), 
            full=True,
        )
        .subquery()
    )

    # Filter label_key
    if label_key:
        joint = (
            select(
                joint.c.datum_id.label("datum_id"),
                joint.c.gt_id.label("gt_id"),
                joint.c.pd_id.label("pd_id"),
                joint.c.gt_label_id.label("gt_label_id"),
                joint.c.pd_label_id.label("pd_label_id"),
                joint.c.gt_geom.label("gt_geom"),
                joint.c.pd_geom.label("pd_geom"),
                joint.c.score.label("score"),
            )
            .select_from(joint)
            .join(
                models.Label, 
                or_(
                    models.Label.id == joint.c.gt_label_id,
                    models.Label.id == joint.c.pd_label_id,
                ),
                full=True,
            )
            .where(models.Label.key == label_key)
            .subquery()
        )

  

    # Compute IOU
    if target_type == AnnotationType.RASTER:
        pass
    else:
        ious = (
            select(
                joint.c.datum_id.label("datum_id"),
                joint.c.gt_id.label("gt_id"),
                joint.c.pd_id.label("pd_id"),
                joint.c.gt_label_id.label("gt_label_id"),
                joint.c.pd_label_id.label("pd_label_id"),
                joint.c.score.label("score"),
                func.coalesce(
                    gfunc.ST_Area(
                    gfunc.ST_Intersection(joint.c.gt_geom, joint.c.pd_geom)
                ) / gfunc.ST_Area(
                    gfunc.ST_Union(joint.c.gt_geom, joint.c.pd_geom)
                ), 0)
            )
            .select_from(joint)
        )


    print(ious)
    return []


    # # Compute IOU's
    # ious = compute_iou(joint_table, annotation_types[common_task])

    # # Load IOU's into a temporary table
    # ious_table = f"create table iou as ({ious})"
    # try:
    #     db.execute(text("drop table iou"))
    # except ProgrammingError:
    #     db.rollback()
    # db.execute(text(ious_table))

    # # Create 'find_ranked_pairs' function
    # db.execute(text(function_find_ranked_pairs()))

    # # Get a list of all ground truth labels
    # labels = {
    #     row[0]: schemas.Label(key=row[1], value=row[2])
    #     for row in db.execute(
    #         text(get_labels(dataset_id, task_types[gt_type]))
    #     ).fetchall()
    # }

    # # Get the number of ground truths per label id
    # number_of_ground_truths = {
    #     row[0]: row[1]
    #     for row in db.execute(text(get_number_of_ground_truths()))
    # }

    # # Load ranked_pairs
    # pairs = {}
    # for row in db.execute(text(get_sorted_ranked_pairs())).fetchall():
    #     label_id = row[0]
    #     if label_id not in pairs:
    #         pairs[label_id] = []
    #     pairs[label_id].append(
    #         RankedPair(gt_id=row[1], pd_id=row[2], score=row[3], iou=row[4])
    #     )

    # # Clear the session
    # db.execute(text("DROP TABLE iou"))

    # # Compute AP
    # ap_metrics = _ap(
    #     sorted_ranked_pairs=pairs,
    #     number_of_ground_truths=number_of_ground_truths,
    #     labels=labels,
    #     iou_thresholds=iou_thresholds,
    # )

    # # now extend to the averaged AP metrics and mAP metric
    # map_metrics = compute_map_metrics_from_aps(ap_metrics)
    # ap_metrics_ave_over_ious = compute_ap_metrics_ave_over_ious_from_aps(
    #     ap_metrics
    # )
    # map_metrics_ave_over_ious = compute_map_metrics_from_aps(
    #     ap_metrics_ave_over_ious
    # )

    # # filter out only specified ious
    # ap_metrics = [m for m in ap_metrics if m.iou in ious_to_keep]
    # map_metrics = [m for m in map_metrics if m.iou in ious_to_keep]

    # return (
    #     ap_metrics
    #     + map_metrics
    #     + ap_metrics_ave_over_ious
    #     + map_metrics_ave_over_ious
    # )


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
        div0_flag = True
        for x in a:
            if x != -1:
                div0_flag = False
                num += x
                denom += 1
        return -1 if div0_flag else num / denom

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
