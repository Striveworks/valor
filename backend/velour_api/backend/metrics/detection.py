import heapq
from dataclasses import dataclass
from typing import Dict, List

from geoalchemy2 import functions as gfunc
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import core, models
from velour_api.backend.metrics.core import (
    create_metric_mappings,
    get_or_create_row,
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

                precisions.append(
                    cnt_tp / (cnt_tp + cnt_fp) if (cnt_tp + cnt_fp) else 0
                )
                recalls.append(
                    cnt_tp / (cnt_tp + cnt_fn) if (cnt_tp + cnt_fn) else 0
                )

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
    gt_type: enums.AnnotationType,
    pd_type: enums.AnnotationType,
    min_area: float | None = None,
    max_area: float | None = None,
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
        dataset_source_type=gt_type,
        model_source_type=pd_type,
        evaluation_target_type=target_type,
    )

    # Select annotations column
    geometry = {
        AnnotationType.BOX: models.Annotation.box,
        AnnotationType.POLYGON: models.Annotation.polygon,
        AnnotationType.MULTIPOLYGON: models.Annotation.multipolygon,
        AnnotationType.RASTER: models.Annotation.raster,
    }

    # Filter by area
    area_filters = []
    if target_type == AnnotationType.RASTER:
        if min_area:
            area_filters.append(
                gfunc.ST_Count(geometry[target_type]) >= min_area
            )
        if max_area:
            area_filters.append(
                gfunc.ST_Count(geometry[target_type]) <= max_area
            )
    else:
        if min_area:
            area_filters.append(
                gfunc.ST_Area(geometry[target_type]) >= min_area
            )
        if max_area:
            area_filters.append(
                gfunc.ST_Area(geometry[target_type]) <= max_area
            )

    # Join gt, datum, annotation, label
    gt = (
        select(
            models.GroundTruth.id.label("id"),
            models.Datum.id.label("datum_id"),
            geometry[target_type].label("geom"),
            models.Label.id.label("label_id"),
        )
        .select_from(models.GroundTruth)
        .join(
            models.Annotation,
            models.Annotation.id == models.GroundTruth.annotation_id,
        )
        .join(models.Label, models.Label.id == models.GroundTruth.label_id)
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            and_(
                models.Datum.dataset_id == dataset.id,
                models.Annotation.model_id.is_(None),
                *area_filters,
            )
        )
        .subquery()
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
        .join(
            models.Annotation,
            models.Annotation.id == models.Prediction.annotation_id,
        )
        .join(models.Label, models.Label.id == models.Prediction.label_id)
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            and_(
                models.Datum.dataset_id == dataset.id,
                models.Annotation.model_id == model.id,
                *area_filters,
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

    # IOU Computation Block
    if target_type == AnnotationType.RASTER:
        gintersection = gfunc.ST_Count(
            gfunc.ST_Intersection(joint.c.gt_geom, joint.c.pd_geom)
        )
        gunion_gt = gfunc.ST_Count(joint.c.gt_geom)
        gunion_pd = gfunc.ST_Count(joint.c.pd_geom)
        gunion = gunion_gt + gunion_pd - gintersection
        iou_computation = gfunc.ST_Area(gintersection) / gfunc.ST_Area(gunion)
    else:
        gintersection = gfunc.ST_Intersection(joint.c.gt_geom, joint.c.pd_geom)
        gunion = gfunc.ST_Union(joint.c.gt_geom, joint.c.pd_geom)
        iou_computation = gfunc.ST_Area(gintersection) / gfunc.ST_Area(gunion)

    # Compute IOUs
    ious = (
        select(
            joint.c.datum_id.label("datum_id"),
            joint.c.gt_id.label("gt_id"),
            joint.c.pd_id.label("pd_id"),
            joint.c.gt_label_id.label("gt_label_id"),
            joint.c.pd_label_id.label("pd_label_id"),
            joint.c.score.label("score"),
            func.coalesce(iou_computation, 0).label("iou"),
        )
        .select_from(joint)
        .where(
            and_(
                joint.c.gt_id.isnot(None),
                joint.c.pd_id.isnot(None),
            )
        )
        .subquery()
    )

    # Order by score, iou
    ordered_ious = db.query(ious).order_by(-ious.c.score, -ious.c.iou).all()

    # Filter out repeated id's
    gt_set = set()
    pd_set = set()
    ranking = {}

    for row in ordered_ious:
        # datum_id = row[0]
        gt_id = row[1]
        pd_id = row[2]
        gt_label_id = row[3]
        # pd_label_id = row[4]
        score = row[5]
        iou = row[6]

        # Check if gt or pd already found
        if gt_id not in gt_set and pd_id not in pd_set:
            gt_set.add(gt_id)
            pd_set.add(pd_id)

            if gt_label_id not in ranking:
                ranking[gt_label_id] = []

            ranking[gt_label_id].append(
                RankedPair(
                    gt_id=gt_id,
                    pd_id=pd_id,
                    score=score,
                    iou=iou,
                )
            )

    # Filter by geometric type
    geometric_filters = []
    if gt_type == enums.AnnotationType.BOX:
        geometric_filters = [models.Annotation.box.isnot(None)]
    elif gt_type == enums.AnnotationType.POLYGON:
        geometric_filters = [models.Annotation.polygon.isnot(None)]
    elif gt_type == enums.AnnotationType.MULTIPOLYGON:
        geometric_filters = [models.Annotation.multipolygon.isnot(None)]
    elif gt_type == enums.AnnotationType.RASTER:
        geometric_filters = [models.Annotation.raster.isnot(None)]
    elif gt_type == enums.AnnotationType.NONE:
        geometric_filters = [
            models.Annotation.box.is_(None),
            models.Annotation.polygon.is_(None),
            models.Annotation.multipolygon.is_(None),
            models.Annotation.raster.is_(None),
        ]
    else:
        raise RuntimeError("Unknown Type")

    # Filter by label key
    label_key_filter = []
    if label_key:
        label_key_filter.append(models.Label.key == label_key)

    # Merge filters
    filters = geometric_filters + label_key_filter

    # Get groundtruth labels
    labels = {
        label[0]: schemas.Label(key=label[1], value=label[2])
        for label in (
            db.query(models.Label.id, models.Label.key, models.Label.value)
            .join(
                models.GroundTruth,
                models.GroundTruth.label_id == models.Label.id,
            )
            .join(
                models.Annotation,
                models.Annotation.id == models.GroundTruth.annotation_id,
            )
            .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
            .where(
                and_(
                    models.Datum.dataset_id == dataset.id,
                    *filters,
                )
            )
            .all()
        )
    }

    # Get the number of ground truths per label id
    number_of_ground_truths = {
        id: db.scalar(
            select(func.count(models.GroundTruth.id)).where(
                models.GroundTruth.label_id == id
            )
        )
        for id in labels
    }

    # Compute AP
    ap_metrics = _ap(
        sorted_ranked_pairs=ranking,
        number_of_ground_truths=number_of_ground_truths,
        labels=labels,
        iou_thresholds=iou_thresholds,
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


def create_ap_evaluation(
    db: Session,
    request_info: schemas.APRequest,
) -> int:
    """This will always run in foreground.

    Returns
        Evaluations settings id.
    """
    dataset = core.get_dataset(db, request_info.settings.dataset)
    model = core.get_model(db, request_info.settings.model)

    es = get_or_create_row(
        db,
        models.EvaluationSettings,
        mapping={
            "dataset_id": dataset.id,
            "model_id": model.id,
            "task_type": enums.TaskType.DETECTION,
            "pd_type": request_info.settings.pd_type,
            "gt_type": request_info.settings.gt_type,
            "label_key": request_info.settings.label_key,
            "min_area": request_info.settings.min_area,
            "max_area": request_info.settings.max_area,
        },
    )

    return es.id


def create_ap_metrics(
    db: Session,
    request_info: schemas.APRequest,
    evaluation_settings_id: int,
):
    """
    Intended to run as background
    """

    # @TODO: This is hacky, fix schemas.APRequest
    # START HACKY
    dataset_name = request_info.settings.dataset
    model_name = request_info.settings.model
    gt_type = request_info.settings.gt_type
    pd_type = request_info.settings.pd_type
    label_key = request_info.settings.label_key
    min_area = request_info.settings.min_area
    max_area = request_info.settings.max_area
    # END HACKY

    target_type = gt_type if gt_type < pd_type else pd_type

    metrics = compute_ap_metrics(
        db=db,
        dataset_name=dataset_name,
        model_name=model_name,
        iou_thresholds=request_info.iou_thresholds,
        ious_to_keep=request_info.ious_to_keep,
        label_key=label_key,
        target_type=target_type,
        gt_type=gt_type,
        pd_type=pd_type,
        min_area=min_area,
        max_area=max_area,
    )

    metric_mappings = create_metric_mappings(
        db=db, metrics=metrics, evaluation_settings_id=evaluation_settings_id
    )

    for mapping in metric_mappings:
        # ignore value since the other columns are unique identifiers
        # and have empircally noticed value can slightly change due to floating
        # point errors

        get_or_create_row(
            db, models.Metric, mapping, columns_to_ignore=["value"]
        )
    db.commit()
