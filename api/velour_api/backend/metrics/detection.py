import heapq
from dataclasses import dataclass
from typing import Dict, List

from geoalchemy2 import functions as gfunc
from sqlalchemy import and_, case, func, select
from sqlalchemy.orm import Session, aliased

from velour_api import enums, schemas
from velour_api.backend import core, models
from velour_api.backend.metrics.metric_utils import (
    create_grouper_mappings,
    create_metric_mappings,
    get_or_create_row,
    validate_computation,
)
from velour_api.backend.ops import Query
from velour_api.enums import AnnotationType


@dataclass
class RankedPair:
    gt_id: int
    pd_id: int
    score: float
    iou: float


def _calculate_101_pt_interp(precisions, recalls) -> float:
    """Use the 101 point interpolation method (following torchmetrics)"""

    assert len(precisions) == len(recalls)
    if len(precisions) == 0:
        return 0

    data = list(zip(precisions, recalls))
    data.sort(key=lambda x: x[1])
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


def _ap(
    sorted_ranked_pairs: Dict[int, List[RankedPair]],
    number_of_ground_truths: Dict[int, int],
    grouper_mappings: dict,
    iou_thresholds: list[float],
) -> list[schemas.APMetric]:
    """
    Computes the average precision. Return is a dict with keys
    `f"IoU={iou_thres}"` for each `iou_thres` in `iou_thresholds` as well as
    `f"IoU={min(iou_thresholds)}:{max(iou_thresholds)}"` which is the average
    of the scores across all of the IoU thresholds.
    """

    detection_metrics = []
    for iou_threshold in iou_thresholds:
        for grouper_id, grouper_label in grouper_mappings[
            "grouper_id_to_grouper_label_mapping"
        ].items():
            precisions = []
            recalls = []
            cnt_tp = 0
            cnt_fp = 0

            if grouper_id in sorted_ranked_pairs:
                for row in sorted_ranked_pairs[grouper_id]:
                    if row.score > 0 and row.iou >= iou_threshold:
                        cnt_tp += 1
                    else:
                        cnt_fp += 1
                    cnt_fn = number_of_ground_truths[grouper_id] - cnt_tp

                    precisions.append(
                        cnt_tp / (cnt_tp + cnt_fp) if (cnt_tp + cnt_fp) else 0
                    )
                    recalls.append(
                        cnt_tp / (cnt_tp + cnt_fn) if (cnt_tp + cnt_fn) else 0
                    )
            else:
                precisions = [0]
                recalls = [0]

            detection_metrics.append(
                schemas.APMetric(
                    iou=iou_threshold,
                    value=_calculate_101_pt_interp(
                        precisions=precisions, recalls=recalls
                    ),
                    label=grouper_label,
                )
            )
    return detection_metrics


def _compute_detection_metrics(
    db: Session,
    parameters: schemas.EvaluationParameters,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    target_type: enums.AnnotationType,
) -> list[
    schemas.APMetric
    | schemas.APMetricAveragedOverIOUs
    | schemas.mAPMetric
    | schemas.mAPMetricAveragedOverIOUs
]:
    """
    Compute detection metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    parameters : schemas.EvaluationParameters
        Any user-defined parameters.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    target_type: enums.AnnotationType
        The annotation type to compute metrics for.


    Returns
    ----------
    List[schemas.APMetric | schemas.APMetricAveragedOverIOUs | schemas.mAPMetric | schemas.mAPMetricAveragedOverIOUs]
        A list of average precision metrics.

    """

    def _annotation_type_to_column(
        annotation_type: AnnotationType,
        table,
    ):
        match annotation_type:
            case AnnotationType.BOX:
                return table.box
            case AnnotationType.POLYGON:
                return table.polygon
            case AnnotationType.MULTIPOLYGON:
                return table.multipolygon
            case _:
                raise RuntimeError

    labels = core.get_label_rows(
        db=db,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
    )

    grouper_mappings = create_grouper_mappings(
        labels=labels,
        label_map=parameters.label_map,
        evaluation_type="detection",
    )

    # Join gt, datum, annotation, label. Map grouper_ids to each label_id.
    gt = (
        Query(
            models.GroundTruth.id.label("id"),
            models.GroundTruth.annotation_id.label("annotation_id"),
            models.GroundTruth.label_id.label("label_id"),
            models.Annotation.datum_id.label("datum_id"),
            case(
                grouper_mappings["label_id_to_grouper_id_mapping"],
                value=models.GroundTruth.label_id,
            ).label("label_id_grouper"),
        )
        .filter(groundtruth_filter)
        .groundtruths("groundtruths")
    )

    # Join pd, datum, annotation, label
    pd = (
        Query(
            models.Prediction.id.label("id"),
            models.Prediction.annotation_id.label("annotation_id"),
            models.Prediction.label_id.label("label_id"),
            models.Prediction.score.label("score"),
            models.Annotation.datum_id.label("datum_id"),
            case(
                grouper_mappings["label_id_to_grouper_id_mapping"],
                value=models.Prediction.label_id,
            ).label("label_id_grouper"),
        )
        .filter(prediction_filter)
        .predictions("predictions")
    )

    # Create joint table
    joint = (
        select(
            func.coalesce(gt.c.datum_id, pd.c.datum_id).label("datum_id"),
            gt.c.id.label("gt_id"),
            pd.c.id.label("pd_id"),
            gt.c.label_id.label("gt_label_id"),
            pd.c.label_id.label("pd_label_id"),
            gt.c.label_id_grouper.label("gt_label_id_grouper"),
            pd.c.label_id_grouper.label("pd_label_id_grouper"),
            gt.c.annotation_id.label("gt_ann_id"),
            pd.c.annotation_id.label("pd_ann_id"),
            pd.c.score.label("score"),
        )
        .select_from(gt)
        .join(
            pd,
            and_(
                pd.c.datum_id == gt.c.datum_id,
                pd.c.label_id_grouper == gt.c.label_id_grouper,
            ),
        )
        .subquery()
    )

    # Alias the annotation table (required for joining twice)
    gt_annotation = aliased(models.Annotation)
    pd_annotation = aliased(models.Annotation)

    # IOU Computation Block
    if target_type == AnnotationType.RASTER:
        gintersection = gfunc.ST_Count(
            gfunc.ST_Intersection(gt_annotation.raster, pd_annotation.raster)
        )
        gunion_gt = gfunc.ST_Count(gt_annotation.raster)
        gunion_pd = gfunc.ST_Count(pd_annotation.raster)
        gunion = gunion_gt + gunion_pd - gintersection
        iou_computation = gintersection / gunion
    else:
        gt_geom = _annotation_type_to_column(target_type, gt_annotation)
        pd_geom = _annotation_type_to_column(target_type, pd_annotation)
        gintersection = gfunc.ST_Intersection(gt_geom, pd_geom)
        gunion = gfunc.ST_Union(gt_geom, pd_geom)
        iou_computation = gfunc.ST_Area(gintersection) / gfunc.ST_Area(gunion)

    # Compute IOUs
    ious = (
        select(
            joint.c.datum_id.label("datum_id"),
            joint.c.gt_id.label("gt_id"),
            joint.c.pd_id.label("pd_id"),
            joint.c.gt_label_id.label("gt_label_id"),
            joint.c.pd_label_id.label("pd_label_id"),
            joint.c.gt_label_id_grouper.label("gt_label_id_grouper"),
            joint.c.pd_label_id_grouper.label("pd_label_id_grouper"),
            joint.c.score.label("score"),
            func.coalesce(iou_computation, 0).label("iou"),
        )
        .select_from(joint)
        .join(gt_annotation, gt_annotation.id == joint.c.gt_ann_id)
        .join(pd_annotation, pd_annotation.id == joint.c.pd_ann_id)
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
        # gt_label_id = row[3]
        # pd_label_id = row[4]
        gt_label_id_grouper = row[5]
        # pd_label_id_grouper = row[6]
        score = row[7]
        iou = row[8]

        # Check if gt or pd already found
        if gt_id not in gt_set and pd_id not in pd_set:
            gt_set.add(gt_id)
            pd_set.add(pd_id)

            if gt_label_id_grouper not in ranking:
                ranking[gt_label_id_grouper] = []

            ranking[gt_label_id_grouper].append(
                RankedPair(
                    gt_id=gt_id,
                    pd_id=pd_id,
                    score=score,
                    iou=iou,
                )
            )

    # Get the number of ground truths per grouper_id
    number_of_ground_truths_per_grouper = {}

    for grouper_id in ranking.keys():
        label_ids = grouper_mappings["grouper_id_to_label_ids_mapping"][
            grouper_id
        ]
        groundtruth_filter.label_ids = label_ids
        number_of_ground_truths_per_grouper[grouper_id] = db.query(
            Query(func.count(models.GroundTruth.id))
            .filter(groundtruth_filter)
            .groundtruths()
        ).scalar()

    # Compute AP
    detection_metrics = _ap(
        sorted_ranked_pairs=ranking,
        number_of_ground_truths=number_of_ground_truths_per_grouper,
        iou_thresholds=parameters.iou_thresholds_to_compute,
        grouper_mappings=grouper_mappings,
    )

    # now extend to the averaged AP metrics and mAP metric
    mean_detection_metrics = _compute_mean_detection_metrics_from_aps(
        detection_metrics
    )

    detection_metrics_ave_over_ious = (
        _compute_detection_metrics_averaged_over_ious_from_aps(
            detection_metrics
        )
    )
    mean_detection_metrics_ave_over_ious = (
        _compute_mean_detection_metrics_from_aps(
            detection_metrics_ave_over_ious
        )
    )

    # filter out only specified ious
    detection_metrics = [
        m
        for m in detection_metrics
        if m.iou in parameters.iou_thresholds_to_return
    ]
    mean_detection_metrics = [
        m
        for m in mean_detection_metrics
        if m.iou in parameters.iou_thresholds_to_return
    ]

    return (
        detection_metrics
        + mean_detection_metrics
        + detection_metrics_ave_over_ious
        + mean_detection_metrics_ave_over_ious
    )


def _compute_detection_metrics_averaged_over_ious_from_aps(
    ap_scores: list[schemas.APMetric],
) -> list[schemas.APMetricAveragedOverIOUs]:
    """Average AP metrics over IOU thresholds using a list of AP metrics."""
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


def _average_ignore_minus_one(a):
    """Average a list of metrics, ignoring values of -1"""
    num, denom = 0.0, 0.0
    div0_flag = True
    for x in a:
        if x != -1:
            div0_flag = False
            num += x
            denom += 1
    return -1 if div0_flag else num / denom


def _compute_mean_detection_metrics_from_aps(
    ap_scores: list[schemas.APMetric | schemas.APMetricAveragedOverIOUs],
) -> list[schemas.mAPMetric]:
    """Calculate the mean of a list of AP metrics."""

    if len(ap_scores) == 0:
        return []

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
    mean_detection_metrics = [
        schemas.mAPMetric(iou=iou, value=_average_ignore_minus_one(vals[iou]))
        if isinstance(iou, float)
        else schemas.mAPMetricAveragedOverIOUs(
            ious=iou, value=_average_ignore_minus_one(vals[iou]), labels=labels
        )
        for iou in vals.keys()
    ]

    return mean_detection_metrics


def _convert_annotations_to_common_type(
    db: Session,
    datasets: list[models.Dataset],
    model: models.Model,
    target_type: enums.AnnotationType | None = None,
) -> enums.AnnotationType:
    """Convert all annotations to a common type."""

    if target_type is None:
        # find the greatest common type
        groundtruth_type = AnnotationType.RASTER
        prediction_type = AnnotationType.RASTER
        for dataset in datasets:
            dataset_type = core.get_annotation_type(
                db=db, dataset=dataset, task_type=enums.TaskType.DETECTION
            )
            model_type = core.get_annotation_type(
                db=db,
                dataset=dataset,
                model=model,
                task_type=enums.TaskType.DETECTION,
            )
            groundtruth_type = (
                dataset_type
                if dataset_type < groundtruth_type
                else groundtruth_type
            )
            prediction_type = (
                model_type if model_type < prediction_type else prediction_type
            )
        target_type = min([groundtruth_type, prediction_type])

    for dataset in datasets:
        # dataset
        source_type = core.get_annotation_type(
            db=db, dataset=dataset, task_type=enums.TaskType.DETECTION
        )
        core.convert_geometry(
            db=db,
            dataset=dataset,
            source_type=source_type,
            target_type=target_type,
        )
        # model
        source_type = core.get_annotation_type(
            db=db,
            dataset=dataset,
            model=model,
            task_type=enums.TaskType.DETECTION,
        )
        core.convert_geometry(
            db=db,
            dataset=dataset,
            model=model,
            source_type=source_type,
            target_type=target_type,
        )

    return target_type


@validate_computation
def compute_detection_metrics(
    *,
    db: Session,
    evaluation_id: int,
):
    """
    Create detection metrics. This function is intended to be run using FastAPI's `BackgroundTasks`.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_id : int
        The job ID to create metrics for.
    """

    # fetch evaluation
    evaluation = core.fetch_evaluation_from_id(db, evaluation_id)

    # unpack filters and params
    groundtruth_filter = schemas.Filter(**evaluation.datum_filter)
    prediction_filter = groundtruth_filter.model_copy()
    prediction_filter.model_names = [evaluation.model_name]
    parameters = schemas.EvaluationParameters(**evaluation.parameters)

    # load task type into filters
    groundtruth_filter.task_types = [parameters.task_type]
    prediction_filter.task_types = [parameters.task_type]

    # fetch model and datasets
    datasets = (
        db.query(Query(models.Dataset).filter(groundtruth_filter).any())
        .distinct()
        .all()
    )
    model = (
        db.query(Query(models.Model).filter(prediction_filter).any())
        .distinct()
        .one_or_none()
    )

    # ensure that all annotations have a common type to operate over
    target_type = _convert_annotations_to_common_type(
        db=db,
        datasets=datasets,
        model=model,
        target_type=parameters.convert_annotations_to_type,
    )
    groundtruth_filter.annotation_types = [target_type]
    prediction_filter.annotation_types = [target_type]

    metrics = _compute_detection_metrics(
        db=db,
        parameters=parameters,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
        target_type=target_type,
    )

    metric_mappings = create_metric_mappings(
        db=db,
        metrics=metrics,
        evaluation_id=evaluation_id,
    )

    for mapping in metric_mappings:
        # ignore value since the other columns are unique identifiers
        # and have empircally noticed value can slightly change due to floating
        # point errors

        get_or_create_row(
            db, models.Metric, mapping, columns_to_ignore=["value"]
        )
    db.commit()
