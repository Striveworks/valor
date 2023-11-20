import heapq
from dataclasses import dataclass
from typing import Dict, List

from geoalchemy2 import functions as gfunc
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import core, models, query
from velour_api.backend.metrics.core import (
    create_metric_mappings,
    get_or_create_row,
)
from velour_api.backend.ops import Query
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

    detection_metrics = []
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

            detection_metrics.append(
                schemas.APMetric(
                    iou=iou_threshold,
                    value=_calculate_101_pt_interp(
                        precisions=precisions, recalls=recalls
                    ),
                    label=labels[label_id],
                )
            )
    return detection_metrics


def _calculate_101_pt_interp(precisions, recalls) -> float:
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


def compute_detection_metrics(
    db: Session,
    dataset: models.Dataset,
    model: models.Model,
    settings: schemas.EvaluationSettings,
    target_type: enums.AnnotationType,
) -> list[
    schemas.APMetric
    | schemas.APMetricAveragedOverIOUs
    | schemas.mAPMetric
    | schemas.mAPMetricAveragedOverIOUs
]:
    """Computes average precision metrics."""

    # Create groundtruth filter
    gt_filter = settings.filters.model_copy()
    gt_filter.dataset_names = [dataset.name]
    gt_filter.models_names = None
    gt_filter.models_metadata = None
    gt_filter.models_geospatial = None
    gt_filter.prediction_scores = None

    # Create prediction filter
    pd_filter = settings.filters.model_copy()
    pd_filter.dataset_names = [dataset.name]
    pd_filter.models_names = [model.name]

    # Join gt, datum, annotation, label
    gt = (
        Query(
            models.GroundTruth.id.label("id"),
            models.Datum.id.label("datum_id"),
            models.annotation_type_to_geometry[target_type].label("geom"),
            models.Label.id.label("label_id"),
        )
        .filter(gt_filter)
        .groundtruths("groundtruths")
    )

    # Join pd, datum, annotation, label
    pd = (
        Query(
            models.Prediction.id.label("id"),
            models.Datum.id.label("datum_id"),
            models.annotation_type_to_geometry[target_type].label("geom"),
            models.Label.id.label("label_id"),
            models.Prediction.score.label("score"),
        )
        .filter(pd_filter)
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

    # Get groundtruth labels
    labels = {
        label.id: schemas.Label(key=label.key, value=label.value)
        for label in db.query(
            Query(models.Label).filter(gt_filter).groundtruths()
        ).all()
    }

    # Get the number of ground truths per label id
    number_of_ground_truths = {}
    for id in labels:
        gt_filter.label_ids = [id]
        number_of_ground_truths[id] = db.query(
            Query(func.count(models.GroundTruth.id))
            .filter(gt_filter)
            .groundtruths()
        ).scalar()

    # Compute AP
    detection_metrics = _ap(
        sorted_ranked_pairs=ranking,
        number_of_ground_truths=number_of_ground_truths,
        labels=labels,
        iou_thresholds=settings.parameters.iou_thresholds_to_compute,
    )

    # now extend to the averaged AP metrics and mAP metric
    mean_detection_metrics = compute_mean_detection_metrics_from_aps(
        detection_metrics
    )
    detection_metrics_ave_over_ious = (
        compute_detection_metrics_ave_over_ious_from_aps(detection_metrics)
    )
    mean_detection_metrics_ave_over_ious = (
        compute_mean_detection_metrics_from_aps(
            detection_metrics_ave_over_ious
        )
    )

    # filter out only specified ious
    detection_metrics = [
        m
        for m in detection_metrics
        if m.iou in settings.parameters.iou_thresholds_to_keep
    ]
    mean_detection_metrics = [
        m
        for m in mean_detection_metrics
        if m.iou in settings.parameters.iou_thresholds_to_keep
    ]

    return (
        detection_metrics
        + mean_detection_metrics
        + detection_metrics_ave_over_ious
        + mean_detection_metrics_ave_over_ious
    )


def compute_detection_metrics_ave_over_ious_from_aps(
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


def compute_mean_detection_metrics_from_aps(
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
    mean_detection_metrics = [
        schemas.mAPMetric(iou=iou, value=_ave_ignore_minus_one(vals[iou]))
        if isinstance(iou, float)
        else schemas.mAPMetricAveragedOverIOUs(
            ious=iou, value=_ave_ignore_minus_one(vals[iou]), labels=labels
        )
        for iou in vals.keys()
    ]

    return mean_detection_metrics


def _get_annotation_type_for_computation(
    db: Session,
    dataset: models.Dataset,
    model: models.Model,
    job_filter: schemas.Filter | None = None,
) -> AnnotationType:
    # get dominant type
    groundtruth_type = core.get_annotation_type(db, dataset, None)
    prediction_type = core.get_annotation_type(db, dataset, model)
    gct = (
        groundtruth_type
        if groundtruth_type < prediction_type
        else prediction_type
    )
    if job_filter.annotation_types:
        if gct not in job_filter.annotation_types:
            sorted_types = sorted(
                job_filter.annotation_types,
                key=lambda x: x,
                reverse=True,
            )
            for annotation_type in sorted_types:
                if gct <= annotation_type:
                    return annotation_type, groundtruth_type, prediction_type
            raise RuntimeError(
                f"Annotation type filter is too restrictive. Attempted filter `{gct}` over `{groundtruth_type, prediction_type}`."
            )
    return gct, groundtruth_type, prediction_type


def _get_disjoint_label_sets(
    db: Session,
    groundtruth_filter: schemas.Filter,
    prediction_filters: schemas.Filter,
) -> tuple:

    # get disjoint label sets
    groundtruth_labels = query.get_groundtruth_labels(db, groundtruth_filter)
    prediction_labels = query.get_prediction_labels(db, prediction_filters)
    groundtruth_unique = list(groundtruth_labels - prediction_labels)
    prediction_unique = list(prediction_labels - groundtruth_labels)
    return groundtruth_unique, prediction_unique


def create_detection_evaluation(
    db: Session,
    job_request: schemas.EvaluationJob,
) -> int:
    """
    This will always run in foreground.

    Returns
        Evaluations settings id.
    """
    # check matching task_type
    if job_request.task_type != enums.TaskType.DETECTION:
        raise TypeError(
            "Invalid task_type, please choose an evaluation method that supports object detection"
        )

    # validate parameters
    if not job_request.settings.parameters:
        job_request.settings.parameters = schemas.DetectionParameters()
    else:
        if not isinstance(
            job_request.settings.parameters, schemas.DetectionParameters
        ):
            raise TypeError(
                "expected evaluation settings to have parameters of type `DetectionParameters` for task type `DETECTION`"
            )

    # validate filters
    if not job_request.settings.filters:
        job_request.settings.filters = schemas.Filter()
    else:
        if (
            job_request.settings.filters.dataset_names is not None
            or job_request.settings.filters.dataset_metadata is not None
            or job_request.settings.filters.dataset_geospatial is not None
            or job_request.settings.filters.models_names is not None
            or job_request.settings.filters.models_metadata is not None
            or job_request.settings.filters.models_geospatial is not None
            or job_request.settings.filters.prediction_scores is not None
            or job_request.settings.filters.task_types is not None
        ):
            raise ValueError(
                "Evaluation filter objects should not include any dataset, model, prediction score or task type filters."
            )

    # load sql objects
    dataset = core.get_dataset(db, job_request.dataset)
    model = core.get_model(db, job_request.model)

    # determine annotation types
    (
        gct,
        groundtruth_type,
        prediction_type,
    ) = _get_annotation_type_for_computation(
        db, dataset, model, job_request.settings.filters
    )

    # create groundtruth label filter
    groundtruth_label_filter = job_request.settings.filters.model_copy()
    groundtruth_label_filter.dataset_names = [job_request.dataset]
    groundtruth_label_filter.annotation_types = [groundtruth_type]

    # create prediction label filter
    prediction_label_filter = job_request.settings.filters.model_copy()
    prediction_label_filter.dataset_names = [job_request.dataset]
    prediction_label_filter.models_names = [model.name]
    prediction_label_filter.annotation_types = [prediction_type]

    # get disjoint sets
    groundtruth_unique, prediction_unique = _get_disjoint_label_sets(
        db, groundtruth_label_filter, prediction_label_filter
    )

    # create evaluation settings row
    es = get_or_create_row(
        db,
        models.Evaluation,
        mapping={
            "dataset_id": dataset.id,
            "model_id": model.id,
            "task_type": enums.TaskType.DETECTION,
            "settings": job_request.settings.model_dump(),
        },
    )

    return es.id, groundtruth_unique, prediction_unique


def create_detection_metrics(
    db: Session,
    evaluation_id: int,
):
    """
    Intended to run as background
    """
    evaluation = db.scalar(
        select(models.Evaluation).where(models.Evaluation.id == evaluation_id)
    )

    # unpack job request
    job_request = schemas.EvaluationJob(
        dataset=evaluation.dataset.name,
        model=evaluation.model.name,
        task_type=evaluation.task_type,
        settings=schemas.EvaluationSettings(**evaluation.settings),
        id=evaluation.id,
    )

    # configure filters
    if not job_request.settings.filters:
        job_request.settings.filters = schemas.Filter()
    job_request.settings.filters.task_types = [enums.TaskType.DETECTION]

    dataset = core.get_dataset(db, job_request.dataset)
    model = core.get_model(db, job_request.model)

    groundtruth_type = core.get_annotation_type(db, dataset, None)
    prediction_type = core.get_annotation_type(db, dataset, model)

    # Get user-specified annotation type
    if job_request.settings.filters.annotation_types:
        target_type = max(
            job_request.settings.filters.annotation_types, key=lambda x: x
        )
    else:
        target_type = min([groundtruth_type, prediction_type])

    # Convert geometries to target type (if required)
    core.convert_geometry(
        db,
        dataset=dataset,
        model=model,
        dataset_source_type=groundtruth_type,
        model_source_type=prediction_type,
        evaluation_target_type=target_type,
    )

    metrics = compute_detection_metrics(
        db=db,
        dataset=dataset,
        model=model,
        settings=job_request.settings,
        target_type=target_type,
    )

    metric_mappings = create_metric_mappings(
        db=db, metrics=metrics, evaluation_id=evaluation_id
    )

    for mapping in metric_mappings:
        # ignore value since the other columns are unique identifiers
        # and have empircally noticed value can slightly change due to floating
        # point errors

        get_or_create_row(
            db, models.Metric, mapping, columns_to_ignore=["value"]
        )
    db.commit()
