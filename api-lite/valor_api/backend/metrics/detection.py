import bisect
from collections import defaultdict
from typing import Sequence

from geoalchemy2 import functions as gfunc
from sqlalchemy import CTE, and_, case, func, or_, select
from sqlalchemy.orm import Session, aliased

from valor_api import enums, schemas
from valor_api.backend import core, models
from valor_api.backend.metrics.metric_utils import (
    LabelMapType,
    commit_results,
    create_label_mapping,
    log_evaluation_duration,
    log_evaluation_item_counts,
    prepare_filter_for_evaluation,
    validate_computation,
)
from valor_api.backend.query import generate_query, generate_select
from valor_api.enums import AnnotationType


def _aggregate_data(
    db: Session,
    groundtruth_filter: schemas.Filter,
    prediction_filter: schemas.Filter,
    target_type: enums.AnnotationType,
    label_map: LabelMapType | None = None,
) -> tuple[CTE, CTE, dict[int, tuple[str, str]]]:
    """
    Aggregates data for an object detection task.

    This function returns a tuple containing CTE's used to gather groundtruths, predictions and a
    dictionary that maps label_id to a key-value pair.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    target_type : enums.AnnotationType
        The annotation type used by the object detection evaluation.
    label_map: LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.

    Returns
    ----------
    tuple[CTE, CTE, dict[int, tuple[str, str]]]:
        A tuple with form (groundtruths, predictions, labels).
    """
    labels = core.fetch_union_of_labels(
        db=db,
        lhs=groundtruth_filter,
        rhs=prediction_filter,
    )

    label_mapping = create_label_mapping(
        db=db,
        labels=labels,
        label_map=label_map,
    )

    groundtruths_subquery = generate_select(
        models.Annotation.datum_id.label("datum_id"),
        models.Datum.uid.label("datum_uid"),
        models.Dataset.name.label("dataset_name"),
        models.GroundTruth.annotation_id.label("annotation_id"),
        models.GroundTruth.id.label("groundtruth_id"),
        models.Label.id,
        label_mapping,
        _annotation_type_to_geojson(target_type, models.Annotation).label(
            "geojson"
        ),
        filters=groundtruth_filter,
        label_source=models.GroundTruth,
    ).subquery()
    groundtruths_cte = (
        select(
            groundtruths_subquery.c.datum_id,
            groundtruths_subquery.c.datum_uid,
            groundtruths_subquery.c.dataset_name,
            groundtruths_subquery.c.annotation_id,
            groundtruths_subquery.c.groundtruth_id,
            groundtruths_subquery.c.geojson,
            models.Label.id.label("label_id"),
            models.Label.key,
            models.Label.value,
        )
        .select_from(groundtruths_subquery)
        .join(
            models.Label,
            models.Label.id == groundtruths_subquery.c.label_id,
        )
        .cte()
    )

    predictions_subquery = generate_select(
        models.Annotation.datum_id.label("datum_id"),
        models.Datum.uid.label("datum_uid"),
        models.Dataset.name.label("dataset_name"),
        models.Prediction.annotation_id.label("annotation_id"),
        models.Prediction.id.label("prediction_id"),
        models.Prediction.score.label("score"),
        models.Label.id,
        label_mapping,
        _annotation_type_to_geojson(target_type, models.Annotation).label(
            "geojson"
        ),
        filters=prediction_filter,
        label_source=models.Prediction,
    ).subquery()
    predictions_cte = (
        select(
            predictions_subquery.c.datum_id,
            predictions_subquery.c.datum_uid,
            predictions_subquery.c.dataset_name,
            predictions_subquery.c.annotation_id,
            predictions_subquery.c.prediction_id,
            predictions_subquery.c.score,
            predictions_subquery.c.geojson,
            models.Label.id.label("label_id"),
            models.Label.key,
            models.Label.value,
        )
        .select_from(predictions_subquery)
        .join(
            models.Label,
            models.Label.id == predictions_subquery.c.label_id,
        )
        .cte()
    )

    # get all labels
    groundtruth_labels = {
        (key, value, label_id)
        for label_id, key, value in db.query(
            groundtruths_cte.c.label_id,
            groundtruths_cte.c.key,
            groundtruths_cte.c.value,
        )
        .distinct()
        .all()
    }
    prediction_labels = {
        (key, value, label_id)
        for label_id, key, value in db.query(
            predictions_cte.c.label_id,
            predictions_cte.c.key,
            predictions_cte.c.value,
        )
        .distinct()
        .all()
    }
    labels = groundtruth_labels.union(prediction_labels)
    labels = {label_id: (key, value) for key, value, label_id in labels}

    return (groundtruths_cte, predictions_cte, labels)


def _compute_detection_metrics(
    db: Session,
    parameters: schemas.EvaluationParameters,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    target_type: enums.AnnotationType,
) -> Sequence[schemas.Metric]:
    """
    Compute detection metrics. This version of _compute_detection_metrics only does IOU calculations for every groundtruth-prediction pair that shares a common grouper id. It also runs _compute_curves to calculate the PrecisionRecallCurve.

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
    List[schemas.APMetric | schemas.ARMetric | schemas.APMetricAveragedOverIOUs | schemas.mAPMetric | schemas.mARMetric | schemas.mAPMetricAveragedOverIOUs | schemas.PrecisionRecallCurve]
        A list of metrics to return to the user.

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
            case AnnotationType.RASTER:
                return table.raster
            case _:
                raise RuntimeError

    if (
        parameters.iou_thresholds_to_return is None
        or parameters.iou_thresholds_to_compute is None
        or parameters.recall_score_threshold is None
        or parameters.pr_curve_iou_threshold is None
    ):
        raise ValueError(
            "iou_thresholds_to_return, iou_thresholds_to_compute, recall_score_threshold, and pr_curve_iou_threshold are required attributes of EvaluationParameters when evaluating detections."
        )

    if (
        parameters.recall_score_threshold > 1
        or parameters.recall_score_threshold < 0
    ):
        raise ValueError(
            "recall_score_threshold should exist in the range 0 <= threshold <= 1."
        )

    gt, pd, labels = _aggregate_data(
        db=db,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        target_type=target_type,
        label_map=parameters.label_map,
    )

    # Alias the annotation table (required for joining twice)
    gt_annotation = aliased(models.Annotation)
    pd_annotation = aliased(models.Annotation)

    # Get distinct annotations
    gt_pd_pairs = (
        select(
            gt.c.annotation_id.label("gt_annotation_id"),
            pd.c.annotation_id.label("pd_annotation_id"),
        )
        .select_from(pd)
        .join(
            gt,
            and_(
                pd.c.datum_id == gt.c.datum_id,
                pd.c.label_id == gt.c.label_id,
            ),
        )
        .distinct()
        .cte()
    )

    gt_distinct = (
        select(gt_pd_pairs.c.gt_annotation_id.label("annotation_id"))
        .distinct()
        .subquery()
    )

    pd_distinct = (
        select(gt_pd_pairs.c.pd_annotation_id.label("annotation_id"))
        .distinct()
        .subquery()
    )

    # IOU Computation Block
    if target_type == AnnotationType.RASTER:

        gt_counts = (
            select(
                gt_distinct.c.annotation_id,
                gfunc.ST_Count(models.Annotation.raster).label("count"),
            )
            .select_from(gt_distinct)
            .join(
                models.Annotation,
                models.Annotation.id == gt_distinct.c.annotation_id,
            )
            .subquery()
        )

        pd_counts = (
            select(
                pd_distinct.c.annotation_id,
                gfunc.ST_Count(models.Annotation.raster).label("count"),
            )
            .select_from(pd_distinct)
            .join(
                models.Annotation,
                models.Annotation.id == pd_distinct.c.annotation_id,
            )
            .subquery()
        )

        gt_pd_counts = (
            select(
                gt_pd_pairs.c.gt_annotation_id,
                gt_pd_pairs.c.pd_annotation_id,
                gt_counts.c.count.label("gt_count"),
                pd_counts.c.count.label("pd_count"),
                func.coalesce(
                    gfunc.ST_Count(
                        gfunc.ST_Intersection(
                            gt_annotation.raster, pd_annotation.raster
                        )
                    ),
                    0,
                ).label("intersection"),
            )
            .select_from(gt_pd_pairs)
            .join(
                gt_annotation,
                gt_annotation.id == gt_pd_pairs.c.gt_annotation_id,
            )
            .join(
                pd_annotation,
                pd_annotation.id == gt_pd_pairs.c.pd_annotation_id,
            )
            .join(
                gt_counts,
                gt_counts.c.annotation_id == gt_pd_pairs.c.gt_annotation_id,
            )
            .join(
                pd_counts,
                pd_counts.c.annotation_id == gt_pd_pairs.c.pd_annotation_id,
            )
            .subquery()
        )

        gt_pd_ious = (
            select(
                gt_pd_counts.c.gt_annotation_id,
                gt_pd_counts.c.pd_annotation_id,
                case(
                    (
                        gt_pd_counts.c.gt_count
                        + gt_pd_counts.c.pd_count
                        - gt_pd_counts.c.intersection
                        == 0,
                        0,
                    ),
                    else_=(
                        gt_pd_counts.c.intersection
                        / (
                            gt_pd_counts.c.gt_count
                            + gt_pd_counts.c.pd_count
                            - gt_pd_counts.c.intersection
                        )
                    ),
                ).label("iou"),
            )
            .select_from(gt_pd_counts)
            .subquery()
        )

    else:
        gt_geom = _annotation_type_to_column(target_type, gt_annotation)
        pd_geom = _annotation_type_to_column(target_type, pd_annotation)
        gintersection = gfunc.ST_Intersection(gt_geom, pd_geom)
        gunion = gfunc.ST_Union(gt_geom, pd_geom)
        iou_computation = gfunc.ST_Area(gintersection) / gfunc.ST_Area(gunion)

        gt_pd_ious = (
            select(
                gt_pd_pairs.c.gt_annotation_id,
                gt_pd_pairs.c.pd_annotation_id,
                case(
                    (gfunc.ST_Area(gunion) == 0, 0),
                    else_=iou_computation,
                ).label("iou"),
            )
            .select_from(gt_pd_pairs)
            .join(
                gt_annotation,
                gt_annotation.id == gt_pd_pairs.c.gt_annotation_id,
            )
            .join(
                pd_annotation,
                pd_annotation.id == gt_pd_pairs.c.pd_annotation_id,
            )
            .cte()
        )

    ious = (
        select(
            func.coalesce(pd.c.dataset_name, gt.c.dataset_name).label(
                "dataset_name"
            ),
            pd.c.datum_uid.label("pd_datum_uid"),
            gt.c.datum_uid.label("gt_datum_uid"),
            gt.c.groundtruth_id.label("gt_id"),
            pd.c.prediction_id.label("pd_id"),
            gt.c.label_id.label("gt_label_id"),
            pd.c.label_id.label("pd_label_id"),
            pd.c.score.label("score"),
            gt_pd_ious.c.iou,
            gt.c.geojson.label("gt_geojson"),
        )
        .select_from(pd)
        .outerjoin(
            gt,
            and_(
                pd.c.datum_id == gt.c.datum_id,
                pd.c.label_id == gt.c.label_id,
            ),
        )
        .outerjoin(
            gt_pd_ious,
            and_(
                gt_pd_ious.c.gt_annotation_id == gt.c.annotation_id,
                gt_pd_ious.c.pd_annotation_id == pd.c.annotation_id,
            ),
        )
        .subquery()
    )

    ordered_ious = (
        db.query(ious).order_by(-ious.c.score, -ious.c.iou, ious.c.gt_id).all()
    )

    matched_pd_set = set()
    matched_sorted_ranked_pairs = defaultdict(list)
    predictions_not_in_sorted_ranked_pairs = list()

    for row in ordered_ious:
        (
            dataset_name,
            pd_datum_uid,
            gt_datum_uid,
            gt_id,
            pd_id,
            gt_label_id,
            pd_label_id,
            score,
            iou,
            gt_geojson,
        ) = row

        if gt_id is None:
            predictions_not_in_sorted_ranked_pairs.append(
                (
                    pd_id,
                    score,
                    dataset_name,
                    pd_datum_uid,
                    pd_label_id,
                )
            )
            continue

        if pd_id not in matched_pd_set:
            matched_pd_set.add(pd_id)
            matched_sorted_ranked_pairs[gt_label_id].append(
                RankedPair(
                    dataset_name=dataset_name,
                    pd_datum_uid=pd_datum_uid,
                    gt_datum_uid=gt_datum_uid,
                    gt_geojson=gt_geojson,
                    gt_id=gt_id,
                    pd_id=pd_id,
                    score=score,
                    iou=iou,
                    is_match=True,  # we're joining on grouper IDs, so only matches are included in matched_sorted_ranked_pairs
                )
            )

    for (
        pd_id,
        score,
        dataset_name,
        pd_datum_uid,
        label_id,
    ) in predictions_not_in_sorted_ranked_pairs:
        if (
            label_id in matched_sorted_ranked_pairs
            and pd_id not in matched_pd_set
        ):
            # add to sorted_ranked_pairs in sorted order
            bisect.insort(  # type: ignore - bisect type issue
                matched_sorted_ranked_pairs[label_id],
                RankedPair(
                    dataset_name=dataset_name,
                    pd_datum_uid=pd_datum_uid,
                    gt_datum_uid=None,
                    gt_geojson=None,
                    gt_id=None,
                    pd_id=pd_id,
                    score=score,
                    iou=0,
                    is_match=False,
                ),
                key=lambda rp: -rp.score,  # bisect assumes decreasing order
            )

    groundtruths_per_label = defaultdict(list)
    number_of_groundtruths_per_label = defaultdict(int)
    for label_id, dataset_name, datum_uid, groundtruth_id in db.query(
        gt.c.label_id, gt.c.dataset_name, gt.c.datum_uid, gt.c.groundtruth_id
    ).all():
        groundtruths_per_label[label_id].append(
            (dataset_name, datum_uid, groundtruth_id)
        )
        number_of_groundtruths_per_label[label_id] += 1

    if (
        parameters.metrics_to_return
        and enums.MetricType.PrecisionRecallCurve
        in parameters.metrics_to_return
    ):
        false_positive_entries = db.query(
            select(
                ious.c.dataset_name,
                ious.c.gt_datum_uid,
                ious.c.pd_datum_uid,
                ious.c.gt_label_id,
                ious.c.pd_label_id,
                ious.c.score.label("score"),
            )
            .select_from(ious)
            .where(
                or_(
                    ious.c.gt_id.is_(None),
                    ious.c.pd_id.is_(None),
                )
            )
            .subquery()
        ).all()

        pr_curves = _compute_curves(
            sorted_ranked_pairs=matched_sorted_ranked_pairs,
            labels=labels,
            groundtruths_per_label=groundtruths_per_label,
            false_positive_entries=false_positive_entries,
            iou_threshold=parameters.pr_curve_iou_threshold,
        )
    else:
        pr_curves = []

    ap_ar_output = []

    ap_metrics, ar_metrics = _calculate_ap_and_ar(
        sorted_ranked_pairs=matched_sorted_ranked_pairs,
        labels=labels,
        number_of_groundtruths_per_label=number_of_groundtruths_per_label,
        iou_thresholds=parameters.iou_thresholds_to_compute,
        recall_score_threshold=parameters.recall_score_threshold,
    )

    ap_ar_output += [
        m for m in ap_metrics if m.iou in parameters.iou_thresholds_to_return
    ]
    ap_ar_output += ar_metrics

    # calculate averaged metrics
    mean_ap_metrics = _compute_mean_detection_metrics_from_aps(ap_metrics)
    mean_ar_metrics = _compute_mean_ar_metrics(ar_metrics)

    ap_metrics_ave_over_ious = list(
        _compute_detection_metrics_averaged_over_ious_from_aps(ap_metrics)
    )

    ap_ar_output += [
        m
        for m in mean_ap_metrics
        if isinstance(m, schemas.mAPMetric)
        and m.iou in parameters.iou_thresholds_to_return
    ]
    ap_ar_output += mean_ar_metrics
    ap_ar_output += ap_metrics_ave_over_ious

    mean_ap_metrics_ave_over_ious = list(
        _compute_mean_detection_metrics_from_aps(ap_metrics_ave_over_ious)
    )
    ap_ar_output += mean_ap_metrics_ave_over_ious

    return ap_ar_output + pr_curves


@validate_computation
def compute_detection_metrics(*_, db: Session, evaluation_id: int):
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
    parameters = schemas.EvaluationParameters(**evaluation.parameters)
    groundtruth_filter, prediction_filter = prepare_filter_for_evaluation(
        filters=schemas.Filter(**evaluation.filters),
        dataset_names=evaluation.dataset_names,
        model_name=evaluation.model_name,
        task_type=parameters.task_type,
    )

    log_evaluation_item_counts(
        db=db,
        evaluation=evaluation,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
    )

    # fetch model and datasets
    datasets = (
        generate_query(
            models.Dataset,
            db=db,
            filters=groundtruth_filter,
            label_source=models.GroundTruth,
        )
        .distinct()
        .all()
    )
    model = (
        generate_query(
            models.Model,
            db=db,
            filters=prediction_filter,
            label_source=models.Prediction,
        )
        .distinct()
        .one_or_none()
    )

    # verify datums exist
    if not datasets:
        raise RuntimeError(
            "No datasets could be found that meet filter requirements."
        )

    # no predictions exist
    if model is not None:
        # ensure that all annotations have a common type to operate over
        target_type = _convert_annotations_to_common_type(
            db=db,
            datasets=datasets,
            model=model,
            target_type=parameters.convert_annotations_to_type,
        )
    else:
        target_type = min(
            [
                core.get_annotation_type(
                    db=db, task_type=parameters.task_type, dataset=dataset
                )
                for dataset in datasets
            ]
        )

    match target_type:
        case AnnotationType.BOX:
            symbol = schemas.Symbol(name=schemas.SupportedSymbol.BOX)
        case AnnotationType.POLYGON:
            symbol = schemas.Symbol(name=schemas.SupportedSymbol.POLYGON)
        case AnnotationType.RASTER:
            symbol = schemas.Symbol(name=schemas.SupportedSymbol.RASTER)
        case _:
            raise TypeError(
                f"'{target_type}' is not a valid type for object detection."
            )

    groundtruth_filter.annotations = schemas.LogicalFunction.and_(
        groundtruth_filter.annotations,
        schemas.Condition(
            lhs=symbol,
            op=schemas.FilterOperator.ISNOTNULL,
        ),
    )
    prediction_filter.annotations = schemas.LogicalFunction.and_(
        prediction_filter.annotations,
        schemas.Condition(
            lhs=symbol,
            op=schemas.FilterOperator.ISNOTNULL,
        ),
    )

    if (
        parameters.metrics_to_return
        and enums.MetricType.DetailedPrecisionRecallCurve
        in parameters.metrics_to_return
    ):
        # this function is more computationally expensive since it calculates IOUs for every groundtruth-prediction pair that shares a label key
        metrics = (
            _compute_detection_metrics_with_detailed_precision_recall_curve(
                db=db,
                parameters=parameters,
                prediction_filter=prediction_filter,
                groundtruth_filter=groundtruth_filter,
                target_type=target_type,
            )
        )
    else:
        # this function is much faster since it only calculates IOUs for every groundtruth-prediction pair that shares a label id
        metrics = _compute_detection_metrics(
            db=db,
            parameters=parameters,
            prediction_filter=prediction_filter,
            groundtruth_filter=groundtruth_filter,
            target_type=target_type,
        )

    # add metrics to database
    commit_results(
        db=db,
        metrics=metrics,
        evaluation_id=evaluation_id,
    )

    log_evaluation_duration(
        evaluation=evaluation,
        db=db,
    )

    return evaluation_id
