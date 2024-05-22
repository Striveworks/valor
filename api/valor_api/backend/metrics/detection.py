import bisect
import heapq
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence, Tuple

from geoalchemy2 import functions as gfunc
from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.orm import Session, aliased

from valor_api import enums, schemas
from valor_api.backend import core, models
from valor_api.backend.metrics.metric_utils import (
    create_grouper_mappings,
    create_metric_mappings,
    get_or_create_row,
    log_evaluation_duration,
    log_evaluation_item_counts,
    validate_computation,
)
from valor_api.backend.query import Query
from valor_api.enums import AnnotationType


@dataclass
class RankedPair:
    dataset_name: str
    pd_datum_uid: str | None
    gt_datum_uid: str | None
    gt_geojson: str | None
    gt_id: int | None
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
    heapq.heapify(prec_heap)

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


def _compute_curves(
    metrics: list[str],
    sorted_ranked_pairs: dict[int, list[RankedPair]],
    grouper_mappings: dict[str, dict[str, schemas.Label]],
    groundtruths_per_grouper: dict[int, list],
    false_positive_entries: list[tuple],
    iou_threshold: float,
    pr_curve_max_examples: int,
) -> list[schemas.PrecisionRecallCurve | schemas.DetailedPrecisionRecallCurve]:
    """
    Calculates precision-recall curves for each class.

    Parameters
    ----------
    metrics: List[str]
        The list of metrics to compute, store, and return to the user.
    sorted_ranked_pairs: dict[int, list[RankedPair]]
        The ground truth-prediction matches from psql, grouped by grouper_id.
    grouper_mappings: dict[str, dict[str, schemas.Label]]
        A dictionary of mappings that connect groupers to their related labels.
    groundtruths_per_grouper: dict[int, int]
        A dictionary containing the (dataset_name, datum_id, gt_id) for all groundtruths associated with a grouper.
    false_positive_entries: list[tuple]
        A list of predictions that don't have an associated ground truth. Used to calculate false positives.
    iou_threshold: float
        The IOU threshold to use as a cut-off for our predictions.
    pr_curve_max_examples: int
        The maximum number of datum examples to store per true positive, false negative, etc.


    Returns
    -------
    dict
        A nested dictionary where the first key is the class label, the second key is the confidence threshold (e.g., 0.05), the third key is the metric name (e.g., "precision"), and the final key is either the value itself (for precision, recall, etc.) or a list of tuples containing the (dataset_name, datum_id, bounding boxes) for each observation.
    """

    pr_output = defaultdict(dict)
    detailed_pr_output = defaultdict(dict)

    # create sets of all datums for which there is a prediction / groundtruth
    # used when separating hallucinations/misclassifications/missed_detections

    gt_datums = set()

    for grouper_id, groundtruths in groundtruths_per_grouper.items():
        for dset_name, datum_uid, gt_id, gt_geojson in groundtruths:
            gt_datums.add(
                (
                    dset_name,
                    datum_uid,
                    grouper_mappings["grouper_id_to_grouper_label_mapping"][
                        grouper_id
                    ].value,  # type: ignore - typechecker is over the dict keys being ints
                )
            )

    pd_datums = set(
        [
            (
                ranked_pair.dataset_name,
                ranked_pair.pd_datum_uid,
                grouper_mappings["grouper_id_to_grouper_label_mapping"][
                    grouper_id
                ].value,  # type: ignore
            )
            for grouper_id, ranked_pairs in sorted_ranked_pairs.items()
            for ranked_pair in ranked_pairs
        ]
    )

    for grouper_id, grouper_label in grouper_mappings[
        "grouper_id_to_grouper_label_mapping"
    ].items():

        pr_curves = defaultdict(lambda: defaultdict(dict))
        detailed_pr_curves = defaultdict(lambda: defaultdict(dict))

        label_key = grouper_label.key
        label_value = grouper_label.value

        for confidence_threshold in [x / 100 for x in range(5, 100, 5)]:
            if grouper_id not in sorted_ranked_pairs:
                tp = []
                fp = defaultdict(list)
                fn = {
                    "missed_detections": [
                        (
                            [
                                (dataset_name, datum_uid, gt_geojson)
                                for dataset_name, datum_uid, _, gt_geojson in groundtruths_per_grouper[
                                    grouper_id
                                ]
                            ]
                            if grouper_id in groundtruths_per_grouper
                            else []
                        )
                    ],
                    "misclassifications": [],
                }

            else:
                tp, fp, fn = [], defaultdict(list), defaultdict(list)
                seen_gts = set()

                for row in sorted_ranked_pairs[grouper_id]:
                    if (
                        row.score >= confidence_threshold
                        and row.iou >= iou_threshold
                        and row.gt_id not in seen_gts
                    ):
                        tp += [
                            (
                                row.dataset_name,
                                row.gt_datum_uid,
                                row.gt_geojson,
                            )
                        ]
                        seen_gts.add(row.gt_id)

                for (
                    dataset_name,
                    datum_uid,
                    gt_id,
                    gt_geojson,
                ) in groundtruths_per_grouper[grouper_id]:
                    if gt_id not in seen_gts:
                        # if there was a prediction for a given datum, then it was a misclassification
                        if (
                            dataset_name,
                            datum_uid,
                            grouper_mappings[
                                "grouper_id_to_grouper_label_mapping"
                            ][grouper_id].value,
                        ) in pd_datums:
                            fn["misclassifications"].append(
                                (dataset_name, datum_uid, gt_geojson)
                            )
                        else:
                            fn["missed_detections"].append(
                                (dataset_name, datum_uid, gt_geojson)
                            )

            for (
                dset_name,
                _,
                pd_datum_uid,
                gt_label_id,
                pd_label_id,
                pd_score,
                pd_geojson,
            ) in false_positive_entries:
                if (
                    pd_score >= confidence_threshold
                    and pd_label_id == grouper_id
                    and gt_label_id is None
                ):  # if there was a groundtruth for a given datum, then it was a misclassification
                    if (
                        dset_name,
                        pd_datum_uid,
                        grouper_mappings[
                            "grouper_id_to_grouper_label_mapping"
                        ][pd_label_id].value,
                    ) in gt_datums:
                        fp["misclassifications"].append(
                            (dset_name, pd_datum_uid, pd_geojson)
                        )
                    else:
                        fp["hallucinations"].append(
                            (dset_name, pd_datum_uid, pd_geojson)
                        )

            # calculate metrics
            tp_cnt, fp_cnt, fn_cnt = (
                len(tp),
                len(fp["hallucinations"]) + len(fp["misclassifications"]),
                len(fn["missed_detections"]) + len(fn["misclassifications"]),
            )
            precision = (
                tp_cnt / (tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else -1
            )
            recall = (
                tp_cnt / (tp_cnt + fn_cnt) if (tp_cnt + fn_cnt) > 0 else -1
            )
            f1_score = (
                (2 * precision * recall) / (precision + recall)
                if precision and recall
                else -1
            )

            if "PrecisionRecallCurve" in metrics:
                pr_curves[label_value][confidence_threshold] = {
                    "tp": tp_cnt,
                    "fp": fp_cnt,
                    "fn": fn_cnt,
                    "tn": None,  # tn and accuracy aren't applicable to detection tasks because there's an infinite number of true negatives
                    "precision": precision,
                    "recall": recall,
                    "accuracy": None,
                    "f1_score": f1_score,
                }

            if "DetailedPrecisionRecallCurve" in metrics:
                detailed_pr_curves[label_value][confidence_threshold] = {
                    "tp": {
                        "total": tp_cnt,
                        "observations": {
                            "all": {
                                "count": tp_cnt,
                                "examples": (
                                    random.sample(tp, pr_curve_max_examples)
                                    if len(tp) >= pr_curve_max_examples
                                    else tp
                                ),
                            }
                        },
                    },
                    "fn": {
                        "total": fn_cnt,
                        "observations": {
                            "misclassifications": {
                                "count": len(fn["misclassifications"]),
                                "examples": (
                                    random.sample(
                                        fn["misclassifications"],
                                        pr_curve_max_examples,
                                    )
                                    if len(fn["misclassifications"])
                                    >= pr_curve_max_examples
                                    else fn["misclassifications"]
                                ),
                            },
                            "missed_detections": {
                                "count": len(fn["missed_detections"]),
                                "examples": (
                                    random.sample(
                                        fn["missed_detections"],
                                        pr_curve_max_examples,
                                    )
                                    if len(fn["missed_detections"])
                                    >= pr_curve_max_examples
                                    else fn["missed_detections"]
                                ),
                            },
                        },
                    },
                    "fp": {
                        "total": fp_cnt,
                        "observations": {
                            "misclassifications": {
                                "count": len(fp["misclassifications"]),
                                "examples": (
                                    random.sample(
                                        fp["misclassifications"],
                                        pr_curve_max_examples,
                                    )
                                    if len(fp["misclassifications"])
                                    >= pr_curve_max_examples
                                    else fp["misclassifications"]
                                ),
                            },
                            "hallucinations": {
                                "count": len(fp["hallucinations"]),
                                "examples": (
                                    random.sample(
                                        fp["hallucinations"],
                                        pr_curve_max_examples,
                                    )
                                    if len(fp["hallucinations"])
                                    >= pr_curve_max_examples
                                    else fp["hallucinations"]
                                ),
                            },
                        },
                    },
                }

        pr_output[label_key].update(dict(pr_curves))
        detailed_pr_output[label_key].update(dict(detailed_pr_curves))

    output = []

    if "PrecisionRecallCurve" in metrics:
        output += [
            schemas.PrecisionRecallCurve(
                label_key=key,
                value=dict(value),
                pr_curve_iou_threshold=iou_threshold,
            )
            for key, value in pr_output.items()
        ]

    if "DetailedPrecisionRecallCurve" in metrics:
        output += [
            schemas.DetailedPrecisionRecallCurve(
                label_key=key,
                value=dict(value),
                pr_curve_iou_threshold=iou_threshold,
            )
            for key, value in detailed_pr_output.items()
        ]

    return output


def _calculate_ap_and_ar(
    sorted_ranked_pairs: dict[str, list[RankedPair]],
    number_of_groundtruths_per_grouper: dict[str, int],
    grouper_mappings: dict[str, dict[str, schemas.Label]],
    iou_thresholds: list[float],
    recall_score_threshold: float,
) -> Tuple[list[schemas.APMetric], list[schemas.ARMetric]]:
    """
    Computes the average precision. Return is a dict with keys
    `f"IoU={iou_thres}"` for each `iou_thres` in `iou_thresholds` as well as
    `f"IoU={min(iou_thresholds)}:{max(iou_thresholds)}"` which is the average
    of the scores across all of the IoU thresholds.
    """
    if recall_score_threshold < 0 or recall_score_threshold > 1.0:
        raise ValueError(
            "recall_score_threshold should exist in the range 0 <= threshold <= 1."
        )
    if min(iou_thresholds) <= 0 or max(iou_thresholds) > 1.0:
        raise ValueError(
            "IOU thresholds should exist in the range 0 < threshold <= 1."
        )

    ap_metrics = []
    ar_metrics = []

    for grouper_id, grouper_label in grouper_mappings[
        "grouper_id_to_grouper_label_mapping"
    ].items():
        recalls_across_thresholds = []

        for iou_threshold in iou_thresholds:
            if grouper_id not in number_of_groundtruths_per_grouper.keys():
                continue

            precisions = []
            recalls = []
            # recall true positives require a confidence score above recall_score_threshold, while precision
            # true positives only require a confidence score above 0
            recall_cnt_tp = 0
            recall_cnt_fp = 0
            recall_cnt_fn = 0
            precision_cnt_tp = 0
            precision_cnt_fp = 0

            if grouper_id in sorted_ranked_pairs:
                matched_gts_for_precision = set()
                matched_gts_for_recall = set()
                for row in sorted_ranked_pairs[grouper_id]:

                    precision_score_conditional = row.score > 0

                    recall_score_conditional = (
                        row.score > recall_score_threshold
                        or (
                            math.isclose(row.score, recall_score_threshold)
                            and recall_score_threshold > 0
                        )
                    )

                    iou_conditional = (
                        row.iou >= iou_threshold and iou_threshold > 0
                    )

                    if (
                        recall_score_conditional
                        and iou_conditional
                        and row.gt_id not in matched_gts_for_recall
                    ):
                        recall_cnt_tp += 1
                        matched_gts_for_recall.add(row.gt_id)
                    else:
                        recall_cnt_fp += 1

                    if (
                        precision_score_conditional
                        and iou_conditional
                        and row.gt_id not in matched_gts_for_precision
                    ):
                        matched_gts_for_precision.add(row.gt_id)
                        precision_cnt_tp += 1
                    else:
                        precision_cnt_fp += 1

                    recall_cnt_fn = (
                        number_of_groundtruths_per_grouper[grouper_id]
                        - recall_cnt_tp
                    )

                    precision_cnt_fn = (
                        number_of_groundtruths_per_grouper[grouper_id]
                        - precision_cnt_tp
                    )

                    precisions.append(
                        precision_cnt_tp
                        / (precision_cnt_tp + precision_cnt_fp)
                        if (precision_cnt_tp + precision_cnt_fp)
                        else 0
                    )
                    recalls.append(
                        precision_cnt_tp
                        / (precision_cnt_tp + precision_cnt_fn)
                        if (precision_cnt_tp + precision_cnt_fn)
                        else 0
                    )

                recalls_across_thresholds.append(
                    recall_cnt_tp / (recall_cnt_tp + recall_cnt_fn)
                    if (recall_cnt_tp + recall_cnt_fn)
                    else 0
                )
            else:
                precisions = [0]
                recalls = [0]
                recalls_across_thresholds.append(0)

            ap_metrics.append(
                schemas.APMetric(
                    iou=iou_threshold,
                    value=_calculate_101_pt_interp(
                        precisions=precisions, recalls=recalls
                    ),
                    label=grouper_label,
                )
            )

        ar_metrics.append(
            schemas.ARMetric(
                ious=set(iou_thresholds),
                value=(
                    sum(recalls_across_thresholds)
                    / len(recalls_across_thresholds)
                    if recalls_across_thresholds
                    else -1
                ),
                label=grouper_label,
            )
        )

    return ap_metrics, ar_metrics


def _compute_detection_metrics(
    db: Session,
    parameters: schemas.EvaluationParameters,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    target_type: enums.AnnotationType,
) -> Sequence[
    schemas.APMetric
    | schemas.ARMetric
    | schemas.APMetricAveragedOverIOUs
    | schemas.mAPMetric
    | schemas.mARMetric
    | schemas.mAPMetricAveragedOverIOUs
    | schemas.PrecisionRecallCurve
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
    List[schemas.APMetric | schemas.ARMetric | schemas.APMetricAveragedOverIOUs | schemas.mAPMetric | schemas.mARMetric | schemas.mAPMetricAveragedOverIOUs | schemas.PrecisionRecallCurve]
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
            case AnnotationType.RASTER:
                return table.raster
            case _:
                raise RuntimeError

    def _annotation_type_to_geojson(
        annotation_type: AnnotationType,
        table,
    ):
        match annotation_type:
            case AnnotationType.BOX:
                box = table.box
            case AnnotationType.POLYGON:
                box = gfunc.ST_Envelope(table.polygon)
            case AnnotationType.RASTER:
                box = gfunc.ST_Envelope(gfunc.ST_MinConvexHull(table.raster))
            case _:
                raise RuntimeError
        return gfunc.ST_AsGeoJSON(box)

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

    labels = core.fetch_union_of_labels(
        db=db,
        rhs=prediction_filter,
        lhs=groundtruth_filter,
    )

    grouper_mappings = create_grouper_mappings(
        labels=labels,
        label_map=parameters.label_map,
        evaluation_type=enums.TaskType.OBJECT_DETECTION,
    )

    # Join gt, datum, annotation, label. Map grouper_ids to each label_id.
    gt = (
        Query(
            models.Dataset.name.label("dataset_name"),
            models.GroundTruth.id.label("id"),
            models.GroundTruth.annotation_id.label("annotation_id"),
            models.GroundTruth.label_id.label("label_id"),
            models.Annotation.datum_id.label("datum_id"),
            models.Datum.uid.label("datum_uid"),
            case(
                grouper_mappings["label_id_to_grouper_id_mapping"],
                value=models.GroundTruth.label_id,
            ).label("label_id_grouper"),
            _annotation_type_to_geojson(target_type, models.Annotation).label(
                "geojson"
            ),
        )
        .filter(groundtruth_filter)
        .groundtruths("groundtruths")
    )

    # Join pd, datum, annotation, label
    pd = (
        Query(
            models.Dataset.name.label("dataset_name"),
            models.Prediction.id.label("id"),
            models.Prediction.annotation_id.label("annotation_id"),
            models.Prediction.label_id.label("label_id"),
            models.Prediction.score.label("score"),
            models.Annotation.datum_id.label("datum_id"),
            models.Datum.uid.label("datum_uid"),
            case(
                grouper_mappings["label_id_to_grouper_id_mapping"],
                value=models.Prediction.label_id,
            ).label("label_id_grouper"),
            _annotation_type_to_geojson(target_type, models.Annotation).label(
                "geojson"
            ),
        )
        .filter(prediction_filter)
        .predictions("predictions")
    )

    joint = (
        select(
            func.coalesce(pd.c.dataset_name, gt.c.dataset_name).label(
                "dataset_name"
            ),
            gt.c.datum_id.label("gt_datum_id"),
            pd.c.datum_id.label("pd_datum_id"),
            gt.c.datum_uid.label("gt_datum_uid"),
            pd.c.datum_uid.label("pd_datum_uid"),
            gt.c.geojson.label("gt_geojson"),
            pd.c.geojson.label("pd_geojson"),
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
        .select_from(pd)  # type: ignore - SQLAlchemy type issue
        .outerjoin(
            gt,  # type: ignore - SQLAlchemy type issue
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
            joint.c.dataset_name,
            joint.c.pd_datum_id,
            joint.c.gt_datum_id,
            joint.c.pd_datum_uid,
            joint.c.gt_datum_uid,
            joint.c.gt_id.label("gt_id"),
            joint.c.pd_id.label("pd_id"),
            joint.c.gt_label_id.label("gt_label_id"),
            joint.c.pd_label_id.label("pd_label_id"),
            joint.c.gt_label_id_grouper.label("gt_label_id_grouper"),
            joint.c.pd_label_id_grouper.label("pd_label_id_grouper"),
            joint.c.score.label("score"),
            func.coalesce(iou_computation, 0).label("iou"),
            joint.c.gt_geojson,
            joint.c.pd_geojson,
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
    ordered_ious = (
        db.query(ious).order_by(-ious.c.score, -ious.c.iou, ious.c.gt_id).all()
    )

    # Filter out repeated predictions
    pd_set = set()
    ranking = {}
    for row in ordered_ious:
        (
            dataset_name,
            _,
            _,
            pd_datum_uid,
            gt_datum_uid,
            gt_id,
            pd_id,
            _,
            _,
            gt_label_id_grouper,
            _,
            score,
            iou,
            gt_geojson,
            _,
        ) = row

        # there should only be one rankedpair per prediction but
        # there can be multiple rankedpairs per groundtruth at this point (i.e. before
        # an iou threshold is specified)
        if pd_id not in pd_set:
            pd_set.add(pd_id)
            if gt_label_id_grouper not in ranking:
                ranking[gt_label_id_grouper] = []

            ranking[gt_label_id_grouper].append(
                RankedPair(
                    dataset_name=dataset_name,
                    pd_datum_uid=pd_datum_uid,
                    gt_datum_uid=gt_datum_uid,
                    gt_geojson=gt_geojson,
                    gt_id=gt_id,
                    pd_id=pd_id,
                    score=score,
                    iou=iou,
                )
            )

    # get pds not appearing
    predictions = db.query(
        Query(
            models.Prediction.id,
            models.Prediction.score,
            models.Dataset.name,
            models.Datum.uid.label("pd_datum_uid"),
            case(
                grouper_mappings["label_id_to_grouper_id_mapping"],
                value=models.Prediction.label_id,
            ).label("label_id_grouper"),
        )
        .filter(prediction_filter)
        .predictions(as_subquery=False)
        .where(models.Prediction.id.notin_(pd_set))  # type: ignore - SQLAlchemy type issue
        .subquery()
    ).all()

    for pd_id, score, dataset_name, pd_datum_uid, grouper_id in predictions:
        if grouper_id in ranking and pd_id not in pd_set:
            # add to ranking in sorted order
            bisect.insort(  # type: ignore - bisect type issue
                ranking[grouper_id],
                RankedPair(
                    dataset_name=dataset_name,
                    pd_datum_uid=pd_datum_uid,
                    gt_datum_uid=None,
                    gt_geojson=None,
                    gt_id=None,
                    pd_id=pd_id,
                    score=score,
                    iou=0,
                ),
                key=lambda rp: -rp.score,  # bisect assumes decreasing order
            )

    # Get the groundtruths per grouper_id
    groundtruths_per_grouper = defaultdict(list)
    number_of_groundtruths_per_grouper = defaultdict(int)

    groundtruths = db.query(
        Query(
            models.GroundTruth.id,
            case(
                grouper_mappings["label_id_to_grouper_id_mapping"],
                value=models.GroundTruth.label_id,
            ).label("label_id_grouper"),
            models.Datum.uid.label("datum_uid"),
            models.Dataset.name.label("dataset_name"),
            _annotation_type_to_geojson(target_type, models.Annotation).label(
                "gt_geojson"
            ),
        )
        .filter(groundtruth_filter)
        .groundtruths()  # type: ignore - SQLAlchemy type issue
    ).all()  # type: ignore - SQLAlchemy type issue

    for gt_id, grouper_id, datum_uid, dset_name, gt_geojson in groundtruths:
        # we're ok with duplicates since they indicate multiple groundtruths for a given dataset/datum_id
        groundtruths_per_grouper[grouper_id].append(
            (dset_name, datum_uid, gt_id, gt_geojson)
        )
        number_of_groundtruths_per_grouper[grouper_id] += 1

    # Optionally compute precision-recall curves
    if parameters.metrics and (
        "PrecisionRecallCurve" in parameters.metrics
        or "DetailedPrecisionRecallCurve" in parameters.metrics
    ):
        false_positive_entries = db.query(
            select(
                joint.c.dataset_name,
                joint.c.gt_datum_uid,
                joint.c.pd_datum_uid,
                joint.c.gt_label_id_grouper.label("gt_label_id_grouper"),
                joint.c.pd_label_id_grouper.label("pd_label_id_grouper"),
                joint.c.score.label("score"),
                joint.c.pd_geojson,
            )
            .select_from(joint)
            .where(
                or_(
                    joint.c.gt_id.is_(None),
                    joint.c.pd_id.is_(None),
                )
            )
            .subquery()
        ).all()

        pr_curves = _compute_curves(
            metrics=parameters.metrics,
            sorted_ranked_pairs=ranking,
            grouper_mappings=grouper_mappings,
            groundtruths_per_grouper=groundtruths_per_grouper,
            false_positive_entries=false_positive_entries,
            iou_threshold=parameters.pr_curve_iou_threshold,
            pr_curve_max_examples=(
                parameters.pr_curve_max_examples
                if parameters.pr_curve_max_examples
                else 1
            ),
        )
    else:
        pr_curves = []

    if any(
        [
            m in parameters.metrics
            for m in [
                "AP",
                "AR",
                "mAP",
                "APAveragedOverIOUs",
                "mAR",
                "mAPAveragedOverIOUs",
            ]
        ]
    ):
        ap_ar_output = []

        ap_metrics, ar_metrics = _calculate_ap_and_ar(
            sorted_ranked_pairs=ranking,
            number_of_groundtruths_per_grouper=number_of_groundtruths_per_grouper,
            iou_thresholds=parameters.iou_thresholds_to_compute,
            grouper_mappings=grouper_mappings,
            recall_score_threshold=parameters.recall_score_threshold,
        )

        if "AR" in parameters.metrics:
            ap_ar_output += ar_metrics

        # calculate averaged metrics
        mean_ap_metrics = _compute_mean_detection_metrics_from_aps(ap_metrics)
        mean_ar_metrics = _compute_mean_ar_metrics(ar_metrics)

        ap_metrics_ave_over_ious = list(
            _compute_detection_metrics_averaged_over_ious_from_aps(ap_metrics)
        )

        if "mAR" in parameters.metrics:
            ap_ar_output += mean_ar_metrics

        if "APAveragedOverIOUs" in parameters.metrics:
            ap_ar_output += ap_metrics_ave_over_ious

        if "mAPAveragedOverIOUs" in parameters.metrics:
            mean_ap_metrics_ave_over_ious = list(
                _compute_mean_detection_metrics_from_aps(
                    ap_metrics_ave_over_ious
                )
            )
            ap_ar_output += mean_ap_metrics_ave_over_ious

        if "AP" in parameters.metrics:
            ap_metrics = [
                m
                for m in ap_metrics
                if m.iou in parameters.iou_thresholds_to_return
            ]
            ap_ar_output += ap_metrics

        if "mAP" in parameters.metrics:
            mean_ap_metrics = [
                m
                for m in mean_ap_metrics
                if isinstance(m, schemas.mAPMetric)
                and m.iou in parameters.iou_thresholds_to_return
            ]
            ap_ar_output += mean_ap_metrics
    else:
        ap_ar_output = []

    return ap_ar_output + pr_curves


def _compute_detection_metrics_averaged_over_ious_from_aps(
    ap_scores: Sequence[schemas.APMetric],
) -> Sequence[schemas.APMetricAveragedOverIOUs]:
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


def _compute_mean_ar_metrics(
    ar_metrics: Sequence[schemas.ARMetric],
) -> list[schemas.mARMetric]:
    """Calculate the mean of a list of AR metrics."""

    if len(ar_metrics) == 0:
        return []

    value_dict = defaultdict(lambda: defaultdict(list))
    for metric in ar_metrics:
        value_dict[metric.label.key][frozenset(metric.ious)].append(
            metric.value
        )

    mean_metrics = []
    for label_key, nested_dict in value_dict.items():
        for ious, values in nested_dict.items():
            mean_metrics.append(
                schemas.mARMetric(
                    ious=ious,
                    value=_average_ignore_minus_one(values),
                    label_key=label_key,
                )
            )

    return mean_metrics


def _compute_mean_detection_metrics_from_aps(
    ap_scores: Sequence[schemas.APMetric | schemas.APMetricAveragedOverIOUs],
) -> Sequence[schemas.mAPMetric | schemas.mAPMetricAveragedOverIOUs]:
    """Calculate the mean of a list of AP metrics."""

    if len(ap_scores) == 0:
        return []

    # dictionary for mapping an iou threshold to set of APs
    vals = defaultdict(lambda: defaultdict(list))
    for ap in ap_scores:
        if hasattr(ap, "iou"):
            iou = ap.iou  # type: ignore - pyright doesn't consider hasattr checks
        else:
            iou = frozenset(ap.ious)  # type: ignore - pyright doesn't consider hasattr checks
        vals[ap.label.key][iou].append(ap.value)

    # get mAP metrics at the individual IOUs
    mean_detection_metrics = []

    for label_key, nested_dict in vals.items():
        for iou, values in nested_dict.items():
            if isinstance(iou, float):
                mean_detection_metrics.append(
                    schemas.mAPMetric(
                        iou=iou,
                        value=_average_ignore_minus_one(values),
                        label_key=label_key,
                    )
                )
            else:
                mean_detection_metrics.append(
                    schemas.mAPMetricAveragedOverIOUs(
                        ious=iou,
                        value=_average_ignore_minus_one(
                            values,
                        ),
                        label_key=label_key,
                    )
                )

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
                db=db,
                dataset=dataset,
                task_type=enums.TaskType.OBJECT_DETECTION,
            )
            model_type = core.get_annotation_type(
                db=db,
                dataset=dataset,
                model=model,
                task_type=enums.TaskType.OBJECT_DETECTION,
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
            db=db, dataset=dataset, task_type=enums.TaskType.OBJECT_DETECTION
        )
        core.convert_geometry(
            db=db,
            dataset=dataset,
            source_type=source_type,
            target_type=target_type,
            task_type=enums.TaskType.OBJECT_DETECTION,
        )
        # model
        source_type = core.get_annotation_type(
            db=db,
            dataset=dataset,
            model=model,
            task_type=enums.TaskType.OBJECT_DETECTION,
        )
        core.convert_geometry(
            db=db,
            dataset=dataset,
            model=model,
            source_type=source_type,
            target_type=target_type,
            task_type=enums.TaskType.OBJECT_DETECTION,
        )

    return target_type


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
    groundtruth_filter = schemas.Filter(**evaluation.datum_filter)
    prediction_filter = groundtruth_filter.model_copy()
    prediction_filter.model_names = [evaluation.model_name]
    parameters = schemas.EvaluationParameters(**evaluation.parameters)

    # load task type into filters
    groundtruth_filter.task_types = [parameters.task_type]
    prediction_filter.task_types = [parameters.task_type]

    log_evaluation_item_counts(
        db=db,
        evaluation=evaluation,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
    )

    # fetch model and datasets
    datasets = (
        db.query(Query(models.Dataset).filter(groundtruth_filter).any())  # type: ignore - SQLAlchemy type issue
        .distinct()
        .all()
    )
    model = (
        db.query(Query(models.Model).filter(prediction_filter).any())  # type: ignore - SQLAlchemy type issue
        .distinct()
        .one_or_none()
    )

    # verify
    if not datasets:
        raise RuntimeError(
            "No datasets could be found that meet filter requirements."
        )
    if model is None:
        raise RuntimeError(
            f"Model '{evaluation.model_name}' does not meet filter requirements."
        )

    # ensure that all annotations have a common type to operate over
    target_type = _convert_annotations_to_common_type(
        db=db,
        datasets=datasets,
        model=model,
        target_type=parameters.convert_annotations_to_type,
    )
    match target_type:
        case AnnotationType.BOX:
            groundtruth_filter.require_bounding_box = True
            prediction_filter.require_bounding_box = True
        case AnnotationType.POLYGON:
            groundtruth_filter.require_polygon = True
            prediction_filter.require_polygon = True
        case AnnotationType.RASTER:
            groundtruth_filter.require_raster = True
            prediction_filter.require_raster = True

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

    log_evaluation_duration(
        evaluation=evaluation,
        db=db,
    )

    return evaluation_id
