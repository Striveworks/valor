import bisect
import heapq
import io
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas
from geoalchemy2 import functions as gfunc
from PIL import Image
from sqlalchemy import and_, case, func, select
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session, aliased

from valor_api import enums, schemas
from valor_api.backend import core, models
from valor_api.backend.metrics.metric_utils import (
    commit_results,
    create_grouper_mappings,
    log_evaluation_duration,
    log_evaluation_item_counts,
    prepare_filter_for_evaluation,
    validate_computation,
)
from valor_api.backend.query import generate_query, generate_select
from valor_api.enums import AnnotationType

pandas.set_option("display.max_columns", None)


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
    is_match: bool


def _calculate_101_pt_interp(precisions, recalls) -> float:
    """Use the 101 point interpolation method (following torchmetrics)"""
    assert len(precisions) == len(recalls)

    if len(precisions) == 0:
        return 0

    if all([x == -1 for x in precisions + recalls]):
        return -1

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


def _calculate_ap_and_ar(
    sorted_ranked_pairs: dict[str, list[RankedPair]],
    number_of_groundtruths_per_grouper: dict[str, int],
    grouper_mappings: dict[str, dict[str, schemas.Label]],
    iou_thresholds: list[float],
    recall_score_threshold: float,
) -> Tuple[list[schemas.APMetric], list[schemas.ARMetric]]:
    """
    Computes the average precision and average recall metrics. Returns a dict with keys
    `f"IoU={iou_thres}"` for each `iou_thres` in `iou_thresholds` as well as
    `f"IoU={min(iou_thresholds)}:{max(iou_thresholds)}", which is the average
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


def _compute_curves(
    sorted_ranked_pairs: dict[int, list[RankedPair]],
    grouper_mappings: dict[str, dict[str, schemas.Label]],
    groundtruths_per_grouper: dict[int, list],
    false_positive_entries: list[tuple],
    iou_threshold: float,
) -> list[schemas.PrecisionRecallCurve]:
    """
    Calculates precision-recall curves for each class.

    Parameters
    ----------
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

    Returns
    -------
    list[schemas.PrecisionRecallCurve]
        A list of PrecisionRecallCurve metrics.
    """

    output = defaultdict(dict)

    for grouper_id, grouper_label in grouper_mappings[
        "grouper_id_to_grouper_label_mapping"
    ].items():

        curves = defaultdict(lambda: defaultdict(dict))

        label_key = grouper_label.key
        label_value = grouper_label.value

        for confidence_threshold in [x / 100 for x in range(5, 100, 5)]:
            if grouper_id not in sorted_ranked_pairs:
                tp_cnt = 0
                if grouper_id in groundtruths_per_grouper:
                    fn_cnt = len(groundtruths_per_grouper[grouper_id])
                else:
                    fn_cnt = 0

            else:
                tp_cnt, fn_cnt = 0, 0
                seen_gts = set()

                for row in sorted_ranked_pairs[grouper_id]:
                    if (
                        row.score >= confidence_threshold
                        and row.iou >= iou_threshold
                        and row.gt_id not in seen_gts
                    ):
                        tp_cnt += 1
                        seen_gts.add(row.gt_id)

                for (
                    _,
                    _,
                    gt_id,
                ) in groundtruths_per_grouper[grouper_id]:
                    if gt_id not in seen_gts:
                        fn_cnt += 1

            fp_cnt = 0
            for (
                _,
                _,
                _,
                gt_label_id_grouper,
                pd_label_id_grouper,
                pd_score,
            ) in false_positive_entries:
                if (
                    pd_score >= confidence_threshold
                    and pd_label_id_grouper == grouper_id
                    and gt_label_id_grouper is None
                ):
                    fp_cnt += 1

            # calculate metrics
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

            curves[label_value][confidence_threshold] = {
                "tp": tp_cnt,
                "fp": fp_cnt,
                "fn": fn_cnt,
                "tn": None,  # tn and accuracy aren't applicable to detection tasks because there's an infinite number of true negatives
                "precision": precision,
                "recall": recall,
                "accuracy": None,
                "f1_score": f1_score,
            }

        output[label_key].update(dict(curves))

    return [
        schemas.PrecisionRecallCurve(
            label_key=key,
            value=value,
            pr_curve_iou_threshold=iou_threshold,
        )
        for key, value in output.items()
    ]


def _compute_detailed_curves(
    sorted_ranked_pairs: dict[int, list[RankedPair]],
    grouper_mappings: dict[str, dict[str, schemas.Label]],
    groundtruths_per_grouper: dict[int, list],
    predictions_per_grouper: dict[int, list],
    pr_curve_iou_threshold: float,
    pr_curve_max_examples: int,
) -> list[schemas.PrecisionRecallCurve | schemas.DetailedPrecisionRecallCurve]:
    """
    Calculates precision-recall curves and detailed precision recall curves for each class.

    Parameters
    ----------
    sorted_ranked_pairs: dict[int, list[RankedPair]]
        The ground truth-prediction matches from psql, grouped by grouper_id.
    grouper_mappings: dict[str, dict[str, schemas.Label]]
        A dictionary of mappings that connect groupers to their related labels.
    groundtruths_per_grouper: dict[int, int]
        A dictionary containing the (dataset_name, datum_id, gt_id) for all groundtruths associated with a grouper.
    predictions_per_grouper: dict[int, int]
        A dictionary containing the (dataset_name, datum_id, gt_id) for all predictions associated with a grouper.
    pr_curve_iou_threshold: float
        The IOU threshold to use as a cut-off for our predictions.
    pr_curve_max_examples: int
        The maximum number of datum examples to store per true positive, false negative, etc.

    Returns
    -------
    list[schemas.PrecisionRecallCurve | schemas.DetailedPrecisionRecallCurve]
        A list of PrecisionRecallCurve and DetailedPrecisionRecallCurve metrics.
    """
    pr_output = defaultdict(dict)
    detailed_pr_output = defaultdict(dict)

    # transform sorted_ranked_pairs into two sets (groundtruths and predictions)
    # we'll use these dictionaries to look up the IOU overlap between specific groundtruths and predictions
    # to separate misclassifications
    pd_datums = defaultdict(lambda: defaultdict(list))
    gt_datums = defaultdict(lambda: defaultdict(list))

    for grouper_id, ranked_pairs in sorted_ranked_pairs.items():
        for ranked_pair in ranked_pairs:
            grouper_id_key = hash(
                (
                    ranked_pair.dataset_name,
                    ranked_pair.pd_datum_uid,
                    grouper_mappings["grouper_id_to_grouper_label_mapping"][
                        grouper_id
                    ].key,  # type: ignore
                )
            )
            gt_key = hash(
                (
                    ranked_pair.dataset_name,
                    ranked_pair.gt_datum_uid,
                    ranked_pair.gt_id,
                )
            )
            pd_key = hash(
                (
                    ranked_pair.dataset_name,
                    ranked_pair.pd_datum_uid,
                    ranked_pair.pd_id,
                )
            )
            pd_datums[grouper_id_key][gt_key].append(
                (ranked_pair.iou, ranked_pair.score)
            )
            gt_datums[grouper_id_key][pd_key].append(
                (ranked_pair.iou, ranked_pair.score)
            )

    for grouper_id, grouper_label in grouper_mappings[
        "grouper_id_to_grouper_label_mapping"
    ].items():

        pr_curves = defaultdict(lambda: defaultdict(dict))
        detailed_pr_curves = defaultdict(lambda: defaultdict(dict))

        label_key = grouper_label.key
        label_value = grouper_label.value

        for confidence_threshold in [x / 100 for x in range(5, 100, 5)]:
            seen_pds = set()
            seen_gts = set()

            tp, fp, fn = [], defaultdict(list), defaultdict(list)

            for row in sorted_ranked_pairs[int(grouper_id)]:
                if (
                    row.score >= confidence_threshold
                    and row.iou >= pr_curve_iou_threshold
                    and row.gt_id not in seen_gts
                    and row.is_match is True
                ):
                    tp += [
                        (
                            row.dataset_name,
                            row.gt_datum_uid,
                            row.gt_geojson,
                        )
                    ]
                    seen_gts.add(row.gt_id)
                    seen_pds.add(row.pd_id)

            if grouper_id in groundtruths_per_grouper:
                for (
                    dataset_name,
                    datum_uid,
                    gt_id,
                    gt_geojson,
                ) in groundtruths_per_grouper[int(grouper_id)]:
                    if gt_id not in seen_gts:
                        grouper_id_key = hash(
                            (
                                dataset_name,
                                datum_uid,
                                grouper_mappings[
                                    "grouper_id_to_grouper_label_mapping"
                                ][grouper_id].key,
                            )
                        )
                        gt_key = hash((dataset_name, datum_uid, gt_id))
                        misclassification_detected = any(
                            [
                                score >= confidence_threshold
                                and iou >= pr_curve_iou_threshold
                                for (iou, score) in pd_datums[grouper_id_key][
                                    gt_key
                                ]
                            ]
                        )
                        # if there is at least one prediction overlapping the groundtruth with a sufficient score and iou threshold, then it's a misclassification
                        if misclassification_detected:
                            fn["misclassifications"].append(
                                (dataset_name, datum_uid, gt_geojson)
                            )
                        else:
                            fn["no_predictions"].append(
                                (dataset_name, datum_uid, gt_geojson)
                            )

            if grouper_id in predictions_per_grouper:
                for (
                    dataset_name,
                    datum_uid,
                    pd_id,
                    pd_geojson,
                ) in predictions_per_grouper[int(grouper_id)]:
                    if pd_id not in seen_pds:
                        grouper_id_key = hash(
                            (
                                dataset_name,
                                datum_uid,
                                grouper_mappings[
                                    "grouper_id_to_grouper_label_mapping"
                                ][
                                    grouper_id
                                ].key,  # type: ignore
                            )
                        )
                        pd_key = hash((dataset_name, datum_uid, pd_id))
                        misclassification_detected = any(
                            [
                                iou >= pr_curve_iou_threshold
                                and score >= confidence_threshold
                                for (iou, score) in gt_datums[grouper_id_key][
                                    pd_key
                                ]
                            ]
                        )
                        hallucination_detected = any(
                            [
                                score >= confidence_threshold
                                for (_, score) in gt_datums[grouper_id_key][
                                    pd_key
                                ]
                            ]
                        )
                        # if there is at least one groundtruth overlapping the prediction with a sufficient score and iou threshold, then it's a misclassification
                        if misclassification_detected:
                            fp["misclassifications"].append(
                                (dataset_name, datum_uid, pd_geojson)
                            )
                        elif hallucination_detected:
                            fp["hallucinations"].append(
                                (dataset_name, datum_uid, pd_geojson)
                            )

            # calculate metrics
            tp_cnt, fp_cnt, fn_cnt = (
                len(tp),
                len(fp["hallucinations"]) + len(fp["misclassifications"]),
                len(fn["no_predictions"]) + len(fn["misclassifications"]),
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
                        "no_predictions": {
                            "count": len(fn["no_predictions"]),
                            "examples": (
                                random.sample(
                                    fn["no_predictions"],
                                    pr_curve_max_examples,
                                )
                                if len(fn["no_predictions"])
                                >= pr_curve_max_examples
                                else fn["no_predictions"]
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

    output += [
        schemas.PrecisionRecallCurve(
            label_key=key,
            value=dict(value),
            pr_curve_iou_threshold=pr_curve_iou_threshold,
        )
        for key, value in pr_output.items()
    ]

    output += [
        schemas.DetailedPrecisionRecallCurve(
            label_key=key,
            value=dict(value),
            pr_curve_iou_threshold=pr_curve_iou_threshold,
        )
        for key, value in detailed_pr_output.items()
    ]

    return output


def _annotation_type_to_geojson(
    annotation_type: AnnotationType,
    table,
):
    """Get the appropriate column for calculating IOUs from models.Annotation."""
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


def wkt_to_array(wkt_data) -> np.ndarray:
    """Convert a WKT string to an array of coordinates."""
    coordinates = json.loads(wkt_data)["coordinates"][0]
    return np.array(coordinates)


def _calculate_bbox_intersection(bbox1, bbox2) -> float:
    """Calculate the intersection between two bounding boxes."""

    # Calculate intersection coordinates
    xmin_inter = max(bbox1[:, 0].min(), bbox2[:, 0].min())
    ymin_inter = max(bbox1[:, 1].min(), bbox2[:, 1].min())
    xmax_inter = min(bbox1[:, 0].max(), bbox2[:, 0].max())
    ymax_inter = min(bbox1[:, 1].max(), bbox2[:, 1].max())

    # Calculate width and height of intersection area
    width = max(0, xmax_inter - xmin_inter)
    height = max(0, ymax_inter - ymin_inter)

    # Calculate intersection area
    intersection_area = width * height
    return intersection_area


def _calculate_bbox_union(bbox1, bbox2) -> float:
    """Calculate the union area between two bounding boxes."""
    area1 = (bbox1[:, 0].max() - bbox1[:, 0].min()) * (
        bbox1[:, 1].max() - bbox1[:, 1].min()
    )
    area2 = (bbox2[:, 0].max() - bbox2[:, 0].min()) * (
        bbox2[:, 1].max() - bbox2[:, 1].min()
    )
    union_area = area1 + area2 - _calculate_bbox_intersection(bbox1, bbox2)
    return union_area


def _calculate_bbox_iou(bbox1, bbox2) -> float:
    """Calculate the IOU between two bounding boxes."""
    intersection = _calculate_bbox_intersection(bbox1, bbox2)
    union = _calculate_bbox_union(bbox1, bbox2)
    iou = intersection / union
    return iou


def _get_groundtruth_and_prediction_df(
    db: Session,
    groundtruth_filter: schemas.Filter,
    prediction_filter: schemas.Filter,
    grouper_mappings: dict,
    annotation_type: AnnotationType,
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Create groundtruth and prediction dataframes"""

    gt = generate_select(
        models.Dataset.name.label("dataset_name"),
        models.GroundTruth.id.label("id"),
        models.GroundTruth.annotation_id.label("annotation_id"),
        models.Annotation.datum_id.label("datum_id"),
        models.Datum.uid.label("datum_uid"),
        case(
            grouper_mappings["label_id_to_grouper_id_mapping"],
            value=models.GroundTruth.label_id,
        ).label("label_id_grouper"),
        case(
            grouper_mappings["label_id_to_grouper_key_mapping"],
            value=models.GroundTruth.label_id,
        ).label("label_key_grouper"),
        _annotation_type_to_geojson(annotation_type, models.Annotation).label(
            "geojson"
        ),
        gfunc.ST_AsPNG(models.Annotation.raster).label("raster"),
        filters=groundtruth_filter,
        label_source=models.GroundTruth,
    )

    pd = generate_select(
        models.Dataset.name.label("dataset_name"),
        models.Prediction.id.label("id"),
        models.Prediction.annotation_id.label("annotation_id"),
        models.Prediction.score.label("score"),
        models.Annotation.datum_id.label("datum_id"),
        models.Datum.uid.label("datum_uid"),
        case(
            grouper_mappings["label_id_to_grouper_id_mapping"],
            value=models.Prediction.label_id,
        ).label("label_id_grouper"),
        case(
            grouper_mappings["label_id_to_grouper_key_mapping"],
            value=models.Prediction.label_id,
        ).label("label_key_grouper"),
        _annotation_type_to_geojson(annotation_type, models.Annotation).label(
            "geojson"
        ),
        gfunc.ST_AsPNG(models.Annotation.raster).label("raster"),
        filters=prediction_filter,
        label_source=models.Prediction,
    )

    assert isinstance(db.bind, Engine)
    gt_df = pandas.read_sql(gt, db.bind)
    pd_df = pandas.read_sql(pd, db.bind)

    return (gt_df, pd_df)


def _get_joint_df(
    gt_df: pandas.DataFrame,
    pd_df: pandas.DataFrame,
    grouper_mappings: dict,
) -> pandas.DataFrame:
    """Create a joint dataframe of groundtruths and predictions for calculating AR/AP metrics."""

    # add the number of groundtruth observations per groupere
    number_of_groundtruths_per_grouper_df = (
        gt_df.groupby("label_id_grouper", as_index=False)["id"]
        .nunique()
        .rename({"id": "gts_per_grouper"}, axis=1)
    )

    joint_df = pandas.merge(
        gt_df,
        pd_df,
        on=["datum_id", "label_id_grouper"],
        how="outer",
        suffixes=("_gt", "_pd"),
    )

    joint_df = pandas.merge(
        joint_df,
        number_of_groundtruths_per_grouper_df,
        on="label_id_grouper",
        how="outer",
    ).assign(
        label=lambda chain_df: chain_df["label_id_grouper"].map(
            grouper_mappings["grouper_id_to_grouper_label_mapping"]
        )
    )

    return joint_df


def _calculate_iou(
    joint_df: pandas.DataFrame, annotation_type: AnnotationType
):
    """Calculate the IOUs between predictions and groundtruths in a joint dataframe"""
    if annotation_type == AnnotationType.RASTER:

        # filter out rows with null rasters
        joint_df = joint_df.loc[
            ~joint_df["raster_pd"].isnull() & ~joint_df["raster_gt"].isnull(),
            :,
        ]

        # convert the raster into a numpy array
        joint_df[["raster_pd", "raster_gt"]] = (
            joint_df[["raster_pd", "raster_gt"]]
            .map(io.BytesIO)  # type: ignore pandas typing error
            .map(Image.open)
            .map(np.array)
        )

        iou_calculation_df = (
            joint_df.assign(
                intersection=lambda chain_df: chain_df.apply(
                    lambda row: np.logical_and(
                        row["raster_pd"], row["raster_gt"]
                    ).sum(),
                    axis=1,
                )
            )
            .assign(
                union_=lambda chain_df: chain_df["raster_gt"].apply(np.sum)
                + chain_df["raster_pd"].apply(np.sum)
                - chain_df["intersection"]
            )
            .assign(
                iou_=lambda chain_df: chain_df["intersection"]
                / chain_df["union_"]
            )
        )

        joint_df = joint_df.join(iou_calculation_df["iou_"])

    else:
        iou_calculation_df = (
            joint_df.loc[
                ~joint_df["geojson_gt"].isnull()
                & ~joint_df["geojson_pd"].isnull(),
                ["geojson_gt", "geojson_pd"],
            ]
            .map(wkt_to_array)
            .apply(
                lambda row: _calculate_bbox_iou(
                    row["geojson_gt"], row["geojson_pd"]
                ),
                axis=1,
            )
        )

        if not iou_calculation_df.empty:
            iou_calculation_df = iou_calculation_df.rename("iou_")
            joint_df = joint_df.join(iou_calculation_df)
        else:
            joint_df["iou_"] = 0

    return joint_df


def _calculate_grouper_id_level_metrics(
    calculation_df: pandas.DataFrame, parameters: schemas.EvaluationParameters
):
    """Calculate the flags and metrics needed to compute AP, AR, and PR curves."""

    # create flags where predictions meet the score and IOU criteria
    calculation_df["recall_true_positive_flag"] = (
        calculation_df["iou_"] >= calculation_df["iou_threshold"]
    ) & (calculation_df["score"] >= parameters.recall_score_threshold)
    # only consider the highest scoring true positive as an actual true positive
    calculation_df["recall_true_positive_flag"] = calculation_df[
        "recall_true_positive_flag"
    ] & (
        ~calculation_df.groupby(
            ["label_id_grouper", "iou_threshold", "id_gt"], as_index=False
        )["recall_true_positive_flag"].shift(1, fill_value=False)
    )

    calculation_df["precision_true_positive_flag"] = (
        calculation_df["iou_"] >= calculation_df["iou_threshold"]
    ) & (calculation_df["score"] > 0)
    calculation_df["precision_true_positive_flag"] = calculation_df[
        "precision_true_positive_flag"
    ] & (
        ~calculation_df.groupby(
            ["label_id_grouper", "iou_threshold", "id_gt"], as_index=False
        )["precision_true_positive_flag"].shift(1, fill_value=False)
    )

    calculation_df["recall_false_positive_flag"] = ~calculation_df[
        "recall_true_positive_flag"
    ] & (calculation_df["score"] >= parameters.recall_score_threshold)
    calculation_df["precision_false_positive_flag"] = ~calculation_df[
        "precision_true_positive_flag"
    ] & (calculation_df["score"] > 0)

    # calculate true and false positives
    calculation_df = (
        calculation_df.join(
            calculation_df.groupby(
                ["label_id_grouper", "iou_threshold"], as_index=False
            )["recall_true_positive_flag"]
            .cumsum()
            .rename("rolling_recall_tp")
        )
        .join(
            calculation_df.groupby(
                ["label_id_grouper", "iou_threshold"], as_index=False
            )["recall_false_positive_flag"]
            .cumsum()
            .rename("rolling_recall_fp")
        )
        .join(
            calculation_df.groupby(
                ["label_id_grouper", "iou_threshold"], as_index=False
            )["precision_true_positive_flag"]
            .cumsum()
            .rename("rolling_precision_tp")
        )
        .join(
            calculation_df.groupby(
                ["label_id_grouper", "iou_threshold"], as_index=False
            )["precision_false_positive_flag"]
            .cumsum()
            .rename("rolling_precision_fp")
        )
    )

    # calculate false negatives, then precision / recall
    calculation_df = (
        calculation_df.assign(
            rolling_recall_fn=lambda chain_df: chain_df["gts_per_grouper"]
            - chain_df["rolling_recall_tp"]
        )
        .assign(
            rolling_precision_fn=lambda chain_df: chain_df["gts_per_grouper"]
            - chain_df["rolling_precision_tp"]
        )
        .assign(
            precision=lambda chain_df: chain_df["rolling_precision_tp"]
            / (
                chain_df["rolling_precision_tp"]
                + chain_df["rolling_precision_fp"]
            )
        )
        .assign(
            recall_for_AP=lambda chain_df: chain_df["rolling_precision_tp"]
            / (
                chain_df["rolling_precision_tp"]
                + chain_df["rolling_precision_fn"]
            )
        )
        .assign(
            recall_for_AR=lambda chain_df: chain_df["rolling_recall_tp"]
            / (chain_df["rolling_recall_tp"] + chain_df["rolling_recall_fn"])
        )
    )

    # fill any predictions that are missing groundtruths with -1
    # leave any groundtruths that are missing predictions with 0
    calculation_df.loc[
        calculation_df["id_gt"].isnull(),
        ["precision", "recall_for_AP", "recall_for_AR"],
    ] = -1

    calculation_df.loc[
        calculation_df["id_pd"].isnull(),
        ["precision", "recall_for_AP", "recall_for_AR"],
    ] = 0

    return calculation_df


def _calculate_ap_metrics(
    calculation_df: pandas.DataFrame,
    grouper_mappings: dict,
    parameters: schemas.EvaluationParameters,
) -> list[
    schemas.APMetric
    | schemas.APMetricAveragedOverIOUs
    | schemas.mAPMetric
    | schemas.mAPMetricAveragedOverIOUs
]:
    """Calculates all AP metrics, including aggregated metrics like mAP."""

    ap_metrics_df = (
        calculation_df.loc[
            ~calculation_df[
                "id_gt"
            ].isnull(),  # for AP, we don't include any predictions without groundtruths
            [
                "label_id_grouper",
                "iou_threshold",
                "precision",
                "recall_for_AP",
            ],
        ]
        .groupby(["label_id_grouper", "iou_threshold"], as_index=False)
        .apply(
            lambda x: pandas.Series(
                {
                    "calculated_precision": _calculate_101_pt_interp(
                        x["precision"].tolist(),
                        x["recall_for_AP"].tolist(),
                    )
                }
            )
        )
    )

    # add back "label" after grouping operations are complete
    ap_metrics_df["label"] = ap_metrics_df["label_id_grouper"].map(
        grouper_mappings["grouper_id_to_grouper_label_mapping"]
    )

    ap_metrics = [
        schemas.APMetric(
            iou=row["iou_threshold"],
            value=row["calculated_precision"],
            label=row["label"],
        )
        for row in ap_metrics_df.to_dict(orient="records")
    ]

    # calculate mean AP metrics
    ap_metrics_df["label_key"] = ap_metrics_df["label"].apply(lambda x: x.key)

    ap_over_ious_df = (
        ap_metrics_df.groupby(["label_id_grouper"], as_index=False)[
            "calculated_precision"
        ].apply(_mean_ignoring_negative_one)
    ).assign(
        label=lambda chained_df: chained_df["label_id_grouper"].map(
            grouper_mappings["grouper_id_to_grouper_label_mapping"]
        )
    )

    # handle type errors
    assert parameters.iou_thresholds_to_compute
    assert parameters.iou_thresholds_to_return

    ap_over_ious = [
        schemas.APMetricAveragedOverIOUs(
            ious=set(parameters.iou_thresholds_to_compute),
            value=row["calculated_precision"],
            label=row["label"],
        )
        for row in ap_over_ious_df.to_dict(orient="records")
    ]

    map_metrics_df = ap_metrics_df.groupby(
        ["iou_threshold", "label_key"], as_index=False
    )["calculated_precision"].apply(_mean_ignoring_negative_one)

    map_metrics = [
        schemas.mAPMetric(
            iou=row["iou_threshold"],
            value=row["calculated_precision"],
            label_key=row["label_key"],
        )
        for row in map_metrics_df.to_dict(orient="records")  # type: ignore - pandas typing issue
    ]

    map_over_ious_df = ap_metrics_df.groupby(["label_key"], as_index=False)[
        "calculated_precision"
    ].apply(_mean_ignoring_negative_one)

    map_over_ious = [
        schemas.mAPMetricAveragedOverIOUs(
            ious=set(parameters.iou_thresholds_to_compute),
            value=row["calculated_precision"],
            label_key=row["label_key"],
        )
        for row in map_over_ious_df.to_dict(orient="records")  # type: ignore - pandas typing issue
    ]

    return (
        [m for m in ap_metrics if m.iou in parameters.iou_thresholds_to_return]
        + [
            m
            for m in map_metrics
            if m.iou in parameters.iou_thresholds_to_return
        ]
        + ap_over_ious
        + map_over_ious
    )


def _calculate_ar_metrics(
    calculation_df: pandas.DataFrame,
    grouper_mappings: dict,
    parameters: schemas.EvaluationParameters,
) -> list[schemas.ARMetric | schemas.mARMetric]:
    """Calculates all AR metrics, including aggregated metrics like mAR."""

    # get the max recall_for_AR for each threshold, then take the mean across thresholds
    ar_metrics_df = (
        calculation_df.groupby(
            ["label_id_grouper", "iou_threshold"], as_index=False
        )["recall_for_AR"]
        .max()
        .groupby("label_id_grouper", as_index=False)["recall_for_AR"]
        .mean()
    )

    # add back "label" after grouping operations are complete
    ar_metrics_df["label"] = ar_metrics_df["label_id_grouper"].map(
        grouper_mappings["grouper_id_to_grouper_label_mapping"]
    )

    # resolve typing error
    assert parameters.iou_thresholds_to_compute

    ious_ = set(parameters.iou_thresholds_to_compute)
    ar_metrics = [
        schemas.ARMetric(
            ious=ious_,
            value=row["recall_for_AR"],
            label=row["label"],
        )
        for row in ar_metrics_df.to_dict(orient="records")
    ]

    # calculate mAR
    ar_metrics_df["label_key"] = ar_metrics_df["label"].apply(lambda x: x.key)
    mar_metrics_df = ar_metrics_df.groupby(["label_key"], as_index=False)[
        "recall_for_AR"
    ].apply(_mean_ignoring_negative_one)

    mar_metrics = [
        schemas.mARMetric(
            ious=ious_,
            value=row["recall_for_AR"],
            label_key=row["label_key"],
        )
        for row in mar_metrics_df.to_dict(orient="records")
    ]

    return ar_metrics + mar_metrics


def _calculate_pr_metrics(
    joint_df: pandas.DataFrame,
    grouper_mappings: dict,
    parameters: schemas.EvaluationParameters,
) -> list[schemas.PrecisionRecallCurve]:
    """Calculates all PrecisionRecallCurve metrics."""

    if not (
        parameters.metrics_to_return
        and enums.MetricType.PrecisionRecallCurve
        in parameters.metrics_to_return
    ):
        return []

    confidence_thresholds = [x / 100 for x in range(5, 100, 5)]
    pr_calculation_df = pandas.concat(
        [
            joint_df.assign(confidence_threshold=threshold)
            for threshold in confidence_thresholds
        ],
        ignore_index=True,
    ).sort_values(
        by=[
            "label_id_grouper",
            "confidence_threshold",
            "score",
            "iou_",
        ],
        ascending=False,
    )

    pr_calculation_df["true_positive_flag"] = (
        (pr_calculation_df["iou_"] >= parameters.pr_curve_iou_threshold)
        & (
            pr_calculation_df["score"]
            >= pr_calculation_df["confidence_threshold"]
        )
        & (
            pr_calculation_df.groupby(
                ["label_id_grouper", "confidence_threshold", "id_gt"]
            ).cumcount()
            == 0
        )  # only the first gt_id in this sorted list should be considered a true positive
    )

    pr_calculation_df["false_positive_flag"] = ~pr_calculation_df[
        "true_positive_flag"
    ] & (
        pr_calculation_df["score"] >= pr_calculation_df["confidence_threshold"]
    )

    pr_metrics_df = (
        pr_calculation_df.groupby(
            [
                "label_id_grouper",
                "confidence_threshold",
                "gts_per_grouper",
            ],
            as_index=False,
        )["true_positive_flag"]
        .sum()
        .merge(
            pr_calculation_df.groupby(
                ["label_id_grouper", "confidence_threshold"],
                as_index=False,
            )["false_positive_flag"].sum(),
            on=["label_id_grouper", "confidence_threshold"],
            how="outer",
        )
        .rename(
            columns={
                "true_positive_flag": "true_positives",
                "false_positive_flag": "false_positives",
            }
        )
        .assign(
            false_negatives=lambda chain_df: chain_df["gts_per_grouper"]
            - chain_df["true_positives"]
        )
        .assign(
            precision=lambda chain_df: chain_df["true_positives"]
            / (chain_df["true_positives"] + chain_df["false_positives"])
        )
        .assign(
            recall=lambda chain_df: chain_df["true_positives"]
            / (chain_df["true_positives"] + chain_df["false_negatives"])
        )
        .assign(
            f1_score=lambda chain_df: (
                2 * chain_df["precision"] * chain_df["recall"]
            )
            / (chain_df["precision"] + chain_df["recall"])
        )
    )

    # add back "label" after grouping operations are complete
    pr_metrics_df["label"] = pr_metrics_df["label_id_grouper"].map(
        grouper_mappings["grouper_id_to_grouper_label_mapping"]
    )

    pr_metrics_df.fillna(0, inplace=True)

    curves = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for row in pr_metrics_df.to_dict(orient="records"):
        curves[row["label"].key][row["label"].value][
            row["confidence_threshold"]
        ] = {
            "tp": row["true_positives"],
            "fp": row["false_positives"],
            "fn": row["false_negatives"],
            "tn": None,  # tn and accuracy aren't applicable to detection tasks because there's an infinite number of true negatives
            "precision": row["precision"],
            "recall": row["recall"],
            "accuracy": None,
            "f1_score": row["f1_score"],
        }

    return [
        schemas.PrecisionRecallCurve(
            label_key=key,
            value=value,  # type: ignore - defaultdict doesn't have strict typing
            pr_curve_iou_threshold=parameters.pr_curve_iou_threshold,
        )
        for key, value in curves.items()
    ]


def _get_geojson_samples(
    detailed_pr_calc_df: pandas.DataFrame,
    parameters: schemas.EvaluationParameters,
    flag_column: str,
    label_key: str,
    label_value: str,
    confidence_threshold: float,
    suffix: str = "_gt",
) -> list:
    """Sample a dataframe to get geojsons based on input criteria."""
    samples = detailed_pr_calc_df[
        (detailed_pr_calc_df["label_key_grouper"] == label_key)
        & (
            (detailed_pr_calc_df["label_value_gt"] == label_value)
            | (detailed_pr_calc_df["label_value_pd"] == label_value)
        )
        & (detailed_pr_calc_df["confidence_threshold"] == confidence_threshold)
        & (detailed_pr_calc_df[flag_column])
    ]

    if samples.empty:
        samples = []
    else:
        samples = list(
            samples.sample(parameters.pr_curve_max_examples)[
                [
                    f"dataset_name{suffix}",
                    f"datum_uid{suffix}",
                    f"geojson{suffix}",
                ]
            ].itertuples(index=False, name=None)
        )
    return samples


def _calculate_detailed_pr_metrics(
    gt_df: pandas.DataFrame,
    pd_df: pandas.DataFrame,
    grouper_mappings: dict,
    parameters: schemas.EvaluationParameters,
    annotation_type: enums.AnnotationType,
) -> list[schemas.DetailedPrecisionRecallCurve]:
    """Calculates all DetailedPrecisionRecallCurve metrics."""
    if not (
        parameters.metrics_to_return
        and enums.MetricType.DetailedPrecisionRecallCurve
        in parameters.metrics_to_return
    ):
        return []

    detailed_pr_joint_df = pandas.merge(
        gt_df,
        pd_df,
        on=["datum_id", "label_key_grouper"],
        how="outer",
        suffixes=("_gt", "_pd"),
    ).assign(
        is_label_match=lambda chain_df: (
            chain_df["label_id_grouper_pd"] == chain_df["label_id_grouper_gt"]
        )
    )

    detailed_pr_joint_df = _calculate_iou(
        joint_df=detailed_pr_joint_df, annotation_type=annotation_type
    )

    # assign labels so that we can tell what we're matching
    detailed_pr_joint_df = detailed_pr_joint_df.assign(
        label_pd=lambda chain_df: chain_df["label_id_grouper_pd"].map(
            grouper_mappings["grouper_id_to_grouper_label_mapping"]
        )
    ).assign(
        label_gt=lambda chain_df: chain_df["label_id_grouper_gt"].map(
            grouper_mappings["grouper_id_to_grouper_label_mapping"]
        )
    )
    detailed_pr_joint_df["label_value_gt"] = detailed_pr_joint_df[
        "label_gt"
    ].map(lambda x: x.value if isinstance(x, schemas.Label) else None)
    detailed_pr_joint_df["label_value_pd"] = detailed_pr_joint_df[
        "label_pd"
    ].map(lambda x: x.value if isinstance(x, schemas.Label) else None)

    # add confidence_threshold to the dataframe and sort
    detailed_pr_calc_df = pandas.concat(
        [
            detailed_pr_joint_df.assign(confidence_threshold=threshold)
            for threshold in [x / 100 for x in range(5, 100, 5)]
        ],
        ignore_index=True,
    ).sort_values(
        by=[
            "label_id_grouper_pd",
            "confidence_threshold",
            "score",
            "iou_",
        ],
        ascending=False,
    )

    # create flags where predictions meet the score and IOU criteria
    detailed_pr_calc_df["true_positive_flag"] = (
        (detailed_pr_calc_df["iou_"] >= parameters.pr_curve_iou_threshold)
        & (
            detailed_pr_calc_df["score"]
            >= detailed_pr_calc_df["confidence_threshold"]
        )
        & detailed_pr_calc_df["is_label_match"]
    )

    # for all the false positives, we consider them to be a misclassification if they overlap with a groundtruth of the same label key
    detailed_pr_calc_df["misclassification_false_positive_flag"] = (
        (detailed_pr_calc_df["iou_"] >= parameters.pr_curve_iou_threshold)
        & (
            detailed_pr_calc_df["score"]
            >= detailed_pr_calc_df["confidence_threshold"]
        )
        & ~detailed_pr_calc_df["is_label_match"]
    )

    # if they aren't a true positive nor a misclassification FP but they meet the iou and score conditions, then they are a hallucination
    detailed_pr_calc_df["hallucination_false_positive_flag"] = (
        (detailed_pr_calc_df["iou_"] < parameters.pr_curve_iou_threshold)
        | (detailed_pr_calc_df["iou_"].isnull())
    ) & (
        detailed_pr_calc_df["score"]
        >= detailed_pr_calc_df["confidence_threshold"]
    )

    # any prediction that is considered a misclassification shouldn't be counted as a hallucination, so we go back and remove these flags
    predictions_associated_with_tps_or_misclassification_fps = (
        detailed_pr_calc_df.groupby(
            ["confidence_threshold", "id_pd"], as_index=False
        )
        .filter(
            lambda x: x["true_positive_flag"].any()
            or x["misclassification_false_positive_flag"].any()
        )
        .groupby(["confidence_threshold"], as_index=False)["id_pd"]
        .unique()
    )
    if not predictions_associated_with_tps_or_misclassification_fps.empty:
        predictions_associated_with_tps_or_misclassification_fps.columns = [
            "confidence_threshold",
            "misclassification_fp_pd_ids",
        ]
        detailed_pr_calc_df = detailed_pr_calc_df.merge(
            predictions_associated_with_tps_or_misclassification_fps,
            on=["confidence_threshold"],
            how="left",
        )
        detailed_pr_calc_df[
            "misclassification_fp_pd_ids"
        ] = detailed_pr_calc_df["misclassification_fp_pd_ids"].map(
            lambda x: x if isinstance(x, np.ndarray) else []
        )
        id_pd_in_set = detailed_pr_calc_df.apply(
            lambda row: (row["id_pd"] in row["misclassification_fp_pd_ids"]),
            axis=1,
        )
        detailed_pr_calc_df.loc[
            (id_pd_in_set)
            & (detailed_pr_calc_df["hallucination_false_positive_flag"]),
            "hallucination_false_positive_flag",
        ] = False

    # next, we flag false negatives by declaring any groundtruth that isn't associated with a true positive to be a false negative
    fn_gt_ids = (
        detailed_pr_calc_df.groupby(
            ["confidence_threshold", "id_gt"], as_index=False
        )
        .filter(lambda x: ~x["true_positive_flag"].any())
        .groupby(["confidence_threshold"], as_index=False)["id_gt"]
        .unique()
    )
    fn_gt_ids.columns = ["confidence_threshold", "false_negative_ids"]

    detailed_pr_calc_df = detailed_pr_calc_df.merge(
        fn_gt_ids, on=["confidence_threshold"], how="left"
    )

    detailed_pr_calc_df["false_negative_flag"] = detailed_pr_calc_df.apply(
        lambda row: row["id_gt"] in row["false_negative_ids"], axis=1
    )

    # it's a misclassification if there is a corresponding misclassification false positive
    detailed_pr_calc_df["misclassification_false_negative_flag"] = (
        detailed_pr_calc_df["misclassification_false_positive_flag"]
        & detailed_pr_calc_df["false_negative_flag"]
    )

    # assign all id_gts that aren't misclassifications but are false negatives to be no_predictions
    no_predictions_fn_gt_ids = (
        detailed_pr_calc_df.groupby(
            ["confidence_threshold", "id_gt"], as_index=False
        )
        .filter(lambda x: ~x["misclassification_false_negative_flag"].any())
        .groupby(["confidence_threshold"], as_index=False)["id_gt"]
        .unique()
    )
    no_predictions_fn_gt_ids.columns = [
        "confidence_threshold",
        "no_predictions_fn_gt_ids",
    ]

    detailed_pr_calc_df = detailed_pr_calc_df.merge(
        no_predictions_fn_gt_ids, on=["confidence_threshold"], how="left"
    )

    detailed_pr_calc_df[
        "no_predictions_false_negative_flag"
    ] = detailed_pr_calc_df.apply(
        lambda row: (row["false_negative_flag"])
        & (row["id_gt"] in row["no_predictions_fn_gt_ids"]),
        axis=1,
    )

    # next, we sum up the occurences of each classification and merge them together into one dataframe
    true_positives = (
        detailed_pr_calc_df[detailed_pr_calc_df["true_positive_flag"]]
        .groupby(
            ["label_key_grouper", "label_value_pd", "confidence_threshold"]
        )["id_pd"]
        .nunique()
    )
    true_positives.name = "true_positives"

    hallucination_false_positives = (
        detailed_pr_calc_df[
            detailed_pr_calc_df["hallucination_false_positive_flag"]
        ]
        .groupby(
            ["label_key_grouper", "label_value_pd", "confidence_threshold"]
        )["id_pd"]
        .nunique()
    )
    hallucination_false_positives.name = "hallucinations_false_positives"

    misclassification_false_positives = (
        detailed_pr_calc_df[
            detailed_pr_calc_df["misclassification_false_positive_flag"]
        ]
        .groupby(
            ["label_key_grouper", "label_value_pd", "confidence_threshold"]
        )["id_pd"]
        .nunique()
    )
    misclassification_false_positives.name = (
        "misclassification_false_positives"
    )

    misclassification_false_negatives = (
        detailed_pr_calc_df[
            detailed_pr_calc_df["misclassification_false_negative_flag"]
        ]
        .groupby(
            ["label_key_grouper", "label_value_gt", "confidence_threshold"]
        )["id_gt"]
        .nunique()
    )
    misclassification_false_negatives.name = (
        "misclassification_false_negatives"
    )

    no_predictions_false_negatives = (
        detailed_pr_calc_df[
            detailed_pr_calc_df["no_predictions_false_negative_flag"]
        ]
        .groupby(
            ["label_key_grouper", "label_value_gt", "confidence_threshold"]
        )["id_gt"]
        .nunique()
    )
    no_predictions_false_negatives.name = "no_predictions_false_negatives"

    # combine these outputs
    detailed_pr_curve_counts_df = (
        pandas.concat(
            [
                detailed_pr_calc_df.loc[
                    ~detailed_pr_calc_df["label_value_pd"].isnull(),
                    [
                        "label_key_grouper",
                        "label_value_pd",
                        "confidence_threshold",
                    ],
                ].rename(columns={"label_value_pd": "label_value"}),
                detailed_pr_calc_df.loc[
                    ~detailed_pr_calc_df["label_value_gt"].isnull(),
                    [
                        "label_key_grouper",
                        "label_value_gt",
                        "confidence_threshold",
                    ],
                ].rename(columns={"label_value_gt": "label_value"}),
            ],
            axis=0,
        )
        .drop_duplicates()
        .merge(
            true_positives,
            left_on=[
                "label_key_grouper",
                "label_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            hallucination_false_positives,
            left_on=[
                "label_key_grouper",
                "label_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            misclassification_false_positives,
            left_on=[
                "label_key_grouper",
                "label_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            misclassification_false_negatives,
            left_on=[
                "label_key_grouper",
                "label_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            no_predictions_false_negatives,
            left_on=[
                "label_key_grouper",
                "label_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
    )

    # we're doing an outer join, so any nulls should be zeroes
    detailed_pr_curve_counts_df.fillna(0, inplace=True)

    # create output
    detailed_pr_curves = defaultdict(lambda: defaultdict(dict))
    for _, values in detailed_pr_curve_counts_df.iterrows():
        label_key = values["label_key_grouper"]
        label_value = values["label_value"]
        confidence_threshold = values["confidence_threshold"]

        tp_samples = _get_geojson_samples(
            detailed_pr_calc_df=detailed_pr_calc_df,
            parameters=parameters,
            flag_column="true_positive_flag",
            suffix="_gt",
            label_key=label_key,
            label_value=label_value,
            confidence_threshold=confidence_threshold,
        )
        misclassifications_fn_samples = _get_geojson_samples(
            detailed_pr_calc_df=detailed_pr_calc_df,
            parameters=parameters,
            flag_column="misclassification_false_negative_flag",
            suffix="_gt",
            label_key=label_key,
            label_value=label_value,
            confidence_threshold=confidence_threshold,
        )
        no_predictions_fn_samples = _get_geojson_samples(
            detailed_pr_calc_df=detailed_pr_calc_df,
            parameters=parameters,
            flag_column="no_predictions_false_negative_flag",
            suffix="_gt",
            label_key=label_key,
            label_value=label_value,
            confidence_threshold=confidence_threshold,
        )
        misclassifications_fp_samples = _get_geojson_samples(
            detailed_pr_calc_df=detailed_pr_calc_df,
            parameters=parameters,
            flag_column="misclassification_false_positive_flag",
            suffix="_pd",
            label_key=label_key,
            label_value=label_value,
            confidence_threshold=confidence_threshold,
        )
        hallucinations_fp_samples = _get_geojson_samples(
            detailed_pr_calc_df=detailed_pr_calc_df,
            parameters=parameters,
            flag_column="hallucination_false_positive_flag",
            suffix="_pd",
            label_key=label_key,
            label_value=label_value,
            confidence_threshold=confidence_threshold,
        )

        detailed_pr_curves[label_key][label_value][confidence_threshold] = {
            "tp": {
                "total": values["true_positives"],
                "observations": {
                    "all": {
                        "count": values["true_positives"],
                        "examples": tp_samples,
                    }
                },
            },
            "fn": {
                "total": values["misclassification_false_negatives"]
                + values["no_predictions_false_negatives"],
                "observations": {
                    "misclassifications": {
                        "count": values["misclassification_false_negatives"],
                        "examples": misclassifications_fn_samples,
                    },
                    "no_predictions": {
                        "count": values["no_predictions_false_negatives"],
                        "examples": no_predictions_fn_samples,
                    },
                },
            },
            "fp": {
                "total": values["misclassification_false_positives"]
                + values["hallucinations_false_positives"],
                "observations": {
                    "misclassifications": {
                        "count": values["misclassification_false_positives"],
                        "examples": misclassifications_fp_samples,
                    },
                    "hallucinations": {
                        "count": values["hallucinations_false_positives"],
                        "examples": hallucinations_fp_samples,
                    },
                },
            },
        }

    detailed_pr_metrics = [
        schemas.DetailedPrecisionRecallCurve(
            label_key=key,
            value=dict(value),
            pr_curve_iou_threshold=parameters.pr_curve_iou_threshold,
        )
        for key, value in detailed_pr_curves.items()
    ]

    return detailed_pr_metrics


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
    | schemas.DetailedPrecisionRecallCurve
]:
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

    if (
        parameters.pr_curve_iou_threshold <= 0
        or parameters.pr_curve_iou_threshold > 1.0
    ):
        raise ValueError(
            "IOU thresholds should exist in the range 0 < threshold <= 1."
        )

    labels = core.fetch_union_of_labels(
        db=db,
        rhs=prediction_filter,
        lhs=groundtruth_filter,
    )

    grouper_mappings = create_grouper_mappings(
        db=db,
        labels=labels,
        label_map=parameters.label_map,
        evaluation_type=enums.TaskType.OBJECT_DETECTION,
    )

    gt_df, pd_df = _get_groundtruth_and_prediction_df(
        db=db,
        annotation_type=target_type,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        grouper_mappings=grouper_mappings,
    )

    joint_df = _get_joint_df(
        gt_df=gt_df,
        pd_df=pd_df,
        grouper_mappings=grouper_mappings,
    )

    # store solo groundtruths and predictions such that we can add them back after we calculate IOU
    predictions_missing_groundtruths = joint_df[
        joint_df["id_gt"].isnull()
    ].assign(iou_=0)
    groundtruths_missing_predictions = joint_df[
        joint_df["id_pd"].isnull()
    ].assign(iou_=0)

    joint_df = _calculate_iou(joint_df=joint_df, annotation_type=target_type)

    # filter out null groundtruths and sort by score and iou so that idxmax returns the best row for each prediction
    joint_df = joint_df[~joint_df["id_gt"].isnull()].sort_values(
        by=["score", "iou_"], ascending=[False, False]
    )

    # get the best prediction (in terms of score and iou) for each groundtruth
    prediction_has_best_score = joint_df.groupby(["id_pd"])["score"].idxmax()

    joint_df = joint_df.loc[prediction_has_best_score]

    # add back missing predictions and groundtruths
    joint_df = pandas.concat(
        [
            joint_df,
            predictions_missing_groundtruths,
            groundtruths_missing_predictions,
        ],
        axis=0,
    )

    # add iou_threshold to the dataframe and sort
    calculation_df = pandas.concat(
        [
            joint_df.assign(iou_threshold=threshold)
            for threshold in parameters.iou_thresholds_to_compute
        ],
        ignore_index=True,
    ).sort_values(
        by=["label_id_grouper", "iou_threshold", "score", "iou_"],
        ascending=False,
    )

    calculation_df = _calculate_grouper_id_level_metrics(
        calculation_df=calculation_df, parameters=parameters
    )

    ap_metrics = _calculate_ap_metrics(
        calculation_df=calculation_df,
        grouper_mappings=grouper_mappings,
        parameters=parameters,
    )

    ar_metrics = _calculate_ar_metrics(
        calculation_df=calculation_df,
        grouper_mappings=grouper_mappings,
        parameters=parameters,
    )

    pr_metrics = _calculate_pr_metrics(
        joint_df=joint_df,
        grouper_mappings=grouper_mappings,
        parameters=parameters,
    )

    detailed_pr_metrics = _calculate_detailed_pr_metrics(
        gt_df=gt_df,
        pd_df=pd_df,
        grouper_mappings=grouper_mappings,
        parameters=parameters,
        annotation_type=target_type,
    )

    return ar_metrics + ap_metrics + pr_metrics + detailed_pr_metrics


def _compute_detection_metrics_with_detailed_precision_recall_curve(
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
    | schemas.DetailedPrecisionRecallCurve
]:
    """
    Compute detection metrics via the heaviest possible calculation set. This version of _compute_detection_metrics does IOU calculations for every groundtruth-prediction pair that shares a common grouper key, which is necessary for calculating the DetailedPrecisionRecallCurve metric.

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
    List[schemas.APMetric | schemas.ARMetric | schemas.APMetricAveragedOverIOUs | schemas.mAPMetric | schemas.mARMetric | schemas.mAPMetricAveragedOverIOUs | schemas.PrecisionRecallCurve | schemas.DetailedPrecisionRecallCurve]
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
        db=db,
        labels=labels,
        label_map=parameters.label_map,
        evaluation_type=enums.TaskType.OBJECT_DETECTION,
    )

    gt = generate_query(
        models.Dataset.name.label("dataset_name"),
        models.GroundTruth.id.label("id"),
        models.GroundTruth.annotation_id.label("annotation_id"),
        models.Annotation.datum_id.label("datum_id"),
        models.Datum.uid.label("datum_uid"),
        case(
            grouper_mappings["label_id_to_grouper_id_mapping"],
            value=models.GroundTruth.label_id,
        ).label("label_id_grouper"),
        case(
            grouper_mappings["label_id_to_grouper_key_mapping"],
            value=models.GroundTruth.label_id,
        ).label("label_key_grouper"),
        _annotation_type_to_geojson(target_type, models.Annotation).label(
            "geojson"
        ),
        db=db,
        filters=groundtruth_filter,
        label_source=models.GroundTruth,
    ).subquery("groundtruths")

    pd = generate_query(
        models.Dataset.name.label("dataset_name"),
        models.Prediction.id.label("id"),
        models.Prediction.annotation_id.label("annotation_id"),
        models.Prediction.score.label("score"),
        models.Annotation.datum_id.label("datum_id"),
        models.Datum.uid.label("datum_uid"),
        case(
            grouper_mappings["label_id_to_grouper_id_mapping"],
            value=models.Prediction.label_id,
        ).label("label_id_grouper"),
        case(
            grouper_mappings["label_id_to_grouper_key_mapping"],
            value=models.Prediction.label_id,
        ).label("label_key_grouper"),
        _annotation_type_to_geojson(target_type, models.Annotation).label(
            "geojson"
        ),
        db=db,
        filters=prediction_filter,
        label_source=models.Prediction,
    ).subquery("predictions")

    joint = (
        select(
            func.coalesce(pd.c.dataset_name, gt.c.dataset_name).label(
                "dataset_name"
            ),
            gt.c.datum_uid.label("gt_datum_uid"),
            pd.c.datum_uid.label("pd_datum_uid"),
            gt.c.geojson.label("gt_geojson"),
            gt.c.id.label("gt_id"),
            pd.c.id.label("pd_id"),
            gt.c.label_id_grouper.label("gt_label_id_grouper"),
            pd.c.label_id_grouper.label("pd_label_id_grouper"),
            gt.c.label_key_grouper.label("gt_label_key_grouper"),
            pd.c.label_key_grouper.label("pd_label_key_grouper"),
            gt.c.annotation_id.label("gt_ann_id"),
            pd.c.annotation_id.label("pd_ann_id"),
            pd.c.score.label("score"),
        )
        .select_from(pd)
        .outerjoin(
            gt,
            and_(
                pd.c.datum_id == gt.c.datum_id,
                pd.c.label_key_grouper == gt.c.label_key_grouper,
            ),
        )
        .subquery()
    )

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

    ious = (
        select(
            joint.c.dataset_name,
            joint.c.pd_datum_uid,
            joint.c.gt_datum_uid,
            joint.c.gt_id.label("gt_id"),
            joint.c.pd_id.label("pd_id"),
            joint.c.gt_label_id_grouper,
            joint.c.pd_label_id_grouper,
            joint.c.score.label("score"),
            func.coalesce(iou_computation, 0).label("iou"),
            joint.c.gt_geojson,
            (joint.c.gt_label_id_grouper == joint.c.pd_label_id_grouper).label(
                "is_match"
            ),
        )
        .select_from(joint)
        .join(gt_annotation, gt_annotation.id == joint.c.gt_ann_id)
        .join(pd_annotation, pd_annotation.id == joint.c.pd_ann_id)
        .subquery()
    )

    ordered_ious = (
        db.query(ious)
        .order_by(
            ious.c.is_match.desc(), -ious.c.score, -ious.c.iou, ious.c.gt_id
        )
        .all()
    )

    pd_set = set()
    matched_pd_set = set()
    sorted_ranked_pairs = defaultdict(list)
    matched_sorted_ranked_pairs = defaultdict(list)

    for row in ordered_ious:
        (
            dataset_name,
            pd_datum_uid,
            gt_datum_uid,
            gt_id,
            pd_id,
            gt_label_id_grouper,
            pd_label_id_grouper,
            score,
            iou,
            gt_geojson,
            is_match,
        ) = row

        if pd_id not in pd_set:
            # sorted_ranked_pairs will include all groundtruth-prediction pairs that meet filter criteria
            pd_set.add(pd_id)
            sorted_ranked_pairs[gt_label_id_grouper].append(
                RankedPair(
                    dataset_name=dataset_name,
                    pd_datum_uid=pd_datum_uid,
                    gt_datum_uid=gt_datum_uid,
                    gt_geojson=gt_geojson,
                    gt_id=gt_id,
                    pd_id=pd_id,
                    score=score,
                    iou=iou,
                    is_match=is_match,
                )
            )
            sorted_ranked_pairs[pd_label_id_grouper].append(
                RankedPair(
                    dataset_name=dataset_name,
                    pd_datum_uid=pd_datum_uid,
                    gt_datum_uid=gt_datum_uid,
                    gt_geojson=gt_geojson,
                    gt_id=gt_id,
                    pd_id=pd_id,
                    score=score,
                    iou=iou,
                    is_match=is_match,
                )
            )

        if pd_id not in matched_pd_set and is_match:
            # matched_sorted_ranked_pairs only contains matched groundtruth-prediction pairs
            matched_pd_set.add(pd_id)
            matched_sorted_ranked_pairs[gt_label_id_grouper].append(
                RankedPair(
                    dataset_name=dataset_name,
                    pd_datum_uid=pd_datum_uid,
                    gt_datum_uid=gt_datum_uid,
                    gt_geojson=gt_geojson,
                    gt_id=gt_id,
                    pd_id=pd_id,
                    score=score,
                    iou=iou,
                    is_match=True,
                )
            )

    # get predictions that didn't make it into matched_sorted_ranked_pairs
    # because they didn't have a corresponding groundtruth to pair with
    predictions_not_in_sorted_ranked_pairs = (
        db.query(
            pd.c.id,
            pd.c.score,
            pd.c.dataset_name,
            pd.c.datum_uid,
            pd.c.label_id_grouper,
        )
        .filter(pd.c.id.notin_(matched_pd_set))
        .all()
    )

    for (
        pd_id,
        score,
        dataset_name,
        pd_datum_uid,
        grouper_id,
    ) in predictions_not_in_sorted_ranked_pairs:
        if pd_id not in pd_set:
            # add to sorted_ranked_pairs in sorted order
            bisect.insort(  # type: ignore - bisect type issue
                sorted_ranked_pairs[grouper_id],
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
        bisect.insort(
            matched_sorted_ranked_pairs[grouper_id],
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

    # Get all groundtruths per grouper_id
    groundtruths_per_grouper = defaultdict(list)
    predictions_per_grouper = defaultdict(list)
    number_of_groundtruths_per_grouper = defaultdict(int)

    groundtruths = db.query(
        gt.c.id,
        gt.c.label_id_grouper,
        gt.c.datum_uid,
        gt.c.dataset_name,
        gt.c.geojson,
    )

    predictions = db.query(
        pd.c.id,
        pd.c.label_id_grouper,
        pd.c.datum_uid,
        pd.c.dataset_name,
        pd.c.geojson,
    )

    for gt_id, grouper_id, datum_uid, dset_name, gt_geojson in groundtruths:
        # we're ok with adding duplicates here since they indicate multiple groundtruths for a given dataset/datum_id
        groundtruths_per_grouper[grouper_id].append(
            (dset_name, datum_uid, gt_id, gt_geojson)
        )
        number_of_groundtruths_per_grouper[grouper_id] += 1

    for pd_id, grouper_id, datum_uid, dset_name, pd_geojson in predictions:
        predictions_per_grouper[grouper_id].append(
            (dset_name, datum_uid, pd_id, pd_geojson)
        )
    if parameters.metrics_to_return is None:
        raise RuntimeError("Metrics to return should always contains values.")

    pr_curves = _compute_detailed_curves(
        sorted_ranked_pairs=sorted_ranked_pairs,
        grouper_mappings=grouper_mappings,
        groundtruths_per_grouper=groundtruths_per_grouper,
        predictions_per_grouper=predictions_per_grouper,
        pr_curve_iou_threshold=parameters.pr_curve_iou_threshold,
        pr_curve_max_examples=(
            parameters.pr_curve_max_examples
            if parameters.pr_curve_max_examples
            else 1
        ),
    )

    ap_ar_output = []

    ap_metrics, ar_metrics = _calculate_ap_and_ar(
        sorted_ranked_pairs=matched_sorted_ranked_pairs,
        number_of_groundtruths_per_grouper=number_of_groundtruths_per_grouper,
        iou_thresholds=parameters.iou_thresholds_to_compute,
        grouper_mappings=grouper_mappings,
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


def _mean_ignoring_negative_one(series: pandas.Series) -> float:
    filtered = series[series != -1]
    return filtered.mean() if not filtered.empty else -1.0


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
        if isinstance(ap, schemas.APMetric):
            iou = ap.iou
        else:
            iou = frozenset(ap.ious)
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
