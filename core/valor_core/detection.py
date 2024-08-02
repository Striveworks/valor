import heapq
import io
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
from valor_core import enums, schemas, utilities

# TODO move this out
LabelMapType = Dict[schemas.Label, schemas.Label]


def _convert_wkt_to_array(wkt_data) -> np.ndarray:
    """Convert a WKT string to an array of coordinates."""
    if isinstance(wkt_data, str):
        coordinates = json.loads(wkt_data)["coordinates"][0]
    elif isinstance(wkt_data, dict) and ("coordinates" in wkt_data.keys()):
        coordinates = wkt_data["coordinates"][0]
    else:
        raise ValueError("Unknown format for wkt_data.")

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


def _get_joint_df(
    groundtruth_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    grouper_mappings: dict,
) -> pd.DataFrame:
    """Create a joint dataframe of groundtruths and predictions for calculating AR/AP metrics."""

    # add the number of groundtruth observations per groupere
    number_of_groundtruths_per_grouper_df = (
        groundtruth_df.groupby("label_id_grouper", as_index=False)["id"]
        .nunique()
        .rename({"id": "gts_per_grouper"}, axis=1)
    )

    joint_df = pd.merge(
        groundtruth_df,
        prediction_df,
        on=["datum_id", "label_id_grouper"],
        how="outer",
        suffixes=("_gt", "_pd"),
    )

    joint_df = pd.merge(
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


def _calculate_iou(joint_df: pd.DataFrame):
    """Calculate the IOUs between predictions and groundtruths in a joint dataframe"""
    if joint_df["raster_pd"].notnull().any():

        # filter out rows with null rasters
        joint_df = joint_df.loc[
            ~joint_df["raster_pd"].isnull() & ~joint_df["raster_gt"].isnull(),
            :,
        ]

        # convert the raster into a numpy array
        # joint_df.loc[:, ["raster_pd", "raster_gt"]] = (
        #     joint_df[["raster_pd", "raster_gt"]]
        #     .map(io.BytesIO)  # type: ignore pd typing error
        #     .map(Image.open)
        #     .map(np.array)
        # )

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
            .map(_convert_wkt_to_array)
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
    calculation_df: pd.DataFrame, parameters: schemas.EvaluationParameters
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


def _mean_ignoring_negative_one(series: pd.Series) -> float:
    filtered = series[series != -1]
    return filtered.mean() if not filtered.empty else -1.0


def _calculate_ap_metrics(
    calculation_df: pd.DataFrame,
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
            lambda x: pd.Series(
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
        for row in map_metrics_df.to_dict(orient="records")  # type: ignore - pd typing issue
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
        for row in map_over_ious_df.to_dict(orient="records")  # type: ignore - pd typing issue
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
    calculation_df: pd.DataFrame,
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
    joint_df: pd.DataFrame,
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
    pr_calculation_df = pd.concat(
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


def _add_samples_to_dataframe(
    detailed_pr_curve_counts_df: pd.DataFrame,
    detailed_pr_calc_df: pd.DataFrame,
    max_examples: int,
    flag_column: str,
):
    """Efficienctly gather samples for a given flag."""
    # TODO merge on dataset before this?

    sample_df = pd.concat(
        [
            detailed_pr_calc_df[detailed_pr_calc_df[flag_column]]
            .groupby(
                [
                    "grouper_key",
                    "grouper_value_gt",
                    "confidence_threshold",
                ],
                as_index=False,
            )[["dataset_name_gt", "datum_uid_gt", "geojson_gt"]]
            .agg(lambda x: tuple(x.head(max_examples)))
            .rename(
                columns={
                    "dataset_name_gt": "dataset_name",
                    "datum_uid_gt": "datum_uid",
                    "grouper_value_gt": "grouper_value",
                    "geojson_gt": "geojson",
                }
            ),
            detailed_pr_calc_df[detailed_pr_calc_df[flag_column]]
            .groupby(
                [
                    "grouper_key",
                    "grouper_value_pd",
                    "confidence_threshold",
                ],
                as_index=False,
            )[["dataset_name_pd", "datum_uid_pd", "geojson_pd"]]
            .agg(lambda x: (tuple(x.head(max_examples))))
            .rename(
                columns={
                    "dataset_name_pd": "dataset_name",
                    "datum_uid_pd": "datum_uid",
                    "grouper_value_pd": "grouper_value",
                    "geojson_pd": "geojson",
                }
            ),
        ],
        axis=0,
    )

    sample_df["geojson"] = sample_df["geojson"].astype(str)
    sample_df.drop_duplicates(inplace=True)

    if not sample_df.empty:
        sample_df[f"{flag_column}_samples"] = sample_df.apply(
            lambda row: set(zip(*row[["dataset_name", "datum_uid", "geojson"]])),  # type: ignore - pd typing error
            axis=1,
        )

        detailed_pr_curve_counts_df = detailed_pr_curve_counts_df.merge(
            sample_df[
                [
                    "grouper_key",
                    "grouper_value",
                    "confidence_threshold",
                    f"{flag_column}_samples",
                ]
            ],
            on=["grouper_key", "grouper_value", "confidence_threshold"],
            how="outer",
        )
        detailed_pr_curve_counts_df[
            f"{flag_column}_samples"
        ] = detailed_pr_curve_counts_df[f"{flag_column}_samples"].map(
            lambda x: list(x) if isinstance(x, set) else list()
        )
    else:
        detailed_pr_curve_counts_df[f"{flag_column}_samples"] = [
            list() for _ in range(len(detailed_pr_curve_counts_df))
        ]

    return detailed_pr_curve_counts_df


def _calculate_detailed_pr_metrics(
    groundtruth_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    grouper_mappings: dict,
    parameters: schemas.EvaluationParameters,
) -> list[schemas.DetailedPrecisionRecallCurve]:
    """Calculates all DetailedPrecisionRecallCurve metrics."""

    if not (
        parameters.metrics_to_return
        and enums.MetricType.DetailedPrecisionRecallCurve
        in parameters.metrics_to_return
    ):
        return []

    detailed_pr_joint_df = pd.merge(
        groundtruth_df,
        prediction_df,
        on=["datum_id", "grouper_key"],
        how="outer",
        suffixes=("_gt", "_pd"),
    ).assign(
        is_label_match=lambda chain_df: (
            chain_df["label_id_grouper_pd"] == chain_df["label_id_grouper_gt"]
        )
    )

    detailed_pr_joint_df = _calculate_iou(joint_df=detailed_pr_joint_df)

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
    detailed_pr_joint_df["grouper_value_gt"] = detailed_pr_joint_df[
        "label_gt"
    ].map(lambda x: x.value if isinstance(x, schemas.Label) else None)
    detailed_pr_joint_df["grouper_value_pd"] = detailed_pr_joint_df[
        "label_pd"
    ].map(lambda x: x.value if isinstance(x, schemas.Label) else None)

    # add confidence_threshold to the dataframe and sort
    detailed_pr_calc_df = pd.concat(
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
        detailed_pr_calc_df[
            detailed_pr_calc_df["true_positive_flag"]
            | detailed_pr_calc_df["misclassification_false_positive_flag"]
        ]
        .groupby(["confidence_threshold"], as_index=False)["id_pd"]
        .unique()
    )

    if not predictions_associated_with_tps_or_misclassification_fps.empty:
        predictions_associated_with_tps_or_misclassification_fps.columns = [
            "confidence_threshold",
            "predictions_associated_with_tps_or_misclassification_fps",
        ]
        detailed_pr_calc_df = detailed_pr_calc_df.merge(
            predictions_associated_with_tps_or_misclassification_fps,
            on=["confidence_threshold"],
            how="left",
        )
        misclassification_id_pds = detailed_pr_calc_df[
            "predictions_associated_with_tps_or_misclassification_fps"
        ].map(lambda x: set(x) if isinstance(x, np.ndarray) else [])
        misclassification_id_pds = np.array(
            [
                id_pd in misclassification_id_pds[i]
                for i, id_pd in enumerate(detailed_pr_calc_df["id_pd"].values)
            ]
        )
        detailed_pr_calc_df.loc[
            (misclassification_id_pds)
            & (detailed_pr_calc_df["hallucination_false_positive_flag"]),
            "hallucination_false_positive_flag",
        ] = False

    # next, we flag false negatives by declaring any groundtruth that isn't associated with a true positive to be a false negative
    groundtruths_associated_with_true_positives = (
        detailed_pr_calc_df[detailed_pr_calc_df["true_positive_flag"]]
        .groupby(["confidence_threshold"], as_index=False)["id_gt"]
        .unique()
    )

    if not groundtruths_associated_with_true_positives.empty:
        groundtruths_associated_with_true_positives.columns = [
            "confidence_threshold",
            "groundtruths_associated_with_true_positives",
        ]
        detailed_pr_calc_df = detailed_pr_calc_df.merge(
            groundtruths_associated_with_true_positives,
            on=["confidence_threshold"],
            how="left",
        )
        true_positive_sets = detailed_pr_calc_df[
            "groundtruths_associated_with_true_positives"
        ].map(lambda x: set(x) if isinstance(x, np.ndarray) else set())

        detailed_pr_calc_df["false_negative_flag"] = np.array(
            [
                id_gt not in true_positive_sets[i]
                for i, id_gt in enumerate(detailed_pr_calc_df["id_gt"].values)
            ]
        )

    else:
        detailed_pr_calc_df["false_negative_flag"] = False

    # it's a misclassification if there is a corresponding misclassification false positive
    detailed_pr_calc_df["misclassification_false_negative_flag"] = (
        detailed_pr_calc_df["misclassification_false_positive_flag"]
        & detailed_pr_calc_df["false_negative_flag"]
    )

    # assign all id_gts that aren't misclassifications but are false negatives to be no_predictions
    groundtruths_associated_with_misclassification_false_negatives = (
        detailed_pr_calc_df[
            detailed_pr_calc_df["misclassification_false_negative_flag"]
        ]
        .groupby(["confidence_threshold"], as_index=False)["id_gt"]
        .unique()
    )

    if (
        not groundtruths_associated_with_misclassification_false_negatives.empty
    ):
        groundtruths_associated_with_misclassification_false_negatives.columns = [
            "confidence_threshold",
            "groundtruths_associated_with_misclassification_false_negatives",
        ]
        detailed_pr_calc_df = detailed_pr_calc_df.merge(
            groundtruths_associated_with_misclassification_false_negatives,
            on=["confidence_threshold"],
            how="left",
        )
        misclassification_sets = (
            detailed_pr_calc_df[
                "groundtruths_associated_with_misclassification_false_negatives"
            ]
            .map(lambda x: set(x) if isinstance(x, np.ndarray) else set())
            .values
        )
        detailed_pr_calc_df["no_predictions_false_negative_flag"] = (
            np.array(
                [
                    id_gt not in misclassification_sets[i]
                    for i, id_gt in enumerate(
                        detailed_pr_calc_df["id_gt"].values
                    )
                ]
            )
            & detailed_pr_calc_df["false_negative_flag"]
        )
    else:
        detailed_pr_calc_df[
            "no_predictions_false_negative_flag"
        ] = detailed_pr_calc_df["false_negative_flag"]

    # next, we sum up the occurences of each classification and merge them together into one dataframe
    true_positives = (
        detailed_pr_calc_df[detailed_pr_calc_df["true_positive_flag"]]
        .groupby(["grouper_key", "grouper_value_pd", "confidence_threshold"])[
            "id_pd"
        ]
        .nunique()
    )
    true_positives.name = "true_positives"

    hallucination_false_positives = (
        detailed_pr_calc_df[
            detailed_pr_calc_df["hallucination_false_positive_flag"]
        ]
        .groupby(["grouper_key", "grouper_value_pd", "confidence_threshold"])[
            "id_pd"
        ]
        .nunique()
    )
    hallucination_false_positives.name = "hallucinations_false_positives"

    misclassification_false_positives = (
        detailed_pr_calc_df[
            detailed_pr_calc_df["misclassification_false_positive_flag"]
        ]
        .groupby(["grouper_key", "grouper_value_pd", "confidence_threshold"])[
            "id_pd"
        ]
        .nunique()
    )
    misclassification_false_positives.name = (
        "misclassification_false_positives"
    )

    misclassification_false_negatives = (
        detailed_pr_calc_df[
            detailed_pr_calc_df["misclassification_false_negative_flag"]
        ]
        .groupby(["grouper_key", "grouper_value_gt", "confidence_threshold"])[
            "id_gt"
        ]
        .nunique()
    )
    misclassification_false_negatives.name = (
        "misclassification_false_negatives"
    )

    no_predictions_false_negatives = (
        detailed_pr_calc_df[
            detailed_pr_calc_df["no_predictions_false_negative_flag"]
        ]
        .groupby(["grouper_key", "grouper_value_gt", "confidence_threshold"])[
            "id_gt"
        ]
        .nunique()
    )
    no_predictions_false_negatives.name = "no_predictions_false_negatives"

    # combine these outputs
    detailed_pr_curve_counts_df = (
        pd.concat(
            [
                detailed_pr_calc_df.loc[
                    ~detailed_pr_calc_df["grouper_value_pd"].isnull(),
                    [
                        "grouper_key",
                        "grouper_value_pd",
                        "confidence_threshold",
                    ],
                ].rename(columns={"grouper_value_pd": "grouper_value"}),
                detailed_pr_calc_df.loc[
                    ~detailed_pr_calc_df["grouper_value_gt"].isnull(),
                    [
                        "grouper_key",
                        "grouper_value_gt",
                        "confidence_threshold",
                    ],
                ].rename(columns={"grouper_value_gt": "grouper_value"}),
            ],
            axis=0,
        )
        .drop_duplicates()
        .merge(
            true_positives,
            left_on=[
                "grouper_key",
                "grouper_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            hallucination_false_positives,
            left_on=[
                "grouper_key",
                "grouper_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            misclassification_false_positives,
            left_on=[
                "grouper_key",
                "grouper_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            misclassification_false_negatives,
            left_on=[
                "grouper_key",
                "grouper_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            no_predictions_false_negatives,
            left_on=[
                "grouper_key",
                "grouper_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
    )

    # we're doing an outer join, so any nulls should be zeroes
    detailed_pr_curve_counts_df.fillna(0, inplace=True)

    # add samples to the dataframe for DetailedPrecisionRecallCurves
    for flag in [
        "true_positive_flag",
        "misclassification_false_negative_flag",
        "no_predictions_false_negative_flag",
        "misclassification_false_positive_flag",
        "hallucination_false_positive_flag",
    ]:
        detailed_pr_curve_counts_df = _add_samples_to_dataframe(
            detailed_pr_calc_df=detailed_pr_calc_df,
            detailed_pr_curve_counts_df=detailed_pr_curve_counts_df,
            max_examples=parameters.pr_curve_max_examples,
            flag_column=flag,
        )

    # create output
    detailed_pr_curves = defaultdict(lambda: defaultdict(dict))
    for _, row in detailed_pr_curve_counts_df.iterrows():
        grouper_key = row["grouper_key"]
        grouper_value = row["grouper_value"]
        confidence_threshold = row["confidence_threshold"]

        detailed_pr_curves[grouper_key][grouper_value][
            confidence_threshold
        ] = {
            "tp": {
                "total": row["true_positives"],
                "observations": {
                    "all": {
                        "count": row["true_positives"],
                        "examples": row["true_positive_flag_samples"],
                    }
                },
            },
            "fn": {
                "total": row["misclassification_false_negatives"]
                + row["no_predictions_false_negatives"],
                "observations": {
                    "misclassifications": {
                        "count": row["misclassification_false_negatives"],
                        "examples": row[
                            "misclassification_false_negative_flag_samples"
                        ],
                    },
                    "no_predictions": {
                        "count": row["no_predictions_false_negatives"],
                        "examples": row[
                            "no_predictions_false_negative_flag_samples"
                        ],
                    },
                },
            },
            "fp": {
                "total": row["misclassification_false_positives"]
                + row["hallucinations_false_positives"],
                "observations": {
                    "misclassifications": {
                        "count": row["misclassification_false_positives"],
                        "examples": row[
                            "misclassification_false_positive_flag_samples"
                        ],
                    },
                    "hallucinations": {
                        "count": row["hallucinations_false_positives"],
                        "examples": row[
                            "hallucination_false_positive_flag_samples"
                        ],
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


def _create_detection_grouper_mappings(
    label_map: Optional[LabelMapType],
    labels: list,
) -> Dict[str, dict]:
    """Create grouper mappings for use when evaluating classifications."""
    mapping_dict = dict()
    if label_map:
        for label, grouper in label_map.items():
            mapping_dict[(label.key, label.value)] = (
                grouper.key,
                grouper.value,
            )

    # define mappers to connect groupers with labels
    grouper_id_to_grouper_label_mapping = dict()
    label_tuple_to_grouper_id_mapping = dict()
    label_to_grouper_key_mapping = dict()

    for label in labels:
        # the grouper should equal the (label.key, label.value) if it wasn't mapped by the user
        grouper_key, grouper_value = mapping_dict.get(label, label)
        grouper_id = hash((grouper_key, grouper_value))

        grouper_id_to_grouper_label_mapping[grouper_id] = schemas.Label(
            key=grouper_key, value=grouper_value
        )
        label_tuple_to_grouper_id_mapping[label] = grouper_id
        label_to_grouper_key_mapping[label] = grouper_key

    return {
        "grouper_id_to_grouper_label_mapping": grouper_id_to_grouper_label_mapping,
        "label_tuple_to_grouper_id_mapping": label_tuple_to_grouper_id_mapping,
        "label_to_grouper_key_mapping": label_to_grouper_key_mapping,
    }


def _compute_detection_metrics(
    groundtruth_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    parameters: schemas.EvaluationParameters,
    unique_labels: list,
) -> List[dict]:
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

    grouper_mappings = _create_detection_grouper_mappings(
        label_map=parameters.label_map,
        labels=unique_labels,
    )

    metrics_to_output = []

    # assign a unique ID per label to each dataframe
    groundtruth_df["label_id_grouper"] = groundtruth_df.apply(
        lambda row: grouper_mappings["label_tuple_to_grouper_id_mapping"].get(
            (row["label_key"], row["label_value"])
        ),  # type: ignore - pandas typing error
        axis=1,
    )

    prediction_df["label_id_grouper"] = prediction_df.apply(
        lambda row: grouper_mappings["label_tuple_to_grouper_id_mapping"].get(
            (row["label_key"], row["label_value"])
        ),  # type: ignore - pandas typing error
        axis=1,
    )

    # assign grouper_key to dataframes
    groundtruth_df["grouper_key"] = groundtruth_df.apply(
        lambda row: grouper_mappings["label_to_grouper_key_mapping"].get(
            (row["label_key"], row["label_value"])
        ),  # type: ignore - pandas typing error
        axis=1,
    )

    prediction_df["grouper_key"] = prediction_df.apply(
        lambda row: grouper_mappings["label_to_grouper_key_mapping"].get(
            (row["label_key"], row["label_value"])
        ),  # type: ignore - pandas typing error
        axis=1,
    )

    joint_df = _get_joint_df(
        groundtruth_df=groundtruth_df,
        prediction_df=prediction_df,
        grouper_mappings=grouper_mappings,
    )

    # store solo groundtruths and predictions such that we can add them back after we calculate IOU
    predictions_missing_groundtruths = joint_df[
        joint_df["id_gt"].isnull()
    ].assign(iou_=0)
    groundtruths_missing_predictions = joint_df[
        joint_df["id_pd"].isnull()
    ].assign(iou_=0)

    joint_df = _calculate_iou(joint_df=joint_df)

    # filter out null groundtruths and sort by score and iou so that idxmax returns the best row for each prediction
    joint_df = joint_df[~joint_df["id_gt"].isnull()].sort_values(
        by=["score", "iou_"], ascending=[False, False]
    )

    # get the best prediction (in terms of score and iou) for each groundtruth
    prediction_has_best_score = joint_df.groupby(["id_pd"])["score"].idxmax()

    joint_df = joint_df.loc[prediction_has_best_score]

    # add back missing predictions and groundtruths
    joint_df = pd.concat(
        [
            joint_df,
            predictions_missing_groundtruths,
            groundtruths_missing_predictions,
        ],
        axis=0,
    )

    # add iou_threshold to the dataframe and sort
    assert parameters.iou_thresholds_to_compute
    calculation_df = pd.concat(
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

    metrics_to_output += _calculate_ap_metrics(
        calculation_df=calculation_df,
        grouper_mappings=grouper_mappings,
        parameters=parameters,
    )

    metrics_to_output += _calculate_ar_metrics(
        calculation_df=calculation_df,
        grouper_mappings=grouper_mappings,
        parameters=parameters,
    )

    metrics_to_output += _calculate_pr_metrics(
        joint_df=joint_df,
        grouper_mappings=grouper_mappings,
        parameters=parameters,
    )

    metrics_to_output += _calculate_detailed_pr_metrics(
        groundtruth_df=groundtruth_df,
        prediction_df=prediction_df,
        grouper_mappings=grouper_mappings,
        parameters=parameters,
    )

    # convert objects to dictionaries and only return what was asked for
    metrics_to_output = [
        m.to_dict()
        for m in metrics_to_output
        if m.to_dict()["type"] in parameters.metrics_to_return
    ]

    return metrics_to_output


def evaluate_detection(
    groundtruths: Union[pd.DataFrame, List[schemas.GroundTruth]],
    predictions: Union[pd.DataFrame, List[schemas.Prediction]],
    parameters: schemas.EvaluationParameters = schemas.EvaluationParameters(),
) -> schemas.Evaluation:
    """
    Create object detection metrics.
    # TODO
    """
    start_time = time.time()

    # TODO move this into some shared function
    parameters = utilities.validate_parameters(
        parameters, task_type=enums.TaskType.OBJECT_DETECTION
    )

    prediction_df = utilities.validate_prediction_dataframe(
        predictions, task_type=enums.TaskType.OBJECT_DETECTION
    )
    groundtruth_df = utilities.validate_groundtruth_dataframe(
        groundtruths, task_type=enums.TaskType.OBJECT_DETECTION
    )

    (
        missing_pred_labels,
        ignored_pred_labels,
    ) = utilities.get_disjoint_labels(
        groundtruth_df=groundtruth_df,
        prediction_df=prediction_df,
        label_map=parameters.label_map,
    )

    unique_labels = list(
        set(zip(groundtruth_df["label_key"], groundtruth_df["label_value"]))
        | set(zip(prediction_df["label_key"], prediction_df["label_value"]))
    )
    unique_datums_cnt = len(
        set(groundtruth_df["datum_uid"]) | set(prediction_df["datum_uid"])
    )
    unique_annotations_cnt = len(
        set(groundtruth_df["annotation_id"])
        | set(prediction_df["annotation_id"])
    )

    # handle type errors
    assert parameters.metrics_to_return

    # TODO just pass parameters to the classification equivalent of this function
    metrics = _compute_detection_metrics(
        groundtruth_df=groundtruth_df,
        prediction_df=prediction_df,
        parameters=parameters,
        unique_labels=unique_labels,
    )

    return schemas.Evaluation(
        parameters=parameters,
        metrics=metrics,
        confusion_matrices=[],
        ignored_pred_labels=ignored_pred_labels,
        missing_pred_labels=missing_pred_labels,
        meta={
            "labels": len(unique_labels),
            "datums": unique_datums_cnt,
            "annotations": unique_annotations_cnt,
            "duration": time.time() - start_time,
        },
    )
