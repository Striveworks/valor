import heapq
import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from valor_core import enums, geometry, metrics, schemas, utilities

pd.set_option("display.max_columns", None)


def _get_joint_df(
    groundtruth_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a joint dataframe of groundtruths and predictions for calculating AR/AP metrics."""

    # add the number of groundtruth observations per groupere
    number_of_groundtruths_per_grouper_df = (
        groundtruth_df.groupby(["label_id", "label"], as_index=False)["id"]
        .nunique()
        .rename({"id": "gts_per_grouper"}, axis=1)
    )

    joint_df = pd.merge(
        groundtruth_df,
        prediction_df,
        on=["datum_id", "label_id", "label"],
        how="outer",
        suffixes=("_gt", "_pd"),
    )

    joint_df = pd.merge(
        joint_df,
        number_of_groundtruths_per_grouper_df,
        on=["label_id", "label"],
        how="outer",
    )

    return joint_df


def _get_dtypes_in_series_of_arrays(series: pd.Series):
    """Get the data type inside of a 2D numpy array. Used to check if a np.array contains coordinates or a mask."""
    if not isinstance(series, pd.Series) or not all(
        series.apply(lambda x: x.ndim == 2)
    ):
        raise ValueError(
            "series must be a pandas Series filled with two-dimensional arrays."
        )

    unique_primitives = series.map(lambda x: x.dtype).unique()

    if len(unique_primitives) > 1:
        raise ValueError("series contains more than one type of primitive.")

    return unique_primitives[0]


def _check_if_series_contains_masks(series: pd.Series) -> bool:
    """Check if any element in a pandas.Series is a mask."""
    if series.empty:
        return False

    primitive = _get_dtypes_in_series_of_arrays(series=series)

    if np.issubdtype(primitive, np.bool_):
        return True

    return False


def _calculate_iou(
    joint_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the IOUs between predictions and groundtruths in a joint dataframe."""
    if _check_if_series_contains_masks(
        joint_df.loc[
            joint_df["converted_geometry_pd"].notnull(),
            "converted_geometry_pd",
        ]
    ):
        iou_calculation_df = (
            joint_df.assign(
                intersection=lambda chain_df: chain_df.apply(
                    lambda row: (
                        0
                        if row["converted_geometry_pd"] is None
                        or row["converted_geometry_gt"] is None
                        else np.logical_and(
                            row["converted_geometry_pd"],
                            row["converted_geometry_gt"],
                        ).sum()
                    ),
                    axis=1,
                )
            )
            .assign(
                union_=lambda chain_df: chain_df.apply(
                    lambda row: (
                        0
                        if row["converted_geometry_pd"] is None
                        or row["converted_geometry_gt"] is None
                        else np.sum(row["converted_geometry_gt"])
                        + np.sum(row["converted_geometry_pd"])
                        - row["intersection"]
                    ),
                    axis=1,
                )
            )
            .assign(
                iou_=lambda chain_df: chain_df["intersection"]
                / chain_df["union_"]
            )
        )

        joint_df = joint_df.join(iou_calculation_df["iou_"])

    else:
        iou_calculation_df = joint_df.loc[
            ~joint_df["converted_geometry_gt"].isnull()
            & ~joint_df["converted_geometry_pd"].isnull(),
            ["converted_geometry_gt", "converted_geometry_pd"],
        ].apply(
            lambda row: geometry.calculate_bbox_iou(
                row["converted_geometry_gt"], row["converted_geometry_pd"]
            ),
            axis=1,
        )

        if not iou_calculation_df.empty:
            iou_calculation_df = iou_calculation_df.rename("iou_")
            joint_df = joint_df.join(iou_calculation_df)
        else:
            joint_df["iou_"] = 0

    return joint_df


def _calculate_label_id_level_metrics(
    calculation_df: pd.DataFrame, recall_score_threshold: float
) -> pd.DataFrame:
    """Calculate the flags and metrics needed to compute AP, AR, and PR curves."""

    # create flags where predictions meet the score and IOU criteria
    calculation_df["recall_true_positive_flag"] = (
        calculation_df["iou_"] >= calculation_df["iou_threshold"]
    ) & (calculation_df["score"] >= recall_score_threshold)
    # only consider the highest scoring true positive as an actual true positive
    calculation_df["recall_true_positive_flag"] = calculation_df[
        "recall_true_positive_flag"
    ] & (
        ~calculation_df.groupby(
            ["label_id", "label", "iou_threshold", "id_gt"], as_index=False
        )["recall_true_positive_flag"].shift(1, fill_value=False)
    )

    calculation_df["precision_true_positive_flag"] = (
        calculation_df["iou_"] >= calculation_df["iou_threshold"]
    ) & (calculation_df["score"] > 0)
    calculation_df["precision_true_positive_flag"] = calculation_df[
        "precision_true_positive_flag"
    ] & (
        ~calculation_df.groupby(
            ["label_id", "iou_threshold", "id_gt"], as_index=False
        )["precision_true_positive_flag"].shift(1, fill_value=False)
    )

    calculation_df["recall_false_positive_flag"] = ~calculation_df[
        "recall_true_positive_flag"
    ] & (calculation_df["score"] >= recall_score_threshold)
    calculation_df["precision_false_positive_flag"] = ~calculation_df[
        "precision_true_positive_flag"
    ] & (calculation_df["score"] > 0)

    # calculate true and false positives
    calculation_df = (
        calculation_df.join(
            calculation_df.groupby(
                ["label_id", "label", "iou_threshold"], as_index=False
            )["recall_true_positive_flag"]
            .cumsum()
            .rename("rolling_recall_tp")
        )
        .join(
            calculation_df.groupby(
                ["label_id", "label", "iou_threshold"], as_index=False
            )["recall_false_positive_flag"]
            .cumsum()
            .rename("rolling_recall_fp")
        )
        .join(
            calculation_df.groupby(
                ["label_id", "label", "iou_threshold"], as_index=False
            )["precision_true_positive_flag"]
            .cumsum()
            .rename("rolling_precision_tp")
        )
        .join(
            calculation_df.groupby(
                ["label_id", "label", "iou_threshold"], as_index=False
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
    """Use the 101 point interpolation method (following torchmetrics)."""
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
        while (
            cutoff_idx < len(data)
            and data[cutoff_idx][1] < r
            and not math.isclose(data[cutoff_idx][1], r)
        ):
            cutoff_idx += 1
        while prec_heap and prec_heap[0][1] < cutoff_idx:
            heapq.heappop(prec_heap)
        if cutoff_idx >= len(data):
            continue
        ret -= prec_heap[0][0]

    return ret / 101


def _calculate_mean_ignoring_negative_one(series: pd.Series) -> float:
    """Calculate the mean of a series, ignoring any values that are -1."""
    filtered = series[series != -1]
    return filtered.mean() if not filtered.empty else -1.0


def _calculate_ap_metrics(
    calculation_df: pd.DataFrame,
    iou_thresholds_to_compute: List[float],
    iou_thresholds_to_return: List[float],
) -> list[
    metrics.APMetric
    | metrics.APMetricAveragedOverIOUs
    | metrics.mAPMetric
    | metrics.mAPMetricAveragedOverIOUs
]:
    """Calculates all AP metrics, including aggregated metrics like mAP."""
    ap_metrics_df = (
        calculation_df.loc[
            ~calculation_df[
                "id_gt"
            ].isnull(),  # for AP, we don't include any predictions without groundtruths
            [
                "label_id",
                "label",
                "iou_threshold",
                "precision",
                "recall_for_AP",
            ],
        ]
        .groupby(["label_id", "label", "iou_threshold"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "calculated_precision": _calculate_101_pt_interp(
                        x["precision"].tolist(),
                        x["recall_for_AP"].tolist(),
                    )
                }
            ),
            include_groups=False,
        )
    )

    ap_metrics = [
        metrics.APMetric(
            iou=row["iou_threshold"],
            value=row["calculated_precision"],
            label=schemas.Label(key=row["label"][0], value=row["label"][1]),
        )
        for row in ap_metrics_df.to_dict(orient="records")
    ]

    # calculate mean AP metrics
    ap_metrics_df["label_key"] = ap_metrics_df["label"].apply(lambda x: x[0])

    ap_over_ious_df = ap_metrics_df.groupby(
        ["label_id", "label"], as_index=False
    )["calculated_precision"].apply(_calculate_mean_ignoring_negative_one)

    ap_over_ious = [
        metrics.APMetricAveragedOverIOUs(
            ious=set(iou_thresholds_to_compute),
            value=row["calculated_precision"],
            label=schemas.Label(key=row["label"][0], value=row["label"][1]),
        )
        for row in ap_over_ious_df.to_dict(
            orient="records"
        )  # pyright: ignore - pandas .to_dict() typing error
    ]

    map_metrics_df = ap_metrics_df.groupby(
        ["iou_threshold", "label_key"], as_index=False
    )["calculated_precision"].apply(_calculate_mean_ignoring_negative_one)

    map_metrics = [
        metrics.mAPMetric(
            iou=row["iou_threshold"],
            value=row["calculated_precision"],
            label_key=row["label_key"],
        )
        for row in map_metrics_df.to_dict(
            orient="records"
        )  # pyright: ignore - pandas .to_dict() typing error
    ]

    map_over_ious_df = ap_metrics_df.groupby(["label_key"], as_index=False)[
        "calculated_precision"
    ].apply(_calculate_mean_ignoring_negative_one)

    map_over_ious = [
        metrics.mAPMetricAveragedOverIOUs(
            ious=set(iou_thresholds_to_compute),
            value=row["calculated_precision"],
            label_key=row["label_key"],
        )
        for row in map_over_ious_df.to_dict(
            orient="records"
        )  # pyright: ignore - pandas .to_dict() typing error
    ]

    return (
        [m for m in ap_metrics if m.iou in iou_thresholds_to_return]
        + [m for m in map_metrics if m.iou in iou_thresholds_to_return]
        + ap_over_ious
        + map_over_ious
    )


def _calculate_ar_metrics(
    calculation_df: pd.DataFrame,
    iou_thresholds_to_compute: List[float],
) -> list[metrics.ARMetric | metrics.mARMetric]:
    """Calculates all AR metrics, including aggregated metrics like mAR."""

    # get the max recall_for_AR for each threshold, then take the mean across thresholds
    ar_metrics_df = (
        calculation_df.groupby(
            ["label_id", "label", "iou_threshold"], as_index=False
        )["recall_for_AR"]
        .max()
        .groupby(["label_id", "label"], as_index=False)["recall_for_AR"]
        .mean()
    )

    ious_ = set(iou_thresholds_to_compute)
    ar_metrics = [
        metrics.ARMetric(
            ious=ious_,
            value=row["recall_for_AR"],
            label=schemas.Label(key=row["label"][0], value=row["label"][1]),
        )
        for row in ar_metrics_df.to_dict(orient="records")
    ]

    # calculate mAR
    ar_metrics_df["label_key"] = ar_metrics_df["label"].apply(lambda x: x[0])
    mar_metrics_df = ar_metrics_df.groupby(["label_key"], as_index=False)[
        "recall_for_AR"
    ].apply(_calculate_mean_ignoring_negative_one)

    mar_metrics = [
        metrics.mARMetric(
            ious=ious_,
            value=row["recall_for_AR"],
            label_key=row["label_key"],
        )
        for row in mar_metrics_df.to_dict(orient="records")
    ]

    return ar_metrics + mar_metrics


def _calculate_pr_metrics(
    joint_df: pd.DataFrame,
    metrics_to_return: List[enums.MetricType],
    pr_curve_iou_threshold: float,
) -> list[metrics.PrecisionRecallCurve]:
    """Calculates all PrecisionRecallCurve metrics."""

    if not (
        metrics_to_return
        and enums.MetricType.PrecisionRecallCurve in metrics_to_return
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
            "label_id",
            "confidence_threshold",
            "score",
            "iou_",
        ],
        ascending=False,
    )

    pr_calculation_df["true_positive_flag"] = (
        (pr_calculation_df["iou_"] >= pr_curve_iou_threshold)
        & (
            pr_calculation_df["score"]
            >= pr_calculation_df["confidence_threshold"]
        )
        & (
            pr_calculation_df.groupby(
                ["label_id", "confidence_threshold", "id_gt"]
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
                "label_id",
                "label",
                "confidence_threshold",
                "gts_per_grouper",
            ],
            as_index=False,
        )["true_positive_flag"]
        .sum()
        .merge(
            pr_calculation_df.groupby(
                ["label_id", "label", "confidence_threshold"],
                as_index=False,
            )["false_positive_flag"].sum(),
            on=["label_id", "label", "confidence_threshold"],
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

    pr_metrics_df.fillna(0, inplace=True)

    curves = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for row in pr_metrics_df.to_dict(orient="records"):
        curves[row["label"][0]][row["label"][1]][
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
        metrics.PrecisionRecallCurve(
            label_key=key,
            value=value,  # #type: ignore - defaultdict doesn't have strict typing
            pr_curve_iou_threshold=pr_curve_iou_threshold,
        )
        for key, value in curves.items()
    ]


def _add_samples_to_dataframe(
    detailed_pr_curve_counts_df: pd.DataFrame,
    detailed_pr_calc_df: pd.DataFrame,
    max_examples: int,
    flag_column: str,
) -> pd.DataFrame:
    """Efficienctly gather samples for a given flag."""

    sample_df = pd.concat(
        [
            detailed_pr_calc_df[detailed_pr_calc_df[flag_column]]
            .groupby(
                [
                    "label_key",
                    "label_value_gt",
                    "confidence_threshold",
                ],
                as_index=False,
            )[["datum_uid_gt", "converted_geometry_gt"]]
            .agg(lambda x: tuple(x.head(max_examples)))
            .rename(
                columns={
                    "datum_uid_gt": "datum_uid",
                    "label_value_gt": "label_value",
                    "converted_geometry_gt": "converted_geometry",
                }
            ),
            detailed_pr_calc_df[detailed_pr_calc_df[flag_column]]
            .groupby(
                [
                    "label_key",
                    "label_value_pd",
                    "confidence_threshold",
                ],
                as_index=False,
            )[["datum_uid_pd", "converted_geometry_pd"]]
            .agg(lambda x: (tuple(x.head(max_examples))))
            .rename(
                columns={
                    "datum_uid_pd": "datum_uid",
                    "label_value_pd": "label_value",
                    "converted_geometry_pd": "converted_geometry",
                }
            ),
        ],
        axis=0,
    )

    sample_df["converted_geometry"] = sample_df["converted_geometry"].astype(
        str
    )
    sample_df.drop_duplicates(inplace=True)

    if not sample_df.empty:
        sample_df[f"{flag_column}_samples"] = sample_df.apply(
            lambda row: set(zip(*row[["datum_uid", "converted_geometry"]])),  # type: ignore - pd typing error
            axis=1,
        )

        detailed_pr_curve_counts_df = detailed_pr_curve_counts_df.merge(
            sample_df[
                [
                    "label_key",
                    "label_value",
                    "confidence_threshold",
                    f"{flag_column}_samples",
                ]
            ],
            on=["label_key", "label_value", "confidence_threshold"],
            how="outer",
        )
        detailed_pr_curve_counts_df[
            f"{flag_column}_samples"
        ] = detailed_pr_curve_counts_df[f"{flag_column}_samples"].apply(
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
    metrics_to_return: List[enums.MetricType],
    pr_curve_iou_threshold: float,
    pr_curve_max_examples: int,
) -> list[metrics.DetailedPrecisionRecallCurve]:
    """Calculates all DetailedPrecisionRecallCurve metrics."""

    if not (
        metrics_to_return
        and enums.MetricType.DetailedPrecisionRecallCurve in metrics_to_return
    ):
        return []

    detailed_pr_joint_df = pd.merge(
        groundtruth_df,
        prediction_df,
        on=["datum_id", "label_key"],
        how="outer",
        suffixes=("_gt", "_pd"),
    ).assign(
        is_label_match=lambda chain_df: (
            chain_df["label_id_pd"] == chain_df["label_id_gt"]
        )
    )

    detailed_pr_joint_df = _calculate_iou(joint_df=detailed_pr_joint_df)

    # add confidence_threshold to the dataframe and sort
    detailed_pr_calc_df = pd.concat(
        [
            detailed_pr_joint_df.assign(confidence_threshold=threshold)
            for threshold in [x / 100 for x in range(5, 100, 5)]
        ],
        ignore_index=True,
    ).sort_values(
        by=[
            "label_id_pd",
            "confidence_threshold",
            "score",
            "iou_",
        ],
        ascending=False,
    )

    # create flags where predictions meet the score and IOU criteria
    detailed_pr_calc_df["true_positive_flag"] = (
        (detailed_pr_calc_df["iou_"] >= pr_curve_iou_threshold)
        & (
            detailed_pr_calc_df["score"]
            >= detailed_pr_calc_df["confidence_threshold"]
        )
        & detailed_pr_calc_df["is_label_match"]
    )

    # for all the false positives, we consider them to be a misclassification if they overlap with a groundtruth of the same label key
    detailed_pr_calc_df["misclassification_false_positive_flag"] = (
        (detailed_pr_calc_df["iou_"] >= pr_curve_iou_threshold)
        & (
            detailed_pr_calc_df["score"]
            >= detailed_pr_calc_df["confidence_threshold"]
        )
        & ~detailed_pr_calc_df["is_label_match"]
    )

    # if they aren't a true positive nor a misclassification FP but they meet the iou and score conditions, then they are a hallucination
    detailed_pr_calc_df["hallucination_false_positive_flag"] = (
        (detailed_pr_calc_df["iou_"] < pr_curve_iou_threshold)
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
        ].apply(lambda x: set(x) if isinstance(x, np.ndarray) else [])
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
        ].apply(lambda x: set(x) if isinstance(x, np.ndarray) else set())

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
            .apply(lambda x: set(x) if isinstance(x, np.ndarray) else set())
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
        .groupby(["label_key", "label_value_pd", "confidence_threshold"])[
            "id_pd"
        ]
        .nunique()
    )
    true_positives.name = "true_positives"

    hallucination_false_positives = (
        detailed_pr_calc_df[
            detailed_pr_calc_df["hallucination_false_positive_flag"]
        ]
        .groupby(["label_key", "label_value_pd", "confidence_threshold"])[
            "id_pd"
        ]
        .nunique()
    )
    hallucination_false_positives.name = "hallucinations_false_positives"

    misclassification_false_positives = (
        detailed_pr_calc_df[
            detailed_pr_calc_df["misclassification_false_positive_flag"]
        ]
        .groupby(["label_key", "label_value_pd", "confidence_threshold"])[
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
        .groupby(["label_key", "label_value_gt", "confidence_threshold"])[
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
        .groupby(["label_key", "label_value_gt", "confidence_threshold"])[
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
                    ~detailed_pr_calc_df["label_value_pd"].isnull(),
                    [
                        "label_key",
                        "label_value_pd",
                        "confidence_threshold",
                    ],
                ].rename(columns={"label_value_pd": "label_value"}),
                detailed_pr_calc_df.loc[
                    ~detailed_pr_calc_df["label_value_gt"].isnull(),
                    [
                        "label_key",
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
                "label_key",
                "label_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            hallucination_false_positives,
            left_on=[
                "label_key",
                "label_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            misclassification_false_positives,
            left_on=[
                "label_key",
                "label_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            misclassification_false_negatives,
            left_on=[
                "label_key",
                "label_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            no_predictions_false_negatives,
            left_on=[
                "label_key",
                "label_value",
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
            max_examples=pr_curve_max_examples,
            flag_column=flag,
        )

    # create output
    detailed_pr_curves = defaultdict(lambda: defaultdict(dict))
    for _, row in detailed_pr_curve_counts_df.iterrows():
        label_key = row["label_key"]
        label_value = row["label_value"]
        confidence_threshold = row["confidence_threshold"]

        detailed_pr_curves[label_key][label_value][confidence_threshold] = {
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
        metrics.DetailedPrecisionRecallCurve(
            label_key=key,
            value=dict(value),
            pr_curve_iou_threshold=pr_curve_iou_threshold,
        )
        for key, value in detailed_pr_curves.items()
    ]

    return detailed_pr_metrics


def _compute_detection_metrics(
    groundtruth_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    label_map: Dict[schemas.Label, schemas.Label],
    metrics_to_return: List[enums.MetricType],
    iou_thresholds_to_compute: List[float],
    iou_thresholds_to_return: List[float],
    recall_score_threshold: float,
    pr_curve_iou_threshold: float,
    pr_curve_max_examples: int,
    unique_labels: list,
) -> List[dict]:
    """
    Compute detection metrics for evaluating object detection models. This function calculates Intersection over Union (IoU) for each ground truth-prediction pair that shares a common grouper id, and computes metrics such as Average Precision (AP), Average Recall (AR), and Precision-Recall (PR) curves.

    Parameters
    ----------
    groundtruth_df : pd.DataFrame
        DataFrame containing ground truth annotations with columns including bounding boxes, polygons, or rasters.
    prediction_df : pd.DataFrame
        DataFrame containing predicted annotations with similar columns as `groundtruth_df`.
    label_map : Dict[schemas.Label, schemas.Label]
        Mapping of ground truth labels to prediction labels.
    metrics_to_return : List[enums.MetricType]
        List of metric types to calculate and return, such as AP, AR, or PR curves.
    iou_thresholds_to_compute : List[float]
        List of IoU thresholds for which metrics should be computed.
    iou_thresholds_to_return : List[float]
        List of IoU thresholds for which metrics should be returned.
    recall_score_threshold : float
        Threshold for the recall score to consider in metric calculations.
    pr_curve_iou_threshold : float
        IoU threshold for computing Precision-Recall curves.
    pr_curve_max_examples : int
        Maximum number of examples to use for Precision-Recall curve calculations.
    unique_labels : list
        List of unique labels present in the datasets.

    Returns
    -------
    List[dict]
        A list of dictionaries containing computed metrics, including AP, AR, and PR curves, filtered according to `metrics_to_return`.

    Raises
    ------
    ValueError
        If there is an issue with the data or parameters provided.
    """

    groundtruth_df, prediction_df = utilities.replace_labels_using_label_map(
        groundtruth_df=groundtruth_df,
        prediction_df=prediction_df,
        label_map=label_map,
    )

    # add label as a column
    for df in (groundtruth_df, prediction_df):
        df.loc[:, "label"] = df.apply(
            lambda chain_df: (chain_df["label_key"], chain_df["label_value"]),
            axis=1,
        )

    metrics_to_output = []

    joint_df = _get_joint_df(
        groundtruth_df=groundtruth_df,
        prediction_df=prediction_df,
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
    calculation_df = pd.concat(
        [
            joint_df.assign(iou_threshold=threshold)
            for threshold in iou_thresholds_to_compute
        ],
        ignore_index=True,
    ).sort_values(
        by=["label_id", "label", "iou_threshold", "score", "iou_"],
        ascending=False,
    )

    # calculate metrics
    calculation_df = _calculate_label_id_level_metrics(
        calculation_df=calculation_df,
        recall_score_threshold=recall_score_threshold,
    )

    metrics_to_output += _calculate_ap_metrics(
        calculation_df=calculation_df,
        iou_thresholds_to_compute=iou_thresholds_to_compute,
        iou_thresholds_to_return=iou_thresholds_to_return,
    )

    metrics_to_output += _calculate_ar_metrics(
        calculation_df=calculation_df,
        iou_thresholds_to_compute=iou_thresholds_to_compute,
    )

    metrics_to_output += _calculate_pr_metrics(
        joint_df=joint_df,
        metrics_to_return=metrics_to_return,
        pr_curve_iou_threshold=pr_curve_iou_threshold,
    )

    metrics_to_output += _calculate_detailed_pr_metrics(
        groundtruth_df=groundtruth_df,
        prediction_df=prediction_df,
        metrics_to_return=metrics_to_return,
        pr_curve_iou_threshold=pr_curve_iou_threshold,
        pr_curve_max_examples=pr_curve_max_examples,
    )

    # convert objects to dictionaries and only return what was asked for
    metrics_to_output = [
        m.to_dict()
        for m in metrics_to_output
        if m.to_dict()["type"] in metrics_to_return
    ]

    return metrics_to_output


def evaluate_detection(
    groundtruths: Union[pd.DataFrame, List[schemas.GroundTruth]],
    predictions: Union[pd.DataFrame, List[schemas.Prediction]],
    label_map: Optional[Dict[schemas.Label, schemas.Label]] = None,
    metrics_to_return: Optional[List[enums.MetricType]] = None,
    convert_annotations_to_type: Optional[enums.AnnotationType] = None,
    iou_thresholds_to_compute: Optional[List[float]] = None,
    iou_thresholds_to_return: Optional[List[float]] = None,
    recall_score_threshold: float = 0.0,
    pr_curve_iou_threshold: float = 0.5,
    pr_curve_max_examples: int = 1,
) -> schemas.Evaluation:
    """
    Evaluate an object detection task using some set of groundtruths and predictions.

    The groundtruths and predictions can be inputted as a pandas DataFrame or as a list of GroundTruth/Prediction objects. A dataframe of groundtruths / predictions should contain the following columns:
    - datum_uid (str): The unique identifier for the datum.
    - datum_id (int): A hashed identifier that's unique to each datum.
    - datum_metadata (dict): Metadata associated with the datum.
    - annotation_id (int): A hashed identifier for each unique (datum_uid, annotation) combination.
    - annotation_metadata (dict): Metadata associated with the annotation.
    - bounding_box (tuple): The bounding box coordinates of the annotation, if available.
    - raster (schemas.Raster): The raster representation of the annotation, if available.
    - polygon (schemas.Polygon): The polygon coordinates of the annotation, if available.
    - embedding (schemas.Embedding): The embedding vector associated with the annotation, if available.
    - is_instance (bool): A boolean indicating whether the annotation is an instance segjmentation (True) or not (False).
    - label_key (str): The key associated with the label.
    - label_value (str): The value associated with the label.
    - score (float): The confidence score of the prediction. Should be bound between 0 and 1. Should only be included for prediction dataframes.
    - label_id (int): A hashed identifier for each unique label.
    - id (str): A unique identifier for the combination of datum, annotation, and label, created by concatenating the indices of these components.


    Parameters
    ----------
    groundtruths : Union[pd.DataFrame, List[schemas.GroundTruth]]
        A list of GroundTruth objects or a pandas DataFrame describing your ground truths.
    predictions : Union[pd.DataFrame, List[schemas.Prediction]]
        A list of Prediction objects or a pandas DataFrame describing your predictions.
    label_map : Optional[Dict[schemas.Label, schemas.Label]], default=None
        Mapping of ground truth labels to prediction labels.
    metrics_to_return : Optional[List[enums.MetricType]], default=None
        List of metric types to calculate and return.
    convert_annotations_to_type : Optional[enums.AnnotationType], default=None
        Annotation type to convert all annotations to.
    iou_thresholds_to_compute : Optional[List[float]], default=None
        IoU thresholds for which metrics should be computed.
    iou_thresholds_to_return : Optional[List[float]], default=None
        IoU thresholds for which metrics should be returned.
    recall_score_threshold : float, default=0.0
        Threshold for recall score to consider in metric calculations.
    pr_curve_iou_threshold : float, default=0.5
        IoU threshold for computing Precision-Recall curves.
    pr_curve_max_examples : int, default=1
        Maximum number of examples for Precision-Recall curve calculations.

    Returns
    -------
    schemas.Evaluation
        An Evaluation object containing the calculated metrics and other details.

    Raises
    ------
    ValueError
        If there is an issue with the provided parameters or data.
    """
    start_time = time.time()

    if not label_map:
        label_map = {}

    if metrics_to_return is None:
        metrics_to_return = [
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
        ]

    if iou_thresholds_to_compute is None:
        iou_thresholds_to_compute = [
            round(0.5 + 0.05 * i, 2) for i in range(10)
        ]
    if iou_thresholds_to_return is None:
        iou_thresholds_to_return = [0.5, 0.75]

    utilities.validate_label_map(label_map=label_map)
    utilities.validate_metrics_to_return(
        metrics_to_return=metrics_to_return,
        task_type=enums.TaskType.OBJECT_DETECTION,
    )
    utilities.validate_parameters(
        pr_curve_iou_threshold=pr_curve_iou_threshold,
        pr_curve_max_examples=pr_curve_max_examples,
        recall_score_threshold=recall_score_threshold,
    )

    groundtruth_df = utilities.create_filtered_and_validated_groundtruth_df(
        groundtruths, task_type=enums.TaskType.OBJECT_DETECTION
    )
    prediction_df = utilities.create_filtered_and_validated_prediction_df(
        predictions, task_type=enums.TaskType.OBJECT_DETECTION
    )

    # ensure that all annotations have a common type to operate over
    (
        groundtruth_df,
        prediction_df,
    ) = utilities.convert_annotations_to_common_type(
        groundtruth_df=groundtruth_df,
        prediction_df=prediction_df,
        target_type=convert_annotations_to_type,
    )

    (
        missing_pred_labels,
        ignored_pred_labels,
    ) = utilities.get_disjoint_labels(
        groundtruth_df=groundtruth_df,
        prediction_df=prediction_df,
        label_map=label_map,
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

    metrics = _compute_detection_metrics(
        groundtruth_df=groundtruth_df,
        prediction_df=prediction_df,
        label_map=label_map,
        metrics_to_return=metrics_to_return,
        iou_thresholds_to_compute=iou_thresholds_to_compute,
        iou_thresholds_to_return=iou_thresholds_to_return,
        recall_score_threshold=recall_score_threshold,
        pr_curve_iou_threshold=pr_curve_iou_threshold,
        pr_curve_max_examples=pr_curve_max_examples,
        unique_labels=unique_labels,
    )

    return schemas.Evaluation(
        parameters=schemas.EvaluationParameters(
            label_map=label_map,
            metrics_to_return=metrics_to_return,
            iou_thresholds_to_compute=iou_thresholds_to_compute,
            iou_thresholds_to_return=iou_thresholds_to_return,
            recall_score_threshold=recall_score_threshold,
            pr_curve_iou_threshold=pr_curve_iou_threshold,
            pr_curve_max_examples=pr_curve_max_examples,
        ),
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
