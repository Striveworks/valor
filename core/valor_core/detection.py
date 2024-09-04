import heapq
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
from valor_core import enums, geometry, metrics, schemas, utilities


def _merge_dataframes_polars(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    suffixes: tuple[str, str],
    on: list[str],
    how: str = "outer",
) -> pl.DataFrame:
    """Handle differences in behavior between polars and pandas"""
    df1 = df1.with_columns(
        [
            pl.col(col).alias(f"{col}{suffixes[0]}")
            for col in df1.columns
            if col not in on
        ]
    )
    df2 = df2.with_columns(
        [
            pl.col(col).alias(f"{col}{suffixes[1]}")
            for col in df2.columns
            if col not in on
        ]
    )

    merged_df = df1.join(df2, on=on, how=how)
    return merged_df


# TODO this could be useful later instead of _get_joint_df
def _get_joint_df_TODO(
    groundtruth_df: pl.DataFrame,
    prediction_df: pl.DataFrame,
) -> pl.DataFrame:
    """Create a joint dataframe of groundtruths and predictions for calculating AR/AP metrics."""

    return _merge_dataframes_polars(
        df1=groundtruth_df,
        df2=prediction_df,
        suffixes=("_gt", "_pd"),
        on=["datum_id", "label_id", "label"],
    )


def _get_joint_df(
    groundtruth_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a joint dataframe of groundtruths and predictions for calculating AR/AP metrics."""

    joint_df = pd.merge(
        groundtruth_df,
        prediction_df,
        on=["datum_id", "label_id", "label"],
        how="outer",
        suffixes=("_gt", "_pd"),
    )

    return joint_df


def _get_dtypes_in_series_of_arrays(series: pl.Series):
    """Get the data type inside of a 2D numpy array. Used to check if a np.array contains coordinates or a mask."""
    if not isinstance(series, pd.Series) or not all(
        series.map(lambda x: x.ndim == 2)
    ):
        raise ValueError(
            "series must be a pandas Series filled with two-dimensional arrays."
        )

    unique_primitives = series.map(lambda x: x.dtype).unique()

    if len(unique_primitives) > 1:
        raise ValueError("series contains more than one type of primitive.")

    return unique_primitives[0]


def _check_if_series_contains_masks(series: pl.Series) -> bool:
    """Check if any element in a pandas.Series is a mask."""
    if series.empty:
        return False

    primitive = _get_dtypes_in_series_of_arrays(series=series)

    if np.issubdtype(primitive, np.bool_):
        return True

    return False


def _check_if_series_contains_axis_aligned_bboxes(series: pl.Series) -> bool:
    """Check if all elements in a pandas.Series are axis-aligned bounding boxes."""

    return series.map(lambda x: x.tolist()).map(geometry.is_axis_aligned).all()


def _calculate_iou(
    joint_df: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate the IOUs between predictions and groundtruths in a joint dataframe."""
    filtered_df = joint_df.loc[
        ~joint_df["converted_geometry_gt"].isnull()
        & ~joint_df["converted_geometry_pd"].isnull(),
        ["converted_geometry_gt", "converted_geometry_pd"],
    ]

    if filtered_df.empty:
        joint_df["iou_"] = 0
        return joint_df

    if not _check_if_series_contains_masks(
        filtered_df["converted_geometry_pd"]
    ):
        if _check_if_series_contains_axis_aligned_bboxes(
            filtered_df["converted_geometry_pd"]
        ) & _check_if_series_contains_axis_aligned_bboxes(
            filtered_df["converted_geometry_gt"]
        ):
            series_of_iou_calculations = (
                geometry.calculate_axis_aligned_bbox_iou(
                    filtered_df["converted_geometry_gt"],
                    filtered_df["converted_geometry_pd"],
                )
            )

        else:
            iou_func = np.vectorize(geometry.calculate_iou)

            series_of_iou_calculations = pl.Series(
                iou_func(
                    filtered_df["converted_geometry_gt"],
                    filtered_df["converted_geometry_pd"],
                ),
                index=filtered_df.index,
            )

        series_of_iou_calculations = series_of_iou_calculations.rename("iou_")

        joint_df = joint_df.join(series_of_iou_calculations)

    else:

        joint_df["iou_"] = geometry.calculate_raster_ious(
            joint_df["converted_geometry_gt"],
            joint_df["converted_geometry_pd"],
        )

    return joint_df


def _calculate_label_id_level_metrics(
    calculation_df: pl.DataFrame, recall_score_threshold: float
) -> pl.DataFrame:
    """Calculate the flags and metrics needed to compute AP, AR, and PR curves."""

    # create flags where predictions meet the score and IOU criteria
    calculation_df = calculation_df.with_columns(
        pl.when(
            (pl.col("iou_") >= pl.col("iou_threshold"))
            & (pl.col("score") >= recall_score_threshold)
        )
        .then(True)
        .otherwise(False)
        .alias("recall_true_positive_flag")
    )

    # only consider the highest scoring true positive as an actual true positive

    calculate_ids_with_highest_score = calculation_df.group_by(
        ["label_id", "label", "iou_threshold", "id_gt"]
    ).agg(
        pl.col("id_gt")
        .filter(pl.col("score") == pl.col("score").max())
        .first()
        .is_not_null()
        .alias("id_with_highest_score")
    )

    calculation_df = calculation_df.join(
        calculate_ids_with_highest_score,
        on=["label_id", "label", "iou_threshold", "id_gt"],
        how="left",
    )

    calculation_df = calculation_df.with_columns(
        pl.when(
            pl.col("recall_true_positive_flag")
            & pl.col("id_with_highest_score")
        )
        .then(True)
        .otherwise(False)
        .alias("recall_true_positive_flag")
    )

    # calculate precision_true_positive_flag and filter to only include the highest scoring prediction
    calculation_df = calculation_df.with_columns(
        pl.when(
            (pl.col("iou_") >= pl.col("iou_threshold")) & (pl.col("score") > 0)
        )
        .then(True)
        .otherwise(False)
        .alias("precision_true_positive_flag")
    )

    calculation_df = calculation_df.with_columns(
        pl.when(
            pl.col("precision_true_positive_flag")
            & pl.col("id_with_highest_score")
        )
        .then(True)
        .otherwise(False)
        .alias("precision_true_positive_flag")
    )

    calculation_df = calculation_df.with_columns(
        pl.when(
            ~pl.col("recall_true_positive_flag")
            & (pl.col("score") >= recall_score_threshold)
        )
        .then(True)
        .otherwise(False)
        .alias("recall_false_positive_flag"),
        pl.when(
            ~pl.col("precision_true_positive_flag") & (pl.col("score") > 0)
        )
        .then(True)
        .otherwise(False)
        .alias("precision_false_positive_flag"),
    )

    # calculate true and false positives

    cumulative_sums = calculation_df.group_by(
        ["label_id", "label", "iou_threshold"]
    ).agg(
        [
            pl.col("recall_true_positive_flag")
            .cum_sum()
            .last()
            .alias("rolling_recall_tp"),
            pl.col("recall_false_positive_flag")
            .cum_sum()
            .last()
            .alias("rolling_recall_fp"),
            pl.col("precision_true_positive_flag")
            .cum_sum()
            .last()
            .alias("rolling_precision_tp"),
            pl.col("precision_false_positive_flag")
            .cum_sum()
            .last()
            .alias("rolling_precision_fp"),
        ]
    )

    calculation_df = calculation_df.join(
        cumulative_sums, on=["label_id", "label", "iou_threshold"], how="left"
    )

    # calculate false negatives, then precision / recall
    calculation_df = calculation_df.with_columns(
        [
            (pl.col("gts_per_grouper") - pl.col("rolling_recall_tp")).alias(
                "rolling_recall_fn"
            ),
            (pl.col("gts_per_grouper") - pl.col("rolling_precision_tp")).alias(
                "rolling_precision_fn"
            ),
            (
                pl.col("rolling_precision_tp")
                / (
                    pl.col("rolling_precision_tp")
                    + pl.col("rolling_precision_fp")
                )
            ).alias("precision"),
        ],
    ).with_columns(
        (
            pl.col("rolling_precision_tp")
            / (pl.col("rolling_precision_tp") + pl.col("rolling_precision_fn"))
        ).alias("recall_for_AP"),
        (
            pl.col("rolling_recall_tp")
            / (pl.col("rolling_recall_tp") + pl.col("rolling_recall_fn"))
        ).alias("recall_for_AR"),
    )

    # fill any predictions that are missing groundtruths with -1
    # leave any groundtruths that are missing predictions with 0
    calculation_df = calculation_df.with_columns(
        [
            pl.when(pl.col("id_gt").is_null())
            .then(-1)
            .otherwise(pl.col("precision"))
            .alias("precision"),
            pl.when(pl.col("id_gt").is_null())
            .then(-1)
            .otherwise(pl.col("recall_for_AP"))
            .alias("recall_for_AP"),
            pl.when(pl.col("id_gt").is_null())
            .then(-1)
            .otherwise(pl.col("recall_for_AR"))
            .alias("recall_for_AR"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("id_pd").is_null())
            .then(0)
            .otherwise(pl.col("precision"))
            .alias("precision"),
            pl.when(pl.col("id_pd").is_null())
            .then(0)
            .otherwise(pl.col("recall_for_AP"))
            .alias("recall_for_AP"),
            pl.when(pl.col("id_pd").is_null())
            .then(0)
            .otherwise(pl.col("recall_for_AR"))
            .alias("recall_for_AR"),
        ]
    )

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


def _calculate_mean_ignoring_negative_one(series: pl.Series) -> float:
    """Calculate the mean of a series, ignoring any values that are -1."""
    filtered = series.filter(series != -1)
    return filtered.mean() if not filtered.is_empty() else -1.0


def _calculate_ap_metrics(
    calculation_df: pl.DataFrame,
    iou_thresholds_to_compute: list[float],
    iou_thresholds_to_return: list[float],
) -> list[
    metrics.APMetric
    | metrics.APMetricAveragedOverIOUs
    | metrics.mAPMetric
    | metrics.mAPMetricAveragedOverIOUs
]:
    """Calculates all AP metrics, including aggregated metrics like mAP."""

    ap_metrics_df = (
        calculation_df.filter(
            ~pl.col("id_gt").is_null()
        )  # for AP, we don't include any predictions without groundtruths
        .group_by(["label_id", "label", "iou_threshold"])
        .agg(
            [
                pl.col("precision").explode().alias("precision_arr"),
                pl.col("recall_for_AP").explode().alias("recall_for_AP_arr"),
            ]
        )
    ).with_columns(
        [
            pl.struct(["precision_arr", "recall_for_AP_arr"])
            .map_elements(
                lambda s: _calculate_101_pt_interp(
                    s["precision_arr"], s["recall_for_AP_arr"]
                ),
                return_dtype=pl.Float64,
            )
            .alias("calculated_precision")
        ]
    )

    # TODO figure out how to not treat labels like strings. this was causing an issue when joining on label. might be as easy as using tuple instead of list?
    # TODO ctrl + f for "lambda s" and "lambda x"
    ap_metrics_df = ap_metrics_df.with_columns(
        pl.col("label")
        .map_elements(
            lambda x: x.strip("[]").split(", "),
            return_dtype=pl.List(pl.Utf8),
        )
        .alias("label")
    ).with_columns(
        pl.col("label")
        .map_elements(
            lambda x: x[0],
            return_dtype=pl.Utf8,
        )
        .alias("label_key")
    )

    ap_metrics = [
        metrics.APMetric(
            iou=row["iou_threshold"],
            value=row["calculated_precision"],
            label=schemas.Label(key=row["label"][0], value=row["label"][1]),
        )
        for row in ap_metrics_df.rows(named=True)
    ]

    ap_over_ious_df = ap_metrics_df.group_by(["label_id", "label"]).agg(
        [
            pl.col("calculated_precision")
            .map_elements(lambda x: _calculate_mean_ignoring_negative_one(x))
            .alias("mean_precision")
        ]
    )

    # TODO try to clear out pyright type ignores
    ap_over_ious = [
        metrics.APMetricAveragedOverIOUs(
            ious=set(iou_thresholds_to_compute),
            value=row["mean_precision"],
            label=schemas.Label(key=row["label"][0], value=row["label"][1]),
        )
        for row in ap_over_ious_df.rows(named=True)
    ]

    map_metrics_df = ap_metrics_df.group_by(
        ["iou_threshold", "label_key"]
    ).agg(
        [
            pl.col("calculated_precision")
            .map_elements(lambda x: _calculate_mean_ignoring_negative_one(x))
            .alias("mean_precision")
        ]
    )

    map_metrics = [
        metrics.mAPMetric(
            iou=row["iou_threshold"],
            value=row["mean_precision"],
            label_key=row["label_key"],
        )
        for row in map_metrics_df.rows(named=True)
    ]

    map_over_ious_df = ap_metrics_df.group_by(["label_key"]).agg(
        [
            pl.col("calculated_precision")
            .map_elements(lambda x: _calculate_mean_ignoring_negative_one(x))
            .alias("mean_precision")
        ]
    )

    map_over_ious = [
        metrics.mAPMetricAveragedOverIOUs(
            ious=set(iou_thresholds_to_compute),
            value=row["mean_precision"],
            label_key=row["label_key"],
        )
        for row in map_over_ious_df.rows(named=True)
    ]

    return (
        [m for m in ap_metrics if m.iou in iou_thresholds_to_return]
        + [m for m in map_metrics if m.iou in iou_thresholds_to_return]
        + ap_over_ious
        + map_over_ious
    )


def _calculate_ar_metrics(
    calculation_df: pl.DataFrame,
    iou_thresholds_to_compute: list[float],
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
    ar_metrics_df["label_key"] = ar_metrics_df["label"].str[0]
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
    joint_df: pl.DataFrame,
    metrics_to_return: list[enums.MetricType],
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
            value=value,  # type: ignore - defaultdict doesn't have strict typing
            pr_curve_iou_threshold=pr_curve_iou_threshold,
        )
        for key, value in curves.items()
    ]


def _add_samples_to_dataframe(
    detailed_pr_curve_counts_df: pl.DataFrame,
    detailed_pr_calc_df: pl.DataFrame,
    max_examples: int,
    flag_column: str,
) -> pl.DataFrame:
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
            .agg(tuple)
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
            .agg(tuple)
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

    sample_df["converted_geometry"] = sample_df["converted_geometry"].apply(
        lambda row: tuple(str(x.tolist()) for x in row)
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
            lambda x: list(x)[:max_examples] if isinstance(x, set) else list()
        )
    else:
        detailed_pr_curve_counts_df[f"{flag_column}_samples"] = [
            list() for _ in range(len(detailed_pr_curve_counts_df))
        ]

    return detailed_pr_curve_counts_df


def _calculate_detailed_pr_metrics(
    detailed_pr_joint_df: pl.DataFrame | None,
    metrics_to_return: list[enums.MetricType],
    pr_curve_iou_threshold: float,
    pr_curve_max_examples: int,
) -> list[metrics.DetailedPrecisionRecallCurve]:
    """Calculates all DetailedPrecisionRecallCurve metrics."""

    if not (
        metrics_to_return
        and enums.MetricType.DetailedPrecisionRecallCurve in metrics_to_return
    ) or (detailed_pr_joint_df is None):
        return []

    if _check_if_series_contains_masks(
        detailed_pr_joint_df.loc[
            detailed_pr_joint_df["converted_geometry_gt"].notnull(),
            "converted_geometry_gt",
        ]
    ) or _check_if_series_contains_masks(
        detailed_pr_joint_df.loc[
            detailed_pr_joint_df["converted_geometry_pd"].notnull(),
            "converted_geometry_pd",
        ]
    ):
        raise NotImplementedError(
            "DetailedPrecisionRecallCurves are not yet implemented when dealing with rasters."
        )

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
        confidence_interval_to_predictions_associated_with_tps_or_misclassification_fps_dict = (
            predictions_associated_with_tps_or_misclassification_fps.set_index(
                "confidence_threshold"
            )["id_pd"]
            .apply(set)
            .to_dict()
        )

        mask = pl.Series(False, index=detailed_pr_calc_df.index)

        for (
            threshold,
            elements,
        ) in (
            confidence_interval_to_predictions_associated_with_tps_or_misclassification_fps_dict.items()
        ):
            threshold_mask = (
                detailed_pr_calc_df["confidence_threshold"] == threshold
            )
            membership_mask = detailed_pr_calc_df["id_pd"].isin(elements)
            mask |= (
                threshold_mask
                & membership_mask
                & detailed_pr_calc_df["hallucination_false_positive_flag"]
            )

        detailed_pr_calc_df.loc[
            mask,
            "hallucination_false_positive_flag",
        ] = False

    # next, we flag false negatives by declaring any groundtruth that isn't associated with a true positive to be a false negative
    groundtruths_associated_with_true_positives = (
        detailed_pr_calc_df[detailed_pr_calc_df["true_positive_flag"]]
        .groupby(["confidence_threshold"], as_index=False)["id_gt"]
        .unique()
    )

    if not groundtruths_associated_with_true_positives.empty:
        confidence_interval_to_groundtruths_associated_with_true_positives_dict = (
            groundtruths_associated_with_true_positives.set_index(
                "confidence_threshold"
            )["id_gt"]
            .apply(set)
            .to_dict()
        )

        mask = pl.Series(False, index=detailed_pr_calc_df.index)

        for (
            threshold,
            elements,
        ) in (
            confidence_interval_to_groundtruths_associated_with_true_positives_dict.items()
        ):
            threshold_mask = (
                detailed_pr_calc_df["confidence_threshold"] == threshold
            )
            membership_mask = detailed_pr_calc_df["id_gt"].isin(elements)
            mask |= threshold_mask & membership_mask

        detailed_pr_calc_df["false_negative_flag"] = ~mask

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
        confidence_interval_to_groundtruths_associated_with_misclassification_fn_dict = (
            groundtruths_associated_with_misclassification_false_negatives.set_index(
                "confidence_threshold"
            )[
                "id_gt"
            ]
            .apply(set)
            .to_dict()
        )

        mask = pl.Series(False, index=detailed_pr_calc_df.index)

        for (
            threshold,
            elements,
        ) in (
            confidence_interval_to_groundtruths_associated_with_misclassification_fn_dict.items()
        ):
            threshold_mask = (
                detailed_pr_calc_df["confidence_threshold"] == threshold
            )
            membership_mask = detailed_pr_calc_df["id_gt"].isin(elements)
            mask |= threshold_mask & membership_mask

        detailed_pr_calc_df["no_predictions_false_negative_flag"] = (
            ~mask & detailed_pr_calc_df["false_negative_flag"]
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


def _create_detailed_joint_df(
    groundtruth_df: pl.DataFrame, prediction_df: pl.DataFrame
):
    """Create the dataframe needed to calculate DetailedPRCurves from a groundtruth and prediction dataframe."""
    detailed_joint_df = pd.merge(
        groundtruth_df,
        prediction_df,
        on=["datum_id", "label_key"],
        how="outer",
        suffixes=("_gt", "_pd"),
    )

    detailed_joint_df["is_label_match"] = (
        detailed_joint_df["label_id_pd"] == detailed_joint_df["label_id_gt"]
    )

    detailed_joint_df = _calculate_iou(joint_df=detailed_joint_df)
    return detailed_joint_df


def create_detection_evaluation_inputs(
    groundtruths: list[schemas.GroundTruth] | pl.DataFrame,
    predictions: list[schemas.Prediction] | pl.DataFrame,
    metrics_to_return: list[enums.MetricType],
    label_map: dict[schemas.Label, schemas.Label],
    convert_annotations_to_type: enums.AnnotationType | None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    """
    Creates and validates the inputs needed to run a detection evaluation.

    Parameters
    ----------
    groundtruths : list[schemas.GroundTruth] | pl.DataFrame
        A list or pandas DataFrame describing the groundtruths.
    predictions : list[schemas.GroundTruth] | pl.DataFrame
        A list or pandas DataFrame describing the predictions.
    metrics_to_return : list[enums.MetricType]
        A list of metrics to calculate during the evaluation.
    label_map : dict[schemas.Label, schemas.Label]
        A mapping from one label schema to another.
    convert_annotations_to_type : AnnotationType, optional
        The target annotation type to convert the data to.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        A tuple of input dataframes.
    """

    groundtruth_df = utilities.create_validated_groundtruth_df(
        groundtruths, task_type=enums.TaskType.OBJECT_DETECTION
    )
    prediction_df = utilities.create_validated_prediction_df(
        predictions, task_type=enums.TaskType.OBJECT_DETECTION
    )

    # filter dataframes based on task type
    groundtruth_df = utilities.filter_dataframe_by_task_type(
        df=groundtruth_df, task_type=enums.TaskType.OBJECT_DETECTION
    )

    if not prediction_df.empty:
        prediction_df = utilities.filter_dataframe_by_task_type(
            df=prediction_df, task_type=enums.TaskType.OBJECT_DETECTION
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

    # apply label map
    groundtruth_df, prediction_df = utilities.replace_labels_using_label_map(
        groundtruth_df=groundtruth_df,
        prediction_df=prediction_df,
        label_map=label_map,
    )

    # add label as a column
    for df in (groundtruth_df, prediction_df):
        df["label"] = list(zip(df["label_key"], df["label_value"]))

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

    if (
        metrics_to_return
        and enums.MetricType.DetailedPrecisionRecallCurve in metrics_to_return
    ):
        detailed_joint_df = _create_detailed_joint_df(
            groundtruth_df=groundtruth_df, prediction_df=prediction_df
        )
    else:
        detailed_joint_df = None

    # remove unnecessary columns to save memory
    groundtruth_df = groundtruth_df.loc[
        :,
        [
            "datum_uid",
            "label_key",
            "annotation_id",
            "label_value",
            "id",
            "label",
        ],
    ]

    prediction_df = prediction_df.loc[
        :,
        [
            "datum_uid",
            "annotation_id",
            "label_key",
            "label_value",
        ],
    ]

    joint_df = joint_df.loc[
        :,
        [
            "label_id",
            "id_gt",
            "label",
            "score",
            "id_pd",
            "iou_",
        ],
    ]

    if detailed_joint_df is not None:
        detailed_joint_df = detailed_joint_df.loc[
            :,
            [
                "datum_uid_gt",
                "label_key",
                "label_value_gt",
                "id_gt",
                "converted_geometry_gt",
                "datum_uid_pd",
                "label_value_pd",
                "score",
                "label_id_pd",
                "id_pd",
                "converted_geometry_pd",
                "is_label_match",
                "iou_",
            ],
        ]

    return groundtruth_df, prediction_df, joint_df, detailed_joint_df


def compute_detection_metrics(
    joint_df: pl.DataFrame,
    detailed_joint_df: pl.DataFrame | None,
    metrics_to_return: list[enums.MetricType],
    iou_thresholds_to_compute: list[float],
    iou_thresholds_to_return: list[float],
    recall_score_threshold: float,
    pr_curve_iou_threshold: float,
    pr_curve_max_examples: int,
) -> list[dict]:
    """
    Compute detection metrics for evaluating object detection models. This function calculates Intersection over Union (IoU) for each ground truth-prediction pair that shares a common grouper id, and computes metrics such as Average Precision (AP), Average Recall (AR), and Precision-Recall (PR) curves.

    Parameters
    ----------
    joint_df : pl.DataFrame
        Dataframe containing merged groundtruths and predictions, joined by label.
    detailed_joint_df : pl.DataFrame
        Dataframe containing merged groundtruths and predictions, joined by label key.
    metrics_to_return : list[enums.MetricType]
        List of metric types to calculate and return, such as AP, AR, or PR curves.
    iou_thresholds_to_compute : list[float]
        List of IoU thresholds for which metrics should be computed.
    iou_thresholds_to_return : list[float]
        List of IoU thresholds for which metrics should be returned.
    recall_score_threshold : float
        Threshold for the recall score to consider in metric calculations.
    pr_curve_iou_threshold : float
        IoU threshold for computing Precision-Recall curves.
    pr_curve_max_examples : int
        Maximum number of examples to use for Precision-Recall curve calculations.

    Returns
    -------
    list[dict]
        A list of dictionaries containing computed metrics, including AP, AR, and PR curves, filtered according to `metrics_to_return`.

    Raises
    ------
    ValueError
        If there is an issue with the data or parameters provided.
    """

    metrics_to_output = []

    dfs = [
        joint_df.with_columns(pl.lit(threshold).alias("iou_threshold"))
        for threshold in iou_thresholds_to_compute
    ]

    calculation_df = pl.concat(dfs).sort(
        by=["label_id", "label", "iou_threshold", "score", "iou_"],
        nulls_last=True,
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
        detailed_pr_joint_df=detailed_joint_df,
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
    groundtruths: pl.DataFrame | list[schemas.GroundTruth],
    predictions: pl.DataFrame | list[schemas.Prediction],
    label_map: dict[schemas.Label, schemas.Label] | None = None,
    metrics_to_return: list[enums.MetricType] | None = None,
    convert_annotations_to_type: enums.AnnotationType | None = None,
    iou_thresholds_to_compute: list[float] | None = None,
    iou_thresholds_to_return: list[float] | None = None,
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
    groundtruths : pl.DataFrame | list[schemas.GroundTruth]
        A list of GroundTruth objects or a pandas DataFrame describing your ground truths.
    predictions : pl.DataFrame | list[schemas.Prediction]
        A list of Prediction objects or a pandas DataFrame describing your predictions.
    label_map : dict[schemas.Label, schemas.Label], optional
        Mapping of ground truth labels to prediction labels.
    metrics_to_return : list[enums.MetricType], optional
        List of metric types to calculate and return.
    convert_annotations_to_type : enums.AnnotationType, optional
        Annotation type to convert all annotations to.
    iou_thresholds_to_compute : list[float], optional
        IoU thresholds for which metrics should be computed.
    iou_thresholds_to_return : list[float], optional
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

    (
        groundtruth_df,
        prediction_df,
        joint_df,
        detailed_joint_df,
    ) = create_detection_evaluation_inputs(
        groundtruths=groundtruths,
        predictions=predictions,
        metrics_to_return=metrics_to_return,
        label_map=label_map,
        convert_annotations_to_type=convert_annotations_to_type,
    )

    # TODO eventually make it so joint_df is natively created as a polars DF, rather than being converted
    joint_df = pl.from_pandas(joint_df)

    # TODO
    joint_df = joint_df.with_columns(
        pl.format(
            "[{}]", pl.col("label").cast(pl.List(pl.Utf8)).list.join(", ")
        ).alias("label")
    )

    if detailed_joint_df is not None:
        detailed_joint_df = pl.from_pandas(detailed_joint_df)
        detailed_joint_df = detailed_joint_df.with_columns(
            pl.format(
                "[{}]", pl.col("label").cast(pl.List(pl.Utf8)).list.join(", ")
            ).alias("label")
        )

    # add the number of groundtruth observations per grouper
    number_of_groundtruths_per_label_df = (
        groundtruth_df.groupby(["label"], as_index=False)["id"]
        .nunique()
        .rename({"id": "gts_per_grouper"}, axis=1)
    )

    # TODO don't use from_pandas anywhere
    number_of_groundtruths_per_label_df = pl.from_pandas(
        number_of_groundtruths_per_label_df
    )
    number_of_groundtruths_per_label_df = (
        number_of_groundtruths_per_label_df.with_columns(
            pl.format(
                "[{}]", pl.col("label").cast(pl.List(pl.Utf8)).list.join(", ")
            ).alias("label")
        )
    )

    joint_df = joint_df.join(
        number_of_groundtruths_per_label_df,
        on=["label"],
        how="outer",
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

    metrics = compute_detection_metrics(
        joint_df=joint_df,
        detailed_joint_df=detailed_joint_df,
        metrics_to_return=metrics_to_return,
        iou_thresholds_to_compute=iou_thresholds_to_compute,
        iou_thresholds_to_return=iou_thresholds_to_return,
        recall_score_threshold=recall_score_threshold,
        pr_curve_iou_threshold=pr_curve_iou_threshold,
        pr_curve_max_examples=pr_curve_max_examples,
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
