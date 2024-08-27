import gc
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from valor_core import enums, metrics, schemas, utilities


def _calculate_confusion_matrix_df(
    merged_groundtruths_and_predictions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[metrics.ConfusionMatrix]]:
    """Calculate our confusion matrix dataframe."""

    cm_counts_df = (
        merged_groundtruths_and_predictions_df[
            ["label_key", "pd_label_value", "gt_label_value"]
        ]
        .groupby(
            ["label_key", "pd_label_value", "gt_label_value"],
            as_index=False,
            dropna=False,
        )
        .size()
    )

    cm_counts_df["true_positive_flag"] = (
        cm_counts_df["pd_label_value"] == cm_counts_df["gt_label_value"]
    )

    # resolve pandas typing error
    if not isinstance(cm_counts_df, pd.DataFrame):
        raise TypeError(
            f"Expected a pd.DataFrame, but got {type(cm_counts_df)}"
        )

    # count of predictions per grouper key
    cm_counts_df = cm_counts_df.merge(
        cm_counts_df.groupby(
            ["label_key", "pd_label_value"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename({"size": "number_of_predictions"}, axis=1),
        on=["label_key", "pd_label_value"],
    )

    # count of groundtruths per grouper key
    cm_counts_df = cm_counts_df.merge(
        cm_counts_df.groupby(
            ["label_key", "gt_label_value"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename({"size": "number_of_groundtruths"}, axis=1),
    )

    cm_counts_df = cm_counts_df.merge(
        cm_counts_df[
            [
                "label_key",
                "pd_label_value",
                "true_positive_flag",
            ]
        ]
        .groupby(
            ["label_key", "pd_label_value"],
            as_index=False,
            dropna=False,
        )
        .sum()
        .rename(
            columns={"true_positive_flag": "true_positives_per_pd_label_value"}
        ),
        on=["label_key", "pd_label_value"],
    )

    cm_counts_df = cm_counts_df.merge(
        cm_counts_df[["label_key", "gt_label_value", "true_positive_flag"]]
        .groupby(
            ["label_key", "gt_label_value"],
            as_index=False,
            dropna=False,
        )
        .sum()
        .rename(
            columns={"true_positive_flag": "true_positives_per_gt_label_value"}
        ),
        on=["label_key", "gt_label_value"],
    )

    cm_counts_df = cm_counts_df.merge(
        cm_counts_df[["label_key", "true_positive_flag"]]
        .groupby("label_key", as_index=False, dropna=False)
        .sum()
        .rename(
            columns={"true_positive_flag": "true_positives_per_label_key"}
        ),
        on="label_key",
    )

    # create ConfusionMatrix objects
    confusion_matrices = []
    for label_key in cm_counts_df.loc[:, "label_key"].unique():
        revelant_rows = cm_counts_df.loc[
            (cm_counts_df["label_key"] == label_key)
            & cm_counts_df["gt_label_value"].notnull()
        ]
        relevant_confusion_matrices = metrics.ConfusionMatrix(
            label_key=label_key,
            entries=[
                metrics.ConfusionMatrixEntry(
                    prediction=row["pd_label_value"],
                    groundtruth=row["gt_label_value"],
                    count=row["size"],
                )
                for row in revelant_rows.to_dict(orient="records")
                if isinstance(row["pd_label_value"], str)
                and isinstance(row["gt_label_value"], str)
            ],
        )
        confusion_matrices.append(relevant_confusion_matrices)

    return cm_counts_df, confusion_matrices


def _calculate_metrics_at_label_value_level(
    cm_counts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate metrics using the confusion matix dataframe."""

    # create base dataframe that's unique at the (grouper key, grouper value level)
    unique_label_values_per_label_key_df = pd.DataFrame(
        np.concatenate(
            [
                cm_counts_df[["label_key", "pd_label_value"]].values,
                cm_counts_df.loc[
                    cm_counts_df["gt_label_value"].notnull(),
                    ["label_key", "gt_label_value"],
                ].values,
            ]
        ),
        columns=[
            "label_key",
            "label_value",
        ],
    ).drop_duplicates()

    # compute metrics using confusion matrices
    metrics_per_label_key_and_label_value_df = (
        unique_label_values_per_label_key_df.assign(
            number_true_positives=lambda df: df.apply(
                lambda chain_df: (
                    cm_counts_df[
                        (
                            cm_counts_df["gt_label_value"]
                            == chain_df["label_value"]
                        )
                        & (cm_counts_df["label_key"] == chain_df["label_key"])
                        & (cm_counts_df["true_positive_flag"])
                    ]["size"].sum()
                ),
                axis=1,
            )
        )
        .assign(
            number_of_groundtruths=unique_label_values_per_label_key_df.apply(
                lambda chain_df: (
                    cm_counts_df[
                        (
                            cm_counts_df["gt_label_value"]
                            == chain_df["label_value"]
                        )
                        & (cm_counts_df["label_key"] == chain_df["label_key"])
                    ]["size"].sum()
                ),
                axis=1,
            )
        )
        .assign(
            number_of_predictions=unique_label_values_per_label_key_df.apply(
                lambda chain_df: (
                    cm_counts_df[
                        (
                            cm_counts_df["pd_label_value"]
                            == chain_df["label_value"]
                        )
                        & (cm_counts_df["label_key"] == chain_df["label_key"])
                    ]["size"].sum()
                ),
                axis=1,
            )
        )
        .assign(
            precision=lambda chain_df: chain_df["number_true_positives"]
            / chain_df["number_of_predictions"]
        )
        .assign(
            recall=lambda chain_df: chain_df["number_true_positives"]
            / chain_df["number_of_groundtruths"]
        )
        .assign(
            f1=lambda chain_df: (
                2 * chain_df["precision"] * chain_df["recall"]
            )
            / (chain_df["precision"] + chain_df["recall"])
        )
    )

    # replace nulls and infinities
    metrics_per_label_key_and_label_value_df[
        ["precision", "recall", "f1"]
    ] = metrics_per_label_key_and_label_value_df.loc[
        :, ["precision", "recall", "f1"]
    ].replace(
        [np.inf, -np.inf, np.nan], 0
    )

    # replace values of labels that only exist in predictions (not groundtruths) with -1
    labels_to_replace = cm_counts_df.loc[
        cm_counts_df["gt_label_value"].isnull(),
        ["label_key", "pd_label_value"],
    ].values.tolist()

    for key, value in labels_to_replace:
        metrics_per_label_key_and_label_value_df.loc[
            (metrics_per_label_key_and_label_value_df["label_key"] == key)
            & (
                metrics_per_label_key_and_label_value_df["label_value"]
                == value
            ),
            ["precision", "recall", "f1"],
        ] = -1

    return metrics_per_label_key_and_label_value_df


def _calculate_precision_recall_f1_metrics(
    metrics_per_label_key_and_label_value_df: pd.DataFrame,
) -> list[metrics.PrecisionMetric | metrics.RecallMetric | metrics.F1Metric]:
    """Calculate Precision, Recall, and F1 metrics."""
    # create metric objects
    output = []

    for row in metrics_per_label_key_and_label_value_df.loc[
        ~metrics_per_label_key_and_label_value_df["label_value"].isnull(),
        ["label_key", "label_value", "precision", "recall", "f1"],
    ].to_dict(orient="records"):
        pydantic_label = schemas.Label(
            key=row["label_key"], value=row["label_value"]
        )

        output += [
            metrics.PrecisionMetric(
                label=pydantic_label,
                value=row["precision"],
            ),
            metrics.RecallMetric(
                label=pydantic_label,
                value=row["recall"],
            ),
            metrics.F1Metric(
                label=pydantic_label,
                value=row["f1"],
            ),
        ]
    return output


def _calculate_accuracy_metrics(
    cm_counts_df: pd.DataFrame,
) -> list[metrics.AccuracyMetric]:
    """Calculate Accuracy metrics."""
    accuracy_calculations = (
        cm_counts_df.loc[
            (
                cm_counts_df["gt_label_value"].notnull()
                & cm_counts_df["true_positive_flag"]
            ),
            ["label_key", "size"],
        ]
        .groupby(["label_key"], as_index=False)
        .sum()
        .rename({"size": "true_positives_per_label_key"}, axis=1)
    ).merge(
        cm_counts_df.loc[
            (cm_counts_df["gt_label_value"].notnull()),
            ["label_key", "size"],
        ]
        .groupby(["label_key"], as_index=False)
        .sum()
        .rename({"size": "observations_per_label_key"}, axis=1),
        on="label_key",
        how="outer",
    )

    accuracy_calculations["accuracy"] = (
        accuracy_calculations["true_positives_per_label_key"]
        / accuracy_calculations["observations_per_label_key"]
    )

    # some elements may be np.nan if a given grouper key has no true positives
    # replace those accuracy scores with 0
    accuracy_calculations["accuracy"] = accuracy_calculations[
        "accuracy"
    ].fillna(value=0)

    return [
        metrics.AccuracyMetric(
            label_key=values["label_key"], value=values["accuracy"]
        )
        for _, values in accuracy_calculations.iterrows()
    ]


def _get_joint_df(
    prediction_df: pd.DataFrame, groundtruth_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge the ground truth and prediction dataframes into one, joint dataframe."""
    max_scores_by_label_key_and_datum_id = (
        prediction_df[["label_key", "datum_id", "score"]]
        .groupby(
            [
                "label_key",
                "datum_id",
            ],
            as_index=False,
        )
        .max()
    )

    # catch pandas typing error
    if not isinstance(prediction_df, pd.DataFrame) or not isinstance(
        max_scores_by_label_key_and_datum_id, pd.DataFrame
    ):
        raise ValueError(
            "prediction_df and max_scores_by_label_key_and_datum_id must be pandas Dataframes."
        )

    best_prediction_id_per_label_key_and_datum_id = (
        pd.merge(
            prediction_df,
            max_scores_by_label_key_and_datum_id,
            on=["label_key", "datum_id", "score"],
            how="inner",
        )[["label_key", "datum_id", "id", "score"]]
        .groupby(["label_key", "datum_id"], as_index=False)
        .min()
        .rename(columns={"score": "best_score"})
    )

    best_prediction_label_for_each_label_key_and_datum = pd.merge(
        prediction_df[["label_key", "label_value", "datum_id", "id"]],
        best_prediction_id_per_label_key_and_datum_id,
        on=["label_key", "datum_id", "id"],
        how="inner",
    )[["label_key", "datum_id", "label_value", "best_score"]]

    # count the number of matches for each (pd_label_value, gt_label_value) for each label_key
    merged_groundtruths_and_predictions_df = pd.merge(
        groundtruth_df[["datum_id", "label_key", "label_value"]].rename(
            columns={"label_value": "gt_label_value"}
        ),
        best_prediction_label_for_each_label_key_and_datum.rename(
            columns={"label_value": "pd_label_value"}
        ),
        on=["datum_id", "label_key"],
        how="left",
    )

    # add back any labels that appear in predictions but not groundtruths
    missing_labels_from_predictions = list(
        set(
            zip(
                [None] * len(prediction_df),
                prediction_df["label_key"],
                [None] * len(prediction_df),
                prediction_df["label_value"],
                [None] * len(prediction_df),
            )
        ).difference(
            set(
                zip(
                    [None] * len(merged_groundtruths_and_predictions_df),
                    merged_groundtruths_and_predictions_df["label_key"],
                    [None] * len(merged_groundtruths_and_predictions_df),
                    merged_groundtruths_and_predictions_df["pd_label_value"],
                    [None] * len(prediction_df),
                )
            ).union(
                set(
                    zip(
                        [None] * len(merged_groundtruths_and_predictions_df),
                        merged_groundtruths_and_predictions_df["label_key"],
                        [None] * len(merged_groundtruths_and_predictions_df),
                        merged_groundtruths_and_predictions_df[
                            "gt_label_value"
                        ],
                        [None] * len(prediction_df),
                    )
                )
            )
        )
    )

    missing_label_df = pd.DataFrame(
        missing_labels_from_predictions,
        columns=merged_groundtruths_and_predictions_df.columns,
    )

    merged_groundtruths_and_predictions_df = (
        merged_groundtruths_and_predictions_df.copy()
        if missing_label_df.empty
        else (
            missing_label_df.copy()
            if merged_groundtruths_and_predictions_df.empty
            else pd.concat(
                [
                    merged_groundtruths_and_predictions_df,
                    missing_label_df,
                ],
                ignore_index=True,
            )
        )
    )

    return merged_groundtruths_and_predictions_df


def _calculate_rocauc(
    prediction_df: pd.DataFrame, groundtruth_df: pd.DataFrame
) -> list[metrics.ROCAUCMetric]:
    """Calculate ROC AUC metrics."""
    # if there are no predictions, then ROCAUC should be 0 for all groundtruth grouper keys
    if prediction_df.empty:
        return [
            metrics.ROCAUCMetric(label_key=label_key, value=float(0))
            for label_key in groundtruth_df["label_key"].unique()
        ]

    merged_predictions_and_groundtruths = (
        prediction_df[["datum_id", "label_key", "label_value", "score"]]
        .merge(
            groundtruth_df[["datum_id", "label_key", "label_value"]].rename(
                columns={
                    "label_value": "gt_label_value",
                }
            ),
            on=["datum_id", "label_key"],
            how="left",
        )
        .assign(
            is_true_positive=lambda chain_df: chain_df["label_value"]
            == chain_df["gt_label_value"],
        )
        .assign(
            is_false_positive=lambda chain_df: chain_df["label_value"]
            != chain_df["gt_label_value"],
        )
    ).sort_values(
        by=["score", "label_key", "gt_label_value"],
        ascending=[False, False, True],
    )

    # count the number of observations (i.e., predictions) and true positives for each grouper key
    total_observations_per_label_key_and_label_value = (
        merged_predictions_and_groundtruths.groupby(
            ["label_key", "label_value"], as_index=False
        )["gt_label_value"]
        .size()
        .rename({"size": "n"}, axis=1)
    )

    total_true_positves_per_label_key_and_label_value = (
        merged_predictions_and_groundtruths.loc[
            merged_predictions_and_groundtruths["is_true_positive"], :
        ]
        .groupby(["label_key", "label_value"], as_index=False)[
            "gt_label_value"
        ]
        .size()
        .rename({"size": "n_true_positives"}, axis=1)
    )

    merged_counts = merged_predictions_and_groundtruths.merge(
        total_observations_per_label_key_and_label_value,
        on=["label_key", "label_value"],
        how="left",
    ).merge(
        total_true_positves_per_label_key_and_label_value,
        on=["label_key", "label_value"],
        how="left",
    )

    cumulative_sums = (
        merged_counts[
            [
                "label_key",
                "label_value",
                "is_true_positive",
                "is_false_positive",
            ]
        ]
        .groupby(["label_key", "label_value"], as_index=False)
        .cumsum()
    ).rename(
        columns={
            "is_true_positive": "cum_true_positive_cnt",
            "is_false_positive": "cum_false_positive_cnt",
        }
    )

    rates = pd.concat([merged_counts, cumulative_sums], axis=1)

    # correct cumulative sums to be the max value for a given datum_id / label_key / label_value (this logic brings pandas' cumsum logic in line with psql's sum().over())
    max_cum_sums = (
        rates.groupby(["label_key", "label_value", "score"], as_index=False)[
            ["cum_true_positive_cnt", "cum_false_positive_cnt"]
        ]
        .max()
        .rename(
            columns={
                "cum_true_positive_cnt": "max_cum_true_positive_cnt",
                "cum_false_positive_cnt": "max_cum_false_positive_cnt",
            }
        )
    )
    rates = rates.merge(max_cum_sums, on=["label_key", "label_value", "score"])
    rates["cum_true_positive_cnt"] = rates[
        ["cum_true_positive_cnt", "max_cum_true_positive_cnt"]
    ].max(axis=1)
    rates["cum_false_positive_cnt"] = rates[
        ["cum_false_positive_cnt", "max_cum_false_positive_cnt"]
    ].max(axis=1)

    # calculate tpr and fpr
    rates = rates.assign(
        tpr=lambda chain_df: chain_df["cum_true_positive_cnt"]
        / chain_df["n_true_positives"]
    ).assign(
        fpr=lambda chain_df: chain_df["cum_false_positive_cnt"]
        / (chain_df["n"] - chain_df["n_true_positives"])
    )

    # sum trapezoidal areas by grouper key and grouper value
    trap_areas_per_label_value = pd.concat(
        [
            rates[
                [
                    "label_key",
                    "label_value",
                    "n",
                    "n_true_positives",
                    "tpr",
                    "fpr",
                ]
            ],
            rates.groupby(["label_key", "label_value"], as_index=False)[
                ["tpr", "fpr"]
            ]
            .shift(1)
            .rename(columns={"tpr": "lagged_tpr", "fpr": "lagged_fpr"}),
        ],
        axis=1,
    ).assign(
        trap_area=lambda chain_df: 0.5
        * (
            (chain_df["tpr"] + chain_df["lagged_tpr"])
            * (chain_df["fpr"] - chain_df["lagged_fpr"])
        )
    )

    summed_trap_areas_per_label_value = trap_areas_per_label_value.groupby(
        ["label_key", "label_value"], as_index=False
    )[["n", "n_true_positives", "trap_area"]].sum(min_count=1)

    # replace values if specific conditions are met
    summed_trap_areas_per_label_value = (
        summed_trap_areas_per_label_value.assign(
            trap_area=lambda chain_df: np.select(
                [
                    chain_df["n_true_positives"].isnull(),
                    ((chain_df["n"] - chain_df["n_true_positives"]) == 0),
                ],
                [1, 1],
                default=chain_df["trap_area"],
            )
        )
    )

    # take the average across grouper keys
    average_across_label_keys = summed_trap_areas_per_label_value.groupby(
        "label_key", as_index=False
    )["trap_area"].mean()

    return [
        metrics.ROCAUCMetric(
            label_key=values["label_key"], value=values["trap_area"]
        )
        for _, values in average_across_label_keys.iterrows()
    ]


def _add_samples_to_dataframe(
    pr_curve_counts_df: pd.DataFrame,
    joint_df: pd.DataFrame,
    max_examples: int,
    flag_column: str,
) -> pd.DataFrame:
    """Efficienctly gather samples for a given flag."""

    samples = []
    sample_df = pd.DataFrame()
    if flag_column == "true_positive_flag":
        for threshold_index in range(1, 20):
            temp = (
                joint_df[
                    joint_df["is_label_match"]
                    & (joint_df["threshold_index"] >= threshold_index)
                ]
                .groupby(["label_key", "label_value"], as_index=False)[
                    "datum_uid"
                ]
                .agg(lambda x: tuple(x.head(max_examples)))
            )
            if not temp.empty:
                temp["threshold_index"] = threshold_index
                samples.append(temp)
    elif flag_column == "misclassification_false_positive_flag":
        for threshold_index in range(1, 20):
            temp = (
                joint_df[
                    ~joint_df["is_label_match"]
                    & (joint_df["threshold_index"] >= threshold_index)
                ]
                .groupby(["label_key", "label_value"], as_index=False)[
                    "datum_uid"
                ]
                .agg(lambda x: tuple(x.head(max_examples)))
            )
            if not temp.empty:
                temp["threshold_index"] = threshold_index
                samples.append(temp)
    elif flag_column == "true_negative_flag":
        for threshold_index in range(1, 20):
            temp = (
                joint_df[
                    ~joint_df["is_label_match"]
                    & (joint_df["threshold_index"] < threshold_index)
                ]
                .groupby(["label_key", "label_value"], as_index=False)[
                    "datum_uid"
                ]
                .agg(lambda x: tuple(x.head(max_examples)))
            )
            if not temp.empty:
                temp["threshold_index"] = threshold_index
                samples.append(temp)
    elif flag_column == "misclassification_false_negative_flag":
        for threshold_index in range(1, 20):
            temp = (
                joint_df[
                    joint_df["is_label_match"]
                    & (joint_df["threshold_index"] < threshold_index)
                    & (threshold_index <= joint_df["threshold_index_max"])
                ]
                .groupby(["label_key", "label_value"], as_index=False)[
                    "datum_uid"
                ]
                .agg(lambda x: tuple(x.head(max_examples)))
            )
            if not temp.empty:
                temp["threshold_index"] = threshold_index
                samples.append(temp)
    elif flag_column == "no_predictions_false_negative_flag":
        for threshold_index in range(1, 20):
            temp = (
                joint_df[
                    joint_df["is_label_match"]
                    & (joint_df["threshold_index_max"] < threshold_index)
                ]
                .groupby(["label_key", "label_value"], as_index=False)[
                    "datum_uid"
                ]
                .agg(lambda x: tuple(x.head(max_examples)))
            )
            if not temp.empty:
                temp["threshold_index"] = threshold_index
                samples.append(temp)

    if len(samples):
        sample_df = pd.concat(samples)

    if not sample_df.empty:
        sample_df[f"{flag_column}_samples"] = sample_df.apply(
            lambda row: set(zip(*row[["datum_uid"]])),  # type: ignore - pandas typing error
            axis=1,
        )
        pr_curve_counts_df = pr_curve_counts_df.merge(
            sample_df[
                [
                    "label_key",
                    "label_value",
                    "threshold_index",
                    f"{flag_column}_samples",
                ]
            ],
            on=["label_key", "label_value", "threshold_index"],
            how="left",
        )
        pr_curve_counts_df[f"{flag_column}_samples"] = pr_curve_counts_df[
            f"{flag_column}_samples"
        ].apply(lambda x: list(x) if isinstance(x, set) else list())

    else:
        pr_curve_counts_df[f"{flag_column}_samples"] = [[]] * len(
            pr_curve_counts_df
        )

    return pr_curve_counts_df


def _calculate_pr_curves(
    prediction_df: pd.DataFrame,
    groundtruth_df: pd.DataFrame,
    metrics_to_return: list,
    pr_curve_max_examples: int,
):
    if not (
        enums.MetricType.PrecisionRecallCurve in metrics_to_return
        or enums.MetricType.DetailedPrecisionRecallCurve in metrics_to_return
    ):
        return []

    joint_df = pd.merge(
        groundtruth_df,
        prediction_df,
        on=["datum_id", "datum_uid", "label_key"],
        how="inner",
        suffixes=("_gt", "_pd"),
    ).loc[
        :,
        [
            "datum_uid",
            "datum_id",
            "label_key",
            "label_value_gt",
            "id_gt",
            "label_value_pd",
            "score",
        ],
    ]

    del groundtruth_df
    del prediction_df
    gc.collect()

    total_label_values_per_label_key = joint_df.groupby(
        ["label_key", "label_value_gt"]
    )["id_gt"].nunique()

    total_datums_per_label_key = total_label_values_per_label_key.groupby(
        "label_key"
    ).sum()

    joint_df["threshold_index"] = ((100 * joint_df["score"]) // 5).astype(
        "uint8"
    )
    joint_df["is_label_match"] = (
        joint_df["label_value_pd"] == joint_df["label_value_gt"]
    )

    joint_df = joint_df.merge(
        joint_df[["datum_id", "datum_uid", "label_key", "threshold_index"]]
        .groupby(["datum_id", "datum_uid", "label_key"])
        .max(),
        on=["datum_id", "datum_uid", "label_key"],
        how="left",
        suffixes=(None, "_max"),
    )

    # Count all true positives at each threshold
    true_positives = (
        joint_df[joint_df["is_label_match"]][
            ["label_key", "label_value_gt", "threshold_index"]
        ]
        .value_counts()
        .reset_index(name="true_positive_count")
        .rename(columns={"label_value_gt": "label_value"})
    )

    # Count all false positives at each threshold
    false_positives = (
        joint_df[~joint_df["is_label_match"]][
            ["label_key", "label_value_pd", "threshold_index"]
        ]
        .value_counts()
        .reset_index(name="false_positive_count")
        .rename(columns={"label_value_pd": "label_value"})
    )

    false_negative_misclassification = (
        joint_df[joint_df["is_label_match"]][
            ["label_key", "label_value_gt", "threshold_index_max"]
        ]
        .value_counts()
        .reset_index(name="false_negative_misclass_count")
        .rename(
            columns={
                "label_value_gt": "label_value",
                "threshold_index_max": "threshold_index",
            }
        )
    )

    pr_curve_counts_df = (
        pd.merge(
            true_positives,
            false_positives,
            on=["label_key", "label_value", "threshold_index"],
            how="outer",
        )
        .merge(
            false_negative_misclassification,
            on=["label_key", "label_value", "threshold_index"],
            how="outer",
        )
        .fillna(0)
    )

    del true_positives
    del false_positives
    gc.collect()

    pr_curve_counts_df = pr_curve_counts_df.merge(
        pr_curve_counts_df[["label_value", "label_key"]]
        .drop_duplicates()
        .merge(pd.DataFrame({"threshold_index": range(21)}), how="cross"),
        on=["label_key", "label_value", "threshold_index"],
        how="outer",
    ).fillna(0)

    # Total number of datums per label key
    total_datums_per_label_key = pr_curve_counts_df["label_key"].map(
        total_datums_per_label_key
    )

    # Total number of datums with groundtruth label per label key
    total_label_values_per_label_key = (
        total_label_values_per_label_key.to_dict(into=defaultdict(lambda: 0))
    )
    total_label_values_per_label_key = pr_curve_counts_df[  # type: ignore - pandas typing error
        ["label_key", "label_value"]
    ].apply(
        lambda row: total_label_values_per_label_key[  # type: ignore - pandas typing error
            (row["label_key"], row["label_value"])
        ],
        axis=1,
    )

    # Compute cumulative sum of true positives to get the total number of true positives at each threshold
    pr_curve_counts_df["true_positives"] = (
        pr_curve_counts_df.loc[::-1, :]
        .groupby(["label_key", "label_value"])["true_positive_count"]
        .transform(pd.Series.cumsum)
    )

    # Compute cumulative sum of false positives to get the total number of true positives at each threshold
    pr_curve_counts_df["false_positives"] = (
        pr_curve_counts_df.loc[::-1, :]
        .groupby(["label_key", "label_value"])["false_positive_count"]
        .transform(pd.Series.cumsum)
    )

    # False negatives = total_ground_truth_datum_with_label - true_positives_with_label
    pr_curve_counts_df["false_negatives"] = (
        total_label_values_per_label_key - pr_curve_counts_df["true_positives"]
    )

    # True negatives = total_predictions_without_correct_label - false_positives_with_label
    pr_curve_counts_df["true_negatives"] = (
        total_datums_per_label_key - total_label_values_per_label_key
    ) - pr_curve_counts_df["false_positives"]

    # Compute Metrics
    pr_curve_counts_df["precision"] = pr_curve_counts_df["true_positives"] / (
        pr_curve_counts_df["true_positives"]
        + pr_curve_counts_df["false_positives"]
    )
    pr_curve_counts_df["recall"] = pr_curve_counts_df["true_positives"] / (
        pr_curve_counts_df["true_positives"]
        + pr_curve_counts_df["false_negatives"]
    )
    pr_curve_counts_df["accuracy"] = (
        pr_curve_counts_df["true_positives"]
        + pr_curve_counts_df["true_negatives"]
    ) / total_datums_per_label_key

    pr_curve_counts_df["f1_score"] = (
        2 * pr_curve_counts_df["precision"] * pr_curve_counts_df["recall"]
    ) / (pr_curve_counts_df["precision"] + pr_curve_counts_df["recall"])

    # any NaNs that are left are from division by zero errors
    pr_curve_counts_df.fillna(-1, inplace=True)

    # add samples to the dataframe for DetailedPrecisionRecallCurves
    if enums.MetricType.DetailedPrecisionRecallCurve in metrics_to_return:
        # Compute total number of no prediction false negatives at each threshold
        pr_curve_counts_df[
            "false_negative_no_predict"
        ] = total_label_values_per_label_key - pr_curve_counts_df.loc[
            ::-1, :
        ].groupby(
            ["label_key", "label_value"]
        )[
            "false_negative_misclass_count"
        ].transform(
            pd.Series.cumsum
        )

        # Misclassification false negatives = total_false_negatives - no_prediction_false_negatives
        pr_curve_counts_df["false_negative_misclassification"] = (
            pr_curve_counts_df["false_negatives"]
            - pr_curve_counts_df["false_negative_no_predict"]
        )

        joint_df = joint_df[
            [
                "datum_uid",
                "label_key",
                "label_value_pd",
                "threshold_index",
                "threshold_index_max",
                "is_label_match",
            ]
        ].rename(
            columns={
                "label_value_pd": "label_value",
            }
        )

        del total_datums_per_label_key
        del total_label_values_per_label_key
        gc.collect()

        for flag in [
            "true_positive_flag",
            "true_negative_flag",
            "misclassification_false_negative_flag",
            "no_predictions_false_negative_flag",
            "misclassification_false_positive_flag",
        ]:
            pr_curve_counts_df = _add_samples_to_dataframe(
                pr_curve_counts_df=pr_curve_counts_df,
                joint_df=joint_df,
                max_examples=pr_curve_max_examples,
                flag_column=flag,
            )

    pr_output = defaultdict(lambda: defaultdict(dict))
    detailed_pr_output = defaultdict(lambda: defaultdict(dict))

    for _, row in pr_curve_counts_df.iterrows():
        if row["threshold_index"] < 1:
            continue
        if row["threshold_index"] > 19:
            continue

        if enums.MetricType.PrecisionRecallCurve in metrics_to_return:
            pr_output[row["label_key"]][row["label_value"]][
                row["threshold_index"] * 5 / 100
            ] = {
                "tp": row["true_positives"],
                "fp": row["false_positives"],
                "fn": row["false_negatives"],
                "tn": row["true_negatives"],
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1_score": row["f1_score"],
            }

        if enums.MetricType.DetailedPrecisionRecallCurve in metrics_to_return:
            detailed_pr_output[row["label_key"]][row["label_value"]][
                row["threshold_index"] * 5 / 100
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
                "tn": {
                    "total": row["true_negatives"],
                    "observations": {
                        "all": {
                            "count": row["true_negatives"],
                            "examples": row["true_negative_flag_samples"],
                        }
                    },
                },
                "fn": {
                    "total": row["false_negatives"],
                    "observations": {
                        "misclassifications": {
                            "count": row["false_negative_misclassification"],
                            "examples": row[
                                "misclassification_false_negative_flag_samples"
                            ],
                        },
                        "no_predictions": {
                            "count": row["false_negative_no_predict"],
                            "examples": row[
                                "no_predictions_false_negative_flag_samples"
                            ],
                        },
                    },
                },
                "fp": {
                    "total": row["false_positives"],
                    "observations": {
                        "misclassifications": {
                            "count": row["false_positives"],
                            "examples": row[
                                "misclassification_false_positive_flag_samples"
                            ],
                        },
                    },
                },
            }

    output = []

    if enums.MetricType.PrecisionRecallCurve in metrics_to_return:
        output += [
            metrics.PrecisionRecallCurve(
                label_key=key, value=dict(value), pr_curve_iou_threshold=None
            )
            for key, value in pr_output.items()
        ]

    if enums.MetricType.DetailedPrecisionRecallCurve in metrics_to_return:
        output += [
            metrics.DetailedPrecisionRecallCurve(
                label_key=key, value=dict(value), pr_curve_iou_threshold=None
            )
            for key, value in detailed_pr_output.items()
        ]

    return output


def _compute_clf_metrics(
    groundtruth_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    metrics_to_return: list[enums.MetricType] | None = None,
    pr_curve_max_examples: int = 1,
) -> tuple[list[dict], list[dict]]:
    """
    Compute classification metrics including confusion matrices and various performance metrics.

    Parameters
    ----------
    groundtruth_df : pd.DataFrame
        DataFrame containing ground truth annotations with necessary columns.
    prediction_df : pd.DataFrame
        DataFrame containing predictions with necessary columns.
    metrics_to_return : list[enums.MetricType], optional
        list of metric types to return. If None, default metrics are used.
    pr_curve_max_examples : int
        Maximum number of examples to use for Precision-Recall curve calculations.

    Returns
    -------
    tuple[list[dict], list[dict]]
        A tuple where:
        - The first element is a list of dictionaries representing confusion matrices.
        - The second element is a list of dictionaries representing the requested classification metrics.
    """

    # add label as a column
    for df in (groundtruth_df, prediction_df):
        df.loc[:, "label"] = df.apply(
            lambda chain_df: (chain_df["label_key"], chain_df["label_value"]),
            axis=1,
        )

    confusion_matrices, metrics_to_output = [], []

    merged_groundtruths_and_predictions_df = _get_joint_df(
        prediction_df=prediction_df, groundtruth_df=groundtruth_df
    )

    cm_counts_df, confusion_matrices = _calculate_confusion_matrix_df(
        merged_groundtruths_and_predictions_df=merged_groundtruths_and_predictions_df
    )

    metrics_per_label_key_and_label_value_df = (
        _calculate_metrics_at_label_value_level(cm_counts_df=cm_counts_df)
    )

    metrics_to_output += _calculate_precision_recall_f1_metrics(
        metrics_per_label_key_and_label_value_df=metrics_per_label_key_and_label_value_df
    )

    metrics_to_output += _calculate_accuracy_metrics(cm_counts_df=cm_counts_df)

    metrics_to_output += _calculate_rocauc(
        prediction_df=prediction_df, groundtruth_df=groundtruth_df
    )

    # handle type error
    if not metrics_to_return:
        raise ValueError("metrics_to_return must be defined.")

    metrics_to_output += _calculate_pr_curves(
        prediction_df=prediction_df,
        groundtruth_df=groundtruth_df,
        metrics_to_return=metrics_to_return,
        pr_curve_max_examples=pr_curve_max_examples,
    )

    # convert objects to dictionaries and only return what was asked for
    metrics_to_output = [
        m.to_dict()
        for m in metrics_to_output
        if m.to_dict()["type"] in metrics_to_return
    ]
    confusion_matrices = [cm.to_dict() for cm in confusion_matrices]

    return confusion_matrices, metrics_to_output


def evaluate_classification(
    groundtruths: pd.DataFrame | list[schemas.GroundTruth],
    predictions: pd.DataFrame | list[schemas.Prediction],
    label_map: dict[schemas.Label, schemas.Label] | None = None,
    metrics_to_return: list[enums.MetricType] | None = None,
    pr_curve_max_examples: int = 1,
) -> schemas.Evaluation:
    """
    Evaluate an object detection task using some set of groundtruths and predictions.

    The groundtruths and predictions can be inputted as a pandas DataFrame or as a list of GroundTruth/Prediction objects. When passing a dataframe of groundtruths / predictions, the dataframe should contain the following columns:
    - datum_uid (str): The unique identifier for the datum.
    - datum_id (int): A hashed identifier that's unique to each datum.
    - datum_metadata (dict): Metadata associated with the datum.
    - annotation_id (int): A hashed identifier for each unique (datum_uid, annotation) combination.
    - annotation_metadata (dict): Metadata associated with the annotation.
    - is_instance (bool): A boolean indicating whether the annotation is an instance segjmentation (True) or not (False).
    - label_key (str): The key associated with the label.
    - label_value (str): The value associated with the label.
    - score (float): The confidence score of the prediction. Should be bound between 0 and 1. Should only be included for prediction dataframes.
    - label_id (int): A hashed identifier for each unique label.
    - id (str): A unique identifier for the combination of datum, annotation, and label, created by concatenating the indices of these components.

    Parameters
    ----------
    groundtruths : pd.DataFrame | list[schemas.GroundTruth]
        Ground truth annotations as either a DataFrame or a list of GroundTruth objects.
    predictions : pd.DataFrame | list[schemas.Prediction]
        Predictions as either a DataFrame or a list of Prediction objects.
    label_map : dict[schemas.Label, schemas.Label], optional
        Optional dictionary mapping ground truth labels to prediction labels.
    metrics_to_return : list[enums.MetricType], optional
        List of metric types to return. Defaults to Precision, Recall, F1, Accuracy, ROCAUC if None.
    pr_curve_max_examples : int, default=1
        Maximum number of examples to use for Precision-Recall curve calculations.

    Returns
    -------
    schemas.Evaluation
        An Evaluation object containing:
        - parameters: EvaluationParameters used for the calculation.
        - metrics: List of dictionaries representing the calculated classification metrics.
        - confusion_matrices: List of dictionaries representing the confusion matrices.
        - meta: Dictionary with metadata including the count of labels, datums, annotations, and duration of the evaluation.
        - ignored_pred_labels: List of ignored prediction labels (empty in this context).
        - missing_pred_labels: List of missing prediction labels (empty in this context).
    """
    start_time = time.time()

    if not label_map:
        label_map = {}

    if metrics_to_return is None:
        metrics_to_return = [
            enums.MetricType.Precision,
            enums.MetricType.Recall,
            enums.MetricType.F1,
            enums.MetricType.Accuracy,
            enums.MetricType.ROCAUC,
        ]

    utilities.validate_label_map(label_map=label_map)
    utilities.validate_metrics_to_return(
        metrics_to_return=metrics_to_return,
        task_type=enums.TaskType.CLASSIFICATION,
    )
    utilities.validate_parameters(pr_curve_max_examples=pr_curve_max_examples)

    groundtruth_df = utilities.create_validated_groundtruth_df(
        groundtruths, task_type=enums.TaskType.CLASSIFICATION
    )
    prediction_df = utilities.create_validated_prediction_df(
        predictions, task_type=enums.TaskType.CLASSIFICATION
    )

    # filter dataframes based on task type
    groundtruth_df = utilities.filter_dataframe_by_task_type(
        df=groundtruth_df, task_type=enums.TaskType.CLASSIFICATION
    )

    if not prediction_df.empty:
        prediction_df = utilities.filter_dataframe_by_task_type(
            df=prediction_df, task_type=enums.TaskType.CLASSIFICATION
        )

    # drop intermediary columns that are no longer needed
    groundtruth_df = groundtruth_df.loc[
        :,
        [
            "datum_uid",
            "datum_id",
            "annotation_id",
            "label_key",
            "label_value",
            "label_id",
            "id",
        ],
    ]

    prediction_df = prediction_df.loc[
        :,
        [
            "datum_uid",
            "datum_id",
            "annotation_id",
            "label_key",
            "label_value",
            "score",
            "label_id",
            "id",
        ],
    ]

    utilities.validate_matching_label_keys(
        groundtruths=groundtruth_df,
        predictions=prediction_df,
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

    groundtruth_df, prediction_df = utilities.replace_labels_using_label_map(
        groundtruth_df=groundtruth_df,
        prediction_df=prediction_df,
        label_map=label_map,
    )

    confusion_matrices, metrics = _compute_clf_metrics(
        groundtruth_df=groundtruth_df,
        prediction_df=prediction_df,
        metrics_to_return=metrics_to_return,
        pr_curve_max_examples=pr_curve_max_examples,
    )

    return schemas.Evaluation(
        parameters=schemas.EvaluationParameters(
            metrics_to_return=metrics_to_return,
            label_map=label_map,
            pr_curve_max_examples=pr_curve_max_examples,
        ),
        metrics=metrics,
        confusion_matrices=confusion_matrices,
        meta={
            "labels": len(unique_labels),
            "datums": unique_datums_cnt,
            "annotations": unique_annotations_cnt,
            "duration": time.time() - start_time,
        },
        ignored_pred_labels=[],
        missing_pred_labels=[],
    )
