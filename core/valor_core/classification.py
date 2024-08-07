import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from valor_core import enums, metrics, schemas, utilities


def _create_classification_grouper_mappings(
    label_map: Optional[schemas.LabelMapType],
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
    label_value_to_grouper_value = dict()
    label_to_grouper_key_mapping = dict()

    for label_key, label_value in labels:
        # the grouper should equal the (label.key, label.value) if it wasn't mapped by the user
        grouper_key, grouper_value = mapping_dict.get(
            (label_key, label_value), (label_key, label_value)
        )

        label_value_to_grouper_value[label_value] = grouper_value
        label_to_grouper_key_mapping[(label_key, label_value)] = grouper_key

    return {
        "label_value_to_grouper_value": label_value_to_grouper_value,
        "label_to_grouper_key_mapping": label_to_grouper_key_mapping,
    }


def _add_columns_to_groundtruth_and_prediction_table(
    df: pd.DataFrame, grouper_mappings: dict
) -> None:
    """Add label, grouper_key, and grouper_value columns to a particular dataframe. Modifies the dataframe in place."""

    df["label"] = df.apply(
        lambda chain_df: (chain_df["label_key"], chain_df["label_value"]),
        axis=1,
    )
    df["grouper_key"] = df["label"].map(
        grouper_mappings["label_to_grouper_key_mapping"]
    )
    df["grouper_value"] = df["label_value"].map(
        grouper_mappings["label_value_to_grouper_value"]
    )


def _calculate_confusion_matrix_df(
    merged_groundtruths_and_predictions_df: pd.DataFrame,
) -> tuple:
    """Calculate our confusion matrix dataframe."""

    cm_counts_df = (
        merged_groundtruths_and_predictions_df[
            ["grouper_key", "pd_grouper_value", "gt_grouper_value"]
        ]
        .groupby(
            ["grouper_key", "pd_grouper_value", "gt_grouper_value"],
            as_index=False,
            dropna=False,
        )
        .size()
    )

    cm_counts_df["true_positive_flag"] = (
        cm_counts_df["pd_grouper_value"] == cm_counts_df["gt_grouper_value"]
    )

    # resolve pandas typing error
    assert isinstance(cm_counts_df, pd.DataFrame)

    # count of predictions per grouper key
    cm_counts_df = cm_counts_df.merge(
        cm_counts_df.groupby(
            ["grouper_key", "pd_grouper_value"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename({"size": "number_of_predictions"}, axis=1),
        on=["grouper_key", "pd_grouper_value"],
    )

    # count of groundtruths per grouper key
    cm_counts_df = cm_counts_df.merge(
        cm_counts_df.groupby(
            ["grouper_key", "gt_grouper_value"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename({"size": "number_of_groundtruths"}, axis=1),
    )

    cm_counts_df = cm_counts_df.merge(
        cm_counts_df[
            [
                "grouper_key",
                "pd_grouper_value",
                "true_positive_flag",
            ]
        ]
        .groupby(
            ["grouper_key", "pd_grouper_value"],
            as_index=False,
            dropna=False,
        )
        .sum()
        .rename(
            columns={
                "true_positive_flag": "true_positives_per_pd_grouper_value"
            }
        ),
        on=["grouper_key", "pd_grouper_value"],
    )

    cm_counts_df = cm_counts_df.merge(
        cm_counts_df[["grouper_key", "gt_grouper_value", "true_positive_flag"]]
        .groupby(
            ["grouper_key", "gt_grouper_value"],
            as_index=False,
            dropna=False,
        )
        .sum()
        .rename(
            columns={
                "true_positive_flag": "true_positives_per_gt_grouper_value"
            }
        ),
        on=["grouper_key", "gt_grouper_value"],
    )

    cm_counts_df = cm_counts_df.merge(
        cm_counts_df[["grouper_key", "true_positive_flag"]]
        .groupby("grouper_key", as_index=False, dropna=False)
        .sum()
        .rename(
            columns={"true_positive_flag": "true_positives_per_grouper_key"}
        ),
        on="grouper_key",
    )

    # create ConfusionMatrix objects
    confusion_matrices = []
    for grouper_key in cm_counts_df.loc[:, "grouper_key"].unique():
        revelant_rows = cm_counts_df.loc[
            (cm_counts_df["grouper_key"] == grouper_key)
            & cm_counts_df["gt_grouper_value"].notnull()
        ]
        relevant_confusion_matrices = metrics.ConfusionMatrix(
            label_key=grouper_key,
            entries=[
                metrics.ConfusionMatrixEntry(
                    prediction=row["pd_grouper_value"],
                    groundtruth=row["gt_grouper_value"],
                    count=row["size"],
                )
                for row in revelant_rows.to_dict(orient="records")
                if isinstance(row["pd_grouper_value"], str)
                and isinstance(row["gt_grouper_value"], str)
            ],
        )
        confusion_matrices.append(relevant_confusion_matrices)

    return cm_counts_df, confusion_matrices


def _calculate_metrics_at_grouper_value_level(
    cm_counts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate metrics using the confusion matix dataframe."""

    # create base dataframe that's unique at the (grouper key, grouper value level)
    unique_grouper_values_per_grouper_key_df = pd.DataFrame(
        np.concatenate(
            [
                cm_counts_df[["grouper_key", "pd_grouper_value"]].values,
                cm_counts_df.loc[
                    cm_counts_df["gt_grouper_value"].notnull(),
                    ["grouper_key", "gt_grouper_value"],
                ].values,
            ]
        ),
        columns=[
            "grouper_key",
            "grouper_value",
        ],
    ).drop_duplicates()

    # compute metrics using confusion matrices
    metrics_per_grouper_key_and_grouper_value_df = (
        unique_grouper_values_per_grouper_key_df.assign(
            number_true_positives=lambda df: df.apply(
                lambda chain_df: (
                    cm_counts_df[
                        (
                            cm_counts_df["gt_grouper_value"]
                            == chain_df["grouper_value"]
                        )
                        & (
                            cm_counts_df["grouper_key"]
                            == chain_df["grouper_key"]
                        )
                        & (cm_counts_df["true_positive_flag"])
                    ]["size"].sum()
                ),
                axis=1,
            )
        )
        .assign(
            number_of_groundtruths=unique_grouper_values_per_grouper_key_df.apply(
                lambda chain_df: (
                    cm_counts_df[
                        (
                            cm_counts_df["gt_grouper_value"]
                            == chain_df["grouper_value"]
                        )
                        & (
                            cm_counts_df["grouper_key"]
                            == chain_df["grouper_key"]
                        )
                    ]["size"].sum()
                ),
                axis=1,
            )
        )
        .assign(
            number_of_predictions=unique_grouper_values_per_grouper_key_df.apply(
                lambda chain_df: (
                    cm_counts_df[
                        (
                            cm_counts_df["pd_grouper_value"]
                            == chain_df["grouper_value"]
                        )
                        & (
                            cm_counts_df["grouper_key"]
                            == chain_df["grouper_key"]
                        )
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
    metrics_per_grouper_key_and_grouper_value_df[
        ["precision", "recall", "f1"]
    ] = metrics_per_grouper_key_and_grouper_value_df.loc[
        :, ["precision", "recall", "f1"]
    ].replace(
        [np.inf, -np.inf, np.nan], 0
    )

    # replace values of labels that only exist in predictions (not groundtruths) with -1
    labels_to_replace = cm_counts_df.loc[
        cm_counts_df["gt_grouper_value"].isnull(),
        ["grouper_key", "pd_grouper_value"],
    ].values.tolist()

    for key, value in labels_to_replace:
        metrics_per_grouper_key_and_grouper_value_df.loc[
            (
                metrics_per_grouper_key_and_grouper_value_df["grouper_key"]
                == key
            )
            & (
                metrics_per_grouper_key_and_grouper_value_df["grouper_value"]
                == value
            ),
            ["precision", "recall", "f1"],
        ] = -1

    return metrics_per_grouper_key_and_grouper_value_df


def _calculate_precision_recall_f1_metrics(
    metrics_per_grouper_key_and_grouper_value_df: pd.DataFrame,
) -> List[
    Union[metrics.PrecisionMetric, metrics.RecallMetric, metrics.F1Metric]
]:
    # create metric objects
    output = []

    for row in metrics_per_grouper_key_and_grouper_value_df.loc[
        ~metrics_per_grouper_key_and_grouper_value_df[
            "grouper_value"
        ].isnull(),
        ["grouper_key", "grouper_value", "precision", "recall", "f1"],
    ].to_dict(orient="records"):
        pydantic_label = schemas.Label(
            key=row["grouper_key"], value=row["grouper_value"]
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
) -> List[metrics.AccuracyMetric]:

    accuracy_calculations = (
        cm_counts_df.loc[
            (
                cm_counts_df["gt_grouper_value"].notnull()
                & cm_counts_df["true_positive_flag"]
            ),
            ["grouper_key", "size"],
        ]
        .groupby(["grouper_key"], as_index=False)
        .sum()
        .rename({"size": "true_positives_per_grouper_key"}, axis=1)
    ).merge(
        cm_counts_df.loc[
            (cm_counts_df["gt_grouper_value"].notnull()),
            ["grouper_key", "size"],
        ]
        .groupby(["grouper_key"], as_index=False)
        .sum()
        .rename({"size": "observations_per_grouper_key"}, axis=1),
        on="grouper_key",
        how="outer",
    )

    accuracy_calculations["accuracy"] = (
        accuracy_calculations["true_positives_per_grouper_key"]
        / accuracy_calculations["observations_per_grouper_key"]
    )

    # some elements may be np.nan if a given grouper key has no true positives
    # replace those accuracy scores with 0
    accuracy_calculations["accuracy"] = accuracy_calculations[
        "accuracy"
    ].fillna(value=0)

    return [
        metrics.AccuracyMetric(
            label_key=values["grouper_key"], value=values["accuracy"]
        )
        for _, values in accuracy_calculations.iterrows()
    ]


def _get_merged_dataframe(
    prediction_df: pd.DataFrame, groundtruth_df: pd.DataFrame
):
    max_scores_by_grouper_key_and_datum_id = (
        prediction_df[["grouper_key", "datum_id", "score"]]
        .groupby(
            [
                "grouper_key",
                "datum_id",
            ],
            as_index=False,
        )
        .max()
    )

    # catch pandas typing error
    if not isinstance(prediction_df, pd.DataFrame) or not isinstance(
        max_scores_by_grouper_key_and_datum_id, pd.DataFrame
    ):
        raise ValueError

    best_prediction_id_per_grouper_key_and_datum_id = (
        pd.merge(
            prediction_df,
            max_scores_by_grouper_key_and_datum_id,
            on=["grouper_key", "datum_id", "score"],
            how="inner",
        )[["grouper_key", "datum_id", "id", "score"]]
        .groupby(["grouper_key", "datum_id"], as_index=False)
        .min()
        .rename(columns={"score": "best_score"})
    )

    best_prediction_label_for_each_grouper_key_and_datum = pd.merge(
        prediction_df[["grouper_key", "grouper_value", "datum_id", "id"]],
        best_prediction_id_per_grouper_key_and_datum_id,
        on=["grouper_key", "datum_id", "id"],
        how="inner",
    )[["grouper_key", "datum_id", "grouper_value", "best_score"]]

    # count the number of matches for each (pd_label_value, gt_label_value) for each grouper_key
    merged_groundtruths_and_predictions_df = pd.merge(
        groundtruth_df[["datum_id", "grouper_key", "grouper_value"]].rename(
            columns={"grouper_value": "gt_grouper_value"}
        ),
        best_prediction_label_for_each_grouper_key_and_datum.rename(
            columns={"grouper_value": "pd_grouper_value"}
        ),
        on=["datum_id", "grouper_key"],
        how="left",
    )

    # add back any labels that appear in predictions but not groundtruths
    missing_grouper_labels_from_predictions = list(
        set(
            zip(
                [None] * len(prediction_df),
                prediction_df["grouper_key"],
                [None] * len(prediction_df),
                prediction_df["grouper_value"],
                [None] * len(prediction_df),
            )
        ).difference(
            set(
                zip(
                    [None] * len(merged_groundtruths_and_predictions_df),
                    merged_groundtruths_and_predictions_df["grouper_key"],
                    [None] * len(merged_groundtruths_and_predictions_df),
                    merged_groundtruths_and_predictions_df["pd_grouper_value"],
                    [None] * len(prediction_df),
                )
            ).union(
                set(
                    zip(
                        [None] * len(merged_groundtruths_and_predictions_df),
                        merged_groundtruths_and_predictions_df["grouper_key"],
                        [None] * len(merged_groundtruths_and_predictions_df),
                        merged_groundtruths_and_predictions_df[
                            "gt_grouper_value"
                        ],
                        [None] * len(prediction_df),
                    )
                )
            )
        )
    )

    missing_label_df = pd.DataFrame(
        missing_grouper_labels_from_predictions,
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
) -> List[metrics.ROCAUCMetric]:

    # if there are no predictions, then ROCAUC should be 0 for all groundtruth grouper keys
    if prediction_df.empty:
        return [
            metrics.ROCAUCMetric(label_key=grouper_key, value=float(0))
            for grouper_key in groundtruth_df["grouper_key"].unique()
        ]

    merged_predictions_and_groundtruths = (
        prediction_df[["datum_id", "grouper_key", "grouper_value", "score"]]
        .merge(
            groundtruth_df[
                ["datum_id", "grouper_key", "grouper_value"]
            ].rename(
                columns={
                    "grouper_value": "gt_grouper_value",
                }
            ),
            on=["datum_id", "grouper_key"],
            how="left",
        )
        .assign(
            is_true_positive=lambda chain_df: chain_df["grouper_value"]
            == chain_df["gt_grouper_value"],
        )
        .assign(
            is_false_positive=lambda chain_df: chain_df["grouper_value"]
            != chain_df["gt_grouper_value"],
        )
    ).sort_values(
        by=["score", "grouper_key", "gt_grouper_value"],
        ascending=[False, False, True],
    )

    # count the number of observations (i.e., predictions) and true positives for each grouper key
    total_observations_per_grouper_key_and_grouper_value = (
        merged_predictions_and_groundtruths.groupby(
            ["grouper_key", "grouper_value"], as_index=False
        )["gt_grouper_value"]
        .size()
        .rename({"size": "n"}, axis=1)
    )

    total_true_positves_per_grouper_key_and_grouper_value = (
        merged_predictions_and_groundtruths.loc[
            merged_predictions_and_groundtruths["is_true_positive"], :
        ]
        .groupby(["grouper_key", "grouper_value"], as_index=False)[
            "gt_grouper_value"
        ]
        .size()
        .rename({"size": "n_true_positives"}, axis=1)
    )

    merged_counts = merged_predictions_and_groundtruths.merge(
        total_observations_per_grouper_key_and_grouper_value,
        on=["grouper_key", "grouper_value"],
        how="left",
    ).merge(
        total_true_positves_per_grouper_key_and_grouper_value,
        on=["grouper_key", "grouper_value"],
        how="left",
    )

    cumulative_sums = (
        merged_counts[
            [
                "grouper_key",
                "grouper_value",
                "is_true_positive",
                "is_false_positive",
            ]
        ]
        .groupby(["grouper_key", "grouper_value"], as_index=False)
        .cumsum()
    ).rename(
        columns={
            "is_true_positive": "cum_true_positive_cnt",
            "is_false_positive": "cum_false_positive_cnt",
        }
    )

    rates = pd.concat([merged_counts, cumulative_sums], axis=1)

    # correct cumulative sums to be the max value for a given datum_id / grouper_key / grouper_value (this logic brings pandas' cumsum logic in line with psql's sum().over())
    max_cum_sums = (
        rates.groupby(
            ["grouper_key", "grouper_value", "score"], as_index=False
        )[["cum_true_positive_cnt", "cum_false_positive_cnt"]]
        .max()
        .rename(
            columns={
                "cum_true_positive_cnt": "max_cum_true_positive_cnt",
                "cum_false_positive_cnt": "max_cum_false_positive_cnt",
            }
        )
    )
    rates = rates.merge(
        max_cum_sums, on=["grouper_key", "grouper_value", "score"]
    )
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
    trap_areas_per_grouper_value = pd.concat(
        [
            rates[
                [
                    "grouper_key",
                    "grouper_value",
                    "n",
                    "n_true_positives",
                    "tpr",
                    "fpr",
                ]
            ],
            rates.groupby(["grouper_key", "grouper_value"], as_index=False)[
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

    summed_trap_areas_per_grouper_value = trap_areas_per_grouper_value.groupby(
        ["grouper_key", "grouper_value"], as_index=False
    )[["n", "n_true_positives", "trap_area"]].sum(min_count=1)

    # replace values if specific conditions are met
    summed_trap_areas_per_grouper_value = (
        summed_trap_areas_per_grouper_value.assign(
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
    average_across_grouper_keys = summed_trap_areas_per_grouper_value.groupby(
        "grouper_key", as_index=False
    )["trap_area"].mean()

    return [
        metrics.ROCAUCMetric(
            label_key=values["grouper_key"], value=values["trap_area"]
        )
        for _, values in average_across_grouper_keys.iterrows()
    ]


def _add_samples_to_dataframe(
    pr_curve_counts_df: pd.DataFrame,
    pr_calc_df: pd.DataFrame,
    max_examples: int,
    flag_column: str,
):
    """Efficienctly gather samples for a given flag."""

    sample_df = pd.concat(
        [
            pr_calc_df[pr_calc_df[flag_column]]
            .groupby(
                [
                    "grouper_key",
                    "grouper_value_gt",
                    "confidence_threshold",
                ],
                as_index=False,
            )[["datum_uid"]]
            .agg(lambda x: tuple(x.head(max_examples)))
            .rename(columns={"grouper_value_gt": "grouper_value"}),
            pr_calc_df[pr_calc_df[flag_column]]
            .groupby(
                [
                    "grouper_key",
                    "grouper_value_pd",
                    "confidence_threshold",
                ],
                as_index=False,
            )[["datum_uid"]]
            .agg(lambda x: tuple(x.head(max_examples)))
            .rename(columns={"grouper_value_pd": "grouper_value"}),
        ],
        axis=0,
    ).drop_duplicates()

    if not sample_df.empty:
        sample_df[f"{flag_column}_samples"] = sample_df.apply(
            lambda row: set(zip(*row[["datum_uid"]])),  # type: ignore - pandas typing error
            axis=1,
        )

        pr_curve_counts_df = pr_curve_counts_df.merge(
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
        pr_curve_counts_df[f"{flag_column}_samples"] = pr_curve_counts_df[
            f"{flag_column}_samples"
        ].apply(lambda x: list(x) if isinstance(x, set) else list())

    else:
        pr_curve_counts_df[f"{flag_column}_samples"] = [
            list() for _ in range(len(pr_curve_counts_df))
        ]

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

    joint_df = (
        pd.merge(
            groundtruth_df,
            prediction_df,
            on=["datum_id", "datum_uid", "grouper_key"],
            how="outer",
            suffixes=("_gt", "_pd"),
        )
        .assign(
            is_label_match=lambda chain_df: (
                (chain_df["grouper_value_pd"] == chain_df["grouper_value_gt"])
            )
        )
        .drop(
            columns=[
                "annotation_id_gt",
                "created_at_gt",
                "label_key_gt",
                "label_value_gt",
                "label_gt",
                "label_id_gt",
                "annotation_id_pd",
                "created_at_pd",
                "label_key_pd",
                "label_value_pd",
                "label_pd",
                "label_id_pd",
            ],
            errors="ignore",
        )
    )

    # add confidence_threshold to the dataframe and sort
    pr_calc_df = pd.concat(
        [
            joint_df.assign(confidence_threshold=threshold)
            for threshold in [x / 100 for x in range(5, 100, 5)]
        ],
        ignore_index=True,
    ).sort_values(
        by=[
            "grouper_key",
            "grouper_value_pd",
            "confidence_threshold",
            "score",
        ],
        ascending=False,
    )

    # create flags where the predictions meet criteria
    pr_calc_df["true_positive_flag"] = (
        pr_calc_df["score"] >= pr_calc_df["confidence_threshold"]
    ) & pr_calc_df["is_label_match"]

    # for all the false positives, we consider them to be a misclassification if they share a key but not a value with a gt
    pr_calc_df["misclassification_false_positive_flag"] = (
        pr_calc_df["score"] >= pr_calc_df["confidence_threshold"]
    ) & ~pr_calc_df["is_label_match"]

    # next, we flag false negatives by declaring any groundtruth that isn't associated with a true positive to be a false negative
    groundtruths_associated_with_true_positives = (
        pr_calc_df[pr_calc_df["true_positive_flag"]]
        .groupby(["confidence_threshold"], as_index=False)["id_gt"]
        .unique()
    )

    if not groundtruths_associated_with_true_positives.empty:
        groundtruths_associated_with_true_positives.columns = [
            "confidence_threshold",
            "groundtruths_associated_with_true_positives",
        ]
        pr_calc_df = pr_calc_df.merge(
            groundtruths_associated_with_true_positives,
            on=["confidence_threshold"],
            how="left",
        )
        true_positive_sets = pr_calc_df[
            "groundtruths_associated_with_true_positives"
        ].apply(lambda x: set(x) if isinstance(x, np.ndarray) else set())

        pr_calc_df["false_negative_flag"] = np.array(
            [
                id_gt not in true_positive_sets[i]
                for i, id_gt in enumerate(pr_calc_df["id_gt"].values)
            ]
        )

    else:
        pr_calc_df["false_negative_flag"] = False

    # it's a misclassification if there is a corresponding misclassification false positive
    pr_calc_df["misclassification_false_negative_flag"] = (
        pr_calc_df["misclassification_false_positive_flag"]
        & pr_calc_df["false_negative_flag"]
    )

    # assign all id_gts that aren't misclassifications but are false negatives to be no_predictions
    groundtruths_associated_with_misclassification_false_negatives = (
        pr_calc_df[pr_calc_df["misclassification_false_negative_flag"]]
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
        pr_calc_df = pr_calc_df.merge(
            groundtruths_associated_with_misclassification_false_negatives,
            on=["confidence_threshold"],
            how="left",
        )
        misclassification_sets = (
            pr_calc_df[
                "groundtruths_associated_with_misclassification_false_negatives"
            ]
            .apply(lambda x: set(x) if isinstance(x, np.ndarray) else set())
            .values
        )
        pr_calc_df["no_predictions_false_negative_flag"] = (
            np.array(
                [
                    id_gt not in misclassification_sets[i]
                    for i, id_gt in enumerate(pr_calc_df["id_gt"].values)
                ]
            )
            & pr_calc_df["false_negative_flag"]
        )

    else:
        pr_calc_df["no_predictions_false_negative_flag"] = pr_calc_df[
            "false_negative_flag"
        ]

    # true negatives are any rows which don't have another flag
    pr_calc_df["true_negative_flag"] = (
        ~pr_calc_df["true_positive_flag"]
        & ~pr_calc_df["false_negative_flag"]
        & ~pr_calc_df["misclassification_false_positive_flag"]
    )

    # next, we sum up the occurences of each classification and merge them together into one dataframe
    true_positives = (
        pr_calc_df[pr_calc_df["true_positive_flag"]]
        .groupby(["grouper_key", "grouper_value_pd", "confidence_threshold"])[
            "id_pd"
        ]
        .nunique()
    )
    true_positives.name = "true_positives"

    misclassification_false_positives = (
        pr_calc_df[pr_calc_df["misclassification_false_positive_flag"]]
        .groupby(["grouper_key", "grouper_value_pd", "confidence_threshold"])[
            "id_pd"
        ]
        .nunique()
    )
    misclassification_false_positives.name = (
        "misclassification_false_positives"
    )

    misclassification_false_negatives = (
        pr_calc_df[pr_calc_df["misclassification_false_negative_flag"]]
        .groupby(["grouper_key", "grouper_value_gt", "confidence_threshold"])[
            "id_gt"
        ]
        .nunique()
    )
    misclassification_false_negatives.name = (
        "misclassification_false_negatives"
    )

    no_predictions_false_negatives = (
        pr_calc_df[pr_calc_df["no_predictions_false_negative_flag"]]
        .groupby(["grouper_key", "grouper_value_gt", "confidence_threshold"])[
            "id_gt"
        ]
        .nunique()
    )
    no_predictions_false_negatives.name = "no_predictions_false_negatives"

    # combine these outputs
    pr_curve_counts_df = (
        pd.concat(
            [
                pr_calc_df.loc[
                    ~pr_calc_df["grouper_value_pd"].isnull(),
                    [
                        "grouper_key",
                        "grouper_value_pd",
                        "confidence_threshold",
                    ],
                ].rename(columns={"grouper_value_pd": "grouper_value"}),
                pr_calc_df.loc[
                    ~pr_calc_df["grouper_value_gt"].isnull(),
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
    pr_curve_counts_df.fillna(0, inplace=True)

    # find all unique datums for use when identifying true negatives
    unique_datum_ids = set(pr_calc_df["datum_id"].unique())

    # calculate additional metrics
    pr_curve_counts_df["false_positives"] = pr_curve_counts_df[
        "misclassification_false_positives"
    ]  # we don't have any hallucinations for classification
    pr_curve_counts_df["false_negatives"] = (
        pr_curve_counts_df["misclassification_false_negatives"]
        + pr_curve_counts_df["no_predictions_false_negatives"]
    )
    pr_curve_counts_df["true_negatives"] = len(unique_datum_ids) - (
        pr_curve_counts_df["true_positives"]
        + pr_curve_counts_df["false_positives"]
        + pr_curve_counts_df["false_negatives"]
    )
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
    ) / len(unique_datum_ids)
    pr_curve_counts_df["f1_score"] = (
        2 * pr_curve_counts_df["precision"] * pr_curve_counts_df["recall"]
    ) / (pr_curve_counts_df["precision"] + pr_curve_counts_df["recall"])

    # any NaNs that are left are from division by zero errors
    pr_curve_counts_df.fillna(-1, inplace=True)

    pr_output = defaultdict(lambda: defaultdict(dict))
    detailed_pr_output = defaultdict(lambda: defaultdict(dict))

    # add samples to the dataframe for DetailedPrecisionRecallCurves
    if enums.MetricType.DetailedPrecisionRecallCurve in metrics_to_return:
        for flag in [
            "true_positive_flag",
            "true_negative_flag",
            "misclassification_false_negative_flag",
            "no_predictions_false_negative_flag",
            "misclassification_false_positive_flag",
        ]:
            pr_curve_counts_df = _add_samples_to_dataframe(
                pr_curve_counts_df=pr_curve_counts_df,
                pr_calc_df=pr_calc_df,
                max_examples=pr_curve_max_examples,
                flag_column=flag,
            )

    for _, row in pr_curve_counts_df.iterrows():
        pr_output[row["grouper_key"]][row["grouper_value"]][
            row["confidence_threshold"]
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
            detailed_pr_output[row["grouper_key"]][row["grouper_value"]][
                row["confidence_threshold"]
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
                    "total": row["false_positives"],
                    "observations": {
                        "misclassifications": {
                            "count": row["misclassification_false_positives"],
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
    unique_labels: list,
    label_map: Optional[Dict[schemas.Label, schemas.Label]] = None,
    metrics_to_return: Optional[List[enums.MetricType]] = None,
    pr_curve_max_examples: int = 1,
) -> Tuple[List[dict], List[dict]]:
    """
    Compute classification metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    metrics_to_return: list[MetricType]
        The list of metrics to compute, store, and return to the user.
    label_map: schemas.LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
    pr_curve_max_examples: int
        The maximum number of datum examples to store per true positive, false negative, etc.


    Returns
    ----------
    Tuple[List[metrics.ConfusionMatrix], List[metrics.ConfusionMatrix | metrics.AccuracyMetric | metrics.ROCAUCMetric| metrics.PrecisionMetric | metrics.RecallMetric | metrics.F1Metric]]
        A tuple of confusion matrices and metrics.
    """
    grouper_mappings = _create_classification_grouper_mappings(
        label_map=label_map,
        labels=unique_labels,
    )

    confusion_matrices, metrics_to_output = [], []

    _add_columns_to_groundtruth_and_prediction_table(
        df=groundtruth_df, grouper_mappings=grouper_mappings
    )
    _add_columns_to_groundtruth_and_prediction_table(
        df=prediction_df, grouper_mappings=grouper_mappings
    )

    merged_groundtruths_and_predictions_df = _get_merged_dataframe(
        prediction_df=prediction_df, groundtruth_df=groundtruth_df
    )

    cm_counts_df, confusion_matrices = _calculate_confusion_matrix_df(
        merged_groundtruths_and_predictions_df=merged_groundtruths_and_predictions_df
    )

    metrics_per_grouper_key_and_grouper_value_df = (
        _calculate_metrics_at_grouper_value_level(cm_counts_df=cm_counts_df)
    )

    metrics_to_output += _calculate_precision_recall_f1_metrics(
        metrics_per_grouper_key_and_grouper_value_df=metrics_per_grouper_key_and_grouper_value_df
    )

    metrics_to_output += _calculate_accuracy_metrics(cm_counts_df=cm_counts_df)

    metrics_to_output += _calculate_rocauc(
        prediction_df=prediction_df, groundtruth_df=groundtruth_df
    )

    # handle type error
    assert metrics_to_return

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
    groundtruths: Union[pd.DataFrame, List[schemas.GroundTruth]],
    predictions: Union[pd.DataFrame, List[schemas.Prediction]],
    label_map: Optional[Dict[schemas.Label, schemas.Label]] = None,
    metrics_to_return: Optional[List[enums.MetricType]] = None,
    pr_curve_max_examples: int = 1,
) -> schemas.Evaluation:
    """
    Create classification metrics.
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

    groundtruth_df = utilities.validate_groundtruth_dataframe(groundtruths)
    prediction_df = utilities.validate_prediction_dataframe(predictions)

    # filter dataframes to only include those rows with an applicable implied task type
    groundtruth_df = utilities.filter_dataframe_based_on_task_type(
        df=groundtruth_df, task_type=enums.TaskType.CLASSIFICATION
    )
    prediction_df = utilities.filter_dataframe_based_on_task_type(
        df=prediction_df, task_type=enums.TaskType.CLASSIFICATION
    )

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

    confusion_matrices, metrics = _compute_clf_metrics(
        groundtruth_df=groundtruth_df,
        prediction_df=prediction_df,
        label_map=label_map,
        metrics_to_return=metrics_to_return,
        pr_curve_max_examples=pr_curve_max_examples,
        unique_labels=unique_labels,
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
