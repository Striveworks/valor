import gc
import pandas as pd

from collections import defaultdict

def _calculate_pr_curves(
    prediction_df: pd.DataFrame,
    groundtruth_df: pd.DataFrame,
    metrics_to_return: list,
    pr_curve_max_examples: int,
):
    """Calculate PrecisionRecallCurve metrics."""

    if not (
        "PrecisionRecallCurve" in metrics_to_return
        or "DetailedPrecisionRecallCurve" in metrics_to_return
    ):
        return []
    
    joint_df = (
        pd.merge(
            groundtruth_df,
            prediction_df,
            on=["datum_id", "datum_uid", "label_key"],
            how="inner",
            suffixes=("_gt", "_pd"),
        ).assign(
            is_label_match=lambda chain_df: (
                (chain_df["label_value_pd"] == chain_df["label_value_gt"])
            )
        )
        # only keep the columns we need
        .loc[
            :,
            [
                "datum_uid",
                "datum_id",
                "label_key",
                "label_value_gt",
                "id_gt",
                "label_value_pd",
                "score",
                "id_pd",
                "is_label_match",
            ],
        ]
    )

    # free up memory
    del groundtruth_df
    del prediction_df
    gc.collect()

    # add confidence_threshold to the dataframe and sort
    pr_calc_df = pd.concat(
        [
            joint_df.assign(confidence_threshold=threshold)
            for threshold in [x / 100 for x in range(5, 100, 5)]
        ],
        ignore_index=True,
    ).sort_values(
        by=[
            "label_key",
            "label_value_pd",
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
        confidence_interval_to_true_positive_groundtruth_ids_dict = (
            groundtruths_associated_with_true_positives.set_index(
                "confidence_threshold"
            )["id_gt"]
            .apply(set)
            .to_dict()
        )

        mask = pd.Series(False, index=pr_calc_df.index)

        for (
            threshold,
            elements,
        ) in confidence_interval_to_true_positive_groundtruth_ids_dict.items():
            threshold_mask = pr_calc_df["confidence_threshold"] == threshold
            membership_mask = pr_calc_df["id_gt"].isin(elements)
            mask |= threshold_mask & membership_mask

        pr_calc_df["false_negative_flag"] = ~mask

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
        confidence_interval_to_misclassification_fn_groundtruth_ids_dict = (
            groundtruths_associated_with_misclassification_false_negatives.set_index(
                "confidence_threshold"
            )[
                "id_gt"
            ]
            .apply(set)
            .to_dict()
        )

        mask = pd.Series(False, index=pr_calc_df.index)

        for (
            threshold,
            elements,
        ) in (
            confidence_interval_to_misclassification_fn_groundtruth_ids_dict.items()
        ):
            threshold_mask = pr_calc_df["confidence_threshold"] == threshold
            membership_mask = ~pr_calc_df["id_gt"].isin(elements)
            mask |= threshold_mask & membership_mask

        pr_calc_df["no_predictions_false_negative_flag"] = (
            mask & pr_calc_df["false_negative_flag"]
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
        .groupby(["label_key", "label_value_pd", "confidence_threshold"])[
            "id_pd"
        ]
        .nunique()
    )
    true_positives.name = "true_positives"

    misclassification_false_positives = (
        pr_calc_df[pr_calc_df["misclassification_false_positive_flag"]]
        .groupby(["label_key", "label_value_pd", "confidence_threshold"])[
            "id_pd"
        ]
        .nunique()
    )
    misclassification_false_positives.name = (
        "misclassification_false_positives"
    )

    misclassification_false_negatives = (
        pr_calc_df[pr_calc_df["misclassification_false_negative_flag"]]
        .groupby(["label_key", "label_value_gt", "confidence_threshold"])[
            "id_gt"
        ]
        .nunique()
    )
    misclassification_false_negatives.name = (
        "misclassification_false_negatives"
    )

    no_predictions_false_negatives = (
        pr_calc_df[pr_calc_df["no_predictions_false_negative_flag"]]
        .groupby(["label_key", "label_value_gt", "confidence_threshold"])[
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
                    ~pr_calc_df["label_value_pd"].isnull(),
                    [
                        "label_key",
                        "label_value_pd",
                        "confidence_threshold",
                    ],
                ].rename(columns={"label_value_pd": "label_value"}),
                pr_calc_df.loc[
                    ~pr_calc_df["label_value_gt"].isnull(),
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

    '''
    pr_output = defaultdict(lambda: defaultdict(dict))
    detailed_pr_output = defaultdict(lambda: defaultdict(dict))

    # add samples to the dataframe for DetailedPrecisionRecallCurves
    if "DetailedPrecisionRecallCurve" in metrics_to_return:
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
        pr_output[row["label_key"]][row["label_value"]][
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

        if "DetailedPrecisionRecallCurve" in metrics_to_return:
            detailed_pr_output[row["label_key"]][row["label_value"]][
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

    if "PrecisionRecallCurve" in metrics_to_return:
        return pr_output

    if "DetailedPrecisionRecallCurve" in metrics_to_return:
        return detailed_pr_output

def _add_samples_to_dataframe(
    pr_curve_counts_df: pd.DataFrame,
    pr_calc_df: pd.DataFrame,
    max_examples: int,
    flag_column: str,
) -> pd.DataFrame:
    """Efficienctly gather samples for a given flag."""

    sample_df = pd.concat(
        [
            pr_calc_df[pr_calc_df[flag_column]]
            .groupby(
                [
                    "label_key",
                    "label_value_gt",
                    "confidence_threshold",
                ],
                as_index=False,
            )[["datum_uid"]]
            .agg(lambda x: tuple(x.head(max_examples)))
            .rename(columns={"label_value_gt": "label_value"}),
            pr_calc_df[pr_calc_df[flag_column]]
            .groupby(
                [
                    "label_key",
                    "label_value_pd",
                    "confidence_threshold",
                ],
                as_index=False,
            )[["datum_uid"]]
            .agg(lambda x: tuple(x.head(max_examples)))
            .rename(columns={"label_value_pd": "label_value"}),
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
                    "label_key",
                    "label_value",
                    "confidence_threshold",
                    f"{flag_column}_samples",
                ]
            ],
            on=["label_key", "label_value", "confidence_threshold"],
            how="outer",
        )
        pr_curve_counts_df[f"{flag_column}_samples"] = pr_curve_counts_df[
            f"{flag_column}_samples"
        ].apply(lambda x: list(x) if isinstance(x, set) else list())

    else:
        pr_curve_counts_df[f"{flag_column}_samples"] = [
            list() for _ in range(len(pr_curve_counts_df))
        ]
    '''

    return pr_curve_counts_df