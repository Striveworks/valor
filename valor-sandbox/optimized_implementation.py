import pandas as pd
from collections import defaultdict

def _calculate_pr_curves_optimized(
    prediction_df: pd.DataFrame,
    groundtruth_df: pd.DataFrame,
    metrics_to_return: list,
    pr_curve_max_examples: int,
):
    joint_df = (
        pd.merge(
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
                "id_pd",
            ],
        ]
    )

    dd = defaultdict(lambda: 0)

    total_datums_per_label_key = joint_df.drop_duplicates(["datum_uid", "datum_id", "label_key"])["label_key"].value_counts().to_dict(into=dd)
    total_label_values_per_label_key = joint_df.drop_duplicates(["datum_uid", "datum_id", "label_key"])[["label_key", "label_value_gt"]].value_counts().to_dict(into=dd)

    joint_df = joint_df.assign(
        threshold_index=lambda chain_df: (
            ((joint_df["score"] * 100) // 5).astype("int32")
        ),
        is_label_match=lambda chain_df: (
            (chain_df["label_value_pd"] == chain_df["label_value_gt"])
        )
    )

    true_positives = joint_df[joint_df["is_label_match"] == True][["label_key", "label_value_gt", "threshold_index"]].value_counts()
    ## true_positives = true_positives.reset_index(2).sort_values("threshold_index").groupby(["label_key", "label_value_gt"]).cumsum()
    false_positives = joint_df[joint_df["is_label_match"] == False][["label_key", "label_value_pd", "threshold_index"]].value_counts()
    ## false_positives = false_positives.reset_index(2).sort_values("threshold_index").groupby(["label_key", "label_value_pd"]).cumsum()
    
    confidence_thresholds = [x / 100 for x in range(5, 100, 5)]

    tps_keys = []
    tps_values = []
    tps_confidence = []
    tps_cumulative = []
    fns_cumulative = []

    fps_keys = []
    fps_values = []
    fps_confidence = []
    fps_cumulative = []
    tns_cumulative = []

    ## Not sure what the efficient way of doing this is in pandas
    for (label_key, label_value) in true_positives.keys().droplevel(2).unique():
        dd = true_positives[label_key][label_value].to_dict(into=dd)
        cumulative_true_positive = [0] * 21
        cumulative_false_negative = [0] * 21
        for threshold_index in range(19, -1, -1):
            cumulative_true_positive[threshold_index] = cumulative_true_positive[threshold_index + 1] + dd[threshold_index]
            cumulative_false_negative[threshold_index] = total_label_values_per_label_key[(label_key, label_value)] - cumulative_true_positive[threshold_index]
        
        tps_keys += [label_key] * 19
        tps_values += [label_value] * 19
        tps_confidence += confidence_thresholds
        tps_cumulative += cumulative_true_positive[1:-1]
        fns_cumulative += cumulative_false_negative[1:-1]

    ## Not sure what the efficient way of doing this is in pandas
    for (label_key, label_value) in false_positives.keys().droplevel(2).unique():
        dd = false_positives[label_key][label_value].to_dict(into=dd)
        cumulative_false_positive = [0] * 21
        cumulative_true_negative = [0] * 21
        for threshold_index in range(19, -1, -1):
            cumulative_false_positive[threshold_index] = cumulative_false_positive[threshold_index + 1] + dd[threshold_index]
            cumulative_true_negative[threshold_index] = total_datums_per_label_key[label_key] - total_label_values_per_label_key[(label_key, label_value)] - cumulative_false_positive[threshold_index]
        
        fps_keys += [label_key] * 19
        fps_values += [label_value] * 19
        fps_confidence += confidence_thresholds
        fps_cumulative += cumulative_false_positive[1:-1]
        tns_cumulative += cumulative_true_negative[1:-1]

    tps_df = pd.DataFrame({
        "label_key": tps_keys,
        "label_value": tps_values,
        "confidence_threshold": tps_confidence,
        "true_positives": tps_cumulative,
        "false_negatives": fns_cumulative, 
    })

    fps_df = pd.DataFrame({
        "label_key": fps_keys,
        "label_value": fps_values,
        "confidence_threshold": fps_confidence,
        "false_positives": fps_cumulative,
        "true_negatives": tns_cumulative, 
    })

    pr_curve_counts_df = pd.merge(
        tps_df,
        fps_df,
        on=["label_key", "label_value", "confidence_threshold"],
        how="outer",
    )

    pr_curve_counts_df.fillna(0, inplace=True)
    pr_curve_counts_df["total_datums"] = pr_curve_counts_df["label_key"].map(total_datums_per_label_key)

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
    ) / pr_curve_counts_df["total_datums"]
    pr_curve_counts_df["f1_score"] = (
        2 * pr_curve_counts_df["precision"] * pr_curve_counts_df["recall"]
    ) / (pr_curve_counts_df["precision"] + pr_curve_counts_df["recall"])

    # any NaNs that are left are from division by zero errors
    pr_curve_counts_df.fillna(-1, inplace=True)

    return pr_curve_counts_df