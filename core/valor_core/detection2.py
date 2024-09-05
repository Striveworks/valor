import time

import numpy as np
import pandas as pd
from valor_core import enums, metrics, schemas, utilities


def bbox_to_useful(bbox):
    coords = bbox.value[0]
    x_values = {c[0] for c in coords[:3]}
    y_values = {c[1] for c in coords[:3]}
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    area = (x_max - x_min) * (y_max - y_min)

    # Pixel-center axis aligned bounding box format
    # area = (x_max - x_min + 1) * (y_max - y_min + 1)

    return x_min, x_max, y_min, y_max, area


def create_dataframes(
    groundtruths: list[schemas.GroundTruth],
    predictions: list[schemas.Prediction],
):
    datum_ids = {}
    datum_id_counter = 0

    label_keys = {}
    label_keys_counter = 0

    label_values = {}
    label_values_counter = 0

    rows = []

    for gt in groundtruths:
        _datum_id = datum_ids.get(gt.datum.uid)
        if _datum_id is None:
            _datum_id = datum_id_counter
            datum_ids[gt.datum.uid] = datum_id_counter
            datum_id_counter += 1

        gt_id = 0
        for an in gt.annotations:
            x_min, x_max, y_min, y_max, area = bbox_to_useful(an.bounding_box)
            for label in an.labels:
                key = label_keys.get(label.key)
                if key is None:
                    key = label_keys_counter
                    label_keys[label.key] = label_keys_counter
                    label_keys_counter += 1

                value = label_values.get(label.value)
                if value is None:
                    value = label_values_counter
                    label_values[label.value] = label_values_counter
                    label_values_counter += 1

                rows.append(
                    (
                        _datum_id,
                        gt_id,
                        key,
                        value,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        area,
                    )
                )
                gt_id += 1

    groundtruth_dataframe = pd.DataFrame(
        rows,
        columns=[
            "id",
            "id_g",
            "k",
            "v",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "area",
        ],
    )

    # Count number of groundtruths with each (label_key, label_value) pair
    groundtruth_label_counts = (
        groundtruth_dataframe.groupby(["k", "v"])
        .size()
        .reset_index(name="glc")  # type: ignore - pandas typing error
    )

    rows = []
    dropped_preds = []

    for pred in predictions:
        _datum_id = datum_ids.get(pred.datum.uid)
        if _datum_id is None:
            # No groundtruth for prediction
            dropped_preds.append(pred.datum.uid)
            continue

        p_id = 0
        for an in pred.annotations:
            x_min, x_max, y_min, y_max, area = bbox_to_useful(an.bounding_box)
            for label in an.labels:
                key = label_keys.get(label.key)
                if key is None:
                    key = label_keys_counter
                    label_keys[label.key] = label_keys_counter
                    label_keys_counter += 1

                value = label_values.get(label.value)
                if value is None:
                    value = label_values_counter
                    label_values[label.value] = label_values_counter
                    label_values_counter += 1

                rows.append(
                    (
                        _datum_id,
                        p_id,
                        key,
                        value,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        area,
                        label.score,
                    )
                )
                p_id += 1

    prediction_dataframe = pd.DataFrame(
        rows,
        columns=[
            "id",
            "id_p",
            "k",
            "v",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "area",
            "s",
        ],
    )

    # Count number of prediction with each (label_key, label_value) pair
    prediction_label_counts = (
        prediction_dataframe.groupby(["k", "v"]).size().reset_index(name="plc")  # type: ignore - pandas typing error
    )

    datum_id_lookup = [""] * len(datum_ids)
    for k, v in datum_ids.items():
        datum_id_lookup[v] = k

    label_keys_lookup = [""] * len(label_keys)
    for k, v in label_keys.items():
        label_keys_lookup[v] = k

    label_value_lookup = [""] * len(label_values)
    for k, v in label_values.items():
        label_value_lookup[v] = k

    if len(dropped_preds):
        print(
            f"Dropped {len(dropped_preds)} predictions because there is no groundtruth for datums: {dropped_preds}"
        )

    joint_dataframe = _create_joint_df(
        groundtruth_dataframe, prediction_dataframe
    )

    return (
        groundtruth_dataframe,
        prediction_dataframe,
        joint_dataframe,
        datum_id_lookup,
        label_keys_lookup,
        label_value_lookup,
        groundtruth_label_counts,
        prediction_label_counts,
    )


def _create_joint_df(
    groundtruth_dataframe: pd.DataFrame, prediction_dataframe: pd.DataFrame
):
    joint_dataframe = pd.merge(
        groundtruth_dataframe,
        prediction_dataframe,
        on=["id", "k"],
        how="outer",
        suffixes=("_g", "_p"),
    )

    joint_dataframe = _calculate_iou(joint_dataframe=joint_dataframe)

    return joint_dataframe.loc[
        :,
        [
            "id",
            "id_g",
            "id_p",
            "k",
            "v_g",
            "v_p",
            "s",
            "iou",
        ],
    ]


def _calculate_iou(
    joint_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the IOUs between groundtruths and predictions in a joint dataframe."""

    x_min = np.maximum(joint_dataframe["x_min_g"], joint_dataframe["x_min_p"])
    x_max = np.minimum(joint_dataframe["x_max_g"], joint_dataframe["x_max_p"])
    y_min = np.maximum(joint_dataframe["y_min_g"], joint_dataframe["y_min_p"])
    y_max = np.minimum(joint_dataframe["y_max_g"], joint_dataframe["y_max_p"])

    w = np.maximum(0, x_max - x_min)
    h = np.maximum(0, y_max - y_min)

    # Pixel-center axis aligned bounding box format
    # w = np.maximum(0, x_max - x_min + 1)
    # h = np.maximum(0, y_max - y_min + 1)

    intersection = np.nan_to_num(w * h, nan=0)
    joint_dataframe["iou"] = intersection / (
        joint_dataframe["area_g"] + joint_dataframe["area_p"] - intersection
    )

    return joint_dataframe


def _calculate_true_positive_predictions_at_each_iou_threshold(
    prediction_dataframe: pd.DataFrame,
    joint_dataframe: pd.DataFrame,
    iou_thresholds_to_compute: list[float],
) -> pd.DataFrame:
    """Pre-compute some filters and data for reuse."""

    # For each prediction, get the index of the ground truth with the highest iou and has matching labels.
    best_groundtruth_per_prediction = (
        joint_dataframe[
            (joint_dataframe["iou"] >= min(iou_thresholds_to_compute))
            & (joint_dataframe["v_g"] == joint_dataframe["v_p"])
        ]
        .groupby(["id", "id_p"])["iou"]
        .idxmax()
    )

    best_groundtruth_per_prediction = joint_dataframe.loc[
        best_groundtruth_per_prediction, ["id", "id_g", "id_p", "s", "iou"]
    ]

    true_positives_at_iou_thresholds = []
    for i, iou_threshold in enumerate(iou_thresholds_to_compute):
        true_positives = (
            best_groundtruth_per_prediction[
                best_groundtruth_per_prediction["iou"] >= iou_threshold
            ]
            .groupby(["id", "id_g"])["s"]
            .idxmax()
        )
        true_positives_at_iou_thresholds.append(
            pd.Series(True, index=true_positives, name=i, dtype="boolean")
        )

    true_positives_at_iou_thresholds = pd.concat(
        true_positives_at_iou_thresholds, axis=1
    )
    rows_to_keep = true_positives_at_iou_thresholds.any(axis=1)
    best_groundtruth_per_prediction = pd.concat(
        [
            best_groundtruth_per_prediction.loc[
                rows_to_keep.index, ["id", "id_p"]
            ],
            true_positives_at_iou_thresholds.loc[rows_to_keep.index, :],
        ],
        axis=1,
    ).fillna(False)
    del rows_to_keep, true_positives_at_iou_thresholds

    true_positive_predictions = (
        prediction_dataframe[["id", "id_p", "k", "v", "s"]]
        .sort_values(by=["s"], ascending=False)
        .reset_index(drop=True)
    )
    true_positive_predictions["c"] = (
        true_positive_predictions.groupby(["k", "v"]).cumcount() + 1
    )

    true_positive_predictions = true_positive_predictions.merge(
        best_groundtruth_per_prediction, on=["id", "id_p"], how="inner"
    )

    return true_positive_predictions


def _calculate_ap_metrics(
    true_positive_predictions: pd.DataFrame,
    groundtruth_label_counts: pd.DataFrame,
    iou_thresholds_to_compute: list[float],
    iou_thresholds_to_return: list[float],
    label_key_lookup: list,
    label_value_lookup: list,
):
    calculation_columns = list(
        range(
            len(iou_thresholds_to_compute), 2 * len(iou_thresholds_to_compute)
        )
    )
    for i, iou_threshold in enumerate(iou_thresholds_to_compute):
        # Calculate max precision for each recall value on PR curve
        true_positive_predictions[calculation_columns[i]] = (
            true_positive_predictions[true_positive_predictions[i]]
            .groupby(["k", "v"])
            .cumcount()
            + 1
        ) / true_positive_predictions["c"]

    # Max precision from the right for PR curve interpolation
    true_positive_predictions[calculation_columns] = (
        true_positive_predictions[::-1]
        .groupby(["k", "v"])[calculation_columns]
        .cummax()[::-1]
    )

    # Sum PR curve approximation to approximate total area under the curve
    average_precision_metrics = true_positive_predictions.groupby(
        ["k", "v"], as_index=False
    )[calculation_columns].sum()
    true_positive_predictions.drop(columns=calculation_columns, inplace=True)

    # Map groundtruth count for each label key-value pair to 'glc' column for each row
    average_precision_metrics = groundtruth_label_counts.merge(
        average_precision_metrics,
        on=["k", "v"],
        how="left",
    ).fillna(0)

    # Divide by groundtruth count for each label key-value pair to calculate AP
    average_precision_metrics[calculation_columns] = (
        average_precision_metrics[calculation_columns]
        .div(average_precision_metrics["glc"], axis=0)
        .fillna(0)
    )

    output_metrics = []
    iou_set = set(iou_thresholds_to_compute)
    iou_column_lookup = {
        iou_thresholds_to_compute[i]: calculation_columns[i]
        for i in range(len(iou_thresholds_to_compute))
    }

    # Average AP over all IOU thresholds to calculate AP averaged over IOUs
    average_precision_metrics["APAvg"] = average_precision_metrics[
        calculation_columns
    ].mean(axis=1)

    for _, row in average_precision_metrics.iterrows():
        k = label_key_lookup[int(row["k"])]
        v = label_value_lookup[int(row["v"])]
        for _, iou_threshold in enumerate(iou_thresholds_to_return):
            output_metrics.append(
                metrics.APMetric(
                    iou=iou_threshold,
                    value=float(row[iou_column_lookup[iou_threshold]]),
                    label=schemas.Label(key=k, value=v),
                )
            )

        output_metrics.append(
            metrics.APMetricAveragedOverIOUs(
                ious=iou_set,
                value=float(row["APAvg"]),
                label=schemas.Label(key=k, value=v),
            )
        )

    # Average AP over label values in label key to calculate mAP
    mean_average_precision_metrics = average_precision_metrics.groupby(
        "k", as_index=False
    )[calculation_columns].mean()

    # Average mAP over all IOU thresholds to calculate mAP averaged over IOUs
    mean_average_precision_metrics["mAPAvg"] = mean_average_precision_metrics[
        calculation_columns
    ].mean(axis=1)

    for _, row in mean_average_precision_metrics.iterrows():
        k = label_key_lookup[int(row["k"])]
        for _, iou_threshold in enumerate(iou_thresholds_to_return):
            output_metrics.append(
                metrics.mAPMetric(
                    iou=iou_threshold,
                    value=float(row[iou_column_lookup[iou_threshold]]),
                    label_key=k,
                )
            )

        output_metrics.append(
            metrics.mAPMetricAveragedOverIOUs(
                ious=iou_set,
                value=float(row["mAPAvg"]),
                label_key=k,
            )
        )

    return output_metrics


def _calculate_ar_metrics(
    true_positive_predictions: pd.DataFrame,
    groundtruth_label_counts: pd.DataFrame,
    iou_thresholds_to_compute: list[float],
    recall_score_threshold: float,
    label_key_lookup: list,
    label_value_lookup: list,
):
    true_positive_predictions = true_positive_predictions[
        true_positive_predictions["s"] >= recall_score_threshold
    ]
    recall_metrics = pd.concat(
        [
            true_positive_predictions[true_positive_predictions[i]]
            .groupby(["k", "v"])
            .size()
            for i in range(len(iou_thresholds_to_compute))
        ],
        axis=1,
    ).fillna(0)

    # Calculate true positives count averaged over all IOUs
    recall_metrics = recall_metrics.mean(axis=1)
    recall_metrics = recall_metrics.reset_index(name="RAvg")

    recall_metrics = groundtruth_label_counts.merge(
        recall_metrics,
        on=["k", "v"],
        how="left",
    ).fillna(0)

    # Divide average true positive count by groundtruth count for each label key-value pair to calculate recall averaged over IOUs
    recall_metrics["RAvg"] = recall_metrics["RAvg"] / recall_metrics["glc"]

    iou_set = set(iou_thresholds_to_compute)
    output_metrics = [
        metrics.ARMetric(
            ious=iou_set,
            value=float(row["RAvg"]),
            label=schemas.Label(
                key=label_key_lookup[int(row["k"])],
                value=label_value_lookup[int(row["v"])],
            ),
        )
        for _, row in recall_metrics.iterrows()
    ]

    mean_recall_metrics = recall_metrics.groupby("k", as_index=False)[
        "RAvg"
    ].mean()

    output_metrics += [
        metrics.mARMetric(
            ious=iou_set,
            value=float(row["RAvg"]),
            label_key=label_key_lookup[int(row["k"])],
        )
        for _, row in mean_recall_metrics.iterrows()
    ]

    return output_metrics


def evaluate_detections(
    groundtruths: list[schemas.GroundTruth],
    predictions: list[schemas.Prediction],
    label_map: dict[schemas.Label, schemas.Label] | None = None,
    metrics_to_return: list[enums.MetricType] | None = None,
    iou_thresholds_to_compute: list[float] | None = None,
    iou_thresholds_to_return: list[float] | None = None,
    recall_score_threshold: float = 0.0,
    pr_curve_iou_threshold: float = 0.5,
    pr_curve_max_examples: int = 1,
):

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
        groundtruth_dataframe,
        prediction_dataframe,
        joint_dataframe,
        datum_id_lookup,
        label_key_lookup,
        label_value_lookup,
        groundtruth_label_counts,
        prediction_label_counts,
    ) = create_dataframes(groundtruths, predictions)

    true_positive_predictions = (
        _calculate_true_positive_predictions_at_each_iou_threshold(
            prediction_dataframe,
            joint_dataframe,
            iou_thresholds_to_compute,
        )
    )

    metrics_to_output = []
    metrics_to_output += _calculate_ap_metrics(
        true_positive_predictions,
        groundtruth_label_counts,
        iou_thresholds_to_compute,
        iou_thresholds_to_return,
        label_key_lookup,
        label_value_lookup,
    )

    metrics_to_output += _calculate_ar_metrics(
        true_positive_predictions,
        groundtruth_label_counts,
        iou_thresholds_to_compute,
        recall_score_threshold,
        label_key_lookup,
        label_value_lookup,
    )

    metrics_to_output = [
        metric.to_dict()
        for metric in metrics_to_output
        if metric.to_dict()["type"] in metrics_to_return
    ]

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
        metrics=metrics_to_output,
        confusion_matrices=[],
        ignored_pred_labels=[],
        missing_pred_labels=[],
        meta={
            "labels": len(groundtruth_dataframe[["k", "v"]].drop_duplicates()),
            "datums": len(datum_id_lookup),
            "annotations": len(groundtruth_dataframe),
            "duration": time.time() - start_time,
        },
    )
