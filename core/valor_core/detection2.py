import time

import numpy as np
import pandas as pd
from valor_core import enums, metrics, schemas


def bbox_to_useful(bbox):
    coords = bbox.value[0]
    x_values = {c[0] for c in coords[:3]}
    y_values = {c[1] for c in coords[:3]}
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    area = (x_max - x_min) * (y_max - y_min)
    ## area = (x_max - x_min + 1) * (y_max - y_min + 1)

    return x_min, x_max, y_min, y_max, area


def create_dataframes(
    groundtruths: list[schemas.GroundTruth],
    predictions: list[schemas.Prediction],
):
    start = time.time()
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
            "id_gt",
            "k",
            "v",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "area",
        ],
    )

    gt_kv_size = groundtruth_dataframe.groupby(
        ["k", "v"], as_index=False
    ).size()

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
            "id_pd",
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

    pd_kv_size = prediction_dataframe.groupby(
        ["k", "v"], as_index=False
    ).size()

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
            f"Dropped {len(dropped_preds)} predictions because there was no detection groundtruth for datums: {dropped_preds}"
        )

    end = time.time()
    print(
        f"Optimized Created {len(groundtruth_dataframe)} GT Rows and {len(prediction_dataframe)} PD Rows in: {end - start}"
    )

    start = time.time()
    joint_df = _create_joint_df(groundtruth_dataframe, prediction_dataframe)
    end = time.time()
    print(
        f"Optimized Created Detailed Joint DF with {len(joint_df)} rows in: {end - start}"
    )

    return (
        groundtruth_dataframe,
        prediction_dataframe,
        joint_df,
        datum_id_lookup,
        label_keys_lookup,
        label_value_lookup,
        gt_kv_size,
        pd_kv_size,
    )


def _create_joint_df(
    groundtruth_df: pd.DataFrame, prediction_df: pd.DataFrame
):
    joint_df = pd.merge(
        groundtruth_df,
        prediction_df,
        on=["id", "k"],
        how="outer",
        suffixes=("_gt", "_pd"),
    )

    joint_df["iou"] = _calculate_iou(joint_df=joint_df)

    """
    joint_df['size_gt'] = joint_df['size_gt'].fillna(0)
    joint_df['size_pd'] = joint_df['size_pd'].fillna(0)
    joint_df["v_pd"] = joint_df["v_pd"].fillna(joint_df["v_gt"])
    joint_df["s"] = joint_df["s"].fillna(0)
    joint_df["v_gt"] = joint_df["v_gt"].fillna(-1)
    joint_df["id_gt"] = joint_df["id_gt"].fillna(-1)
    """

    return joint_df.loc[
        :,
        [
            "id",
            "id_gt",
            "id_pd",
            "k",
            "v_gt",
            "v_pd",
            "s",
            "iou",
        ],
    ]


def _calculate_iou(
    joint_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the IOUs between predictions and groundtruths in a joint dataframe."""

    x_min = np.maximum(joint_df["x_min_gt"], joint_df["x_min_pd"])
    x_max = np.minimum(joint_df["x_max_gt"], joint_df["x_max_pd"])
    y_min = np.maximum(joint_df["y_min_gt"], joint_df["y_min_pd"])
    y_max = np.minimum(joint_df["y_max_gt"], joint_df["y_max_pd"])

    x = np.maximum(0, x_max - x_min)
    y = np.maximum(0, y_max - y_min)
    ## x = np.maximum(0, x_max - x_min + 1)
    ## y = np.maximum(0, y_max - y_min + 1)

    intersection = np.nan_to_num(x * y, nan=0)
    return intersection / (
        joint_df["area_gt"] + joint_df["area_pd"] - intersection
    )


def _calculate_label_id_level_metrics(
    prediction_df: pd.DataFrame,
    joint_df: pd.DataFrame,
    iou_thresholds: list[float],
) -> pd.DataFrame:

    best_g_per_p = (
        joint_df[joint_df["iou"].notna()]
        .groupby(["id", "id_pd"])["iou"]
        .idxmax()
    )
    best_gp_pair = (
        joint_df.loc[best_g_per_p, :].groupby(["id", "id_gt"])["s"].idxmax()
    )

    sorted_groundtruth_prediction_pairs = pd.merge(
        prediction_df[["id", "id_pd", "k", "v", "s"]],
        joint_df.loc[best_gp_pair, ["id", "id_pd", "id_gt", "iou"]],
        on=["id", "id_pd"],
        how="left",
    ).sort_values(by=["s"], ascending=False)

    sorted_groundtruth_prediction_pairs["count"] = (
        sorted_groundtruth_prediction_pairs.groupby(["k", "v"]).cumcount() + 1
    )
    sorted_groundtruth_prediction_pairs = sorted_groundtruth_prediction_pairs[
        sorted_groundtruth_prediction_pairs["id_gt"].notna()
    ]

    return sorted_groundtruth_prediction_pairs.reset_index(drop=True)


def _calculate_ap_metrics(
    calculation_df: pd.DataFrame,
    gt_kv_size: pd.DataFrame,
    iou_thresholds_to_compute: list[float],
    iou_thresholds_to_return: list[float],
    label_key_lookup: list,
    label_value_lookup: list,
):
    calc_ap_df = {
        "k": calculation_df["k"],
        "v": calculation_df["v"],
        "c": calculation_df["count"],
    }
    cols = [str(i) for i in range(len(iou_thresholds_to_compute))]
    for i, iou_threshold in enumerate(iou_thresholds_to_compute):
        calc_ap_df[cols[i]] = (
            calculation_df[calculation_df["iou"] >= iou_threshold]
            .groupby(["k", "v"])
            .cumcount()
            + 1
        ).astype("float64")

    calc_ap_df = pd.DataFrame(calc_ap_df)

    calc_ap_df[cols] = calc_ap_df[cols].div(calc_ap_df["c"], axis=0)

    calc_ap_df[cols] = (
        calc_ap_df[::-1].groupby(["k", "v"])[cols].cummax()[::-1]
    )

    ap_metrics_df = (calc_ap_df.groupby(["k", "v"])[cols].sum()).reset_index()
    ap_metrics_df = gt_kv_size.merge(
        ap_metrics_df,
        on=["k", "v"],
        how="left",
    ).fillna(0)

    ap_metrics_df[cols] = (
        ap_metrics_df[cols].div(ap_metrics_df["size"], axis=0).fillna(0)
    )

    ap_metrics = []
    col_lookup = {
        iou_thresholds_to_compute[i]: cols[i] for i in range(len(cols))
    }
    ap_metrics_df["AP"] = ap_metrics_df[cols].mean(axis=1)
    temp = set(iou_thresholds_to_compute)

    for _, row in ap_metrics_df.iterrows():
        k = label_key_lookup[int(row["k"])]
        v = label_value_lookup[int(row["v"])]
        for _, iou_threshold in enumerate(iou_thresholds_to_return):
            ap_metrics.append(
                metrics.APMetric(
                    iou=iou_threshold,
                    value=float(row[col_lookup[iou_threshold]]),
                    label=schemas.Label(key=k, value=v),
                )
            )

        ap_metrics.append(
            metrics.APMetricAveragedOverIOUs(
                ious=temp,
                value=float(row["AP"]),
                label=schemas.Label(key=k, value=v),
            )
        )

    map_metrics_df = ap_metrics_df.groupby("k", as_index=False)[cols].mean()
    map_metrics_df["mAP"] = map_metrics_df[cols].mean(axis=1)

    for _, row in map_metrics_df.iterrows():
        k = label_key_lookup[int(row["k"])]
        for _, iou_threshold in enumerate(iou_thresholds_to_return):
            ap_metrics.append(
                metrics.mAPMetric(
                    iou=iou_threshold,
                    value=float(row[col_lookup[iou_threshold]]),
                    label_key=k,
                )
            )

        ap_metrics.append(
            metrics.mAPMetricAveragedOverIOUs(
                ious=temp,
                value=float(row["mAP"]),
                label_key=k,
            )
        )

    return ap_metrics


def _calculate_ar_metrics(
    calculation_df: pd.DataFrame,
    gt_kv_size: pd.DataFrame,
    iou_thresholds_to_compute: list[float],
    recall_score_threshold: float,
    label_key_lookup: list,
    label_value_lookup: list,
):
    ar_metrics_df = []
    for iou_threshold in iou_thresholds_to_compute:
        ar_metrics_df.append(
            calculation_df[
                (calculation_df["iou"] >= iou_threshold)
                & (calculation_df["s"] >= recall_score_threshold)
            ]
            .groupby(["k", "v"])
            .size()
        )

    ar_metrics_df = pd.concat(ar_metrics_df, axis=1).sum(axis=1) / len(
        iou_thresholds_to_compute
    )
    ar_metrics_df = ar_metrics_df.reset_index(name="ar")

    ar_metrics_df = gt_kv_size.merge(
        ar_metrics_df,
        on=["k", "v"],
        how="left",
    ).fillna(0)

    ar_metrics_df["ar"] = ar_metrics_df["ar"] / ar_metrics_df["size"]

    temp = set(iou_thresholds_to_compute)
    ar_metrics = [
        metrics.ARMetric(
            ious=temp,
            value=float(row["ar"]),
            label=schemas.Label(
                key=label_key_lookup[int(row["k"])],
                value=label_value_lookup[int(row["v"])],
            ),
        )
        for _, row in ar_metrics_df.iterrows()
    ]

    mar_metrics_df = ar_metrics_df.groupby("k", as_index=False)["ar"].mean()

    mar_metrics = [
        metrics.mARMetric(
            ious=temp,
            value=float(row["ar"]),
            label_key=label_key_lookup[int(row["k"])],
        )
        for _, row in mar_metrics_df.iterrows()
    ]

    return ar_metrics + mar_metrics


def evaluate_detections(
    groundtruths: pd.DataFrame | list[schemas.GroundTruth],
    predictions: pd.DataFrame | list[schemas.Prediction],
    label_map: dict[schemas.Label, schemas.Label] | None = None,
    metrics_to_return: list[enums.MetricType] | None = None,
    iou_thresholds_to_compute: list[float] | None = None,
    iou_thresholds_to_return: list[float] | None = None,
    recall_score_threshold: float = 0.0,
    pr_curve_iou_threshold: float = 0.5,
    pr_curve_max_examples: int = 1,
):

    if not label_map:
        label_map = {}

    if iou_thresholds_to_compute is None:
        iou_thresholds_to_compute = [
            round(0.5 + 0.05 * i, 2) for i in range(10)
        ]
    if iou_thresholds_to_return is None:
        iou_thresholds_to_return = [0.5, 0.75]

    (
        groundtruth_dataframe,
        prediction_dataframe,
        joint_df,
        datum_id_lookup,
        label_key_lookup,
        label_value_lookup,
        gt_kv_size,
        pd_kv_size,
    ) = create_dataframes(groundtruths, predictions)

    calculation_df = _calculate_label_id_level_metrics(
        prediction_dataframe,
        joint_df,
        iou_thresholds_to_compute,
    )

    metrics_to_output = []
    metrics_to_output += _calculate_ap_metrics(
        calculation_df,
        gt_kv_size,
        iou_thresholds_to_compute,
        iou_thresholds_to_return,
        label_key_lookup,
        label_value_lookup,
    )

    metrics_to_output += _calculate_ar_metrics(
        calculation_df,
        gt_kv_size,
        iou_thresholds_to_compute,
        recall_score_threshold,
        label_key_lookup,
        label_value_lookup,
    )
