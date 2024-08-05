from typing import Dict, List, Optional, Union

import pandas as pd
from valor_core import enums, schemas


def _validate_label_map(
    label_map: Optional[Dict[schemas.Label, schemas.Label]],
) -> Union[List[List[List[str]]], None]:
    """Validate the label mapping if necessary."""

    if not isinstance(label_map, dict) or not all(
        [
            isinstance(key, schemas.Label) and isinstance(value, schemas.Label)
            for key, value in label_map.items()
        ]
    ):
        raise TypeError(
            "label_map should be a dictionary with valid Labels for both the key and value."
        )


def validate_parameters(
    parameters: schemas.EvaluationParameters, task_type: enums.TaskType
) -> schemas.EvaluationParameters:
    if parameters.metrics_to_return is None:
        parameters.metrics_to_return = {
            enums.TaskType.CLASSIFICATION: [
                enums.MetricType.Precision,
                enums.MetricType.Recall,
                enums.MetricType.F1,
                enums.MetricType.Accuracy,
                enums.MetricType.ROCAUC,
            ],
            enums.TaskType.OBJECT_DETECTION: [
                enums.MetricType.AP,
                enums.MetricType.AR,
                enums.MetricType.mAP,
                enums.MetricType.APAveragedOverIOUs,
                enums.MetricType.mAR,
                enums.MetricType.mAPAveragedOverIOUs,
            ],
        }[task_type]

    if parameters.label_map:
        _validate_label_map(parameters.label_map)

    if task_type == enums.TaskType.CLASSIFICATION:
        if not set(parameters.metrics_to_return).issubset(
            enums.MetricType.classification()
        ):
            raise ValueError(
                f"The following metrics are not supported for classification: '{set(parameters.metrics_to_return) - enums.MetricType.classification()}'"
            )

    if task_type == enums.TaskType.OBJECT_DETECTION:

        if not set(parameters.metrics_to_return).issubset(
            enums.MetricType.object_detection()
        ):
            raise ValueError(
                f"The following metrics are not supported for object detection: '{set(parameters.metrics_to_return) - enums.MetricType.object_detection()}'"
            )

        if parameters.iou_thresholds_to_compute is None:
            parameters.iou_thresholds_to_compute = [
                round(0.5 + 0.05 * i, 2) for i in range(10)
            ]
        if parameters.iou_thresholds_to_return is None:
            parameters.iou_thresholds_to_return = [0.5, 0.75]

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

    return parameters


def _add_geojson_column(df: pd.DataFrame) -> pd.DataFrame:
    # TODO write test for this
    if not (
        df[["bounding_box", "polygon", "raster"]].notna().sum(axis=1) == 1
    ).all():
        raise ValueError(
            "Each Annotation must contain either a bounding_box, polygon, raster, or an embedding. One Annotation cannot have multiple of these attributes (for example, one Annotation can't contain both a raster and a bounding box)."
        )

    # if the user wants to use rasters, then we don't care about converting to geojson format
    if df["raster"].notna().any():
        df["raster"] = df["raster"].map(lambda x: x.to_numpy())
        df["geojson"] = None

    else:
        # get the geojson for each object in a column called geojson
        df["geojson"] = (
            df[["bounding_box", "polygon"]]
            .bfill(axis=1)
            .iloc[:, 0]
            .map(lambda x: x.to_dict())
        )

    return df


# TODO this is a bit messy. maybe refactor and move shared parts into a new function
def validate_groundtruth_dataframe(
    obj: Union[pd.DataFrame, List[schemas.GroundTruth]],
    task_type: enums.TaskType,
) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        # TODO check for correct columns
        return obj
    elif (
        obj
        and isinstance(obj, list)
        and isinstance(obj[0], schemas.GroundTruth)
    ):
        df = _convert_groundtruth_or_prediction_to_dataframe(obj)

        if task_type == enums.TaskType.OBJECT_DETECTION:
            df = _add_geojson_column(df)

        return df
    else:
        raise ValueError(
            f"Could not validate object as it's neither a dataframe nor a list of Valor objects. Object is of type {type(obj)}."
        )


def validate_prediction_dataframe(
    obj: Union[pd.DataFrame, List[schemas.Prediction]],
    task_type: enums.TaskType,
) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        # TODO check for correct columns
        return obj
    elif (
        obj
        and isinstance(obj, list)
        and isinstance(obj[0], schemas.Prediction)
    ):
        df = _convert_groundtruth_or_prediction_to_dataframe(obj)

        if task_type == enums.TaskType.OBJECT_DETECTION:
            df = _add_geojson_column(df)

        return df
    else:
        raise ValueError(
            "Could not validate object as it's neither a dataframe nor a list of Valor objects."
        )


def validate_matching_label_keys(
    groundtruths: pd.DataFrame,
    predictions: pd.DataFrame,
    label_map,
) -> None:
    """
    # TODO
    Validates that every datum has the same set of label keys for both ground truths and predictions. This check is only needed for classification tasks.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    label_map: LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.


    Raises
    -------
    ValueError
        If the distinct ground truth label keys don't match the distinct prediction label keys for any datum.
    """
    # allow for case where our predictions don't have any labels
    if len(predictions) == 0:
        return

    if not label_map:
        label_map = dict()

    # get the label keys per datum
    groundtruths["mapped_groundtruth_label_keys"] = groundtruths.apply(
        lambda row: label_map.get(
            schemas.Label(key=row["label_key"], value=row["label_value"]),
            schemas.Label(key=row["label_key"], value=row["label_value"]),
        ).key,
        axis=1,
    )

    predictions["mapped_prediction_label_keys"] = predictions.apply(
        lambda row: label_map.get(
            schemas.Label(key=row["label_key"], value=row["label_value"]),
            schemas.Label(key=row["label_key"], value=row["label_value"]),
        ).key,
        axis=1,
    )

    gt_label_keys_per_datum = groundtruths.groupby(
        ["datum_id"], as_index=False
    )["mapped_groundtruth_label_keys"].unique()

    pd_label_keys_per_datum = predictions.groupby(
        ["datum_id"], as_index=False
    )["mapped_prediction_label_keys"].unique()

    joined = gt_label_keys_per_datum.merge(
        pd_label_keys_per_datum,
        on=["datum_id"],
    )

    if not joined["mapped_groundtruth_label_keys"].equals(
        joined["mapped_prediction_label_keys"]
    ):
        raise ValueError(
            "Ground truth label keys must match prediction label keys for classification tasks."
        )


def _convert_groundtruth_or_prediction_to_dataframe(
    list_of_objects: Union[List[schemas.GroundTruth], List[schemas.Prediction]]
) -> pd.DataFrame:

    output = []

    dataset_name = "delete later"
    # TODO dataset and model number don't really do anything in this framework?

    for i, obj in enumerate(list_of_objects):
        datum_uid = obj.datum.uid
        datum_id = hash(obj.datum.uid)
        datum_metadata = obj.datum.metadata

        for j, ann in enumerate(obj.annotations):
            ann_id = hash(str(datum_uid) + str(ann))
            ann_metadata = ann.metadata
            ann_bbox = ann.bounding_box
            ann_raster = ann.raster
            ann_embeding = ann.embedding
            ann_polygon = ann.polygon
            ann_is_instance = ann.is_instance

            for k, label in enumerate(ann.labels):
                id_ = (
                    str(i) + str(j) + str(k)
                )  # we use indices here, rather than a hash() so that the IDs are sequential. this prevents randomness when two predictions share the same score
                label_key = label.key
                label_value = label.value
                label_score = label.score
                label_id = hash(label_key + label_value + str(label_score))

                # only include scores for predictions
                if isinstance(obj, schemas.Prediction):
                    output.append(
                        {
                            "dataset_name": dataset_name,
                            "datum_uid": datum_uid,
                            "datum_id": datum_id,
                            "datum_metadata": datum_metadata,
                            "annotation_id": ann_id,
                            "annotation_metadata": ann_metadata,
                            "bounding_box": ann_bbox,
                            "raster": ann_raster,
                            "embedding": ann_embeding,
                            "polygon": ann_polygon,
                            "is_instance": ann_is_instance,
                            "label_key": label_key,
                            "label_value": label_value,
                            "score": label_score,
                            "label_id": label_id,
                            "id": id_,
                        }
                    )
                else:
                    output.append(
                        {
                            "dataset_name": dataset_name,
                            "datum_uid": datum_uid,
                            "datum_id": datum_id,
                            "datum_metadata": datum_metadata,
                            "annotation_id": ann_id,
                            "annotation_metadata": ann_metadata,
                            "bounding_box": ann_bbox,
                            "raster": ann_raster,
                            "embedding": ann_embeding,
                            "polygon": ann_polygon,
                            "is_instance": ann_is_instance,
                            "label_key": label_key,
                            "label_value": label_value,
                            "label_id": label_id,
                            "id": id_,
                        }
                    )

    return (
        pd.DataFrame(output)
        if output
        else pd.DataFrame(
            [],
            columns=[
                "dataset_name",
                "datum_uid",
                "datum_id",
                "datum_metadata",
                "annotation_id",
                "annotation_metadata",
                "bounding_box",
                "raster",
                "embedding",
                "polygon",
                "is_instance",
                "label_key",
                "label_value",
                "score",
                "label_id",
                "id",
            ],
        )
    )


def get_disjoint_labels(
    groundtruth_df: pd.DataFrame, prediction_df: pd.DataFrame, label_map
) -> tuple[list[schemas.Label], list[schemas.Label]]:
    """
    Returns all unique labels that are not shared between both filters.

    # TODO
    Parameters
    ----------
    db : Session
        The database Session to query against.
    lhs : list[schemas.Filter]
        Filter defining first label set.
    rhs : list[schemas.Filter]
        Filter defining second label set.
    label_map: LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.

    Returns
    ----------
    Tuple[list[schemas.Label], list[schemas.Label]]
        A tuple of disjoint labels, where the first element is those labels which are present in lhs label set but absent in rhs label set.
    """
    if not label_map:
        label_map = {}

    groundtruth_labels = set(
        groundtruth_df.apply(
            lambda row: (row["label_key"], row["label_value"]),
            axis=1,
        ).values  # type: ignore - pandas typing errors
    )

    prediction_labels = set(
        prediction_df.apply(
            lambda row: (row["label_key"], row["label_value"]),
            axis=1,
        ).values  # type: ignore - pandas typing errors
    )

    # don't count user-mapped labels as disjoint
    mapped_labels = set()
    if label_map:
        for map_from, map_to in label_map.items():
            mapped_labels.add((map_from.key, map_from.value))
            mapped_labels.add((map_to.key, map_to.value))

    groundtruth_unique = list(
        groundtruth_labels - prediction_labels - mapped_labels
    )
    prediction_unique = list(
        prediction_labels - groundtruth_labels - mapped_labels
    )

    return (groundtruth_unique, prediction_unique)
