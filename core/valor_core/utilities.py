from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from valor_core import enums, geometry, schemas


def validate_label_map(
    label_map: Optional[Dict[schemas.Label, schemas.Label]],
) -> None:
    """
    Validate the label mapping if necessary.

    This function checks if the provided label_map is a dictionary with both
    keys and values being instances of schemas.Label. If the label_map is
    invalid, a TypeError is raised.

    Parameters
    ----------
    label_map : Optional[Dict[schemas.Label, schemas.Label]]
        A dictionary mapping labels to other labels, or None if no mapping
        is provided.

    Raises
    ------
    TypeError
        If label_map is not a dictionary or if its keys and values are not
        instances of schemas.Label.
    """
    if label_map and (
        not isinstance(label_map, dict)
        or not all(
            [
                isinstance(key, schemas.Label)
                and isinstance(value, schemas.Label)
                for key, value in label_map.items()
            ]
        )
    ):
        raise TypeError(
            "label_map should be a dictionary with valid Labels for both the key and value."
        )


def validate_metrics_to_return(
    task_type: enums.TaskType, metrics_to_return: List[enums.MetricType]
) -> None:
    """
    Validate that the provided metrics are appropriate for the specified task type.

    This function checks if the provided metrics_to_return are valid for the given
    task_type. It raises a ValueError if any of the metrics are not supported for
    the specified task type.

    Parameters
    ----------
    task_type : enums.TaskType
        The type of task for which the metrics are being validated. This can be
        either `enums.TaskType.CLASSIFICATION` or `enums.TaskType.OBJECT_DETECTION`.
    metrics_to_return : List[enums.MetricType]
        A list of metrics that need to be validated against the task type.

    Raises
    ------
    ValueError
        If any of the provided metrics are not supported for the specified task type.
    """

    if task_type == enums.TaskType.CLASSIFICATION:
        if not set(metrics_to_return).issubset(
            enums.MetricType.classification()
        ):
            raise ValueError(
                f"The following metrics are not supported for classification: '{set(metrics_to_return) - enums.MetricType.classification()}'"
            )

    if task_type == enums.TaskType.OBJECT_DETECTION:
        if not set(metrics_to_return).issubset(
            enums.MetricType.object_detection()
        ):
            raise ValueError(
                f"The following metrics are not supported for object detection: '{set(metrics_to_return) - enums.MetricType.object_detection()}'"
            )


def validate_parameters(
    recall_score_threshold: Optional[float] = None,
    pr_curve_iou_threshold: Optional[float] = None,
    pr_curve_max_examples: Optional[int] = None,
) -> None:
    """
    Validate parameters for scoring and PR curves.

    Parameters
    ----------
    recall_score_threshold : Optional[float]
        The threshold for recall score.
    pr_curve_iou_threshold : Optional[float]
        The IOU threshold for PR curve.
    pr_curve_max_examples : Optional[int]
        The maximum number of examples for PR curve.

    Raises
    ------
    ValueError
        If any of the parameters are out of their valid ranges.
    """

    if recall_score_threshold and (
        recall_score_threshold > 1 or recall_score_threshold < 0
    ):
        raise ValueError(
            "recall_score_threshold should exist in the range 0 <= threshold <= 1."
        )

    if pr_curve_iou_threshold and (
        pr_curve_iou_threshold <= 0 or pr_curve_iou_threshold > 1.0
    ):
        raise ValueError(
            "IOU thresholds should exist in the range 0 < threshold <= 1."
        )

    if pr_curve_max_examples and (pr_curve_max_examples < 0):
        raise ValueError(
            "pr_curve_max_examples should be an integer greater than or equal to zero."
        )


def validate_matching_label_keys(
    groundtruths: pd.DataFrame,
    predictions: pd.DataFrame,
    label_map: Optional[Dict[schemas.Label, schemas.Label]],
) -> None:
    """
    Validates that every datum has the same set of label keys for both ground truths and predictions. This check is only needed for classification tasks.

    Parameters
    ----------
    groundtruths : pd.DataFrame
        The DataFrame containing ground truth data.
    predictions : pd.DataFrame
        The DataFrame containing prediction data.
    label_map : Optional[Dict[schemas.Label, schemas.Label]]
        Optional mapping of individual labels to a grouper label.

    Raises
    ------
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


def _validate_groundtruth_dataframe(
    df: pd.DataFrame, task_type: enums.TaskType
) -> None:
    """Validate the details of a ground truth dataframe."""
    null_placeholder_column = pd.Series([None] * len(df))

    if df.get("score", null_placeholder_column).notnull().any():
        raise ValueError("GroundTruth labels should not have scores.")

    if task_type == enums.TaskType.SEMANTIC_SEGMENTATION:
        if not (df.groupby("label")["annotation_id"].nunique() == 1).all():
            raise ValueError(
                "For semantic segmentation tasks, each label can only be associated with a single annotation id."
            )


def _validate_prediction_dataframe(
    df: pd.DataFrame, task_type: enums.TaskType
) -> None:
    """Validate the details of a prediction dataframe."""

    if task_type == enums.TaskType.CLASSIFICATION:
        if df["score"].isnull().any():
            raise ValueError(
                "All classification predictions must have an associated score."
            )
        if not (
            df.groupby(["datum_id", "label_key"])["score"].sum() - 1.0 <= 1e-6
        ).all():
            raise ValueError(
                "All classification scores must sum to one for each label key."
            )
    if task_type == enums.TaskType.OBJECT_DETECTION:
        if df["score"].isnull().any():
            raise ValueError(
                "All object detection predictions must have an associated score."
            )
    if task_type == enums.TaskType.SEMANTIC_SEGMENTATION:
        if df["score"].notnull().any():
            raise ValueError(
                "All classification predictions must have an associated score."
            )

        if not (df.groupby("label")["annotation_id"].nunique() == 1).all():
            raise ValueError(
                "For semantic segmentation tasks, each label can only be associated with a single annotation id."
            )


def create_filtered_and_validated_groundtruth_df(
    obj: Union[pd.DataFrame, List[schemas.GroundTruth]],
    task_type: enums.TaskType,
) -> pd.DataFrame:
    """
    Create a filtered and validated DataFrame of ground truth data.

    This function takes either a DataFrame or a list of GroundTruth objects,
    validates the input, identifies the implied task types, filters the DataFrame
    to match the specified task type, and then validates the filtered DataFrame.

    Parameters
    ----------
    obj : Union[pd.DataFrame, List[schemas.GroundTruth]]
        The ground truth data to be processed. This can be either a pandas DataFrame
        or a list of GroundTruth objects.
    task_type : enums.TaskType
        The task type for which the ground truth data is being filtered and validated.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the filtered and validated ground truth data.

    Raises
    ------
    ValueError
        If the input object is neither a DataFrame nor a list of GroundTruth objects.
    """
    if not (
        isinstance(obj, pd.DataFrame)
        or (
            obj
            and isinstance(obj, list)
            and isinstance(obj[0], schemas.GroundTruth)
        )
    ):
        raise ValueError(
            f"Could not validate object as it's neither a dataframe nor a list of Valor objects. Object is of type {type(obj)}."
        )
    if isinstance(obj, pd.DataFrame):
        df = obj
    else:
        df = _convert_groundtruth_or_prediction_to_dataframe(obj)

    df = _identify_implied_task_types(df=df)

    filtered_df = df[df["implied_task_type"] == task_type]

    _validate_groundtruth_dataframe(df=filtered_df, task_type=task_type)

    return filtered_df


def create_filtered_and_validated_prediction_df(
    obj: Union[pd.DataFrame, List[schemas.Prediction]],
    task_type: enums.TaskType,
) -> pd.DataFrame:
    """
    Create a filtered and validated DataFrame of prediction data.

    This function takes either a DataFrame or a list of Prediction objects,
    validates the input, identifies the implied task types, filters the DataFrame
    to match the specified task type, and then validates the filtered DataFrame.

    Parameters
    ----------
    obj : Union[pd.DataFrame, List[schemas.Prediction]]
        The prediction data to be processed. This can be either a pandas DataFrame
        or a list of GroundTruth objects.
    task_type : enums.TaskType
        The task type for which the orediction data is being filtered and validated.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the filtered and validated orediction data.

    Raises
    ------
    ValueError
        If the input object is neither a DataFrame nor a list of Prediction objects.
    """
    if not (
        isinstance(obj, pd.DataFrame)
        or (
            obj
            and isinstance(obj, list)
            and isinstance(obj[0], schemas.Prediction)
        )
    ):
        raise ValueError(
            f"Could not validate object as it's neither a dataframe nor a list of Valor objects. Object is of type {type(obj)}."
        )
    if isinstance(obj, pd.DataFrame):
        df = obj
    else:
        df = _convert_groundtruth_or_prediction_to_dataframe(obj)

    if df.empty:
        return df

    df = _identify_implied_task_types(df=df)

    filtered_df = df[df["implied_task_type"] == task_type]

    _validate_prediction_dataframe(df=filtered_df, task_type=task_type)

    return filtered_df


def _convert_groundtruth_or_prediction_to_dataframe(
    list_of_objects: Union[List[schemas.GroundTruth], List[schemas.Prediction]]
) -> pd.DataFrame:
    """
    Convert a list of GroundTruth or Prediction objects to a DataFrame.

    Parameters
    ----------
    list_of_objects : Union[List[schemas.GroundTruth], List[schemas.Prediction]]
        List of GroundTruth or Prediction objects.

    Returns
    -------
    pd.DataFrame
        DataFrame representation of the input list.
    """

    output = []

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
    groundtruth_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    label_map: Optional[Dict[schemas.Label, schemas.Label]],
) -> tuple[list[schemas.Label], list[schemas.Label]]:
    """
    Returns all unique labels that are not shared between two dataframes.

    Parameters
    ----------
    groundtruth_df : pd.DataFrame
        The dataframe representing ground truth objects.
    prediction_df : pd.DataFrame
        The dataframe representing prediction objects.

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


def _identify_implied_task_types(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Match an annotation to an implied task type."""
    # null series for use if the column doesn't exist
    null_placeholder_column = pd.Series([None] * len(df))

    # classification rows only have labels
    classification_rows = df[
        df.get("label_key", null_placeholder_column).notnull()
        & df.get("label_value", null_placeholder_column).notnull()
        & df.get("bounding_box", null_placeholder_column).isnull()
        & df.get("polygon", null_placeholder_column).isnull()
        & df.get("raster", null_placeholder_column).isnull()
        & df.get("embedding", null_placeholder_column).isnull()
    ].index

    # object detection tasks have is_instance=True & one of (bounding_box, polygon, raster)
    object_detection_rows = df[
        df.get("label_key", null_placeholder_column).notnull()
        & df.get("label_value", null_placeholder_column).notnull()
        & (
            df[
                [
                    col
                    for col in ["bounding_box", "polygon", "raster"]
                    if col in df.columns
                ]
            ]
            .notna()
            .sum(axis=1)
            == 1
        )
        & df.get("is_instance", null_placeholder_column).isin([True])
        & df.get("embedding", null_placeholder_column).isnull()
    ].index

    # semantic segmentation tasks only support rasters
    semantic_segmentation_rows = df[
        df.get("label_key", null_placeholder_column).notnull()
        & df.get("label_value", null_placeholder_column).notnull()
        & df.get("bounding_box", null_placeholder_column).isnull()
        & df.get("polygon", null_placeholder_column).isnull()
        & df.get("raster", null_placeholder_column).notnull()
        & df.get("embedding", null_placeholder_column).isnull()
        & df.get("is_instance", null_placeholder_column).isin([None, False])
    ].index

    # empty annotations shouldn't contain anything
    empty_rows = df[
        df.get("label_key", null_placeholder_column).isnull()
        & df.get("label_value", null_placeholder_column).isnull()
        & df.get("bounding_box", null_placeholder_column).isnull()
        & df.get("polygon", null_placeholder_column).isnull()
        & df.get("raster", null_placeholder_column).isnull()
        & df.get("embedding", null_placeholder_column).isnull()
    ].index

    if not classification_rows.empty:
        df.loc[
            classification_rows, "implied_task_type"
        ] = enums.TaskType.CLASSIFICATION

    if not object_detection_rows.empty:
        df.loc[
            object_detection_rows, "implied_task_type"
        ] = enums.TaskType.OBJECT_DETECTION

    if not semantic_segmentation_rows.empty:
        df.loc[
            semantic_segmentation_rows, "implied_task_type"
        ] = enums.TaskType.SEMANTIC_SEGMENTATION

    if not empty_rows.empty:
        df.loc[empty_rows, "implied_task_type"] = enums.TaskType.EMPTY

    if df["implied_task_type"].isnull().any():
        raise ValueError(
            "Input didn't match any known patterns. Classification tasks should only contain labels. Object detection tasks should contain labels and polygons, bounding boxes, or rasters with is_instance == True. Segmentation tasks should contain labels and rasters with is_instance != True. Text generation tasks should only contain text and optionally context."
        )

    return df


def _convert_raster_to_box(raster: np.ndarray) -> geometry.Box:
    """Convert a raster mask to a Box."""
    rows = np.any(raster, axis=1)
    cols = np.any(raster, axis=0)
    if not np.any(rows) or not np.any(cols):
        raise ValueError("Raster is empty, cannot create bounding box.")

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return geometry.Box.from_extrema(xmin, xmax + 1, ymin, ymax + 1)


def _convert_raster_to_polygon(raster: np.ndarray) -> geometry.Polygon:
    """Convert a raster mask to a Polygon."""
    if raster.ndim != 2:
        raise ValueError("Raster must be a 2D array.")

    mask = (raster > 0).astype(np.uint8)
    rows, cols = np.where(mask > 0)

    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("Raster is empty, cannot create a polygon.")

    contours = []
    for r, c in zip(rows, cols):
        if (
            (r > 0 and mask[r - 1, c] == 0)
            or (r < mask.shape[0] - 1 and mask[r + 1, c] == 0)
            or (c > 0 and mask[r, c - 1] == 0)
            or (c < mask.shape[1] - 1 and mask[r, c + 1] == 0)
        ):
            contours.append((c, r))

    if not contours:
        raise ValueError("No contours found in raster.")

    contours = sorted(contours, key=lambda p: (p[1], p[0]))

    polygon = [[(x, y) for x, y in contours] + [contours[0]]]

    return geometry.Polygon.from_dict(
        {"type": "Polygon", "coordinates": polygon}
    )


def _convert_polygon_to_box(polygon: geometry.Polygon) -> geometry.Box:
    """Convert a Polygon to a Box."""

    boundary = polygon.boundary

    xmin = min(point[0] for point in boundary)
    xmax = max(point[0] for point in boundary)
    ymin = min(point[1] for point in boundary)
    ymax = max(point[1] for point in boundary)

    return geometry.Box.from_extrema(xmin, xmax, ymin, ymax)


def _identify_most_detailed_annotation_type(
    df: pd.DataFrame,
) -> enums.AnnotationType:
    """
    Identify the most detailed annotation type present in the DataFrame.

    The function checks the columns in the given DataFrame for non-null values
    to determine the most detailed annotation type in the following order:
    raster, polygon, bounding_box. If none of these types are present,
    it returns AnnotationType.NONE.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the annotations.

    Returns
    -------
    enums.AnnotationType
        The most detailed annotation type present in the DataFrame.
    """

    if df["raster"].notnull().any():
        return enums.AnnotationType.RASTER

    elif df["polygon"].notnull().any():
        return enums.AnnotationType.POLYGON

    elif df["bounding_box"].notnull().any():
        return enums.AnnotationType.BOX

    else:
        return enums.AnnotationType.NONE


def _add_converted_geometry_column(
    df: pd.DataFrame,
    target_type: enums.AnnotationType,
) -> pd.DataFrame:
    """
    Add a column with converted geometries to the DataFrame.

    The function checks that each annotation contains only one type of geometry
    (bounding_box, polygon, or raster) and then converts these geometries to the
    specified target type. The resulting geometries are stored in a new column
    called 'converted_geometry'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the annotations with geometry columns.
    target_type : enums.AnnotationType
        The target annotation type to convert the geometries to.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added column 'converted_geometry' containing the converted geometries.

    Raises
    ------
    ValueError
        If an annotation contains more than one type of geometry.
    """
    if not (
        df[["bounding_box", "polygon", "raster"]].notna().sum(axis=1) == 1
    ).all():
        raise ValueError(
            "Each Annotation must contain either a bounding_box, polygon, raster, or an embedding. One Annotation cannot have multiple of these attributes (for example, one Annotation can't contain both a raster and a bounding box)."
        )

    # converted_geometry will be an array representing the original geometry
    df["converted_geometry"] = (
        df[["raster", "bounding_box", "polygon"]].bfill(axis=1).iloc[:, 0]
    )

    if target_type == enums.AnnotationType.RASTER:
        df["converted_geometry"] = df["converted_geometry"].map(
            lambda x: (
                x.to_array() if isinstance(x, geometry.Raster) else None
            )  # pyright: ignore
        )
    elif target_type == enums.AnnotationType.POLYGON:
        df["converted_geometry"] = df["converted_geometry"].map(
            lambda x: (
                _convert_raster_to_polygon(
                    x.to_array()  # pyright: ignore
                ).to_array()  # pyright: ignore
                if isinstance(x, geometry.Raster)
                else x.to_array()
                if isinstance(x, geometry.Polygon)
                else None
            )  # pyright: ignore
        )

    elif target_type == enums.AnnotationType.BOX:
        df["converted_geometry"] = df["converted_geometry"].map(
            lambda x: (
                _convert_raster_to_box(
                    x.to_array()  # pyright: ignore
                ).to_array()  # pyright: ignore
                if isinstance(x, geometry.Raster)
                else (
                    _convert_polygon_to_box(x).to_array()
                    if isinstance(x, geometry.Polygon)
                    else x.to_array()
                    if isinstance(x, geometry.Box)
                    else None
                )
            )  # pyright: ignore
        )

    return df


def convert_annotations_to_common_type(
    groundtruth_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    target_type: enums.AnnotationType | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert all annotations to a common type.

    This function converts the geometries in the provided groundtruth and prediction
    DataFrames to a common target type. If no target type is specified, it determines
    the most detailed annotation type present in the data and uses that as the target type.

    Parameters
    ----------
    groundtruth_df : pd.DataFrame
        DataFrame containing the groundtruth annotations.
    prediction_df : pd.DataFrame
        DataFrame containing the prediction annotations.
    target_type : enums.AnnotationType, optional
        The target annotation type to convert the geometries to. If None, the most
        detailed type present in the data is used.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, enums.AnnotationType]
        A tuple containing the converted groundtruth DataFrame, the converted prediction
        DataFrame, and the target annotation type used for conversion.

    Raises
    ------
    ValueError
        If the target annotation type is not supported.
    """

    if target_type is None:
        most_detailed_groundtruth_type = (
            _identify_most_detailed_annotation_type(
                df=groundtruth_df,
            )
        )

        most_detailed_prediction_type = (
            _identify_most_detailed_annotation_type(
                df=prediction_df,
            )
        )

        target_type = min(
            [most_detailed_groundtruth_type, most_detailed_prediction_type]
        )

        # Check typing
        valid_geometric_types = [
            enums.AnnotationType.BOX,
            enums.AnnotationType.POLYGON,
            enums.AnnotationType.RASTER,
        ]

        # validate that we can convert geometries successfully
        if target_type not in valid_geometric_types:
            raise ValueError(
                f"Annotation target with type `{target_type}` not supported."
            )

    groundtruth_df = _add_converted_geometry_column(
        df=groundtruth_df, target_type=target_type
    )
    prediction_df = _add_converted_geometry_column(
        df=prediction_df, target_type=target_type
    )

    return (groundtruth_df, prediction_df)
