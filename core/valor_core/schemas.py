import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from valor_core import enums, geometry


@dataclass
class Label:
    """
    An object for labeling datasets, models, and annotations.

    Attributes
    ----------
    key : str
        The label key. (e.g. 'class', 'category')
    value : str
        The label's value. (e.g. 'dog', 'cat')
    score : float, optional
        A score assigned to the label in the case of a prediction.
    """

    key: str
    value: str
    score: Optional[float] = None

    def __eq__(self, other):
        """
        Defines how labels are compared to one another.

        Parameters
        ----------
        other : Label
            The object to compare with the label.

        Returns
        ----------
        bool
            A boolean describing whether the two objects are equal.
        """
        if (
            not hasattr(other, "key")
            or not hasattr(other, "key")
            or not hasattr(other, "score")
        ):
            return False

        # if the scores aren't the same type return False
        if (other.score is None) != (self.score is None):
            return False

        if self.score is None or other.score is None:
            scores_equal = other.score is None and self.score is None
        else:
            scores_equal = math.isclose(self.score, other.score)

        return (
            scores_equal
            and self.key == other.key
            and self.value == other.value
        )

    def __hash__(self) -> int:
        """
        Defines how a 'Label' is hashed.

        Returns
        ----------
        int
            The hashed 'Label'.
        """
        return hash(f"key:{self.key},value:{self.value},score:{self.score}")


@dataclass
class Embedding:
    """
    Represents a model embedding.

    Parameters
    ----------
    value : List[float], optional
        An embedding value.
    """

    value: Optional[Union[List[int], List[float]]] = None

    @classmethod
    def __validate__(cls, value: Any):
        """
        Validates

        Parameters
        ----------
        value : Any
            The value to validate.

        Raises
        ------
        TypeError
            If the value type is not supported.
        """
        if not isinstance(value, list):
            raise TypeError(
                f"Expected type 'Optional[List[float]]' received type '{type(value)}'"
            )
        elif len(value) < 1:
            raise ValueError("embedding should have at least one dimension")


@dataclass
class Annotation:
    """
    A class used to annotate `GroundTruths` and `Predictions`.

    Attributes
    ----------
    metadata: Dictionary
        A dictionary of metadata that describes the `Annotation`.
    labels: List[Label], optional
        A list of labels to use for the `Annotation`.
    bounding_box: geometry.Box
        A bounding box to assign to the `Annotation`.
    polygon: BoundingPolygon
        A polygon to assign to the `Annotation`.
    raster: Raster
        A raster to assign to the `Annotation`.
    embedding: List[float]
        An embedding, described by a list of values with type float and a maximum length of 16,000.
    is_instance: bool, optional
        A boolean describing whether we should treat the Raster attached to an annotation as an instance segmentation or not. If set to true, then the Annotation will be validated for use in object detection tasks. If set to false, then the Annotation will be validated for use in semantic segmentation tasks.
    implied_task_types: list[str], optional
        The validated task types that are applicable to each Annotation. Doesn't need to bet set by the user.

    Examples
    --------

    Classification
    >>> Annotation.create(
    ...     labels=[
    ...         Label(key="class", value="dog"),
    ...         Label(key="category", value="animal"),
    ...     ]
    ... )

    Object-Detection geometry.Box
    >>> annotation = Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     bounding_box=box2,
    ... )

    Object-Detection geometry.Polygon
    >>> annotation = Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     polygon=BoundingPolygon(...),
    ... )

     Raster
    >>> annotation = Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     raster=Raster(...),
    ...     is_instance=True
    ... )

    Object-Detection with all supported Geometries defined.
    >>> Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     bounding_box=geometry.Box(...),
    ...     polygon=BoundingPolygon(...),
    ...     raster=Raster(...),
    ...     is_instance=True,
    ... )

    Semantic-Segmentation Raster
    >>> annotation = Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     raster=Raster(...),
    ...     is_instance=False # or None
    ... )
    """

    labels: List[Label]
    metadata: Optional[dict] = None
    bounding_box: Optional[geometry.Box] = None
    polygon: Optional[geometry.Polygon] = None
    raster: Optional[geometry.Raster] = None
    embedding: Optional[Embedding] = None
    is_instance: Optional[bool] = None
    implied_task_types: Optional[List[str]] = None


@dataclass
class Datum:
    """
    A class used to store information about a datum for either a 'GroundTruth' or a 'Prediction'.

    Attributes
    ----------
    uid : String
        The UID of the datum.
    metadata : Dictionary
        A dictionary of metadata that describes the datum.

    Examples
    --------
    >>> Datum(uid="uid1")
    >>> Datum(uid="uid1", metadata={})
    >>> Datum(uid="uid1", metadata={"foo": "bar", "pi": 3.14})
    """

    uid: Optional[str] = None
    metadata: Optional[dict] = None


@dataclass
class EvaluationParameters:
    """
    Defines optional parameters for evaluation methods.

    Attributes
    ----------
    label_map: Optional[List[List[List[str]]]]
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
    metrics: List[str], optional
        The list of metrics to compute, store, and return to the user.
    convert_annotations_to_type: AnnotationType | None = None
        The type to convert all annotations to.
    iou_thresholds_to_compute: List[float], optional
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_return: List[float], optional
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    recall_score_threshold: float, default=0
        The confidence score threshold for use when determining whether to count a prediction as a true positive or not while calculating Average Recall.
    pr_curve_iou_threshold: float, optional
            The IOU threshold to use when calculating precision-recall curves for object detection tasks. Defaults to 0.5.
    pr_curve_max_examples: int
        The maximum number of datum examples to store when calculating PR curves.
        # TODO
    """

    label_map: Optional[Dict[Label, Label]] = None
    metrics_to_return: Optional[List[enums.MetricType]] = None
    convert_annotations_to_type: Optional[enums.AnnotationType] = None
    iou_thresholds_to_compute: Optional[List[float]] = None
    iou_thresholds_to_return: Optional[List[float]] = None
    recall_score_threshold: float = 0.0
    pr_curve_iou_threshold: float = 0.5
    pr_curve_max_examples: int = 1


@dataclass
class Evaluation:
    # TODO docstring
    parameters: EvaluationParameters
    metrics: List[Dict]
    confusion_matrices: Optional[List[Dict]]
    ignored_pred_labels: Optional[List[Label]]
    missing_pred_labels: Optional[List[Label]]
    meta: Optional[Dict] = None

    def __str__(self) -> str:
        """Dumps the object into a JSON formatted string."""
        return json.dumps(self.__dict__, indent=4)

    def to_dict(self) -> dict:
        """
        Defines how a `valor.Evaluation` object is serialized into a dictionary.

        Returns
        ----------
        dict
            A dictionary describing an evaluation.
        """
        return {
            "parameters": self.parameters.__dict__,
            "metrics": self.metrics,
            "confusion_matrices": self.confusion_matrices,
            # TODO make sure these two get tested at some point
            "ignored_pred_labels": self.ignored_pred_labels,
            "missing_pred_labels": self.missing_pred_labels,
            "meta": self.meta,
        }


@dataclass
class GroundTruth:
    """
    An object describing a ground truth (e.g., a human-drawn bounding box on an image).

    Attributes
    ----------
    datum : Datum
        The datum associated with the groundtruth.
    annotations : List[Annotation]
        The list of annotations associated with the groundtruth.

    Examples
    --------
    >>> GroundTruth(
    ...     datum=Datum(uid="uid1"),
    ...     annotations=[
    ...         Annotation(
    ...             labels=[Label(key="k1", value="v1")],
    ...         )
    ...     ]
    ... )
    """

    datum: Datum
    annotations: list[Annotation]

    def __post_init__(
        self,
    ):
        """
        Creates a ground truth.

        Parameters
        ----------
        datum : Datum
            The datum that the ground truth is operating over.
        annotations : List[Annotation]
            The list of ground truth annotations.
        """

        for annotation in self.annotations:
            if annotation.labels:
                for label in annotation.labels:
                    if label.score is not None:
                        raise ValueError(
                            "GroundTruth labels should not have scores."
                        )


@dataclass
class Prediction:
    """
    An object describing a prediction (e.g., a machine-drawn bounding box on an image).

    Attributes
    ----------
    datum : Datum
        The datum associated with the prediction.
    annotations : List[Annotation]
        The list of annotations associated with the prediction.

    Examples
    --------
    >>> Prediction(
    ...     datum=Datum(uid="uid1"),
    ...     annotations=[
    ...         Annotation(
    ...             labels=[
    ...                 Label(key="k1", value="v1", score=0.9),
    ...                 Label(key="k1", value="v1", score=0.1)
    ...             ],
    ...         )
    ...     ]
    ... )
    """

    datum: Datum
    annotations: list[Annotation]


@dataclass
class _LabelMetricBase:
    """
    Defines a base class for label-level metrics.

    Attributes
    ----------
    label : label
        A label for the metric.
    value : float
        The metric value.
    """

    label: Label
    value: Optional[float]
    __type__ = "BaseClass"

    def to_dict(self):
        """Converts a metric object into a dictionary."""
        return {
            "label": {"key": self.label.key, "value": self.label.value},
            "value": self.value,
            "type": self.__type__,
        }


@dataclass
class _LabelKeyMetricBase:
    """
    Defines a base class for label key-level metrics.

    Attributes
    ----------
    label_key : str
        The label key associated with the metric.
    value : float
        The metric value.
    """

    label_key: str
    value: Optional[float]
    __type__ = "BaseClass"

    def to_dict(self):
        """Converts a metric object into a dictionary."""
        return {
            "parameters": {"label_key": self.label_key},
            "value": self.value,
            "type": self.__type__,
        }


@dataclass
class ARMetric(_LabelMetricBase):
    """
    # TODO
    An AR metric response from the API.

    Attributes
    ----------
    ious : set[float]
        A set of intersect-over-union (IOU) values.
    value : float
        The value of the metric.
    label : Label
        The `Label` for the metric.
    """

    ious: set[float]
    __type__ = "AR"

    def to_dict(self):
        """Converts a metric object into a dictionary."""
        return {
            "label": {"key": self.label.key, "value": self.label.value},
            "parameters": {"ious": sorted(list(self.ious))},
            "value": self.value,
            "type": self.__type__,
        }


@dataclass
class APMetric(_LabelMetricBase):
    """
    # TODO
    An AP metric response from the API.

    Attributes
    ----------
    ious : set[float]
        A set of intersect-over-union (IOU) values.
    value : float
        The value of the metric.
    label : Label
        The `Label` for the metric.
    """

    iou: float
    __type__ = "AP"

    def to_dict(self):
        """Converts a metric object into a dictionary."""
        return {
            "label": {"key": self.label.key, "value": self.label.value},
            "parameters": {"iou": self.iou},
            "value": self.value,
            "type": self.__type__,
        }


@dataclass
class APMetricAveragedOverIOUs(_LabelMetricBase):
    """
    # TODO
    An AR metric response from the API.

    Attributes
    ----------
    ious : set[float]
        A set of intersect-over-union (IOU) values.
    value : float
        The value of the metric.
    label : Label
        The `Label` for the metric.
    """

    ious: set[float]
    __type__ = "APAveragedOverIOUs"

    def to_dict(self):
        """Converts a metric object into a dictionary."""
        return {
            "label": {"key": self.label.key, "value": self.label.value},
            "parameters": {"ious": sorted(list(self.ious))},
            "value": self.value,
            "type": self.__type__,
        }


@dataclass
class mARMetric(_LabelKeyMetricBase):
    """
    An mAR metric response from the API.

    Attributes
    ----------
    ious : set[float]
        A set of intersect-over-union (IOU) values.
    value : float
        The value of the metric.
    label_key : str
        The label key associated with the metric.
    """

    ious: set[float]
    __type__ = "mAR"

    def to_dict(self):
        """Converts a metric object into a dictionary."""
        return {
            "parameters": {
                "label_key": self.label_key,
                "ious": sorted(list(self.ious)),
            },
            "value": self.value,
            "type": self.__type__,
        }


@dataclass
class mAPMetric(_LabelKeyMetricBase):
    """
    An mAR metric response from the API.

    Attributes
    ----------
    ious : set[float]
        A set of intersect-over-union (IOU) values.
    value : float
        The value of the metric.
    label_key : str
        The label key associated with the metric.
    """

    iou: float
    __type__ = "mAP"

    def to_dict(self):
        """Converts a metric object into a dictionary."""
        return {
            "parameters": {"label_key": self.label_key, "iou": self.iou},
            "value": self.value,
            "type": self.__type__,
        }


@dataclass
class mAPMetricAveragedOverIOUs(_LabelKeyMetricBase):
    """
    An mAR metric response from the API.

    Attributes
    ----------
    ious : set[float]
        A set of intersect-over-union (IOU) values.
    value : float
        The value of the metric.
    label_key : str
        The label key associated with the metric.
    """

    ious: set[float]
    __type__ = "mAPAveragedOverIOUs"

    def to_dict(self):
        """Converts a metric object into a dictionary."""
        return {
            "parameters": {
                "label_key": self.label_key,
                "ious": sorted(list(self.ious)),
            },
            "value": self.value,
            "type": self.__type__,
        }


class PrecisionMetric(_LabelMetricBase):
    """
    Describes a precision metric.

    Attributes
    ----------
    label : Label
        A key-value pair.
    value : float, optional
        The metric value.
    """

    __type__ = "Precision"


class RecallMetric(_LabelMetricBase):
    """
    Describes a recall metric.

    Attributes
    ----------
    label : Label
        A key-value pair.
    value : float, optional
        The metric value.
    """

    __type__ = "Recall"


class F1Metric(_LabelMetricBase):
    """
    Describes an F1 metric.

    Attributes
    ----------
    label : Label
        A key-value pair.
    value : float, optional
        The metric value.
    """

    __type__ = "F1"


class ROCAUCMetric(_LabelKeyMetricBase):
    """
    Describes an ROC AUC metric.

    Attributes
    ----------
    label_key : str
        The label key associated with the metric.
    value : float
        The metric value.
    """

    __type__ = "ROCAUC"


class AccuracyMetric(_LabelKeyMetricBase):
    """
    Describes an accuracy metric.

    Attributes
    ----------
    label_key : str
        The label key associated with the metric.
    value : float
        The metric value.
    """

    __type__ = "Accuracy"


@dataclass
class _BasePrecisionRecallCurve:
    """
    Describes the parent class of our precision-recall curve metrics.

    Attributes
    ----------
    label_key: str
        The label key associated with the metric.
    pr_curve_iou_threshold: float, optional
        The IOU threshold to use when calculating precision-recall curves. Defaults to 0.5.
    """

    label_key: str
    value: dict
    pr_curve_iou_threshold: Optional[float]
    __type__ = "BaseClass"

    def to_dict(self):
        """Converts a metric object into a dictionary."""
        return {
            "parameters": {"label_key": self.label_key},
            "value": self.value,
            "type": self.__type__,
        }


class PrecisionRecallCurve(_BasePrecisionRecallCurve):
    """
    Describes a precision-recall curve.

    Attributes
    ----------
    label_key: str
        The label key associated with the metric.
    value: dict
        A nested dictionary where the first key is the class label, the second key is the confidence threshold (e.g., 0.05), the third key is the metric name (e.g., "precision"), and the final key is either the value itself (for precision, recall, etc.) or a list of tuples containing data for each observation.
    pr_curve_iou_threshold: float, optional
        The IOU threshold to use when calculating precision-recall curves. Defaults to 0.5.
    """

    __type__ = "PrecisionRecallCurve"
    value: Dict[
        str,
        Dict[
            float,
            Dict[str, Optional[Union[int, float]]],
        ],
    ]


class DetailedPrecisionRecallCurve(_BasePrecisionRecallCurve):
    """
    Describes a detailed precision-recall curve, which includes datum examples for each classification (e.g., true positive, false negative, etc.).

    Attributes
    ----------
    label_key: str
        The label key associated with the metric.
    value: dict
        A nested dictionary where the first key is the class label, the second key is the confidence threshold (e.g., 0.05), the third key is the metric name (e.g., "precision"), and the final key is either the value itself (for precision, recall, etc.) or a list of tuples containing data for each observation.
    pr_curve_iou_threshold: float, optional
        The IOU threshold to use when calculating precision-recall curves. Defaults to 0.5.
    """

    __type__ = "DetailedPrecisionRecallCurve"
    value: Dict[
        str,  # the label value
        Dict[
            float,  # the score threshold
            Dict[
                str,  # the metric (e.g., "tp" for true positive)
                Dict[
                    str,  # the label for the next level of the dictionary (e.g., "observations" or "total")
                    Union[
                        int,  # the count of classifications
                        Dict[
                            str,  # the subclassification for the label (e.g., "misclassifications")
                            Dict[
                                str,  # the label for the next level of the dictionary (e.g., "count" or "examples")
                                Union[
                                    int,  # the count of subclassifications
                                    List[
                                        Union[
                                            Tuple[str, str],
                                            Tuple[str, str, str],
                                        ]
                                    ],
                                ],  # a list containing examples
                            ],
                        ],
                    ],
                ],
            ],
        ],
    ]


@dataclass
class ConfusionMatrixEntry:
    """
    Describes one element in a confusion matrix.

    Attributes
    ----------
    prediction : str
        The prediction.
    groundtruth : str
        The ground truth.
    count : int
        The value of the element in the matrix.
    """

    prediction: str
    groundtruth: str
    count: int

    def to_dict(self):
        """Converts a ConfusionMatrixEntry object into a dictionary."""
        return {
            "prediction": self.prediction,
            "groundtruth": self.groundtruth,
            "count": self.count,
        }


@dataclass
class _BaseConfusionMatrix:
    """
    Describes a base confusion matrix.

    Attributes
    ----------
    label_ley : str
        A label for the matrix.
    entries : List[ConfusionMatrixEntry]
        A list of entries for the matrix.
    """

    label_key: str
    entries: List[ConfusionMatrixEntry]

    def to_dict(self):
        """Converts a ConfusionMatrix object into a dictionary."""
        return {
            "label_key": self.label_key,
            "entries": [entry.to_dict() for entry in self.entries],
        }


class ConfusionMatrix(_BaseConfusionMatrix):
    """
    Describes a confusion matrix.

    Attributes
    ----------
    label_key : str
        A label for the matrix.
    entries : List[ConfusionMatrixEntry]
        A list of entries for the matrix.

    Attributes
    ----------
    matrix : np.zeroes
        A sparse matrix representing the confusion matrix.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        label_values = set(
            [entry.prediction for entry in self.entries]
            + [entry.groundtruth for entry in self.entries]
        )
        self.label_map = {
            label_value: i
            for i, label_value in enumerate(sorted(label_values))
        }
        n_label_values = len(self.label_map)

        matrix = np.zeros((n_label_values, n_label_values), dtype=int)
        for entry in self.entries:
            matrix[
                self.label_map[entry.groundtruth],
                self.label_map[entry.prediction],
            ] = entry.count

        self.matrix = matrix
