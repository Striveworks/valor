from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from valor_core import schemas


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

    label: schemas.Label
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
