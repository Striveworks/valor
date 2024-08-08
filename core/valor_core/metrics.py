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

    def __post_init__(self):
        """
        Validates the types of the attributes.

        Raises
        ------
        TypeError
            If `label` is not an instance of schemas.Label.
            If `value` is not a float or None.
        """
        if not isinstance(self.label, schemas.Label):
            raise TypeError(
                f"Expected label to be an instance of schemas.Label, got {type(self.label).__name__}"
            )
        if self.value is not None and not isinstance(self.value, float):
            raise TypeError(
                f"Expected value to be a float or None, got {type(self.value).__name__}"
            )

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

    def __post_init__(self):
        """
        Validates the types of the attributes.

        Raises
        ------
        TypeError
            If `label_key` is not a string.
            If `value` is not a float or None.
        """
        if not isinstance(self.label_key, str):
            raise TypeError(
                f"Expected label_key to be a string, got {type(self.label_key).__name__}"
            )
        if self.value is not None and not isinstance(self.value, float):
            raise TypeError(
                f"Expected value to be a float or None, got {type(self.value).__name__}"
            )

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
    Defines an AR metric.

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

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.ious, set):
            raise TypeError(
                f"Expected ious to be a set, got {type(self.ious).__name__}"
            )

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
    Defines an AP metric.

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

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.iou, float):
            raise TypeError(
                f"Expected iou to be a float, got {type(self.iou).__name__}"
            )

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
    Defines an APMetricAveragedOverIOUs metric.

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

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.ious, set):
            raise TypeError(
                f"Expected ious to be a set, got {type(self.ious).__name__}"
            )

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
    Defines a mAR metric.

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

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.ious, set):
            raise TypeError(
                f"Expected ious to be a set, got {type(self.ious).__name__}"
            )

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
    Defines a mAP metric.

    Attributes
    ----------
    iou: float
        An intersect-over-union (IOU) value.
    value : float
        The value of the metric.
    label_key : str
        The label key associated with the metric.
    """

    iou: float
    __type__ = "mAP"

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.iou, float):
            raise TypeError(
                f"Expected iou to be a float, got {type(self.iou).__name__}"
            )

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
    Defines a mAR metric.

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

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.ious, set):
            raise TypeError(
                f"Expected ious to be a set, got {type(self.ious).__name__}"
            )

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
    Defines a Precision metric.

    Attributes
    ----------
    label : Label
        A key-value pair.
    value : float, optional
        The metric value.
    """

    __type__ = "Precision"

    def __post_init__(self):
        super().__post_init__()


class RecallMetric(_LabelMetricBase):
    """
    Defines a Recall metric.

    Attributes
    ----------
    label : Label
        A key-value pair.
    value : float, optional
        The metric value.
    """

    __type__ = "Recall"

    def __post_init__(self):
        super().__post_init__()


class F1Metric(_LabelMetricBase):
    """
    Defines a F1 metric.

    Attributes
    ----------
    label : Label
        A key-value pair.
    value : float, optional
        The metric value.
    """

    __type__ = "F1"

    def __post_init__(self):
        super().__post_init__()


class ROCAUCMetric(_LabelKeyMetricBase):
    """
    Defines a ROC AUC metric.

    Attributes
    ----------
    label_key : str
        The label key associated with the metric.
    value : float
        The metric value.
    """

    __type__ = "ROCAUC"

    def __post_init__(self):
        super().__post_init__()


class AccuracyMetric(_LabelKeyMetricBase):
    """
    Defines a accuracy metric.

    Attributes
    ----------
    label_key : str
        The label key associated with the metric.
    value : float
        The metric value.
    """

    __type__ = "Accuracy"

    def __post_init__(self):
        super().__post_init__()


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

    def __post_init__(self):
        if not isinstance(self.label_key, str):
            raise TypeError(
                f"Expected label_key to be a string, but got {type(self.label_key).__name__}."
            )

        if not isinstance(self.value, dict):
            raise TypeError(
                f"Expected value to be a dictionary, but got {type(self.value).__name__}."
            )

        if self.pr_curve_iou_threshold is not None and not isinstance(
            self.pr_curve_iou_threshold, float
        ):
            raise TypeError(
                f"Expected pr_curve_iou_threshold to be a float or None, but got {type(self.pr_curve_iou_threshold).__name__}."
            )

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

    def __post_init__(self):
        super().__post_init__()


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

    def __post_init__(self):
        super().__post_init__()


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

    def __post_init__(self):
        if not isinstance(self.prediction, str):
            raise TypeError(
                f"Expected prediction to be a string, but got {type(self.prediction).__name__}."
            )

        if not isinstance(self.groundtruth, str):
            raise TypeError(
                f"Expected groundtruth to be a string, but got {type(self.groundtruth).__name__}."
            )

        if not isinstance(self.count, int):
            raise TypeError(
                f"Expected count to be an integer, but got {type(self.count).__name__}."
            )

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

    def __post_init__(self):
        if not isinstance(self.label_key, str):
            raise TypeError(
                f"Expected label_key to be a string, but got {type(self.label_key).__name__}."
            )

        if not isinstance(self.entries, list):
            raise TypeError(
                f"Expected entries to be a list, but got {type(self.entries).__name__}."
            )

        for entry in self.entries:
            if not isinstance(entry, ConfusionMatrixEntry):
                raise TypeError(
                    f"Expected entry to be of type ConfusionMatrixEntry, but got {type(entry).__name__}."
                )

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
