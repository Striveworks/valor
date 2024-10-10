from dataclasses import dataclass
from enum import Enum

from valor_lite.schemas import Metric


class MetricType(Enum):
    Precision = "Precision"
    Recall = "Recall"
    Accuracy = "Accuracy"
    F1 = "F1"
    IoU = "IoU"
    mIoU = "mIoU"
    ConfusionMatrix = "ConfusionMatrix"

    @classmethod
    def base(cls):
        return [
            cls.Precision,
            cls.Recall,
            cls.Accuracy,
            cls.F1,
            cls.IoU,
            cls.mIoU,
            cls.ConfusionMatrix,
        ]


@dataclass
class _LabelValue:
    value: float
    label: str

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


class Precision(_LabelValue):
    """
    Precision metric for a specific class label.

    Precision is calulated using the number of true-positive pixels divided by
    the sum of all true-positive and false-positive pixels.

    Attributes
    ----------
    value : float
        The computed precision value.
    label : str
        The label for which the precision is calculated.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    pass


class Recall(_LabelValue):
    """
    Recall metric for a specific class label.

    Recall is calulated using the number of true-positive pixels divided by
    the sum of all true-positive and false-negative pixels.

    Attributes
    ----------
    value : float
        The computed recall value.
    label : str
        The label for which the recall is calculated.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    pass


class F1(_LabelValue):
    """
    F1 score for a specific class label.

    Attributes
    ----------
    value : float
        The computed F1 score.
    label : str
        The label for which the F1 score is calculated.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    pass


class IoU(_LabelValue):
    """
    Intersection over Union (IoU) ratio for a specific class label.

    Attributes
    ----------
    value : float
        The computed IoU ratio.
    label : str
        The label for which the IoU is calculated.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    pass


@dataclass
class _Value:
    value: float

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={},
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


class Accuracy(_Value):
    """
    Accuracy metric computed over all labels.

    Attributes
    ----------
    value : float
        The accuracy value.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    pass


class mIoU(_Value):
    """
    Mean Intersection over Union (mIoU) ratio.

    The mIoU value is computed by averaging IoU over all labels.

    Attributes
    ----------
    value : float
        The mIoU value.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    pass


@dataclass
class ConfusionMatrix:
    """
    The confusion matrix and related metrics for semantic segmentation tasks.

    This class encapsulates detailed information about the model's performance, including correct
    predictions, misclassifications, hallucinations (false positives), and missing predictions
    (false negatives). It provides counts for each category to facilitate in-depth analysis.

    Confusion Matrix Format:
    {
        <ground truth label>: {
            <prediction label>: {
                'iou': <float>,
            },
        },
    }

    Hallucinations Format:
    {
        <prediction label>: {
            'iou': <float>,
        },
    }

    Missing Predictions Format:
    {
        <ground truth label>: {
            'iou': <float>,
        },
    }

    Attributes
    ----------
    confusion_matrix : dict
        Nested dictionaries representing the Intersection over Union (IoU) scores for each
        ground truth label and prediction label pair.
    hallucinations : dict
        Dictionary representing the pixel ratios for predicted labels that do not correspond
        to any ground truth labels (false positives).
    missing_predictions : dict
        Dictionary representing the pixel ratios for ground truth labels that were not predicted
        (false negatives).

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    confusion_matrix: dict[
        str,  # ground truth label value
        dict[
            str,  # prediction label value
            dict[str, float],  # iou
        ],
    ]
    hallucinations: dict[
        str,  # prediction label value
        dict[str, float],  # pixel ratio
    ]
    missing_predictions: dict[
        str,  # ground truth label value
        dict[str, float],  # pixel ratio
    ]

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value={
                "confusion_matrix": self.confusion_matrix,
                "hallucinations": self.hallucinations,
                "missing_predictions": self.missing_predictions,
            },
            parameters={},
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()
