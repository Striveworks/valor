from dataclasses import dataclass
from enum import Enum

from valor_lite.schemas import BaseMetric


class MetricType(Enum):
    Precision = "Precision"
    Recall = "Recall"
    Accuracy = "Accuracy"
    F1 = "F1"
    IOU = "IOU"
    mIOU = "mIOU"
    ConfusionMatrix = "ConfusionMatrix"


@dataclass
class Metric(BaseMetric):
    """
    Semantic Segmentation Metric.

    Attributes
    ----------
    type : str
        The metric type.
    value : int | float | dict
        The metric value.
    parameters : dict[str, Any]
        A dictionary containing metric parameters.
    """

    def __post_init__(self):
        if not isinstance(self.type, str):
            raise TypeError(
                f"Metric type should be of type 'str': {self.type}"
            )
        elif not isinstance(self.value, (int, float, dict)):
            raise TypeError(
                f"Metric value must be of type 'int', 'float' or 'dict': {self.value}"
            )
        elif not isinstance(self.parameters, dict):
            raise TypeError(
                f"Metric parameters must be of type 'dict[str, Any]': {self.parameters}"
            )
        elif not all([isinstance(k, str) for k in self.parameters.keys()]):
            raise TypeError(
                f"Metric parameter dictionary should only have keys with type 'str': {self.parameters}"
            )

    @classmethod
    def precision(
        cls,
        value: float,
        label: str,
    ):
        """
        Precision metric for a specific class label.

        Precision is calulated using the number of true-positive pixels divided by
        the sum of all true-positive and false-positive pixels.

        Parameters
        ----------
        value : float
            The computed precision value.
        label : str
            The label for which the precision is calculated.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.Precision.value,
            value=value,
            parameters={
                "label": label,
            },
        )

    @classmethod
    def recall(
        cls,
        value: float,
        label: str,
    ):
        """
        Recall metric for a specific class label.

        Recall is calulated using the number of true-positive pixels divided by
        the sum of all true-positive and false-negative pixels.

        Parameters
        ----------
        value : float
            The computed recall value.
        label : str
            The label for which the recall is calculated.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.Recall.value,
            value=value,
            parameters={
                "label": label,
            },
        )

    @classmethod
    def f1_score(
        cls,
        value: float,
        label: str,
    ):
        """
        F1 score for a specific class label.

        Parameters
        ----------
        value : float
            The computed F1 score.
        label : str
            The label for which the F1 score is calculated.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.F1.value,
            value=value,
            parameters={
                "label": label,
            },
        )

    @classmethod
    def iou(
        cls,
        value: float,
        label: str,
    ):
        """
        Intersection over Union (IOU) ratio for a specific class label.

        Parameters
        ----------
        value : float
            The computed IOU ratio.
        label : str
            The label for which the IOU is calculated.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.IOU.value,
            value=value,
            parameters={
                "label": label,
            },
        )

    @classmethod
    def mean_iou(cls, value: float):
        """
        Mean Intersection over Union (mIOU) ratio.

        The mIOU value is computed by averaging IOU over all labels.

        Parameters
        ----------
        value : float
            The mIOU value.

        Returns
        -------
        Metric
        """
        return cls(type=MetricType.mIOU.value, value=value, parameters={})

    @classmethod
    def accuracy(cls, value: float):
        """
        Accuracy metric computed over all labels.

        Parameters
        ----------
        value : float
            The accuracy value.

        Returns
        -------
        Metric
        """
        return cls(type=MetricType.Accuracy.value, value=value, parameters={})

    @classmethod
    def confusion_matrix(
        cls,
        confusion_matrix: dict[
            str,  # ground truth label value
            dict[
                str,  # prediction label value
                dict[str, float],  # iou
            ],
        ],
        unmatched_predictions: dict[
            str,  # prediction label value
            dict[str, float],  # pixel ratio
        ],
        unmatched_ground_truths: dict[
            str,  # ground truth label value
            dict[str, float],  # pixel ratio
        ],
    ):
        """
        The confusion matrix and related metrics for semantic segmentation tasks.

        This class encapsulates detailed information about the model's performance, including correct
        predictions, misclassifications, unmatched_predictions (subset of false positives), and unmatched ground truths
        (subset of false negatives). It provides counts for each category to facilitate in-depth analysis.

        Confusion Matrix Format:
        {
            <ground truth label>: {
                <prediction label>: {
                    'iou': <float>,
                },
            },
        }

        Unmatched Predictions Format:
        {
            <prediction label>: {
                'iou': <float>,
            },
        }

        Unmatched Ground Truths Format:
        {
            <ground truth label>: {
                'iou': <float>,
            },
        }

        Parameters
        ----------
        confusion_matrix : dict
            Nested dictionaries representing the Intersection over Union (IOU) scores for each
            ground truth label and prediction label pair.
        unmatched_predictions : dict
            Dictionary representing the pixel ratios for predicted labels that do not correspond
            to any ground truth labels (false positives).
        unmatched_ground_truths : dict
            Dictionary representing the pixel ratios for ground truth labels that were not predicted
            (false negatives).

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.ConfusionMatrix.value,
            value={
                "confusion_matrix": confusion_matrix,
                "unmatched_predictions": unmatched_predictions,
                "unmatched_ground_truths": unmatched_ground_truths,
            },
            parameters={},
        )
