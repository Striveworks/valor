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
        hallucinations: dict[
            str,  # prediction label value
            dict[str, float],  # pixel ratio
        ],
        missing_predictions: dict[
            str,  # ground truth label value
            dict[str, float],  # pixel ratio
        ],
    ):
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

        Parameters
        ----------
        confusion_matrix : dict
            Nested dictionaries representing the Intersection over Union (IOU) scores for each
            ground truth label and prediction label pair.
        hallucinations : dict
            Dictionary representing the pixel ratios for predicted labels that do not correspond
            to any ground truth labels (false positives).
        missing_predictions : dict
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
                "hallucinations": hallucinations,
                "missing_predictions": missing_predictions,
            },
            parameters={},
        )
