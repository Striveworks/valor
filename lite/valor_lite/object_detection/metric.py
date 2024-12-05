from dataclasses import dataclass
from enum import Enum

from valor_lite.schemas import BaseMetric


class MetricType(str, Enum):
    Counts = "Counts"
    Accuracy = "Accuracy"
    Precision = "Precision"
    Recall = "Recall"
    F1 = "F1"
    AP = "AP"
    AR = "AR"
    mAP = "mAP"
    mAR = "mAR"
    APAveragedOverIOUs = "APAveragedOverIOUs"
    mAPAveragedOverIOUs = "mAPAveragedOverIOUs"
    ARAveragedOverScores = "ARAveragedOverScores"
    mARAveragedOverScores = "mARAveragedOverScores"
    PrecisionRecallCurve = "PrecisionRecallCurve"
    ConfusionMatrix = "ConfusionMatrix"


@dataclass
class Metric(BaseMetric):
    """
    Object Detection Metric.

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
        iou_threshold: float,
        score_threshold: float,
    ):
        """
        Precision metric for a specific class label in object detection.

        This class encapsulates a metric value for a particular class label,
        along with the associated Intersection over Union (IOU) threshold and
        confidence score threshold.

        Parameters
        ----------
        value : float
            The metric value.
        label : str
            The class label for which the metric is calculated.
        iou_threshold : float
            The IOU threshold used to determine matches between predicted and ground truth boxes.
        score_threshold : float
            The confidence score threshold above which predictions are considered.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.Precision.value,
            value=value,
            parameters={
                "label": label,
                "iou_threshold": iou_threshold,
                "score_threshold": score_threshold,
            },
        )

    @classmethod
    def recall(
        cls,
        value: float,
        label: str,
        iou_threshold: float,
        score_threshold: float,
    ):
        """
        Recall metric for a specific class label in object detection.

        This class encapsulates a metric value for a particular class label,
        along with the associated Intersection over Union (IOU) threshold and
        confidence score threshold.

        Parameters
        ----------
        value : float
            The metric value.
        label : str
            The class label for which the metric is calculated.
        iou_threshold : float
            The IOU threshold used to determine matches between predicted and ground truth boxes.
        score_threshold : float
            The confidence score threshold above which predictions are considered.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.Recall.value,
            value=value,
            parameters={
                "label": label,
                "iou_threshold": iou_threshold,
                "score_threshold": score_threshold,
            },
        )

    @classmethod
    def f1_score(
        cls,
        value: float,
        label: str,
        iou_threshold: float,
        score_threshold: float,
    ):
        """
        F1 score for a specific class label in object detection.

        This class encapsulates a metric value for a particular class label,
        along with the associated Intersection over Union (IOU) threshold and
        confidence score threshold.

        Parameters
        ----------
        value : float
            The metric value.
        label : str
            The class label for which the metric is calculated.
        iou_threshold : float
            The IOU threshold used to determine matches between predicted and ground truth boxes.
        score_threshold : float
            The confidence score threshold above which predictions are considered.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.F1.value,
            value=value,
            parameters={
                "label": label,
                "iou_threshold": iou_threshold,
                "score_threshold": score_threshold,
            },
        )

    @classmethod
    def accuracy(
        cls,
        value: float,
        iou_threshold: float,
        score_threshold: float,
    ):
        """
        Accuracy metric for the object detection task type.

        This class encapsulates a metric value at a specific Intersection
        over Union (IOU) threshold and confidence score threshold.

        Parameters
        ----------
        value : float
            The metric value.
        iou_threshold : float
            The IOU threshold used to determine matches between predicted and ground truth boxes.
        score_threshold : float
            The confidence score threshold above which predictions are considered.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.Accuracy.value,
            value=value,
            parameters={
                "iou_threshold": iou_threshold,
                "score_threshold": score_threshold,
            },
        )

    @classmethod
    def average_precision(
        cls,
        value: float,
        iou_threshold: float,
        label: str,
    ):
        """
        Average Precision (AP) metric for object detection tasks.

        The AP computation uses 101-point interpolation, which calculates the average
        precision by interpolating the precision-recall curve at 101 evenly spaced recall
        levels from 0 to 1.

        Parameters
        ----------
        value : float
            The average precision value.
        iou_threshold : float
            The IOU threshold used to compute the AP.
        label : str
            The class label for which the AP is computed.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.AP.value,
            value=value,
            parameters={
                "iou_threshold": iou_threshold,
                "label": label,
            },
        )

    @classmethod
    def mean_average_precision(
        cls,
        value: float,
        iou_threshold: float,
    ):
        """
        Mean Average Precision (mAP) metric for object detection tasks.

        The AP computation uses 101-point interpolation, which calculates the average
        precision for each class by interpolating the precision-recall curve at 101 evenly
        spaced recall levels from 0 to 1. The mAP is then calculated by averaging these
        values across all class labels.

        Parameters
        ----------
        value : float
            The mean average precision value.
        iou_threshold : float
            The IOU threshold used to compute the mAP.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.mAP.value,
            value=value,
            parameters={
                "iou_threshold": iou_threshold,
            },
        )

    @classmethod
    def average_precision_averaged_over_IOUs(
        cls,
        value: float,
        iou_thresholds: list[float],
        label: str,
    ):
        """
        Average Precision (AP) metric averaged over multiple IOU thresholds.

        The AP computation uses 101-point interpolation, which calculates the average precision
        by interpolating the precision-recall curve at 101 evenly spaced recall levels from 0 to 1
        for each IOU threshold specified in `iou_thresholds`. The final APAveragedOverIOUs value is
        obtained by averaging these AP values across all specified IOU thresholds.

        Parameters
        ----------
        value : float
            The average precision value averaged over the specified IOU thresholds.
        iou_thresholds : list[float]
            The list of IOU thresholds used to compute the AP values.
        label : str
            The class label for which the AP is computed.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.APAveragedOverIOUs.value,
            value=value,
            parameters={
                "iou_thresholds": iou_thresholds,
                "label": label,
            },
        )

    @classmethod
    def mean_average_precision_averaged_over_IOUs(
        cls,
        value: float,
        iou_thresholds: list[float],
    ):
        """
        Mean Average Precision (mAP) metric averaged over multiple IOU thresholds.

        The AP computation uses 101-point interpolation, which calculates the average precision
        by interpolating the precision-recall curve at 101 evenly spaced recall levels from 0 to 1
        for each IOU threshold specified in `iou_thresholds`. The final mAPAveragedOverIOUs value is
        obtained by averaging these AP values across all specified IOU thresholds and all class labels.

        Parameters
        ----------
        value : float
            The average precision value averaged over the specified IOU thresholds.
        iou_thresholds : list[float]
            The list of IOU thresholds used to compute the AP values.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.mAPAveragedOverIOUs.value,
            value=value,
            parameters={
                "iou_thresholds": iou_thresholds,
            },
        )

    @classmethod
    def average_recall(
        cls,
        value: float,
        score_threshold: float,
        iou_thresholds: list[float],
        label: str,
    ):
        """
        Average Recall (AR) metric for object detection tasks.

        The AR computation considers detections with confidence scores above the specified
        `score_threshold` and calculates the recall at each IOU threshold in `iou_thresholds`.
        The final AR value is the average of these recall values across all specified IOU
        thresholds.

        Parameters
        ----------
        value : float
            The average recall value averaged over the specified IOU thresholds.
        score_threshold : float
            The detection score threshold; only detections with confidence scores above this
            threshold are considered.
        iou_thresholds : list[float]
            The list of IOU thresholds used to compute the recall values.
        label : str
            The class label for which the AR is computed.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.AR.value,
            value=value,
            parameters={
                "iou_thresholds": iou_thresholds,
                "score_threshold": score_threshold,
                "label": label,
            },
        )

    @classmethod
    def mean_average_recall(
        cls,
        value: float,
        score_threshold: float,
        iou_thresholds: list[float],
    ):
        """
        Mean Average Recall (mAR) metric for object detection tasks.

        The mAR computation considers detections with confidence scores above the specified
        `score_threshold` and calculates recall at each IOU threshold in `iou_thresholds` for
        each label. The final mAR value is obtained by averaging these recall values over the
        specified IOU thresholds and then averaging across all labels.

        Parameters
        ----------
        value : float
            The mean average recall value averaged over the specified IOU thresholds.
        score_threshold : float
            The detection score threshold; only detections with confidence scores above this
            threshold are considered.
        iou_thresholds : list[float]
            The list of IOU thresholds used to compute the recall values.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.mAR.value,
            value=value,
            parameters={
                "iou_thresholds": iou_thresholds,
                "score_threshold": score_threshold,
            },
        )

    @classmethod
    def average_recall_averaged_over_scores(
        cls,
        value: float,
        score_thresholds: list[float],
        iou_thresholds: list[float],
        label: str,
    ):
        """
        Average Recall (AR) metric averaged over multiple score thresholds for a specific object class label.

        The AR computation considers detections across multiple `score_thresholds` and calculates
        recall at each IOU threshold in `iou_thresholds`. The final AR value is obtained by averaging
        the recall values over all specified score thresholds and IOU thresholds.

        Parameters
        ----------
        value : float
            The average recall value averaged over the specified score thresholds and IOU thresholds.
        score_thresholds : list[float]
            The list of detection score thresholds; detections with confidence scores above each threshold are considered.
        iou_thresholds : list[float]
            The list of IOU thresholds used to compute the recall values.
        label : str
            The class label for which the AR is computed.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.ARAveragedOverScores.value,
            value=value,
            parameters={
                "iou_thresholds": iou_thresholds,
                "score_thresholds": score_thresholds,
                "label": label,
            },
        )

    @classmethod
    def mean_average_recall_averaged_over_scores(
        cls,
        value: float,
        score_thresholds: list[float],
        iou_thresholds: list[float],
    ):
        """
        Mean Average Recall (mAR) metric averaged over multiple score thresholds and IOU thresholds.

        The mAR computation considers detections across multiple `score_thresholds`, calculates recall
        at each IOU threshold in `iou_thresholds` for each label, averages these recall values over all
        specified score thresholds and IOU thresholds, and then computes the mean across all labels to
        obtain the final mAR value.

        Parameters
        ----------
        value : float
            The mean average recall value averaged over the specified score thresholds and IOU thresholds.
        score_thresholds : list[float]
            The list of detection score thresholds; detections with confidence scores above each threshold are considered.
        iou_thresholds : list[float]
            The list of IOU thresholds used to compute the recall values.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.mARAveragedOverScores.value,
            value=value,
            parameters={
                "iou_thresholds": iou_thresholds,
                "score_thresholds": score_thresholds,
            },
        )

    @classmethod
    def precision_recall_curve(
        cls,
        precisions: list[float],
        scores: list[float],
        iou_threshold: float,
        label: str,
    ):
        """
        Interpolated precision-recall curve over 101 recall points.

        The precision values are interpolated over recalls ranging from 0.0 to 1.0 in steps of 0.01,
        resulting in 101 points. This is a byproduct of the 101-point interpolation used in calculating
        the Average Precision (AP) metric in object detection tasks.

        Parameters
        ----------
        precisions : list[float]
            Interpolated precision values corresponding to recalls at 0.0, 0.01, ..., 1.0.
        scores : list[float]
            Maximum prediction score for each point on the interpolated curve.
        iou_threshold : float
            The Intersection over Union (IOU) threshold used to determine true positives.
        label : str
            The class label associated with this precision-recall curve.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.PrecisionRecallCurve.value,
            value={
                "precisions": precisions,
                "scores": scores,
            },
            parameters={
                "iou_threshold": iou_threshold,
                "label": label,
            },
        )

    @classmethod
    def counts(
        cls,
        tp: int,
        fp: int,
        fn: int,
        label: str,
        iou_threshold: float,
        score_threshold: float,
    ):
        """
        `Counts` encapsulates the counts of true positives (`tp`), false positives (`fp`),
        and false negatives (`fn`) for object detection evaluation, along with the associated
        class label, Intersection over Union (IOU) threshold, and confidence score threshold.

        Parameters
        ----------
        tp : int
            Number of true positives.
        fp : int
            Number of false positives.
        fn : int
            Number of false negatives.
        label : str
            The class label for which the counts are calculated.
        iou_threshold : float
            The IOU threshold used to determine a match between predicted and ground truth boxes.
        score_threshold : float
            The confidence score threshold above which predictions are considered.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.Counts.value,
            value={
                "tp": tp,
                "fp": fp,
                "fn": fn,
            },
            parameters={
                "iou_threshold": iou_threshold,
                "score_threshold": score_threshold,
                "label": label,
            },
        )

    @classmethod
    def confusion_matrix(
        cls,
        confusion_matrix: dict[
            str,  # ground truth label value
            dict[
                str,  # prediction label value
                dict[
                    str,  # either `count` or `examples`
                    int
                    | list[
                        dict[
                            str,  # either `datum`, `groundtruth`, `prediction` or score
                            str  # datum uid
                            | dict[
                                str, float
                            ]  # bounding box (xmin, xmax, ymin, ymax)
                            | float,  # prediction score
                        ]
                    ],
                ],
            ],
        ],
        unmatched_predictions: dict[
            str,  # prediction label value
            dict[
                str,  # either `count` or `examples`
                int
                | list[
                    dict[
                        str,  # either `datum`, `prediction` or score
                        str  # datum uid
                        | float  # prediction score
                        | dict[
                            str, float
                        ],  # bounding box (xmin, xmax, ymin, ymax)
                    ]
                ],
            ],
        ],
        unmatched_ground_truths: dict[
            str,  # ground truth label value
            dict[
                str,  # either `count` or `examples`
                int
                | list[
                    dict[
                        str,  # either `datum` or `groundtruth`
                        str  # datum uid
                        | dict[
                            str, float
                        ],  # bounding box (xmin, xmax, ymin, ymax)
                    ]
                ],
            ],
        ],
        score_threshold: float,
        iou_threshold: float,
        maximum_number_of_examples: int,
    ):
        """
        Confusion matrix for object detection tasks.

        This class encapsulates detailed information about the model's performance, including correct
        predictions, misclassifications, unmatched_predictions (subset of false positives), and unmatched ground truths
        (subset of false negatives). It provides counts and examples for each category to facilitate in-depth analysis.

        Confusion Matrix Format:
        {
            <ground truth label>: {
                <prediction label>: {
                    'count': int,
                    'examples': [
                        {
                            'datum': str,
                            'groundtruth': dict,  # {'xmin': float, 'xmax': float, 'ymin': float, 'ymax': float}
                            'prediction': dict,   # {'xmin': float, 'xmax': float, 'ymin': float, 'ymax': float}
                            'score': float,
                        },
                        ...
                    ],
                },
                ...
            },
            ...
        }

        Unmatched Predictions Format:
        {
            <prediction label>: {
                'count': int,
                'examples': [
                    {
                        'datum': str,
                        'prediction': dict,  # {'xmin': float, 'xmax': float, 'ymin': float, 'ymax': float}
                        'score': float,
                    },
                    ...
                ],
            },
            ...
        }

        Unmatched Ground Truths Format:
        {
            <ground truth label>: {
                'count': int,
                'examples': [
                    {
                        'datum': str,
                        'groundtruth': dict,  # {'xmin': float, 'xmax': float, 'ymin': float, 'ymax': float}
                    },
                    ...
                ],
            },
            ...
        }

        Parameters
        ----------
        confusion_matrix : dict
            A nested dictionary where the first key is the ground truth label value, the second key
            is the prediction label value, and the innermost dictionary contains either a `count`
            or a list of `examples`. Each example includes the datum UID, ground truth bounding box,
            predicted bounding box, and prediction scores.
        unmatched_predictions : dict
            A dictionary where each key is a prediction label value with no corresponding ground truth
            (subset of false positives). The value is a dictionary containing either a `count` or a list of
            `examples`. Each example includes the datum UID, predicted bounding box, and prediction score.
        unmatched_ground_truths : dict
            A dictionary where each key is a ground truth label value for which the model failed to predict
            (subset of false negatives). The value is a dictionary containing either a `count` or a list of `examples`.
            Each example includes the datum UID and ground truth bounding box.
        score_threshold : float
            The confidence score threshold used to filter predictions.
        iou_threshold : float
            The Intersection over Union (IOU) threshold used to determine true positives.
        maximum_number_of_examples : int
            The maximum number of examples per element.

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
            parameters={
                "score_threshold": score_threshold,
                "iou_threshold": iou_threshold,
                "maximum_number_of_examples": maximum_number_of_examples,
            },
        )
