from dataclasses import dataclass
from enum import Enum

from valor_lite.schemas import Metric


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
class Counts:
    """
    `Counts` encapsulates the counts of true positives (`tp`), false positives (`fp`),
    and false negatives (`fn`) for object detection evaluation, along with the associated
    class label, Intersection over Union (IoU) threshold, and confidence score threshold.

    Attributes
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
        The IoU threshold used to determine a match between predicted and ground truth boxes.
    score_threshold : float
        The confidence score threshold above which predictions are considered.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    tp: int
    fp: int
    fn: int
    label: str
    iou_threshold: float
    score_threshold: float

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value={
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
            },
            parameters={
                "iou_threshold": self.iou_threshold,
                "score_threshold": self.score_threshold,
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


@dataclass
class _ClassMetric:
    value: float
    label: str
    iou_threshold: float
    score_threshold: float

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_threshold": self.iou_threshold,
                "score_threshold": self.score_threshold,
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


class Precision(_ClassMetric):
    """
    Precision metric for a specific class label in object detection.

    This class encapsulates a metric value for a particular class label,
    along with the associated Intersection over Union (IoU) threshold and
    confidence score threshold.

    Attributes
    ----------
    value : float
        The metric value.
    label : str
        The class label for which the metric is calculated.
    iou_threshold : float
        The IoU threshold used to determine matches between predicted and ground truth boxes.
    score_threshold : float
        The confidence score threshold above which predictions are considered.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    pass


class Recall(_ClassMetric):
    """
    Recall metric for a specific class label in object detection.

    This class encapsulates a metric value for a particular class label,
    along with the associated Intersection over Union (IoU) threshold and
    confidence score threshold.

    Attributes
    ----------
    value : float
        The metric value.
    label : str
        The class label for which the metric is calculated.
    iou_threshold : float
        The IoU threshold used to determine matches between predicted and ground truth boxes.
    score_threshold : float
        The confidence score threshold above which predictions are considered.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    pass


class Accuracy(_ClassMetric):
    """
    Accuracy metric for a specific class label in object detection.

    This class encapsulates a metric value for a particular class label,
    along with the associated Intersection over Union (IoU) threshold and
    confidence score threshold.

    Attributes
    ----------
    value : float
        The metric value.
    label : str
        The class label for which the metric is calculated.
    iou_threshold : float
        The IoU threshold used to determine matches between predicted and ground truth boxes.
    score_threshold : float
        The confidence score threshold above which predictions are considered.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    pass


class F1(_ClassMetric):
    """
    F1 score for a specific class label in object detection.

    This class encapsulates a metric value for a particular class label,
    along with the associated Intersection over Union (IoU) threshold and
    confidence score threshold.

    Attributes
    ----------
    value : float
        The metric value.
    label : str
        The class label for which the metric is calculated.
    iou_threshold : float
        The IoU threshold used to determine matches between predicted and ground truth boxes.
    score_threshold : float
        The confidence score threshold above which predictions are considered.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    pass


@dataclass
class AP:
    """
    Average Precision (AP) metric for object detection tasks.

    The AP computation uses 101-point interpolation, which calculates the average
    precision by interpolating the precision-recall curve at 101 evenly spaced recall
    levels from 0 to 1.

    Attributes
    ----------
    value : float
        The average precision value.
    iou_threshold : float
        The IoU threshold used to compute the AP.
    label : str
        The class label for which the AP is computed.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    value: float
    iou_threshold: float
    label: str

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_threshold": self.iou_threshold,
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


@dataclass
class mAP:
    """
    Mean Average Precision (mAP) metric for object detection tasks.

    The AP computation uses 101-point interpolation, which calculates the average
    precision for each class by interpolating the precision-recall curve at 101 evenly
    spaced recall levels from 0 to 1. The mAP is then calculated by averaging these
    values across all class labels.

    Attributes
    ----------
    value : float
        The mean average precision value.
    iou_threshold : float
        The IoU threshold used to compute the mAP.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    value: float
    iou_threshold: float

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_threshold": self.iou_threshold,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


@dataclass
class APAveragedOverIOUs:
    """
    Average Precision (AP) metric averaged over multiple IoU thresholds.

    The AP computation uses 101-point interpolation, which calculates the average precision
    by interpolating the precision-recall curve at 101 evenly spaced recall levels from 0 to 1
    for each IoU threshold specified in `iou_thresholds`. The final APAveragedOverIOUs value is
    obtained by averaging these AP values across all specified IoU thresholds.

    Attributes
    ----------
    value : float
        The average precision value averaged over the specified IoU thresholds.
    iou_thresholds : list[float]
        The list of IoU thresholds used to compute the AP values.
    label : str
        The class label for which the AP is computed.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    value: float
    iou_thresholds: list[float]
    label: str

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_thresholds": self.iou_thresholds,
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


@dataclass
class mAPAveragedOverIOUs:
    """
    Mean Average Precision (mAP) metric averaged over multiple IoU thresholds.

    The AP computation uses 101-point interpolation, which calculates the average precision
    by interpolating the precision-recall curve at 101 evenly spaced recall levels from 0 to 1
    for each IoU threshold specified in `iou_thresholds`. The final mAPAveragedOverIOUs value is
    obtained by averaging these AP values across all specified IoU thresholds and all class labels.

    Attributes
    ----------
    value : float
        The average precision value averaged over the specified IoU thresholds.
    iou_thresholds : list[float]
        The list of IoU thresholds used to compute the AP values.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    value: float
    iou_thresholds: list[float]

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_thresholds": self.iou_thresholds,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


@dataclass
class AR:
    """
    Average Recall (AR) metric for object detection tasks.

    The AR computation considers detections with confidence scores above the specified
    `score_threshold` and calculates the recall at each IoU threshold in `iou_thresholds`.
    The final AR value is the average of these recall values across all specified IoU
    thresholds.

    Attributes
    ----------
    value : float
        The average recall value averaged over the specified IoU thresholds.
    score_threshold : float
        The detection score threshold; only detections with confidence scores above this
        threshold are considered.
    iou_thresholds : list[float]
        The list of IoU thresholds used to compute the recall values.
    label : str
        The class label for which the AR is computed.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    value: float
    score_threshold: float
    iou_thresholds: list[float]
    label: str

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_threshold": self.score_threshold,
                "iou_thresholds": self.iou_thresholds,
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


@dataclass
class mAR:
    """
    Mean Average Recall (mAR) metric for object detection tasks.

    The mAR computation considers detections with confidence scores above the specified
    `score_threshold` and calculates recall at each IoU threshold in `iou_thresholds` for
    each label. The final mAR value is obtained by averaging these recall values over the
    specified IoU thresholds and then averaging across all labels.

    Attributes
    ----------
    value : float
        The mean average recall value averaged over the specified IoU thresholds.
    score_threshold : float
        The detection score threshold; only detections with confidence scores above this
        threshold are considered.
    iou_thresholds : list[float]
        The list of IoU thresholds used to compute the recall values.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    value: float
    score_threshold: float
    iou_thresholds: list[float]

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_threshold": self.score_threshold,
                "iou_thresholds": self.iou_thresholds,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


@dataclass
class ARAveragedOverScores:
    """
    Average Recall (AR) metric averaged over multiple score thresholds for a specific object class label.

    The AR computation considers detections across multiple `score_thresholds` and calculates
    recall at each IoU threshold in `iou_thresholds`. The final AR value is obtained by averaging
    the recall values over all specified score thresholds and IoU thresholds.

    Attributes
    ----------
    value : float
        The average recall value averaged over the specified score thresholds and IoU thresholds.
    score_thresholds : list[float]
        The list of detection score thresholds; detections with confidence scores above each threshold are considered.
    iou_thresholds : list[float]
        The list of IoU thresholds used to compute the recall values.
    label : str
        The class label for which the AR is computed.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    value: float
    score_thresholds: list[float]
    iou_thresholds: list[float]
    label: str

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_thresholds": self.score_thresholds,
                "iou_thresholds": self.iou_thresholds,
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


@dataclass
class mARAveragedOverScores:
    """
    Mean Average Recall (mAR) metric averaged over multiple score thresholds and IoU thresholds.

    The mAR computation considers detections across multiple `score_thresholds`, calculates recall
    at each IoU threshold in `iou_thresholds` for each label, averages these recall values over all
    specified score thresholds and IoU thresholds, and then computes the mean across all labels to
    obtain the final mAR value.

    Attributes
    ----------
    value : float
        The mean average recall value averaged over the specified score thresholds and IoU thresholds.
    score_thresholds : list[float]
        The list of detection score thresholds; detections with confidence scores above each threshold are considered.
    iou_thresholds : list[float]
        The list of IoU thresholds used to compute the recall values.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    value: float
    score_thresholds: list[float]
    iou_thresholds: list[float]

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_thresholds": self.score_thresholds,
                "iou_thresholds": self.iou_thresholds,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


@dataclass
class PrecisionRecallCurve:
    """
    Interpolated precision-recall curve over 101 recall points.

    The precision values are interpolated over recalls ranging from 0.0 to 1.0 in steps of 0.01,
    resulting in 101 points. This is a byproduct of the 101-point interpolation used in calculating
    the Average Precision (AP) metric in object detection tasks.

    Attributes
    ----------
    precisions : list[float]
        Interpolated precision values corresponding to recalls at 0.0, 0.01, ..., 1.0.
    scores : list[float]
        Maximum prediction score for each point on the interpolated curve.
    iou_threshold : float
        The Intersection over Union (IoU) threshold used to determine true positives.
    label : str
        The class label associated with this precision-recall curve.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    precisions: list[float]
    scores: list[float]
    iou_threshold: float
    label: str

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value={
                "precisions": self.precisions,
                "scores": self.scores,
            },
            parameters={
                "iou_threshold": self.iou_threshold,
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


@dataclass
class ConfusionMatrix:
    """
    Confusion matrix for object detection tasks.

    This class encapsulates detailed information about the model's performance, including correct
    predictions, misclassifications, hallucinations (false positives), and missing predictions
    (false negatives). It provides counts and examples for each category to facilitate in-depth analysis.

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

    Hallucinations Format:
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

    Missing Prediction Format:
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

    Attributes
    ----------
    confusion_matrix : dict
        A nested dictionary where the first key is the ground truth label value, the second key
        is the prediction label value, and the innermost dictionary contains either a `count`
        or a list of `examples`. Each example includes the datum UID, ground truth bounding box,
        predicted bounding box, and prediction scores.
    hallucinations : dict
        A dictionary where each key is a prediction label value with no corresponding ground truth
        (false positives). The value is a dictionary containing either a `count` or a list of
        `examples`. Each example includes the datum UID, predicted bounding box, and prediction score.
    missing_predictions : dict
        A dictionary where each key is a ground truth label value for which the model failed to predict
        (false negatives). The value is a dictionary containing either a `count` or a list of `examples`.
        Each example includes the datum UID and ground truth bounding box.
    score_threshold : float
        The confidence score threshold used to filter predictions.
    iou_threshold : float
        The Intersection over Union (IoU) threshold used to determine true positives.
    number_of_examples : int
        The maximum number of examples per element.

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
    ]
    hallucinations: dict[
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
    ]
    missing_predictions: dict[
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
    ]
    score_threshold: float
    iou_threshold: float
    number_of_examples: int

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value={
                "confusion_matrix": self.confusion_matrix,
                "hallucinations": self.hallucinations,
                "missing_predictions": self.missing_predictions,
            },
            parameters={
                "score_threshold": self.score_threshold,
                "iou_threshold": self.iou_threshold,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()
