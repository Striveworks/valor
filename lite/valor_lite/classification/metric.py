from dataclasses import dataclass
from enum import Enum

from valor_lite.schemas import Metric


class MetricType(Enum):
    Counts = "Counts"
    ROCAUC = "ROCAUC"
    mROCAUC = "mROCAUC"
    Precision = "Precision"
    Recall = "Recall"
    Accuracy = "Accuracy"
    F1 = "F1"
    ConfusionMatrix = "ConfusionMatrix"


@dataclass
class Counts:
    """
    Confusion matrix counts at specified score thresholds for binary classification.

    This class stores the true positive (`tp`), false positive (`fp`), false negative (`fn`), and true
    negative (`tn`) counts computed at various score thresholds for a binary classification task.

    Attributes
    ----------
    tp : list[int]
        True positive counts at each score threshold.
    fp : list[int]
        False positive counts at each score threshold.
    fn : list[int]
        False negative counts at each score threshold.
    tn : list[int]
        True negative counts at each score threshold.
    score_thresholds : list[float]
        Score thresholds at which the counts are computed.
    hardmax : bool
        Indicates whether hardmax thresholding was used.
    label : str
        The class label for which the counts are computed.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    tp: list[int]
    fp: list[int]
    fn: list[int]
    tn: list[int]
    score_thresholds: list[float]
    hardmax: bool
    label: str

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value={
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
                "tn": self.tn,
            },
            parameters={
                "score_thresholds": self.score_thresholds,
                "hardmax": self.hardmax,
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


@dataclass
class _ThresholdValue:
    value: list[float]
    score_thresholds: list[float]
    hardmax: bool
    label: str

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_thresholds": self.score_thresholds,
                "hardmax": self.hardmax,
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


class Precision(_ThresholdValue):
    """
    Precision metric for a specific class label.

    This class calculates the precision at various score thresholds for a binary
    classification task. Precision is defined as the ratio of true positives to the
    sum of true positives and false positives.

    Attributes
    ----------
    value : list[float]
        Precision values computed at each score threshold.
    score_thresholds : list[float]
        Score thresholds at which the precision values are computed.
    hardmax : bool
        Indicates whether hardmax thresholding was used.
    label : str
        The class label for which the precision is computed.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    pass


class Recall(_ThresholdValue):
    """
    Recall metric for a specific class label.

    This class calculates the recall at various score thresholds for a binary
    classification task. Recall is defined as the ratio of true positives to the
    sum of true positives and false negatives.

    Attributes
    ----------
    value : list[float]
        Recall values computed at each score threshold.
    score_thresholds : list[float]
        Score thresholds at which the recall values are computed.
    hardmax : bool
        Indicates whether hardmax thresholding was used.
    label : str
        The class label for which the recall is computed.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    pass


class Accuracy(_ThresholdValue):
    """
    Accuracy metric for a specific class label.

    This class calculates the accuracy at various score thresholds for a binary
    classification task. Accuracy is defined as the ratio of the sum of true positives and
    true negatives over all predictions.

    Attributes
    ----------
    value : list[float]
        Accuracy values computed at each score threshold.
    score_thresholds : list[float]
        Score thresholds at which the accuracy values are computed.
    hardmax : bool
        Indicates whether hardmax thresholding was used.
    label : str
        The class label for which the accuracy is computed.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    pass


class F1(_ThresholdValue):
    """
    F1 score for a specific class label.

    This class calculates the F1 score at various score thresholds for a binary
    classification task.

    Attributes
    ----------
    value : list[float]
        F1 scores computed at each score threshold.
    score_thresholds : list[float]
        Score thresholds at which the F1 scores are computed.
    hardmax : bool
        Indicates whether hardmax thresholding was used.
    label : str
        The class label for which the F1 score is computed.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    pass


@dataclass
class ROCAUC:
    """
    Receiver Operating Characteristic Area Under the Curve (ROC AUC).

    This class calculates the ROC AUC score for a specific class label in a multiclass classification task.
    ROC AUC is a performance measurement for classification problems at various threshold settings.
    It reflects the ability of the classifier to distinguish between the positive and negative classes.

    Parameters
    ----------
    value : float
        The computed ROC AUC score.
    label : str
        The class label for which the ROC AUC is computed.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    value: float
    label: str

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={"label": self.label},
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


@dataclass
class mROCAUC:
    """
    Mean Receiver Operating Characteristic Area Under the Curve (mROC AUC).

    This class calculates the mean ROC AUC score over all classes in a multiclass classification task.
    It provides an aggregate measure of the model's ability to distinguish between classes.

    Parameters
    ----------
    value : float
        The computed mean ROC AUC score.

    Methods
    -------
    to_metric()
        Converts the instance to a generic `Metric` object.
    to_dict()
        Converts the instance to a dictionary representation.
    """

    value: float

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={},
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()


@dataclass
class ConfusionMatrix:
    """
    The confusion matrix and related metrics for the classification task.

    This class encapsulates detailed information about the model's performance, including correct
    predictions, misclassifications, hallucinations (false positives), and missing predictions
    (false negatives). It provides counts and examples for each category to facilitate in-depth analysis.

    Confusion Matrix Structure:
    {
        ground_truth_label: {
            predicted_label: {
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

    Hallucinations Structure:
    {
        prediction_label: {
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

    Missing Prediction Structure:
    {
        ground_truth_label: {
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
        or a list of `examples`. Each example includes the datum UID and prediction score.
    missing_predictions : dict
        A dictionary where each key is a ground truth label value for which the model failed to predict
        (false negatives). The value is a dictionary containing either a `count` or a list of `examples`.
        Each example includes the datum UID.
    score_threshold : float
        The confidence score threshold used to filter predictions.
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
                        str,  # either `datum` or `score`
                        str | float,  # datum uid  # prediction score
                    ]
                ],
            ],
        ],
    ]
    missing_predictions: dict[
        str,  # ground truth label value
        dict[
            str,  # either `count` or `examples`
            int | list[dict[str, str]],  # count or datum examples
        ],
    ]
    score_threshold: float
    number_of_examples: int

    def to_metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value={
                "confusion_matrix": self.confusion_matrix,
                "missing_predictions": self.missing_predictions,
            },
            parameters={
                "score_threshold": self.score_threshold,
            },
        )

    def to_dict(self) -> dict:
        return self.to_metric().to_dict()
