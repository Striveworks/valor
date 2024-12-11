from dataclasses import dataclass
from enum import Enum

from valor_lite.schemas import BaseMetric


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
class Metric(BaseMetric):
    """
    Classification Metric.

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
        score_threshold: float,
        hardmax: bool,
        label: str,
    ):
        """
        Precision metric for a specific class label.

        This class calculates the precision at a specific score threshold.
        Precision is defined as the ratio of true positives to the sum of
        true positives and false positives.

        Parameters
        ----------
        value : float
            Precision value computed at a specific score threshold.
        score_threshold : float
            Score threshold at which the precision value is computed.
        hardmax : bool
            Indicates whether hardmax thresholding was used.
        label : str
            The class label for which the precision is computed.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.Precision.value,
            value=value,
            parameters={
                "score_threshold": score_threshold,
                "hardmax": hardmax,
                "label": label,
            },
        )

    @classmethod
    def recall(
        cls,
        value: float,
        score_threshold: float,
        hardmax: bool,
        label: str,
    ):
        """
        Recall metric for a specific class label.

        This class calculates the recall at a specific score threshold.
        Recall is defined as the ratio of true positives to the sum of
        true positives and false negatives.

        Parameters
        ----------
        value : float
            Recall value computed at a specific score threshold.
        score_threshold : float
            Score threshold at which the recall value is computed.
        hardmax : bool
            Indicates whether hardmax thresholding was used.
        label : str
            The class label for which the recall is computed.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.Recall.value,
            value=value,
            parameters={
                "score_threshold": score_threshold,
                "hardmax": hardmax,
                "label": label,
            },
        )

    @classmethod
    def f1_score(
        cls,
        value: float,
        score_threshold: float,
        hardmax: bool,
        label: str,
    ):
        """
        F1 score for a specific class label and confidence score threshold.

        Parameters
        ----------
        value : float
            F1 score computed at a specific score threshold.
        score_threshold : float
            Score threshold at which the F1 score is computed.
        hardmax : bool
            Indicates whether hardmax thresholding was used.
        label : str
            The class label for which the F1 score is computed.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.F1.value,
            value=value,
            parameters={
                "score_threshold": score_threshold,
                "hardmax": hardmax,
                "label": label,
            },
        )

    @classmethod
    def accuracy(
        cls,
        value: float,
        score_threshold: float,
        hardmax: bool,
    ):
        """
        Multiclass accuracy metric.

        This class calculates the accuracy at various score thresholds.

        Parameters
        ----------
        value : float
            Accuracy value computed at a specific score threshold.
        score_threshold : float
            Score threshold at which the accuracy value is computed.
        hardmax : bool
            Indicates whether hardmax thresholding was used.

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.Accuracy.value,
            value=value,
            parameters={
                "score_threshold": score_threshold,
                "hardmax": hardmax,
            },
        )

    @classmethod
    def roc_auc(
        cls,
        value: float,
        label: str,
    ):
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

        Returns
        -------
        Metric
        """
        return cls(
            type=MetricType.ROCAUC.value,
            value=value,
            parameters={
                "label": label,
            },
        )

    @classmethod
    def mean_roc_auc(cls, value: float):
        """
        Mean Receiver Operating Characteristic Area Under the Curve (mROC AUC).

        This class calculates the mean ROC AUC score over all classes in a multiclass classification task.
        It provides an aggregate measure of the model's ability to distinguish between classes.

        Parameters
        ----------
        value : float
            The computed mean ROC AUC score.

        Returns
        -------
        Metric
        """
        return cls(type=MetricType.mROCAUC.value, value=value, parameters={})

    @classmethod
    def counts(
        cls,
        tp: int,
        fp: int,
        fn: int,
        tn: int,
        score_threshold: float,
        hardmax: bool,
        label: str,
    ):
        """
        Confusion matrix counts at specified score thresholds for binary classification.

        This class stores the true positive (`tp`), false positive (`fp`), false negative (`fn`), and true
        negative (`tn`) counts computed at various score thresholds for a binary classification task.

        Parameters
        ----------
        tp : int
            True positive counts at each score threshold.
        fp : int
            False positive counts at each score threshold.
        fn : int
            False negative counts at each score threshold.
        tn : int
            True negative counts at each score threshold.
        score_threshold : float
            Score thresholds at which the counts are computed.
        hardmax : bool
            Indicates whether hardmax thresholding was used.
        label : str
            The class label for which the counts are computed.

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
                "tn": tn,
            },
            parameters={
                "score_threshold": score_threshold,
                "hardmax": hardmax,
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
                            str,  # either `datum` or `score`
                            str | float,  # datum uid  # prediction score
                        ]
                    ],
                ],
            ],
        ],
        unmatched_ground_truths: dict[
            str,  # ground truth label value
            dict[
                str,  # either `count` or `examples`
                int | list[dict[str, str]],  # count or datum examples
            ],
        ],
        score_threshold: float,
        maximum_number_of_examples: int,
    ):
        """
        The confusion matrix and related metrics for the classification task.

        This class encapsulates detailed information about the model's performance, including correct
        predictions, misclassifications, unmatched predictions (subset of false positives), and unmatched ground truths
        (subset of false negatives). It provides counts and examples for each category to facilitate in-depth analysis.

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

        Unmatched Ground Truths Structure:
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

        Parameters
        ----------
        confusion_matrix : dict
            A nested dictionary where the first key is the ground truth label value, the second key
            is the prediction label value, and the innermost dictionary contains either a `count`
            or a list of `examples`. Each example includes the datum UID and prediction score.
        unmatched_ground_truths : dict
            A dictionary where each key is a ground truth label value for which the model failed to predict
            (false negatives). The value is a dictionary containing either a `count` or a list of `examples`.
            Each example includes the datum UID.
        score_threshold : float
            The confidence score threshold used to filter predictions.
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
                "unmatched_ground_truths": unmatched_ground_truths,
            },
            parameters={
                "score_threshold": score_threshold,
                "maximum_number_of_examples": maximum_number_of_examples,
            },
        )
