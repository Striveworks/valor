from dataclasses import dataclass

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
    value: float | None
    __type__ = "BaseClass"

    def __post_init__(self):
        """Validate instantiated class."""

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
    value: float | None
    __type__ = "BaseClass"

    def __post_init__(self):
        """Validate instantiated class."""

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
        """Validate instantiated class."""

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
        """Validate instantiated class."""

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
        """Validate instantiated class."""

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
        """Validate instantiated class."""

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
        """Validate instantiated class."""

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
        """Validate instantiated class."""

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
        """Validate instantiated class."""

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
        """Validate instantiated class."""

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
        """Validate instantiated class."""

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
        """Validate instantiated class."""

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
    pr_curve_iou_threshold: float | None
    __type__ = "BaseClass"

    def __post_init__(self):
        """Validate instantiated class."""

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
    value: dict[
        str,  # the label value
        dict[
            float,  # the score threshold
            dict[
                str,  # the metric (e.g., "tp" for true positive)
                int | float | None,
            ],  # the count or metric value
        ],
    ]

    def __post_init__(self):
        """Validate instantiated class."""

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
    value: dict[
        str,  # the label value
        dict[
            float,  # the score threshold
            dict[
                str,  # the metric (e.g., "tp" for true positive)
                dict[
                    str,  # the label for the next level of the dictionary (e.g., "observations" or "total")
                    int  # the count of classifications
                    | dict[
                        str,  # the subclassification for the label (e.g., "misclassifications")
                        dict[
                            str,  # the label for the next level of the dictionary (e.g., "count" or "examples")
                            int  # the count of subclassifications
                            | list[
                                tuple[str, str] | tuple[str, str, str],
                            ],
                        ],  # a list containing examples
                    ],
                ],
            ],
        ],
    ]

    def __post_init__(self):
        """Validate instantiated class."""

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
        """Validate instantiated class."""

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
    entries : list[ConfusionMatrixEntry]
        A list of entries for the matrix.
    """

    label_key: str
    entries: list[ConfusionMatrixEntry]

    def __post_init__(self):
        """Validate instantiated class."""

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
    entries : list[ConfusionMatrixEntry]
        A list of entries for the matrix.

    Attributes
    ----------
    matrix : np.ndarray
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


@dataclass
class _TextGenerationMetricBase:
    """
    Defines a base class for text generation metrics.

    Attributes
    ----------
    value : int | float | dict[str, float]
        The metric value. Different metrics have different value types.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    value: int | float | dict[str, float]
    parameters: dict
    __type__ = "BaseClass"

    def __post_init__(self):
        """Validate instantiated class."""
        if not isinstance(self.parameters, dict):
            raise TypeError(
                f"Expected parameters to be a dict, got {type(self.parameters).__name__}"
            )

    def to_dict(self):
        """Converts a metric object into a dictionary."""
        return {
            "type": self.__type__,
            "value": self.value,
            "parameters": self.parameters,
        }


class AnswerCorrectnessMetric(_TextGenerationMetricBase):
    """
    Defines an answer correctness metric.

    Attributes
    ----------
    value : float
        The answer correctness score between 0 and 1, with higher values indicating that the answer is more correct. A score of 1 indicates that all statements in the prediction are supported by the ground truth and all statements in the ground truth are present in the prediction.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "AnswerCorrectness"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if not isinstance(self.value, float):
            raise TypeError(
                f"Expected value to be a float, got {type(self.value).__name__}"
            )


class AnswerRelevanceMetric(_TextGenerationMetricBase):
    """
    Defines an answer relevance metric.

    Attributes
    ----------
    value : float
        The number of statements in the answer that are relevant to the query divided by the total number of statements in the answer.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "AnswerRelevance"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if not isinstance(self.value, float):
            raise TypeError(
                f"Expected value to be a float, got {type(self.value).__name__}"
            )


class BLEUMetric(_TextGenerationMetricBase):
    """
    Defines a BLEU metric.

    Attributes
    ----------
    value : float
        The BLEU score for an individual datapoint.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "BLEU"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if not isinstance(self.value, float):
            raise TypeError(
                f"Expected value to be a float, got {type(self.value).__name__}"
            )


class BiasMetric(_TextGenerationMetricBase):
    """
    Defines a bias metric.

    Attributes
    ----------
    value : float
        The bias score for a datum. This is a float between 0 and 1, with 1 indicating that all opinions in the datum text are biased and 0 indicating that there is no bias.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "Bias"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if not isinstance(self.value, float):
            raise TypeError(
                f"Expected value to be a float, got {type(self.value).__name__}"
            )


class ContextPrecisionMetric(_TextGenerationMetricBase):
    """
    Defines a context precision metric.

    Attributes
    ----------
    value : float
        The context precision score for a datum. This is a float between 0 and 1, with 0 indicating that none of the contexts are useful to arrive at the ground truth answer to the query and 1 indicating that all contexts are useful to arrive at the ground truth answer to the query. The score is more heavily influenced by earlier contexts in the list of contexts than later contexts.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "ContextPrecision"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if not isinstance(self.value, float):
            raise TypeError(
                f"Expected value to be a float, got {type(self.value).__name__}"
            )


class ContextRecallMetric(_TextGenerationMetricBase):
    """
    Defines a context recall metric.

    Attributes
    ----------
    value : float
        The context recall score for a datum. This is a float between 0 and 1, with 1 indicating that all ground truth statements are attributable to the context list.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "ContextRecall"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if not isinstance(self.value, float):
            raise TypeError(
                f"Expected value to be a float, got {type(self.value).__name__}"
            )


class ContextRelevanceMetric(_TextGenerationMetricBase):
    """
    Defines a context relevance metric.

    Attributes
    ----------
    value : float
        The context relevance score for a datum. This is a float between 0 and 1, with 0 indicating that none of the contexts are relevant and 1 indicating that all of the contexts are relevant.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "ContextRelevance"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if not isinstance(self.value, float):
            raise TypeError(
                f"Expected value to be a float, got {type(self.value).__name__}"
            )


class FaithfulnessMetric(_TextGenerationMetricBase):
    """
    Defines a faithfulness metric.

    Attributes
    ----------
    value : float
        The faithfulness score for a datum. This is a float between 0 and 1, with 1 indicating that all claims in the text are implied by the contexts.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "Faithfulness"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if not isinstance(self.value, float):
            raise TypeError(
                f"Expected value to be a float, got {type(self.value).__name__}"
            )


class HallucinationMetric(_TextGenerationMetricBase):
    """
    Defines a hallucination metric.

    Attributes
    ----------
    value : float
        The hallucination score for a datum. This is a float between 0 and 1, with 1 indicating that all contexts are contradicted by the text.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "Hallucination"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if not isinstance(self.value, float):
            raise TypeError(
                f"Expected value to be a float, got {type(self.value).__name__}"
            )


class ROUGEMetric(_TextGenerationMetricBase):
    """
    Defines a ROUGE metric.

    Attributes
    ----------
    value : dict[str, float]
        A JSON containing individual ROUGE scores calculated in different ways. `rouge1` is unigram-based scoring, `rouge2` is bigram-based scoring, `rougeL` is scoring based on sentences (i.e., splitting on "." and ignoring "\n"), and `rougeLsum` is scoring based on splitting the text using "\n".
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "ROUGE"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if not isinstance(self.value, dict):
            raise TypeError(
                f"Expected value to be a dict[str, float], got {type(self.value).__name__}"
            )
        if not all(isinstance(k, str) for k in self.value.keys()):
            raise TypeError(
                f"Expected keys in value to be strings, got {type(next(iter(self.value.keys()))).__name__}"
            )
        if not all(isinstance(v, float) for v in self.value.values()):
            raise TypeError(
                f"Expected values in value to be floats, got {type(next(iter(self.value.values()))).__name__}"
            )


class SummaryCoherenceMetric(_TextGenerationMetricBase):
    """
    Defines a summary coherence metric.

    Attributes
    ----------
    value : int
        The summary coherence score for a datum. This is an integer with 1 being the lowest summary coherence and 5 the highest summary coherence.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "SummaryCoherence"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if not isinstance(self.value, int):
            raise TypeError(
                f"Expected value to be a int, got {type(self.value).__name__}"
            )


class ToxicityMetric(_TextGenerationMetricBase):
    """
    Defines a toxicity metric.

    Attributes
    ----------
    value : float
        The toxicity score for a datum. This is a float between 0 and 1, with 1 indicating that all opinions in the datum text are toxic and 0 indicating that there is no toxicity.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "Toxicity"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if not isinstance(self.value, float):
            raise TypeError(
                f"Expected value to be a float, got {type(self.value).__name__}"
            )
