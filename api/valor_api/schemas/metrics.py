import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from valor_api.schemas.types import Label


class Metric(BaseModel):
    """
    A metric response from the API.

    Attributes
    ----------
    type : str
        The type of metric.
    parameters : dict
        The parameters of the metric.
    value : float
        The value of the metric.
    label : Label
        The `Label` for the metric.
    """

    type: str
    parameters: dict | None = None
    value: float | dict | None = None
    label: Label | None = None


class ARMetric(BaseModel):
    """
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
    value: float
    label: Label

    def db_mapping(self, label_id: int, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        label_id : int
            The id of the label.
        evaluation_id : ind
            The ID of the evaluation.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "label_id": label_id,
            "type": "AR",
            "evaluation_id": evaluation_id,
            "parameters": {"ious": list(self.ious)},
        }


class APMetric(BaseModel):
    """
    An AP metric response from the API.

    Attributes
    ----------
    iou : float
        The intersect-over-union (IOU) value.
    value : float
        The value of the metric.
    label : Label
        The `Label` for the metric.
    """

    iou: float
    value: float
    label: Label

    def db_mapping(self, label_id: int, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        label_id : int
            The id of the label.
        evaluation_id : ind
            The ID of the evaluation.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "label_id": label_id,
            "type": "AP",
            "evaluation_id": evaluation_id,
            "parameters": {"iou": self.iou},
        }


class APMetricAveragedOverIOUs(BaseModel):
    """
    An averaged AP metric response from the API.

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
    value: float
    label: Label

    def db_mapping(self, label_id: int, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        label_id : int
            The id of the label.
        evaluation_id : ind
            The ID of the evaluation.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "label_id": label_id,
            "type": "APAveragedOverIOUs",
            "evaluation_id": evaluation_id,
            "parameters": {"ious": list(self.ious)},
        }


class mARMetric(BaseModel):
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
    value: float
    label_key: str

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : ind
            The ID of the evaluation.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "type": "mAR",
            "evaluation_id": evaluation_id,
            "parameters": {
                "ious": list(self.ious),
                "label_key": self.label_key,
            },
        }


class mAPMetric(BaseModel):
    """
    A mAP metric response from the API.

    Attributes
    ----------
    iou : float
        The intersect-over-union (IOU) value.
    value : float
        The value of the metric.
    label_key : str
        The label key associated with the metric.
    """

    iou: float
    value: float
    label_key: str

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : ind
            The ID of the evaluation.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "type": "mAP",
            "evaluation_id": evaluation_id,
            "parameters": {
                "iou": self.iou,
                "label_key": self.label_key,
            },
        }


class mAPMetricAveragedOverIOUs(BaseModel):
    """
    An averaged mAP metric response from the API.

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
    value: float
    label_key: str

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : ind
            The ID of the evaluation.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "type": "mAPAveragedOverIOUs",
            "evaluation_id": evaluation_id,
            "parameters": {
                "ious": list(self.ious),
                "label_key": self.label_key,
            },
        }


class ConfusionMatrixEntry(BaseModel):
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
    model_config = ConfigDict(frozen=True)


class _BaseConfusionMatrix(BaseModel):
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
    entries: list[ConfusionMatrixEntry]


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

    model_config = ConfigDict(extra="allow")

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

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the confusion matrix to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "label_key": self.label_key,
            "value": [entry.model_dump() for entry in self.entries],
            "evaluation_id": evaluation_id,
        }


class ConfusionMatrixResponse(_BaseConfusionMatrix):
    """
    A response object used for HTTP responses since they won't contain matrix or label map attributes.
    """

    pass


class AccuracyMetric(BaseModel):
    """
    Describes an accuracy metric.

    Attributes
    ----------
    label_key : str
        The label key associated with the metric.
    value : float
        The metric value.
    """

    label_key: str
    value: float

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "type": "Accuracy",
            "evaluation_id": evaluation_id,
            "parameters": {"label_key": self.label_key},
        }


class _BasePrecisionRecallCurve(BaseModel):
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
    pr_curve_iou_threshold: float | None = None


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

    value: dict[
        str,
        dict[
            float,
            dict[str, int | float | None],
        ],
    ]

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the curves to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """

        return {
            "value": self.value,
            "type": "PrecisionRecallCurve",
            "evaluation_id": evaluation_id,
            "parameters": {
                "label_key": self.label_key,
                "pr_curve_iou_threshold": self.pr_curve_iou_threshold,
            },
        }


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
                                tuple[str, str] | tuple[str, str, str]
                            ],  # a list containing examples
                        ],
                    ],
                ],
            ],
        ],
    ]

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the curves to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """

        return {
            "value": self.value,
            "type": "DetailedPrecisionRecallCurve",
            "evaluation_id": evaluation_id,
            "parameters": {
                "label_key": self.label_key,
                "pr_curve_iou_threshold": self.pr_curve_iou_threshold,
            },
        }


class _PrecisionRecallF1Base(BaseModel):
    """
    Describes an accuracy metric.

    Attributes
    ----------
    label : label
        A label for the metric.
    value : float
        The metric value.
    """

    label: Label
    value: float | None = None
    __type__ = "BaseClass"

    @field_validator("value")
    @classmethod
    def _replace_nan_with_neg_1(cls, v):
        """Convert null values to -1."""
        if v is None or np.isnan(v):
            return -1
        return v

    def db_mapping(self, label_id: int, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "label_id": label_id,
            "type": self.__type__,
            "evaluation_id": evaluation_id,
        }


class PrecisionMetric(_PrecisionRecallF1Base):
    """
    Describes an precision metric.

    Attributes
    ----------
    label : Label
        A key-value pair.
    value : float, optional
        The metric value.
    """

    __type__ = "Precision"


class RecallMetric(_PrecisionRecallF1Base):
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


class F1Metric(_PrecisionRecallF1Base):
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


class ROCAUCMetric(BaseModel):
    """
    Describes an ROC AUC metric.

    Attributes
    ----------
    label_key : str
        The label key associated with the metric.
    value : float
        The metric value.
    """

    label_key: str
    value: float | None

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """
        value = (
            self.value
            if (self.value is not None and not np.isnan(self.value))
            else -1
        )
        return {
            "value": value,
            "type": "ROCAUC",
            "parameters": {"label_key": self.label_key},
            "evaluation_id": evaluation_id,
        }


class IOUMetric(BaseModel):
    """
    Describes an intersection-over-union (IOU) metric.

    Attributes
    ----------
    value : float
        The metric value.
    label : Label
        A label for the metric.
    """

    value: float
    label: Label

    def db_mapping(self, label_id: int, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "label_id": label_id,
            "type": "IOU",
            "evaluation_id": evaluation_id,
        }


class mIOUMetric(BaseModel):
    """
    Describes a mean intersection-over-union (IOU) metric.

    Attributes
    ----------
    value : float
        The metric value.
    label_key : str
        The label key associated with the metric.
    """

    value: float
    label_key: str

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "type": "mIOU",
            "evaluation_id": evaluation_id,
            "parameters": {"label_key": self.label_key},
        }


class AnswerRelevanceMetric(BaseModel):
    """
    Describes an answer relevance metric.

    Attributes
    ----------
    value : float
        The number of statements in the answer that are relevant to the query divided by the total number of statements in the answer.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    value: float
    parameters: dict

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "parameters": self.parameters,
            "type": "AnswerRelevance",
            "evaluation_id": evaluation_id,
        }


class BLEUMetric(BaseModel):
    """
    Describes a BLEU metric.

    Attributes
    ----------
    value : float
        The BLEU score for an individual datapoint, which is a JSON containing individual ROUGE scores calculated in different ways.
    parameters : dict[str, str | list[int | float]]
        The parameters associated with the metric.
    """

    value: float
    parameters: dict[str, str | list[int | float]]

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "parameters": self.parameters,
            "type": "BLEU",
            "evaluation_id": evaluation_id,
        }


class BiasMetric(BaseModel):
    """
    Describes a bias metric.

    Attributes
    ----------
    value : float
        The bias score for a datum. This is a float between 0 and 1, with 1 indicating that all opinions in the datum text are biased and 0 indicating that there is no bias.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    value: float
    parameters: dict

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "parameters": self.parameters,
            "type": "Bias",
            "evaluation_id": evaluation_id,
        }


class CoherenceMetric(BaseModel):
    """
    Describes a coherence metric.

    Attributes
    ----------
    value : int
        The coherence score for a datum. This is an integer with 1 being the lowest coherence and 5 the highest coherence.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    value: int
    parameters: dict

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "parameters": self.parameters,
            "type": "Coherence",
            "evaluation_id": evaluation_id,
        }


class ContextRelevanceMetric(BaseModel):
    """
    Describes a context relevance metric.

    Attributes
    ----------
    value : float
        The context relevance score for a datum. This is a float between 0 and 1, with 0 indicating that none of the context is relevant and 1 indicating that all of the context is relevant.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    value: float
    parameters: dict

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "parameters": self.parameters,
            "type": "ContextRelevance",
            "evaluation_id": evaluation_id,
        }


class HallucinationMetric(BaseModel):
    """
    Describes a hallucination metric.

    Attributes
    ----------
    value : float
        The hallucination score for a datum. This is a float between 0 and 1, with 1 indicating that all context is contradicted by the text.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    value: float
    parameters: dict

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "parameters": self.parameters,
            "type": "Hallucination",
            "evaluation_id": evaluation_id,
        }


class ROUGEMetric(BaseModel):
    """
    Describes a ROUGE metric.

    Attributes
    ----------
    value : dict[str, float]
        A JSON containing individual ROUGE scores calculated in different ways. `rouge1` is unigram-based scoring, `rouge2` is bigram-based scoring, `rougeL` is scoring based on sentences (i.e., splitting on "." and ignoring "\n"), and `rougeLsum` is scoring based on splitting the text using "\n".
    parameters : dict[str, str | bool | list[str]]
        The parameters associated with the metric.
    """

    value: dict[str, float]
    parameters: dict[str, str | bool | list[str]]

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "parameters": self.parameters,
            "type": "ROUGE",
            "evaluation_id": evaluation_id,
        }


class ToxicityMetric(BaseModel):
    """
    Describes a toxicity metric.

    Attributes
    ----------
    value : float
        The toxicity score for a datum. This is a float between 0 and 1, with 1 indicating that all opinions in the datum text are toxic and 0 indicating that there is no toxicity.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    value: float
    parameters: dict

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : int
            The evaluation id.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "parameters": self.parameters,
            "type": "Toxicity",
            "evaluation_id": evaluation_id,
        }
