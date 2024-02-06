import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from velour_api.schemas.label import Label


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
            The id of the evaluation.

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
            The id of the evaluation.

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


class mAPMetric(BaseModel):
    """
    A mAP metric response from the API.

    Attributes
    ----------
    iou : float
        The intersect-over-union (IOU) value.
    value : float
        The value of the metric.
    """

    iou: float
    value: float

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : ind
            The id of the evaluation.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "type": "mAP",
            "evaluation_id": evaluation_id,
            "parameters": {"iou": self.iou},
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
    """

    ious: set[float]
    value: float

    def db_mapping(self, evaluation_id: int) -> dict:
        """
        Creates a mapping for use when uploading the metric to the database.

        Parameters
        ----------
        evaluation_id : ind
            The id of the evaluation.

        Returns
        ----------
        A mapping dictionary.
        """
        return {
            "value": self.value,
            "type": "mAPAveragedOverIOUs",
            "evaluation_id": evaluation_id,
            "parameters": {"ious": list(self.ious)},
        }


class ConfusionMatrixEntry(BaseModel):
    """
    Describes one element in a confusion matrix.

    Attributes
    ----------
    prediction : str
        The prediction.
    groundtruth : str
        The groundtruth.
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
        A label for the metric.
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
            "type": self.__type__,  # type: ignore - Member "__type__" is unknown because it's only assigned on Child classes
            "evaluation_id": evaluation_id,
        }


class PrecisionMetric(_PrecisionRecallF1Base):
    """Describes an precision metric."""

    __type__ = "Precision"


class RecallMetric(_PrecisionRecallF1Base):
    """Describes a recall metric."""

    __type__ = "Recall"


class F1Metric(_PrecisionRecallF1Base):
    """Describes an F1 metric."""

    __type__ = "F1"


class ROCAUCMetric(BaseModel):
    """
    Describes an ROC AUC metric.

    Attributes
    ----------
    label_key : str
        A label for the metric.
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
        return {
            "value": self.value if not np.isnan(self.value) else -1,  # type: ignore - numpy type error; np.isnan can take None
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
    """

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
            "type": "mIOU",
            "evaluation_id": evaluation_id,
        }
