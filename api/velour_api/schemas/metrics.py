from uuid import uuid4

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from velour_api.enums import JobStatus, TaskType
from velour_api.schemas.filters import Filter
from velour_api.schemas.label import Label


class DetectionParameters(BaseModel):
    """
    Defines important attributes to use when evaluating an object detection model.

    Attributes
    ----------
    iou_thresholds_to_compute : List[float]
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_keep: List[float]
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    """

    # thresholds to iterate over (mutable defaults are ok for pydantic models)
    iou_thresholds_to_compute: list[float] | None = [
        round(0.5 + 0.05 * i, 2) for i in range(10)
    ]
    iou_thresholds_to_keep: list[float] | None = [0.5, 0.75]

    # pydantic setting
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    @classmethod
    def _check_ious(cls, values):
        """Validate the IOU thresholds."""
        if values.iou_thresholds_to_keep and values.iou_thresholds_to_compute:
            for iou in values.iou_thresholds_to_keep:
                if iou not in values.iou_thresholds_to_compute:
                    raise ValueError(
                        "`iou_thresholds_to_keep` must be contained in `iou_thresholds_to_compute`"
                    )
        return values


class EvaluationSettings(BaseModel):
    """
    Defines important attributes of an evaluation's settings.

    Attributes
    ----------
    parameters : DetectionParameters
        The parameter object (e.g., `DetectionParameters) to use when creating an evaluation.
    filters: Filter
        The `Filter`object to use when creating an evaluation.
    """

    parameters: DetectionParameters | None = None
    filters: Filter | None = None

    # pydantic setting
    model_config = ConfigDict(extra="forbid")


class EvaluationJob(BaseModel):
    """
    Defines important attributes of an evaluation job.

    Attributes
    ----------
    model : str
        The name of the `Model` invoked during the evaluation.
    dataset : str
        The name of the `Dataset` invoked during the evaluation.
    task_type : TaskType
        The task type of the evaluation.
    settings : EvaluationSettings
        The `EvaluationSettings` object used to configurate the `EvaluationJob`.
    id : int
        The id of the job.
    """

    model: str
    dataset: str
    task_type: TaskType
    settings: EvaluationSettings = Field(default=EvaluationSettings())
    id: int | None = None

    # pydantic setting
    model_config = ConfigDict(extra="forbid")


class CreateDetectionMetricsResponse(BaseModel):
    """
    The response from a job that creates AP metrics.

    Attributes
    ----------
    missing_pred_labels: list[Label]
        A list of missing prediction labels.
    ignored_pred_labels: list[Label]
        A list of ignored preiction labels.
    job_id: int
        The job ID.
    """

    missing_pred_labels: list[Label]
    ignored_pred_labels: list[Label]
    job_id: int


class CreateSemanticSegmentationMetricsResponse(BaseModel):
    """
    The response from a job that creates segmentation metrics.

    Attributes
    ----------
    missing_pred_labels: list[Label]
        A list of missing prediction labels.
    ignored_pred_labels: list[Label]
        A list of ignored preiction labels.
    job_id: int
        The job ID.
    """

    missing_pred_labels: list[Label]
    ignored_pred_labels: list[Label]
    job_id: int


class CreateClfMetricsResponse(BaseModel):
    """
    The response from a job that creates classification metrics.

    Attributes
    ----------
    missing_pred_keys: list[str]
        A list of missing prediction keys.
    ignored_pred_keys: list[str]
        A list of ignored preiction keys.
    job_id: int
        The job ID.
    """

    missing_pred_keys: list[str]
    ignored_pred_keys: list[str]
    job_id: int


class Job(BaseModel):
    """
    Defines important attributes for a job.

    Attributes
    ----------
    uid : str
        The UID of the job.
    status : JobStatus
        The status of the job.
    """

    uid: str = Field(default_factory=lambda: str(uuid4()))
    status: JobStatus = JobStatus.PENDING
    model_config = ConfigDict(extra="allow")


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
            "type": self.__type__,
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


class Evaluation(BaseModel):
    """
    An object for storing the returned results of a model evaluation (where groundtruths are compared with predictions to measure performance).

    Attributes
    ----------
    dataset : str
        The name of the dataset.
    model : str
        The name of the model.
    settings : EvaluationSettings
        Settings for the evaluation.
    job_id : int
        The ID of the evaluation job.
    status : str
        The status of the evaluation.
    metrics : List[Metric]
        A list of metrics associated with the evaluation.
    confusion_matrices: List[ConfusionMatrixResponse]
        A list of confusion matrices associated with the evaluation.

    """

    dataset: str
    model: str
    settings: EvaluationSettings
    job_id: int
    status: str
    metrics: list[Metric]
    confusion_matrices: list[ConfusionMatrixResponse]

    # pydantic setting
    model_config = ConfigDict(extra="forbid")
