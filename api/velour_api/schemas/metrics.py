from uuid import uuid4

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from velour_api.enums import AnnotationType, JobStatus
from velour_api.schemas.label import Label


class DetectionParameters(BaseModel):
    """Parameters for evaluating a Object Detection.
    ----------
    iou_thresholds_to_compute
        Compute all thresholds in this list to create the mAP value.
    iou_thresholds_to_keep
        Must be subset of `iou_thresholds_to_compute`, returns an `schemas.APMetric`
        for each threshold.
    """

    # thresholds to iterate over (mutable defaults are ok for pydantic models)
    iou_thresholds_to_compute: list[float] | None = [
        round(0.5 + 0.05 * i, 2) for i in range(10)
    ]
    iou_thresholds_to_keep: list[float] | None = [0.5, 0.75]

    # constraints
    annotation_type: AnnotationType | None = None
    label_key: str | None = None
    min_area: float | None = None
    max_area: float | None = None

    # pydantic setting
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    @classmethod
    def check_ious(cls, values):
        for iou in values.iou_thresholds_to_keep:
            if iou not in values.iou_thresholds_to_compute:
                raise ValueError(
                    "`iou_thresholds_to_keep` must be contained in `iou_thresholds_to_compute`"
                )
        return values


class EvaluationSettings(BaseModel):
    """General parameters defining any filters of the data such
    as model, dataset, groundtruth and prediction type, model, dataset,
    size constraints, coincidence/intersection constraints, etc.
    """

    model: str
    dataset: str
    parameters: DetectionParameters | None = None
    id: int | None = None

    # pydantic setting
    model_config = ConfigDict(extra="forbid")


class CreateAPMetricsResponse(BaseModel):
    missing_pred_labels: list[Label]
    ignored_pred_labels: list[Label]
    job_id: int


class CreateSemanticSegmentationMetricsResponse(BaseModel):
    missing_pred_labels: list[Label]
    ignored_pred_labels: list[Label]
    job_id: int


class CreateClfMetricsResponse(BaseModel):
    missing_pred_keys: list[str]
    ignored_pred_keys: list[str]
    job_id: int


class Job(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid4()))
    status: JobStatus = JobStatus.PENDING
    model_config = ConfigDict(extra="allow")


class Metric(BaseModel):
    """This is used for responses from the API"""

    type: str
    parameters: dict | None = None
    value: float | dict | None = None
    label: Label | None = None


class BulkEvaluations(BaseModel):
    """Used to fetch all metrics for a given dataset of model"""

    model: str
    dataset: str
    metrics: list[Metric]
    statuses: dict[str, str]

    # pydantic setting
    model_config = ConfigDict(extra="forbid")


class APMetric(BaseModel):
    iou: float
    value: float
    label: Label

    def db_mapping(self, label_id: int, evaluation_id: int) -> dict:
        return {
            "value": self.value,
            "label_id": label_id,
            "type": "AP",
            "evaluation_id": evaluation_id,
            "parameters": {"iou": self.iou},
        }


class APMetricAveragedOverIOUs(BaseModel):
    ious: set[float]
    value: float
    label: Label

    def db_mapping(self, label_id: int, evaluation_id: int) -> dict:
        return {
            "value": self.value,
            "label_id": label_id,
            "type": "APAveragedOverIOUs",
            "evaluation_id": evaluation_id,
            "parameters": {"ious": list(self.ious)},
        }


class mAPMetric(BaseModel):
    iou: float
    value: float

    def db_mapping(self, evaluation_id: int) -> dict:
        return {
            "value": self.value,
            "type": "mAP",
            "evaluation_id": evaluation_id,
            "parameters": {"iou": self.iou},
        }


class mAPMetricAveragedOverIOUs(BaseModel):
    ious: set[float]
    value: float

    def db_mapping(self, evaluation_id: int) -> dict:
        return {
            "value": self.value,
            "type": "mAPAveragedOverIOUs",
            "evaluation_id": evaluation_id,
            "parameters": {"ious": list(self.ious)},
        }


class ConfusionMatrixEntry(BaseModel):
    prediction: str
    groundtruth: str
    count: int
    # TODO[pydantic]: The following keys were removed: `allow_mutation`.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-config for more information.
    model_config = ConfigDict(frozen=True)


class _BaseConfusionMatrix(BaseModel):
    label_key: str
    entries: list[ConfusionMatrixEntry]


class ConfusionMatrix(_BaseConfusionMatrix):
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
        return {
            "label_key": self.label_key,
            "value": [entry.model_dump() for entry in self.entries],
            "evaluation_id": evaluation_id,
        }


class ConfusionMatrixResponse(_BaseConfusionMatrix):
    """used for http response since it won't have the matrix and
    label map attributes
    """

    pass


class AccuracyMetric(BaseModel):
    label_key: str
    value: float

    def db_mapping(self, evaluation_id: int) -> dict:
        return {
            "value": self.value,
            "type": "Accuracy",
            "evaluation_id": evaluation_id,
            "parameters": {"label_key": self.label_key},
        }


class _PrecisionRecallF1Base(BaseModel):
    label: Label
    value: float | None = None

    @field_validator("value")
    @classmethod
    def replace_nan_with_neg_1(cls, v):
        if np.isnan(v):
            return -1
        return v

    def db_mapping(self, label_id: int, evaluation_id: int) -> dict:
        return {
            "value": self.value,
            "label_id": label_id,
            "type": self.__type__,
            "evaluation_id": evaluation_id,
        }


class PrecisionMetric(_PrecisionRecallF1Base):
    __type__ = "Precision"


class RecallMetric(_PrecisionRecallF1Base):
    __type__ = "Recall"


class F1Metric(_PrecisionRecallF1Base):
    __type__ = "F1"


class ROCAUCMetric(BaseModel):
    label_key: str
    value: float

    def db_mapping(self, evaluation_id: int) -> dict:
        return {
            "value": self.value,
            "type": "ROCAUC",
            "parameters": {"label_key": self.label_key},
            "evaluation_id": evaluation_id,
        }


class IOUMetric(BaseModel):
    value: float
    label: Label

    def db_mapping(self, label_id: int, evaluation_id: int) -> dict:
        return {
            "value": self.value,
            "label_id": label_id,
            "type": "IOU",
            "evaluation_id": evaluation_id,
        }


class mIOUMetric(BaseModel):
    value: float

    def db_mapping(self, evaluation_id: int) -> dict:
        return {
            "value": self.value,
            "type": "mIOU",
            "evaluation_id": evaluation_id,
        }
