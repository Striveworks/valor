from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Extra, Field, root_validator, validator

from velour_api.enums import JobStatus, TaskType
from velour_api.schemas.label import Label
from velour_api.schemas.metadata import MetaDatum


class EvaluationSettings(BaseModel):
    """General parameters defining any filters of the data such
    as model, dataset, groundtruth and prediction type, model, dataset,
    size constraints, coincidence/intersection constraints, etc.
    """

    model_name: str
    dataset_name: str
    model_pred_task_type: TaskType = None
    dataset_gt_task_type: TaskType = None
    min_area: float = None
    max_area: float = None
    group_by: str = None
    label_key: str = None
    id: int = None


class APRequest(BaseModel):
    """Request to compute average precision"""

    settings: EvaluationSettings
    # (mutable defaults are ok for pydantic models)
    iou_thresholds: list[float] = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    ious_to_keep: set[float] = {0.5, 0.75}

    @root_validator
    def check_ious(cls, values):
        for iou in values["ious_to_keep"]:
            if iou not in values["iou_thresholds"]:
                raise ValueError(
                    "`ious_to_keep` must be contained in `iou_thresholds`"
                )
        return values


class CreateAPMetricsResponse(BaseModel):
    missing_pred_labels: list[Label]
    ignored_pred_labels: list[Label]
    job_id: str


class CreateClfMetricsResponse(BaseModel):
    missing_pred_keys: list[str]
    ignored_pred_keys: list[str]
    job_id: str


class Job(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid4()))
    status: JobStatus = JobStatus.PENDING

    class Config:
        extra = Extra.allow


class ClfMetricsRequest(BaseModel):
    settings: EvaluationSettings


class Metric(BaseModel):
    """This is used for responses from the API"""

    type: str
    parameters: dict | None
    value: float | dict | None
    label: Label = None
    group: MetaDatum = None


class APMetric(BaseModel):
    iou: float
    value: float
    label: Label

    def db_mapping(self, label_id: int, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "label_id": label_id,
            "type": "AP",
            "evaluation_settings_id": evaluation_settings_id,
            "parameters": {"iou": self.iou},
        }


class APMetricAveragedOverIOUs(BaseModel):
    ious: set[float]
    value: float
    label: Label

    def db_mapping(self, label_id: int, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "label_id": label_id,
            "type": "APAveragedOverIOUs",
            "evaluation_settings_id": evaluation_settings_id,
            "parameters": {"ious": list(self.ious)},
        }


class mAPMetric(BaseModel):
    iou: float
    value: float

    def db_mapping(self, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "type": "mAP",
            "evaluation_settings_id": evaluation_settings_id,
            "parameters": {"iou": self.iou},
        }


class mAPMetricAveragedOverIOUs(BaseModel):
    ious: set[float]
    value: float

    def db_mapping(self, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "type": "mAPAveragedOverIOUs",
            "evaluation_settings_id": evaluation_settings_id,
            "parameters": {"ious": list(self.ious)},
        }


class ConfusionMatrixEntry(BaseModel):
    prediction: str
    groundtruth: str
    count: int

    class Config:
        allow_mutation = False


class _BaseConfusionMatrix(BaseModel):
    label_key: str
    entries: list[ConfusionMatrixEntry]
    group: MetaDatum = None
    group_id: int = None


class ConfusionMatrix(_BaseConfusionMatrix, extra=Extra.allow):
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

    def db_mapping(self, evaluation_settings_id: int) -> dict:
        return {
            "label_key": self.label_key,
            "value": [entry.dict() for entry in self.entries],
            "evaluation_settings_id": evaluation_settings_id,
        }


class ConfusionMatrixResponse(_BaseConfusionMatrix):
    """used for http response since it won't have the matrix and
    label map attributes
    """

    pass


class AccuracyMetric(BaseModel):
    label_key: str
    value: float
    group: MetaDatum = None
    group_id: int = None

    def db_mapping(self, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "type": "Accuracy",
            "evaluation_settings_id": evaluation_settings_id,
            "parameters": {"label_key": self.label_key},
            "group_id": self.group_id,
        }


class _PrecisionRecallF1Base(BaseModel):
    label: Label
    value: float | None
    group: MetaDatum = None
    group_id: int = None

    @validator("value")
    def replace_nan_with_neg_1(cls, v):
        if np.isnan(v):
            return -1
        return v

    def db_mapping(self, label_id: int, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "label_id": label_id,
            "type": self.__type__,
            "evaluation_settings_id": evaluation_settings_id,
            "group_id": self.group_id,
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
    group: MetaDatum = None
    group_id: int = None

    def db_mapping(self, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "type": "ROCAUC",
            "parameters": {"label_key": self.label_key},
            "evaluation_settings_id": evaluation_settings_id,
            "group_id": self.group_id,
        }
