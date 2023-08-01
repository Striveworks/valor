from uuid import uuid4

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Extra,
    Field,
    field_validator,
    model_validator,
)

from velour_api.enums import AnnotationType, JobStatus, TaskType
from velour_api.schemas.label import Label
from velour_api.schemas.metadata import MetaDatum


# @TODO: Implement later for metrics + metadata overhaul
class MetricFilter(BaseModel):
    datum_uids: list[str] = []
    task_types: list[TaskType] = []
    annotation_types: list[AnnotationType] = []
    labels: list[Label] = []
    label_keys: list[str] = []
    metadata: list[MetaDatum] = []
    min_area: float | None = None
    max_area: float | None = None
    allow_conversion: bool = True


class EvaluationSettings(BaseModel):
    """General parameters defining any filters of the data such
    as model, dataset, groundtruth and prediction type, model, dataset,
    size constraints, coincidence/intersection constraints, etc.
    """

    model: str
    dataset: str
    gt_type: AnnotationType | None = None
    pd_type: AnnotationType | None = None
    task_type: TaskType | None = None
    min_area: float | None = None
    max_area: float | None = None
    group_by: str | None = None
    label_key: str | None = None
    id: int | None = None


class APRequest(BaseModel):
    """Request to compute average precision"""

    settings: EvaluationSettings
    # (mutable defaults are ok for pydantic models)
    iou_thresholds: list[float] = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    ious_to_keep: set[float] = {0.5, 0.75}

    @model_validator(mode="after")
    @classmethod
    def check_ious(cls, values):
        for iou in values.ious_to_keep:
            if iou not in values.iou_thresholds:
                raise ValueError(
                    "`ious_to_keep` must be contained in `iou_thresholds`"
                )
        return values


class CreateAPMetricsResponse(BaseModel):
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


class ClfMetricsRequest(BaseModel):
    settings: EvaluationSettings


class Metric(BaseModel):
    """This is used for responses from the API"""

    type: str
    parameters: dict | None = None
    value: float | dict | None = None
    label: Label | None = None
    group: MetaDatum | None = None


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
    # TODO[pydantic]: The following keys were removed: `allow_mutation`.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-config for more information.
    model_config = ConfigDict(frozen=True)


class _BaseConfusionMatrix(BaseModel):
    label_key: str
    entries: list[ConfusionMatrixEntry]
    group: MetaDatum | None = None


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
    group: MetaDatum | None = None

    def db_mapping(self, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "type": "Accuracy",
            "evaluation_settings_id": evaluation_settings_id,
            "parameters": {"label_key": self.label_key},
            "group": self.group,
        }


class _PrecisionRecallF1Base(BaseModel):
    label: Label
    value: float | None = None
    group: MetaDatum | None = None

    @field_validator("value")
    @classmethod
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
            "group": self.group,
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
    group: MetaDatum | None = None

    def db_mapping(self, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "type": "ROCAUC",
            "parameters": {"label_key": self.label_key},
            "evaluation_settings_id": evaluation_settings_id,
            "group": self.group,
        }
