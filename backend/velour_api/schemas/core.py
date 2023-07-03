import re
from typing import Optional

from pydantic import BaseModel, validator

from velour_api.enums import AnnotationType, DatumTypes, TaskType
from velour_api.schemas.annotation import GeometricAnnotation
from velour_api.schemas.label import Label, ScoredLabel
from velour_api.schemas.metadata import MetaDatum


def format_name(name: str):
    allowed_special = ["-", "_"]
    pattern = re.compile(f"[^a-zA-Z0-9{''.join(allowed_special)}]")
    return re.sub(pattern, "", name)


class DatasetInfo(BaseModel):
    id: int = None
    name: str

    @validator("name")
    def check_name_valid(cls, v):
        v = format_name(v)
        if len(v) == 0:
            raise ValueError
        return v


class ModelInfo(BaseModel):
    id: int = None
    name: str

    @validator("name")
    def check_name_valid(cls, v):
        v = format_name(v)
        if len(v) == 0:
            raise ValueError
        return v


class Datum(BaseModel):
    dataset_id: DatasetInfo
    uid: str
    metadata: list[MetaDatum] = []


class GroundTruth(BaseModel):
    task_type: Optional[TaskType] = None
    labels: list[Label]
    annotation: Optional[GeometricAnnotation] = None


class Prediction(BaseModel):
    model_id: ModelInfo
    task_type: TaskType
    scored_labels: list[ScoredLabel]
    annotation: Optional[GeometricAnnotation] = None

    @validator("scored_labels")
    def check_sum_to_one(cls, v: list[ScoredLabel]):
        label_keys_to_sum = {}
        for scored_label in v:
            label_key = scored_label.label.key
            if label_key not in label_keys_to_sum:
                label_keys_to_sum[label_key] = 0.0
            label_keys_to_sum[label_key] += scored_label.score

        for k, total_score in label_keys_to_sum.items():
            if abs(total_score - 1) > 1e-5:
                raise ValueError(
                    "For each label key, prediction scores must sum to 1, but"
                    f" for label key {k} got scores summing to {total_score}."
                )
        return v


class DatumGroundTruth(BaseModel):
    datum: Datum
    gts: list[GroundTruth]


class DatumPrediction(BaseModel):
    datum: Datum
    pds: list[Prediction]


class Dataset(BaseModel):
    dataset_id: DatasetInfo
    datums: list[DatumGroundTruth]
    metadata: list[MetaDatum]


class Model(BaseModel):
    model_id: ModelInfo
    dataset_id: DatasetInfo
    datums: list[DatumPrediction]
    metadata: list[MetaDatum]
