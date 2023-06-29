from typing import Optional

from pydantic import BaseModel, validator

from velour_api.enums import AnnotationType, DatumTypes, TaskType
from velour_api.schemas.annotation import GeometricAnnotation
from velour_api.schemas.label import Label, ScoredLabel
from velour_api.schemas.metadata import MetaDatum


def _validate_href(v: str | None):
    if v is None:
        return v
    if not (v.startswith("http://") or v.startswith("https://")):
        raise ValueError("`href` must start with http:// or https://")
    return v


class Datum(BaseModel):
    uid: str
    metadata: list[MetaDatum] = []


class Dataset(BaseModel):
    name: str
    href: str = None
    description: str = None
    type: DatumTypes
    label: Label
    metadata: list[MetaDatum]

    @validator("href")
    def validate_href(cls, v):
        return _validate_href(v)


class Model(BaseModel):
    name: str
    href: str = None
    description: str = None
    type: DatumTypes
    metadata: list[MetaDatum]

    @validator("href")
    def validate_href(cls, v):
        return _validate_href(v)


class GroundTruth(BaseModel):
    task_type: Optional[TaskType] = None
    labels: list[Label]
    annotation: Optional[GeometricAnnotation] = None
    metadata: list[MetaDatum]


class Prediction(BaseModel):
    task_type: TaskType
    scored_labels: list[ScoredLabel]
    annotation: Optional[GeometricAnnotation] = None
    metadata: list[MetaDatum]

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


class GroundTruthDatumCreate(BaseModel):
    datum: Datum
    gts: list[GroundTruth]


class DatasetCreate(BaseModel):
    dataset: Dataset
    datums: list[GroundTruthDatumCreate]


class PredictionDatumCreate(BaseModel):
    datum: Datum
    pds: list[Prediction]


class ModelCreate(BaseModel):
    model: Model
    dataset: Dataset
    datums: list[PredictionDatumCreate]
