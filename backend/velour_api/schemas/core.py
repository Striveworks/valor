import re
from typing import Optional

from pydantic import BaseModel, validator

from velour_api.enums import AnnotationType, DatumTypes, TaskType
from velour_api.schemas.geometry import (
    BoundingBox,
    MultiPolygon,
    Polygon,
    Raster,
)
from velour_api.schemas.label import Label, ScoredLabel
from velour_api.schemas.metadata import MetaDatum


def format_name(name: str):
    allowed_special = ["-", "_"]
    pattern = re.compile(f"[^a-zA-Z0-9{''.join(allowed_special)}]")
    return re.sub(pattern, "", name)


class Dataset(BaseModel):
    id: int = None
    name: str
    metadata: list[MetaDatum]

    @validator("name")
    def check_name_valid(cls, v):
        v = format_name(v)
        if len(v) == 0:
            raise ValueError
        return v


class Model(BaseModel):
    id: int = None
    name: str
    metadata: list[MetaDatum]

    @validator("name")
    def check_name_valid(cls, v):
        v = format_name(v)
        if len(v) == 0:
            raise ValueError
        return v


class Datum(BaseModel):
    uid: str
    metadata: list[MetaDatum] = []


class Annotation(BaseModel):
    task_type: str
    metadata: list[MetaDatum] = None

    # Geometric types``
    bounding_box: BoundingBox = None
    polygon: Polygon = None
    multipolygon: MultiPolygon = None
    raster: Raster = None


class GroundTruthAnnotation(BaseModel):
    labels: list[Label]
    annotation: Annotation


class PredictedAnnotation(BaseModel):
    scored_labels: list[ScoredLabel]
    annotation: Annotation

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


class GroundTruth(BaseModel):
    dataset_name: str
    datum: Datum
    annotations: list[GroundTruthAnnotation]

    @validator("dataset_name")
    def check_name_valid(cls, v):
        v = format_name(v)
        if len(v) == 0:
            raise ValueError
        return v


class Prediction(BaseModel):
    model_name: str
    datum: Datum
    annotations: list[PredictedAnnotation]

    @validator("model_name")
    def check_name_valid(cls, v):
        v = format_name(v)
        if len(v) == 0:
            raise ValueError
        return v