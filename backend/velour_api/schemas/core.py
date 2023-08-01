import re

from pydantic import BaseModel, field_validator, model_validator

from velour_api import enums
from velour_api.schemas.geojson import GeoJSON
from velour_api.schemas.geometry import (
    BoundingBox,
    MultiPolygon,
    Polygon,
    Raster,
)
from velour_api.schemas.label import Label, ScoredLabel


def _format_name(name: str):
    allowed_special = ["-", "_"]
    pattern = re.compile(f"[^a-zA-Z0-9{''.join(allowed_special)}]")
    return re.sub(pattern, "", name)


def _format_uid(uid: str):
    allowed_special = ["-", "_", "/", "."]
    pattern = re.compile(f"[^a-zA-Z0-9{''.join(allowed_special)}]")
    return re.sub(pattern, "", uid)


class MetaDatum(BaseModel):
    key: str
    value: float | str | GeoJSON

    @field_validator("key")
    @classmethod
    def check_key(cls, v):
        if not isinstance(v, str):
            raise ValueError
        return v

    @property
    def string_value(self) -> str | None:
        if isinstance(self.value, str):
            return self.value
        return None

    @property
    def numeric_value(self) -> float | None:
        if isinstance(self.value, float):
            return self.value
        return None

    # @property
    # def geo(self) ->


class Dataset(BaseModel):
    id: int = None
    name: str
    metadata: list[MetaDatum] = []

    @field_validator("name")
    @classmethod
    def check_name_valid(cls, v):
        if v != _format_name(v):
            raise ValueError("name includes illegal characters.")
        if not v:
            raise ValueError("invalid string")
        return v


class Model(BaseModel):
    id: int = None
    name: str
    metadata: list[MetaDatum] = []

    @field_validator("name")
    @classmethod
    def check_name_valid(cls, v):
        if v != _format_name(v):
            raise ValueError("name includes illegal characters.")
        if not v:
            raise ValueError("invalid string")
        return v


class Datum(BaseModel):
    uid: str
    metadata: list[MetaDatum] = []

    @field_validator("uid")
    @classmethod
    def format_uid(cls, v):
        if v != _format_uid(v):
            raise ValueError("uid includes illegal characters.")
        if not v:
            raise ValueError("invalid string")
        return v


class Annotation(BaseModel):
    task_type: enums.TaskType
    labels: list[Label]
    metadata: list[MetaDatum] = None

    # Geometric types
    bounding_box: BoundingBox = None
    polygon: Polygon = None
    multipolygon: MultiPolygon = None
    raster: Raster = None

    @field_validator("labels")
    @classmethod
    def check_labels(cls, v):
        if not v:
            raise ValueError
        return v


class ScoredAnnotation(BaseModel):
    task_type: enums.TaskType
    scored_labels: list[ScoredLabel]
    metadata: list[MetaDatum] = None

    # Geometric types
    bounding_box: BoundingBox = None
    polygon: Polygon = None
    multipolygon: MultiPolygon = None
    raster: Raster = None

    @model_validator(skip_on_failure=True)
    @classmethod
    def check_sum_to_one_if_classification(cls, values):
        if values["task_type"] == enums.TaskType.CLASSIFICATION:
            label_keys_to_sum = {}
            for scored_label in values["scored_labels"]:
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
        return values


class GroundTruth(BaseModel):
    dataset_name: str
    datum: Datum
    annotations: list[Annotation]

    @field_validator("dataset_name")
    @classmethod
    def check_name_valid(cls, v):
        if v != _format_name(v):
            raise ValueError("name includes illegal characters.")
        if not v:
            raise ValueError("invalid string")
        return v

    @field_validator("annotations")
    @classmethod
    def check_annotations(cls, v):
        if not v:
            raise ValueError("annotations is empty")
        return v


class Prediction(BaseModel):
    model_name: str
    datum: Datum
    annotations: list[ScoredAnnotation]

    @field_validator("model_name")
    @classmethod
    def check_name_valid(cls, v):
        if v != _format_name(v):
            raise ValueError("name includes illegal characters.")
        if not v:
            raise ValueError("invalid string")
        return v

    @field_validator("annotations")
    @classmethod
    def check_annotations(cls, v):
        if not v:
            raise ValueError("annotations is empty")
        return v
