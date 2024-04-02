from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from valor_api.enums import TaskType
from valor_api.schemas.geojson import Polygon
from valor_api.schemas.validators import (
    validate_annotation_by_task_type,
    validate_dictionary,
    validate_groundtruth_annotations,
    validate_prediction_annotations,
    validate_string,
)


class Deserializer:
    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, data: Any) -> Any:
        if isinstance(data, dict) and set(data.keys()) == {"type", "value"}:
            if not data.get("type") == cls.__name__.lower():
                raise TypeError(
                    f"'{cls.__name__}' received value with type '{data.get('type')}'"
                )
            return data.get("value")
        return data


class Label(BaseModel, Deserializer):
    key: str
    value: str
    score: float | None = None
    model_config = ConfigDict(extra="forbid")


class Annotation(BaseModel, Deserializer):
    task_type: TaskType
    metadata: dict = dict()
    labels: list[Label] = list()
    box: Polygon | None = None
    polygon: Polygon | None = None
    raster: Any | None = None
    embedding: list[float] | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    @classmethod
    def validate_by_task_type(cls, values):
        """Validates the annotation by task type."""
        return validate_annotation_by_task_type(values)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_dictionary(v)
        return v


class Datum(BaseModel, Deserializer):
    uid: str
    metadata: dict = dict()
    model_config = ConfigDict(extra="forbid")

    @field_validator("uid")
    @classmethod
    def validate_uid(cls, v):
        """Validates the 'uid' field."""
        validate_string(v)
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_dictionary(v)
        return v


class GroundTruth(BaseModel, Deserializer):
    dataset_name: str
    datum: Datum
    annotations: list[Annotation]
    model_config = ConfigDict(extra="forbid")

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v):
        """Validates the 'dataset_name' field."""
        validate_string(v)
        return v

    @field_validator("annotations")
    @classmethod
    def validate_annotations(cls, v: list[Annotation]):
        """Validate annotations."""
        if not v:
            v = [Annotation(task_type=TaskType.EMPTY)]
        validate_groundtruth_annotations(v)
        return v


class Prediction(BaseModel, Deserializer):
    dataset_name: str
    model_name: str
    datum: Datum
    annotations: list[Annotation]
    model_config = ConfigDict(extra="forbid")

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v):
        """Validates the 'dataset_name' field."""
        validate_string(v)
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        """Validates the 'model_name' field."""
        validate_string(v)
        return v

    @field_validator("annotations")
    @classmethod
    def validate_annotations(cls, v: list[Annotation]):
        """Validate annotations."""
        if not v:
            v = [Annotation(task_type=TaskType.EMPTY)]
        validate_prediction_annotations(v)
        return v


class Dataset(BaseModel, Deserializer):
    name: str
    metadata: dict = dict()
    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Validates the 'name' field."""
        validate_string(v)
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_dictionary(v)
        return v


class Model(BaseModel, Deserializer):
    name: str
    metadata: dict = dict()
    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Validates the 'name' field."""
        validate_string(v)
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_dictionary(v)
        return v
