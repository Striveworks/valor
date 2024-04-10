import math
from typing import Union

from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)

from valor_api.enums import TaskType
from valor_api.schemas.geometry import Box, Polygon, Raster
from valor_api.schemas.validators import (
    validate_annotation_by_task_type,
    validate_groundtruth_annotations,
    validate_metadata,
    validate_prediction_annotations,
    validate_type_string,
)

GeometryType = Union[
    tuple[float, float],
    list[tuple[float, float]],
    list[list[tuple[float, float]]],
    list[list[list[tuple[float, float]]]],
]
MetadataType = dict[str, dict[str, bool | int | float | str | GeometryType]]


class Label(BaseModel):
    key: str
    value: str
    score: float | None = None
    model_config = ConfigDict(extra="forbid")

    def __eq__(self, other):
        """
        Defines how labels are compared to one another.

        Parameters
        ----------
        other : Label
            The object to compare with the label.

        Returns
        ----------
        bool
            A boolean describing whether the two objects are equal.
        """
        if (
            not hasattr(other, "key")
            or not hasattr(other, "key")
            or not hasattr(other, "score")
        ):
            return False

        # if the scores aren't the same type return False
        if (other.score is None) != (self.score is None):
            return False

        if self.score is None or other.score is None:
            scores_equal = other.score is None and self.score is None
        else:
            scores_equal = math.isclose(self.score, other.score)

        return (
            scores_equal
            and self.key == other.key
            and self.value == other.value
        )

    def __hash__(self) -> int:
        """
        Defines how a `Label` is hashed.

        Returns
        ----------
        int
            The hashed 'Label`.
        """
        return hash(f"key:{self.key},value:{self.value},score:{self.score}")


class Annotation(BaseModel):
    task_type: TaskType
    metadata: dict = dict()
    labels: list[Label] = list()
    bounding_box: Box | None = None
    polygon: Polygon | None = None
    raster: Raster | None = None
    embedding: list[float] | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    @classmethod
    def validate_by_task_type(cls, values):
        """Validates the annotation by task type."""
        return validate_annotation_by_task_type(values)

    @field_validator("metadata")
    @classmethod
    def validate_metadata_values(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_metadata(v)
        return v

    @field_serializer("bounding_box")
    def serialize_bounding_box(bounding_box: Box | None):  # type: ignore - pydantic field_serializer
        if bounding_box is None:
            return None
        return bounding_box.model_dump()["value"]

    @field_serializer("polygon")
    def serialize_polygon(polygon: Polygon | None):  # type: ignore - pydantic field_serializer
        if polygon is None:
            return None
        return polygon.model_dump()["value"]

    @field_serializer("raster")
    def serialize_raster(raster: Raster | None):  # type: ignore - pydantic field_serializer
        if raster is None:
            return None
        return raster.model_dump()["value"]


class Datum(BaseModel):
    uid: str
    metadata: dict = dict()
    model_config = ConfigDict(extra="forbid")

    @field_validator("uid")
    @classmethod
    def validate_uid(cls, v):
        """Validates the 'uid' field."""
        validate_type_string(v)
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata_values(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_metadata(v)
        return v


class GroundTruth(BaseModel):
    dataset_name: str
    datum: Datum
    annotations: list[Annotation]
    model_config = ConfigDict(extra="forbid")

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v):
        """Validates the 'dataset_name' field."""
        validate_type_string(v)
        return v

    @field_validator("annotations")
    @classmethod
    def validate_annotations(cls, v: list[Annotation]):
        """Validate annotations."""
        if not v:
            v = [Annotation(task_type=TaskType.EMPTY)]
        validate_groundtruth_annotations(v)
        return v


class Prediction(BaseModel):
    dataset_name: str
    model_name: str
    datum: Datum
    annotations: list[Annotation]
    model_config = ConfigDict(extra="forbid")

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v):
        """Validates the 'dataset_name' field."""
        validate_type_string(v)
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        """Validates the 'model_name' field."""
        validate_type_string(v)
        return v

    @field_validator("annotations")
    @classmethod
    def validate_annotations(cls, v: list[Annotation]):
        """Validate annotations."""
        if not v:
            v = [Annotation(task_type=TaskType.EMPTY)]
        validate_prediction_annotations(v)
        return v


class Dataset(BaseModel):
    name: str
    metadata: dict = dict()
    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Validates the 'name' field."""
        validate_type_string(v)
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata_values(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_metadata(v)
        return v


class Model(BaseModel):
    name: str
    metadata: dict = dict()
    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Validates the 'name' field."""
        validate_type_string(v)
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata_values(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_metadata(v)
        return v
