from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from valor_api.schemas.metadata import Metadata
from valor_api.schemas.object_detection import (
    BoundingBox,
    InstanceBitmask,
    InstancePolygon,
)
from valor_api.schemas.semantic_segmentation import SemanticBitmask
from valor_api.schemas.validators import validate_string_identifier


class Dataset(BaseModel):
    name: str
    metadata: Metadata
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> Self:
        validate_string_identifier(self.name)
        return self


class Model(BaseModel):
    name: str
    metadata: Metadata
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> Self:
        validate_string_identifier(self.name)
        return self


class Classification(BaseModel):
    uid: str
    groundtruth: str
    predictions: list[str]
    scores: list[float]
    metadata: Metadata
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> Self:
        validate_string_identifier(self.uid)
        validate_string_identifier(self.groundtruth)
        for prediction in self.predictions:
            validate_string_identifier(prediction)
        return self


class ObjectDetection(BaseModel):
    uid: str
    groundtruths: list[BoundingBox] | list[InstancePolygon] | list[
        InstanceBitmask
    ]
    predictions: list[BoundingBox] | list[InstancePolygon] | list[
        InstanceBitmask
    ]
    metadata: Metadata
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> Self:
        validate_string_identifier(self.uid)
        for groundtruth in self.groundtruths:
            for label in groundtruth.labels:
                validate_string_identifier(label)
        for prediction in self.predictions:
            for label in prediction.labels:
                validate_string_identifier(label)
        return self


class SemanticSegmentation(BaseModel):
    uid: str
    groundtruths: list[SemanticBitmask]
    predictions: list[SemanticBitmask]
    metadata: Metadata
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> Self:
        validate_string_identifier(self.uid)
        for groundtruth in self.groundtruths:
            validate_string_identifier(groundtruth.label)
        for prediction in self.predictions:
            validate_string_identifier(prediction.label)
        return self
