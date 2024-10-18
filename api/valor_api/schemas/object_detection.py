import io
from base64 import b64decode
from typing import Any

import numpy as np
import PIL.Image
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from valor_api.schemas.metadata import Metadata
from valor_api.schemas.validators import validate_string_identifier


class BoundingBox(BaseModel):
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    labels: list[str]
    scores: list[float]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_passwords_match(self) -> Self:
        n_labels = len(self.labels)
        n_scores = len(self.scores)
        if n_scores > 0 and n_scores != n_labels:
            raise ValueError("Scores should be empty or enumerate per label.")
        return self


class InstancePolygon(BaseModel):
    shape: list
    labels: list[str]
    scores: list[float]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_passwords_match(self) -> Self:
        n_labels = len(self.labels)
        n_scores = len(self.scores)
        if n_scores > 0 and n_scores != n_labels:
            raise ValueError("Scores should be empty or enumerate per label.")
        return self


class InstanceBitmask(BaseModel):
    mask: np.ndarray
    labels: list[str]
    scores: list[float]
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _decode_mask(cls, kwargs: Any) -> Any:
        """Decode the encoded byte string into a NumPy mask."""
        encoding = kwargs["mask"]
        if not isinstance(encoding, str):
            raise ValueError("Instance bitmask takes an encoded bitmask as input.")
        f = io.BytesIO(b64decode(encoding))
        img = PIL.Image.open(f)
        f.close()
        if img.format != "PNG":
            raise ValueError(
                f"Expected image format PNG but got {img.format}."
            )
        if img.mode != "1":
            raise ValueError(
                f"Expected image mode to be binary but got mode {img.mode}."
            )
        kwargs["mask"] = np.array(img, dtype=np.bool_)
        return kwargs

    @model_validator(mode="after")
    def _check_labels_match_scores(self) -> Self:
        n_labels = len(self.labels)
        n_scores = len(self.scores)
        if n_scores > 0 and n_scores != n_labels:
            raise ValueError("Scores should be empty or enumerate per label.")
        return self


class ObjectDetection(BaseModel):
    groundtruths: list[BoundingBox] | list[InstancePolygon] | list[
        InstanceBitmask
    ]
    predictions: list[BoundingBox] | list[InstancePolygon] | list[
        InstanceBitmask
    ]
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
