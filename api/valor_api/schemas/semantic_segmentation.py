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


class SemanticBitmask(BaseModel):
    mask: np.ndarray
    label: str
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
            raise ValueError("Semantic bitmask takes an encoded bitmask as input.")
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


class SemanticSegmentation(BaseModel):
    groundtruths: list[SemanticBitmask]
    predictions: list[SemanticBitmask]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> Self:
        validate_string_identifier(self.uid)
        for groundtruth in self.groundtruths:
            validate_string_identifier(groundtruth.label)
        for prediction in self.predictions:
            validate_string_identifier(prediction.label)
        return self
