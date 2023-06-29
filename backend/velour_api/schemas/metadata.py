import json
from typing import Optional

from pydantic import BaseModel, validator

from velour_api.schemas.annotation import GeometricAnnotation
from velour_api.schemas.core import Dataset, Datum, Model


class GeographicFeature(BaseModel):
    geography: dict

    @validator("geography")
    def check_value(cls, v):
        if not isinstance(v, dict):
            raise ValueError
        else:
            # TODO: add more validation that the dict is valid geoJSON?
            json.dumps(v)
        return v


class ImageMetadata(BaseModel):
    height: int
    width: int
    frame: int


class MetaDatum(BaseModel):
    name: str
    value: int | float | str | GeographicFeature | ImageMetadata

    @validator("name")
    def check_name(cls, v):
        if not isinstance(v, str):
            raise ValueError
        return v


class Datum(BaseModel):
    uid: str
    metadata: list[MetaDatum] = []
