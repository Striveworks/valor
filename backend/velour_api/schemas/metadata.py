import json
from pydantic import BaseModel, validator


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


class MetaDatum(BaseModel):
    name: str
    value: int | float | str | GeographicFeature

    @validator("name")
    def check_name(cls, v):
        if not isinstance(v, str):
            raise ValueError
        return v

