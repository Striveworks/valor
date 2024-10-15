from datetime import date, datetime, time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from valor_api.schemas.geojson import GeoJSON
from valor_api.schemas.validators import validate_string_identifier


class Metadata(BaseModel):
    """
    Metadata schema.

    Attributes
    ----------
    geospatial : GeoJSON, optional
        A geospatial point or region represented by GeoJSON.
    datetime : str, optional
        A datetime in ISO string format.
    date : str, optional
        A date in ISO string format.
    time : str, optional
        A time in ISO string format.
    string : dict[str, str]
        A dictionary containing string-type values.
    numeric : dict[str, int | float], optional
        A dictionary containing numeric values.
    """

    geospatial: dict[str, GeoJSON]
    datetime: dict[str, "datetime"] = Field(default_factory=dict)
    date: dict[str, "date"] = Field(default_factory=dict)
    time: dict[str, "time"] = Field(default_factory=dict)
    string: dict[str, str] = Field(default_factory=dict)
    integer: dict[str, int] = Field(default_factory=dict)
    floating: dict[str, float] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, kwargs: Any) -> Any:
        for key in kwargs.keys():
            validate_string_identifier(key)
            for k, v in kwargs[key].items():
                if key == "datetime":
                    kwargs[key][k] = datetime.fromisoformat(v)
                elif key == "date":
                    kwargs[key][k] = date.fromisoformat(v)
                elif key == "time":
                    kwargs[key][k] = time.fromisoformat(v)
        return kwargs
